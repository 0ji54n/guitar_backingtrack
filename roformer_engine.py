"""Roformer guitar-separation engine.

Runs the dedicated "MelBand-Roformer Guitar (by becruily)" model, a Mel-Band
RoFormer trained specifically to split a mix into *guitar* and *everything else*.
It is markedly cleaner (less bleed) than a general-purpose stem separator's
guitar output, at roughly 1.3x the track length on CPU.

The model architecture is vendored in ``roformer_arch/`` (from ZFTurbo's
Music-Source-Separation-Training, which is the variant the checkpoint was
trained with). The chunked inference loop below is adapted from that project.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Default location of the downloaded model (see README "Roformer guitar model").
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "mel_band_roformer_guitar"
DEFAULT_CONFIG = DEFAULT_MODEL_DIR / "config_guitar_becruily.yaml"
DEFAULT_CKPT = DEFAULT_MODEL_DIR / "becruily_guitar.ckpt"


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_config(config_path: Path):
    """Load the model's YAML config.

    The config uses a ``!!python/tuple`` tag, so it needs the full (unsafe)
    loader. It is a trusted file we ship/download ourselves.
    """
    import yaml
    from ml_collections import ConfigDict

    with open(config_path) as f:
        cfg = ConfigDict(yaml.unsafe_load(f))
    return cfg


def build_model(config, ckpt_path: Path):
    """Construct the MelBandRoformer and strict-load the checkpoint weights."""
    import inspect

    from roformer_arch.mel_band_roformer import MelBandRoformer

    model_cfg = dict(config.model)
    if "multi_stft_resolutions_window_sizes" in model_cfg:
        model_cfg["multi_stft_resolutions_window_sizes"] = tuple(
            model_cfg["multi_stft_resolutions_window_sizes"]
        )
    # Drop any keys this architecture revision doesn't accept (forward-compat).
    accepted = set(inspect.signature(MelBandRoformer.__init__).parameters)
    model_cfg = {k: v for k, v in model_cfg.items() if k in accepted}

    model = MelBandRoformer(**model_cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _demix(config, model, mix: torch.Tensor, device: torch.device) -> np.ndarray:
    """Chunked separation of one track.

    `mix` is (channels, samples). Returns the target stem as a numpy array of
    the same shape. Adapted from ZFTurbo's Music-Source-Separation-Training.
    """
    chunk = config.audio.chunk_size
    overlap = config.inference.num_overlap
    step = chunk // overlap
    fade = chunk // 10
    border = chunk - step

    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    # Triangular-ish fade window so overlapping chunks cross-fade cleanly.
    window = torch.ones(chunk)
    window[-fade:] *= torch.linspace(1, 0, fade)
    window[:fade] *= torch.linspace(0, 1, fade)
    window = window.to(device)

    with torch.no_grad():
        mix = mix.to(device)
        result = torch.zeros((1,) + tuple(mix.shape), dtype=torch.float32, device=device)
        counter = torch.zeros((1,) + tuple(mix.shape), dtype=torch.float32, device=device)

        total = mix.shape[1]
        num_chunks = (total + step - 1) // step
        chunk_time = None
        i = 0
        while i < total:
            part = mix[:, i:i + chunk]
            length = part.shape[-1]
            if length < chunk:
                pad_mode = "reflect" if length > chunk // 2 + 1 else "constant"
                part = nn.functional.pad(part, (0, chunk - length), mode=pad_mode)

            t0 = time.time()
            x = model(part.unsqueeze(0))[0]
            if chunk_time is None:
                chunk_time = time.time() - t0
                _eprint(f"  ~{chunk_time * num_chunks:.0f}s estimated for this track")

            w = window.clone()
            if i == 0:
                w[:fade] = 1
            elif i + chunk >= total:
                w[-fade:] = 1

            result[..., i:i + length] += x[..., :length] * w[..., :length]
            counter[..., i:i + length] += w[..., :length]
            i += step
            done = i // step
            _eprint(f"\r  chunk {min(done, num_chunks)}/{num_chunks}", end="")
        _eprint()

        est = (result / counter).cpu().numpy()
    np.nan_to_num(est, copy=False, nan=0.0)
    if mix.shape[1] > 2 * border and border > 0:
        est = est[..., border:-border]
    return est[0]  # single target stem -> (channels, samples)


def separate(source_path: Path, ffmpeg: str, decode_to_wav, device: str,
             config_path: Path = DEFAULT_CONFIG, ckpt_path: Path = DEFAULT_CKPT):
    """Run guitar separation on `source_path`.

    `decode_to_wav(src, ffmpeg, samplerate, dst)` is injected from the caller to
    reuse its ffmpeg decode path. Returns ``(stems, samplerate)`` where `stems`
    is ``{"guitar": tensor, "no_guitar": tensor}`` of shape (channels, samples).
    """
    import tempfile

    import soundfile as sf

    if not config_path.exists() or not ckpt_path.exists():
        _eprint(
            "ERROR: Roformer guitar model not found. Expected:\n"
            f"  {config_path}\n  {ckpt_path}\n"
            "Download it once with the commands in the README "
            "(\"Roformer guitar model\")."
        )
        sys.exit(2)

    cfg = load_config(config_path)
    samplerate = int(cfg.audio.sample_rate)

    _eprint("Loading Roformer guitar model...")
    model = build_model(cfg, ckpt_path)
    dev = torch.device(device)
    model = model.to(dev)

    with tempfile.TemporaryDirectory() as tmp:
        wav_in = Path(tmp) / "input.wav"
        _eprint(f"Decoding '{source_path.name}' -> wav ({samplerate} Hz stereo)...")
        decode_to_wav(source_path, ffmpeg, samplerate, wav_in)
        mix, _ = sf.read(str(wav_in), dtype="float32")

    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=-1)
    mix = mix[:, :2]  # ensure stereo

    _eprint("Separating guitar (this can take a while on CPU)...")
    mix_t = torch.tensor(mix.T, dtype=torch.float32)  # (channels, samples)
    guitar = _demix(cfg, model, mix_t, dev)            # (channels, samples)
    no_guitar = mix.T - guitar

    stems = {
        "guitar": torch.from_numpy(guitar),
        "no_guitar": torch.from_numpy(no_guitar),
    }
    return stems, samplerate
