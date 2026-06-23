#!/usr/bin/env python3
"""guitar-bt: remove the electric guitar from a song to make a backing track.

Runs the mix through a dedicated guitar-separation model (MelBand-Roformer
Guitar) and writes back everything *except* the guitar - a backing track you
can play guitar over. Use --isolate to keep only the guitar instead.

This tool does guitar and nothing else. For removing drums / bass / vocals,
use a general stem separator - there are plenty that do it faster and better.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Formats we transparently pre-decode with ffmpeg (video / containers / lossy).
INPUT_SUFFIXES = {
    ".mp3", ".mp4", ".m4a", ".aac", ".wav", ".flac", ".ogg",
    ".opus", ".wma", ".mkv", ".mov", ".webm", ".avi",
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def is_spotify_url(s: str) -> bool:
    return "spotify.com" in s or s.startswith("spotify:")


def _newest_audio_file(directory: Path) -> Path | None:
    files = [p for p in directory.iterdir()
             if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def download_with_ytdlp(url: str, dst_dir: Path) -> Path:
    """Download bestaudio from a YouTube (or other yt-dlp supported) URL.

    Returns the path to the downloaded audio file inside dst_dir. We do NOT
    transcode here - the existing ffmpeg decode step handles whatever container
    yt-dlp gives us (usually .webm/opus or .m4a).
    """
    try:
        import yt_dlp
    except ModuleNotFoundError:
        eprint("ERROR: yt-dlp is not installed. Run: "
               ".\\.venv\\Scripts\\python.exe -m pip install yt-dlp")
        sys.exit(2)

    class _Logger:
        def debug(self, msg):
            if not msg.startswith("[debug]"):
                eprint(msg)
        def info(self, msg): eprint(msg)
        def warning(self, msg): eprint(msg)
        def error(self, msg): eprint(msg)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,      # a playlist URL -> just the one selected track
        "logtostderr": True,     # keep our stdout clean (final path only)
        "logger": _Logger(),
        # YouTube needs a JS runtime for reliable format extraction. Enable any
        # of these that happen to be installed (node/deno/bun); harmless if none.
        "js_runtimes": {"node": {}, "deno": {}, "bun": {}},
    }
    eprint(f"Downloading audio from URL: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info:                # safety: playlist slipped through
                info = info["entries"][0]
            path = Path(ydl.prepare_filename(info))
    except Exception as e:
        eprint(f"ERROR: failed to download from URL: {e}")
        sys.exit(2)

    if not path.exists():
        # prepare_filename can disagree with the real extension; fall back.
        found = _newest_audio_file(dst_dir)
        if found is None:
            eprint("ERROR: download finished but no audio file was found.")
            sys.exit(2)
        path = found
    return path


def download_url(url: str, dst_dir: Path) -> Path:
    """Download a URL via yt-dlp (YouTube, SoundCloud, Bandcamp, ~1800 sites)."""
    if is_spotify_url(url):
        eprint("ERROR: Spotify links aren't supported. Spotify audio is "
               "DRM-protected, so it can only be fetched as a YouTube match "
               "anyway - just paste the YouTube link for the song directly.")
        sys.exit(2)
    return download_with_ytdlp(url, dst_dir)


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        eprint("ERROR: ffmpeg not found on PATH. Install it (https://ffmpeg.org) and retry.")
        sys.exit(2)
    return ffmpeg


def decode_to_wav(src: Path, ffmpeg: str, samplerate: int, dst: Path) -> None:
    """Decode any input (mp3/mp4/m4a/...) to a clean stereo WAV for separation."""
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vn",                      # ignore any video stream
        "-ac", "2",                 # force stereo
        "-ar", str(samplerate),     # match the model's sample rate
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        eprint("ERROR: ffmpeg failed to decode the input file:")
        eprint(proc.stderr.strip())
        sys.exit(2)


def write_output(mix, samplerate: int, out_path: Path, fmt: str,
                 bitrate: int, ffmpeg: str) -> None:
    """Write the separated audio to disk.

    `mix` is a torch tensor of shape (channels, samples). We use soundfile for
    WAV, transcoding to MP3 with ffmpeg when requested.
    """
    import numpy as np
    import soundfile as sf

    wav = mix.detach().cpu().numpy()
    # Prevent clipping: rescale if any sample exceeds full scale.
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    data = wav.T  # soundfile wants (samples, channels)

    if fmt == "wav":
        sf.write(str(out_path), data, samplerate, subtype="PCM_16")
        return

    # mp3: write a temp wav, then encode with ffmpeg.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "mix.wav"
        sf.write(str(tmp_wav), data, samplerate, subtype="PCM_16")
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(tmp_wav),
            "-b:a", f"{bitrate}k",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            eprint("ERROR: ffmpeg failed to encode MP3:")
            eprint(proc.stderr.strip())
            sys.exit(2)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="guitar-bt",
        description="Remove the electric guitar from a song to create a backing track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="Input audio/video file (mp3, mp4, m4a, wav, flac, ...) "
                        "OR a URL (YouTube, SoundCloud, Bandcamp, ...).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output file. Default: <input>_no_guitar.<format> next to the input.")
    p.add_argument("-f", "--format", choices=["mp3", "wav"], default=None,
                   help="Output format. Default: inferred from --output, else mp3.")
    p.add_argument("--bitrate", type=int, default=320,
                   help="MP3 bitrate in kbps (ignored for wav).")
    p.add_argument("--isolate", action="store_true",
                   help="Output ONLY the guitar track (e.g. to study the part) "
                        "instead of the backing track without it.")
    p.add_argument("--roformer-config", type=Path, default=None,
                   help="Path to the guitar model's YAML config "
                        "(default: models/mel_band_roformer_guitar/...).")
    p.add_argument("--roformer-ckpt", type=Path, default=None,
                   help="Path to the guitar model's .ckpt weights "
                        "(default: models/mel_band_roformer_guitar/...).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Compute device. 'auto' uses CUDA if available, else CPU.")
    return p.parse_args(argv)


def resolve_output(args: argparse.Namespace, base: Path) -> tuple[Path, str]:
    """Return (output_path, fmt).

    `base` is the path the default output name is derived from: the input file
    for a local input, or <cwd>/<track title> for a downloaded URL.
    """
    if args.output is not None:
        out = args.output
        ext = out.suffix.lower().lstrip(".")
        fmt = args.format or (ext if ext in ("mp3", "wav") else "mp3")
        if out.suffix.lower() not in (".mp3", ".wav"):
            out = out.with_suffix("." + fmt)
        return out, fmt
    fmt = args.format or "mp3"
    suffix = "guitar_only" if args.isolate else "no_guitar"
    out = base.with_name(f"{base.stem}_{suffix}.{fmt}")
    return out, fmt


def main(argv=None) -> int:
    args = parse_args(argv)

    is_url = looks_like_url(args.input)
    if not is_url:
        input_path = Path(args.input)
        if not input_path.exists():
            eprint(f"ERROR: input file not found: {input_path}")
            return 2
        if input_path.suffix.lower() not in INPUT_SUFFIXES:
            eprint(f"WARNING: unrecognized input extension '{input_path.suffix}'. "
                   "Trying anyway via ffmpeg.")

    ffmpeg = find_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Resolve the actual file to process. For a URL we download into the
        # temp dir and base the default output name on the track title, written
        # to the current working directory.
        if is_url:
            source_path = download_url(args.input, tmp_dir)
            eprint(f"Downloaded: {source_path.name}")
            base = Path.cwd() / source_path.name
        else:
            source_path = input_path
            base = input_path

        out_path, fmt = resolve_output(args, base)

        # Heavy imports deferred so --help stays instant.
        import torch
        import roformer_engine

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        eprint(f"Device: {device}")

        stems, samplerate = roformer_engine.separate(
            source_path, ffmpeg, decode_to_wav, device,
            config_path=args.roformer_config or roformer_engine.DEFAULT_CONFIG,
            ckpt_path=args.roformer_ckpt or roformer_engine.DEFAULT_CKPT,
        )
        mix = stems["guitar"] if args.isolate else stems["no_guitar"]
        eprint("Isolating: guitar" if args.isolate
               else "Removing: guitar  |  Keeping: everything else")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        eprint(f"Writing {fmt.upper()} -> {out_path}")
        write_output(mix, samplerate, out_path, fmt, args.bitrate, ffmpeg)

    eprint("Done.")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
