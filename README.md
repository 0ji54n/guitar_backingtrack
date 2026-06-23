# guitar-bt — Guitar Backing Track Maker

A small CLI that takes a song — a local file (mp3, mp4, m4a, wav, flac, …) **or a
URL** (YouTube, SoundCloud, Bandcamp, …) — and produces a copy with the
**electric guitar removed**, leaving vocals, drums, bass, keys and everything
else — i.e. a backing track you can play guitar over.

It does **one thing: guitar**. The mix is run through a model trained
specifically to separate guitar from everything else (**MelBand-Roformer Guitar**
by becruily), and everything *except* the guitar is written back out. Use
`--isolate` to keep only the guitar instead.

## Requirements

- Python 3.10+ (tested on 3.14)
- [ffmpeg](https://ffmpeg.org) on your `PATH`
- The guitar model (~45 MB) — a **one-time download**, see [below](#guitar-model-one-time-download)
- For URL input: `yt-dlp` (installed via `requirements.txt`)
- **Recommended for YouTube:** a JavaScript runtime so yt-dlp can reliably
  extract formats. Node.js, [Deno](https://deno.com), or
  [Bun](https://bun.sh) all work — the tool auto-detects any of them. With scoop:
  `scoop install deno`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available; otherwise it runs on CPU
(expect a few minutes per song — that's normal).

## Guitar model (one-time download)

The model weights aren't bundled. Download them once into
`models\mel_band_roformer_guitar\`:

```powershell
# Run from the repo root.
$dir = "models\mel_band_roformer_guitar"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$base = "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main"
Invoke-WebRequest "$base/config_guitar_becruily.yaml" -OutFile "$dir\config_guitar_becruily.yaml"
Invoke-WebRequest "$base/becruily_guitar.ckpt"        -OutFile "$dir\becruily_guitar.ckpt"
```

The model architecture lives in `roformer_arch/` (vendored from
[ZFTurbo's Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
the variant these weights were trained with). File paths can be overridden with
`--roformer-config` / `--roformer-ckpt`.

## Usage

```powershell
# Simplest: makes song_no_guitar.mp3 next to the input
python remove_guitar.py "song.mp3"

# Straight from a YouTube link — output lands in the current folder,
# named after the video title (e.g. "Song Title_no_guitar.mp3")
# (also works for SoundCloud, Bandcamp, and ~1800 other sites)
python remove_guitar.py "https://www.youtube.com/watch?v=..."

# From a video, to WAV, at a chosen path
python remove_guitar.py "live.mp4" -o backing.wav

# Get ONLY the guitar track (e.g. to study the part)
python remove_guitar.py "song.mp3" --isolate -o guitar.wav
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Input audio/video file **or a URL** (YouTube, SoundCloud, Bandcamp, …) |
| `-o, --output` | `<input>_no_guitar.<fmt>` | Output file path |
| `-f, --format` | `mp3` (or inferred from `-o`) | `mp3` or `wav` |
| `--bitrate` | `320` | MP3 bitrate (kbps) |
| `--isolate` | off | Output only the guitar track instead of the backing track |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--roformer-config` / `--roformer-ckpt` | auto | Override the model file locations |

## URL input

- **YouTube / SoundCloud / Bandcamp / ~1800 sites** are downloaded with
  [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) (best available audio), then
  processed normally.
- **Spotify links are not supported.** Spotify audio is DRM-protected and can
  only ever be fetched as a YouTube match anyway — just paste the YouTube link
  for the song directly.
- For URL input the output is written to the **current folder**, named after the
  track title, unless you pass `-o`.
- If a YouTube download fails with a "JavaScript runtime" error, install one (see
  Requirements).

## Speed notes (CPU)

Separation on CPU takes roughly **1.3–1.5× the length of the song** — a few
minutes per track is normal, not a hang. The biggest speedup would be a GPU: this
project auto-uses an NVIDIA GPU if present, but on a CPU-only machine a free
cloud GPU (Google Colab) or an online service like [MVSEP](https://mvsep.com)
will be dramatically faster if speed matters more than keeping everything local.

## Notes & limitations

- Source separation is not perfect: some guitar may remain, and a little of
  other instruments can bleed into the guitar track. Acoustic guitar is often
  picked up as guitar too. Results vary by song.
- This tool removes **guitar only**, by design.

## Credits

- **Guitar model:** [MelBand-Roformer Guitar by becruily](https://huggingface.co/becruily/mel-band-roformer-guitar)
  (downloaded at runtime; not redistributed here — check its page for the model's
  own terms of use).
- **Model architecture** (`roformer_arch/`, vendored): [ZFTurbo's Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
  derived from [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer).
  Both MIT-licensed — see [`roformer_arch/NOTICE.md`](roformer_arch/NOTICE.md).
- Downloads via [`yt-dlp`](https://github.com/yt-dlp/yt-dlp); audio I/O via
  [ffmpeg](https://ffmpeg.org).

## License

[MIT](LICENSE) for this project's own code. The vendored model architecture
keeps its upstream MIT notices (see [`roformer_arch/NOTICE.md`](roformer_arch/NOTICE.md)).
The guitar model weights are downloaded separately and governed by their own
terms on Hugging Face.
