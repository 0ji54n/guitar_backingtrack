---
name: backing-track
description: Make a guitar backing track by removing the electric guitar from a song. Use when the user wants to process an audio/video file OR a URL (YouTube, SoundCloud, Bandcamp) with remove_guitar.py, or types /backing-track <file-or-url>. Pass the input path/URL and any flags as arguments.
---

# backing-track

Run the project's guitar-removal CLI on a song to produce a backing track (the mix
with the electric guitar stem removed). The input can be a local file or a URL.

## How to run

Use the repo's virtualenv Python directly — do NOT rely on the venv being activated:

```
.\.venv\Scripts\python.exe remove_guitar.py <ARGUMENTS>
```

`$ARGUMENTS` is the input file path **or URL** plus any optional flags the user passed.

- If the user gave only a file path, run with defaults (outputs `<song>_no_guitar.mp3`
  next to the input).
- If the user gave a URL (YouTube, SoundCloud, Bandcamp, …), pass it through the same
  way — the CLI downloads it automatically. Output lands in the current folder,
  named after the track title. Quote the URL.
- Spotify links are NOT supported — the CLI rejects them. If the user pastes one,
  tell them to paste the YouTube link for the song instead.
- Common flags to pass through if requested: `-o <out>`, `-f wav|mp3`, `--isolate`
  (output only the guitar track instead of the backing track).
- This tool removes **guitar only**. If the user asks to also strip drums/bass/vocals,
  tell them that's out of scope here and point them at a general stem separator.
- See `python remove_guitar.py --help` for the full list.

## Notes

- Quote paths and URLs (they often contain spaces or special characters).
- URL input needs `yt-dlp` (in requirements.txt) and, for YouTube, a JS
  runtime (node/deno/bun) on PATH for reliable extraction.
- The guitar model (~45 MB) must be downloaded once into
  `models\mel_band_roformer_guitar\` (see README "Guitar model"). If the CLI errors
  that the model is missing, run the download commands from the README first.
- Separation runs on CPU and takes a few minutes per song — this is expected, not a hang.
  Let it finish; the progress bar shows it working.
- Requires ffmpeg on PATH (used to decode input and encode MP3).
- After it finishes, the CLI prints the output file path on stdout — report that to the user.
