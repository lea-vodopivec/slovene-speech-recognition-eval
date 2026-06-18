"""
Preprocess audio files for fair ASR comparison.
Converts all files in posnetki/ to 16kHz mono WAV
and saves them to posnetki_preprocessed/.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("posnetki")
OUTPUT_DIR = Path("posnetki_preprocessed")
SAMPLE_RATE = 16000
CHANNELS    = 1
# ─────────────────────────────────────────────────────────────────────────────

EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4"}


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-f", "wav",
        str(output_path),
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {input_path.name}:\n{result.stderr.decode()}"
        )


def main():
    if not INPUT_DIR.exists():
        sys.exit(f"ERROR: input directory '{INPUT_DIR}' not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in EXTENSIONS],
        key=natural_sort_key,
    )

    if not audio_files:
        sys.exit(f"No audio files found in '{INPUT_DIR}'.")

    print(f"Found {len(audio_files)} file(s) in '{INPUT_DIR}'.")
    print(f"Output: '{OUTPUT_DIR}/'  ({SAMPLE_RATE}Hz, mono, WAV)")
    print("=" * 60)

    ok, failed = 0, 0
    for i, src in enumerate(audio_files, 1):
        dst = OUTPUT_DIR / (src.stem + ".wav")
        print(f"[{i}/{len(audio_files)}] {src.name}  →  {dst.name}", end="  ")
        try:
            convert_to_wav(src, dst)
            size_kb = dst.stat().st_size // 1024
            print(f"✓  ({size_kb} KB)")
            ok += 1
        except RuntimeError as e:
            print(f"✗  FAILED\n  {e}")
            failed += 1

    print("=" * 60)
    print(f"Done.  {ok} converted, {failed} failed.")
    print(f"Point both Voxtral and Whisper at '{OUTPUT_DIR}/' for comparison.")


if __name__ == "__main__":
    main()
