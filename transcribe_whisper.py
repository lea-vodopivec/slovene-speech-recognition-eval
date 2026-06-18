"""
Transcribe audio files with Whisper.
"""

import os
import re
import sys
import glob
import time
import whisper

INPUT_DIR = "./posnetki_preprocessed"
OUTPUT_DIR = "./transkripcije"
MODEL_SIZE = "large"  
LANGUAGE = "sl"


def get_audio_files(specific_file=None):
    if specific_file:
        filepath = os.path.join(INPUT_DIR, specific_file)
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            sys.exit(1)
        return [filepath]

    files = []
    for ext in ("mp3", "m4a", "wav", "ogg", "flac", "webm"):
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"posnetek*.{ext}")))

    if not files:
        print(f"[ERROR] No posnetek* audio files found in '{INPUT_DIR}/'")
        sys.exit(1)

    def sort_key(f):
        match = re.search(r"posnetek(\d+)", os.path.basename(f))
        return int(match.group(1)) if match else 0

    files.sort(key=sort_key)
    return files


def extract_number(filename):
    match = re.search(r"posnetek(\d+)", os.path.basename(filename))
    return match.group(1) if match else "0"


def transcribe_file(model, filepath):
    print(f"  Transcribing locally with Whisper ({MODEL_SIZE})...")

    result = model.transcribe(filepath, language=LANGUAGE)

    return result["text"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    specific_file = sys.argv[1] if len(sys.argv) > 1 else None
    audio_files = get_audio_files(specific_file)

    print("=" * 60)
    print("  WHISPER ASR TRANSCRIPTION (LOCAL)")
    print(f"  Model size: {MODEL_SIZE} | Language: Slovenian ({LANGUAGE})")
    print(f"  Files to transcribe: {len(audio_files)}")
    print("=" * 60)

    # Load the Whisper model (downloaded on first run)
    print(f"\nLoading Whisper model '{MODEL_SIZE}'...")
    model = whisper.load_model(MODEL_SIZE)
    print("Model loaded.\n")

    for i, filepath in enumerate(audio_files, 1):
        num = extract_number(filepath)
        filename = os.path.basename(filepath)
        output_filename = f"transkripcija{num}_whisper.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"[{i}/{len(audio_files)}] Processing: {filename}")

        start_time = time.time()

        try:
            text = transcribe_file(model, filepath)
            elapsed = time.time() - start_time

            # Save transcription to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  [OK] Saved: {output_filename} ({elapsed:.2f}s)")
            print(f"  Preview: {text[:100]}{'...' if len(text) > 100 else ''}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  [X] Error after {elapsed:.2f}s: {e}")

    print("\n" + "=" * 60)
    print("  Done! Transcriptions saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
