"""
Transcribe audio files with Voxtral.
"""

import os
import re
import sys
import torch
import tempfile
import subprocess
from pathlib import Path

AUDIO_DIR      = Path("posnetki")
OUTPUT_DIR     = Path("transkripcije")
MODEL_ID       = "mistralai/Voxtral-Small-24B-2507"
LANGUAGE       = "sl"
MAX_NEW_TOKENS = 8192


def convert_to_wav(input_path: Path, tmp_dir: str) -> str:
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    out_path = os.path.join(tmp_dir, input_path.stem + ".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-f", "wav", out_path,
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {input_path.name}:\n{result.stderr.decode()}"
        )
    return out_path


def natural_sort_key(path: Path):
    """Sort posnetek1 ... posnetek27 numerically."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_model():
    print(f"Loading model: {MODEL_ID}")
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def transcribe_file(audio_path: Path, model, processor, tmp_dir: str) -> str:
    """Convert audio and run Voxtral transcription."""
    wav_path = str(audio_path)

    inputs = processor.apply_transcription_request(
        language=LANGUAGE,
        audio=wav_path,
        model_id=MODEL_ID,
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_len:]
    transcript = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return transcript


def main():
    if not AUDIO_DIR.exists():
        sys.exit(f"ERROR: audio directory '{AUDIO_DIR}' not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4"}
    audio_files = sorted(
        [f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in extensions],
        key=natural_sort_key,
    )

    if not audio_files:
        sys.exit(f"No audio files found in '{AUDIO_DIR}'.")

    print(f"Found {len(audio_files)} audio file(s).")
    print("=" * 60)

    model, processor = load_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, audio_path in enumerate(audio_files, start=1):
            out_path = OUTPUT_DIR / f"transkripcija{idx}_voxtral.txt"
            print(f"\n[{idx}/{len(audio_files)}] {audio_path.name}  →  {out_path.name}")

            try:
                transcript = transcribe_file(audio_path, model, processor, tmp_dir)
                out_path.write_text(transcript, encoding="utf-8")
                preview = transcript[:120].replace("\n", " ")
                print(f"    ✓ {len(transcript)} chars  |  \"{preview}...\"")
            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
                out_path.write_text(f"[TRANSCRIPTION FAILED: {e}]", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Done. Transcripts saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
