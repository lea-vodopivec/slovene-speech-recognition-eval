# Comparative Evaluation of ASR Systems for Slovenian

This repository contains a corpus, scripts and evaluation code used to compare two speech-to-text systems on Slovenian speech: OpenAI Whisper (large) and MistralAI / Voxtral Small 24B.

**Project Goal:** Evaluate lexical and semantic transcription quality for Slovenian (low-resource language) and compare a classic transformer ASR (Whisper) with a multimodal LLM-based STT (Voxtral Small 24B).

**Contents**
- **Files & scripts**
  - [preprocess_audio.py](preprocess_audio.py) — convert and normalize audio to 16 kHz mono WAVs.
  - [transcribe_whisper.py](transcribe_whisper.py) — batch transcription using Whisper (large).
  - [transcribe_voxtral.py](transcribe_voxtral.py) — batch transcription using Voxtral Small 24B (Transformers/PyTorch).
  - [evaluate_transcriptions.py](evaluate_transcriptions.py) — computes evaluation metrics comparing automatic transcripts to reference transcripts.
  - [requirements.txt](requirements.txt) — Python dependencies.

- **Data directories**
  - `posnetki/` — original audio files (M4A AAC, stereo, 48 kHz).
  - `posnetki_preprocessed/` — preprocessed WAV (PCM, 16 kHz, mono) used as model inputs.
  - `original-transkript/` — human reference transcripts (UTF-8 TXT files: besedilo1.txt … besedilo27.txt).
  - `transkripcije/` — generated transcripts (files named e.g. transkripcija1_whisper.txt, transkripcija1_voxtral.txt).
  - `rezultati/` — evaluation outputs, summary CSVs and report.

**Setup**
1. Create and activate a Python 3.10+ virtual environment.

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure `ffmpeg` is installed and available in PATH (used by preprocessing scripts).

Usage
1. Preprocess audio (creates WAV files in `posnetki_preprocessed/`):

```bash
python preprocess_audio.py
```

2. Generate transcripts with Whisper:

```bash
python transcribe_whisper.py
```

3. Generate transcripts with Voxtral:

```bash
python transcribe_voxtral.py
```

4. Evaluate transcripts (produces CSVs and `evaluation_report.md` in `rezultati/`):

```bash
python evaluate_transcriptions.py
```

Evaluation metrics
- WER (Word Error Rate)
- CER (Character Error Rate)
- SER (Sentence / Semantic Error Rate) — sentence-level semantic similarity using a multilingual semantic model and cosine similarity (thresholding).
- BLEU — n-gram overlap score.
- WIP / WIL — Word Information Preserved / Lost.
- NER accuracy — named-entity overlap (spaCy multilingual model).
- Morphological accuracy — lemma overlap using `classla` Slovenian pipeline.
- Punctuation accuracy.

Dataset & recording conditions
- 27 single-speaker read speech recordings recorded on 2026-05-01 in a quiet home environment using a standard built-in microphone. Files are short (≈51–66 s), stereo M4A at 48 kHz.
- The speaker is a single female (25 y/o) from Nova Gorica (Primorska region). The corpus covers multiple registers (literary, journalistic, legal, instructional, tourist, conversational, scientific, etc.).
- Reference transcripts are orthographic and do not annotate paralinguistic phenomena (hesitations, nonverbal sounds).

Key findings (summary)
- Both systems perform comparably overall; neither dominates across all metrics.
- Voxtral shows slightly better lexical accuracy (lower WER and CER and higher NER in the evaluated corpus).
- Whisper shows stronger semantic preservation (lower SER, higher semantic similarity) and slightly better punctuation accuracy.
- Differences are generally small (under ~10 percentage points); per-file variability depends strongly on register and lexical/phonetic difficulty.

Notes & reproducibility
- The repository scripts assume local GPU availability for model inference (used in the original experiments). If running on CPU, expect slower execution and possible memory limitations.
- Preprocessing applies only resampling and channel mixing (no denoising or normalization) to keep inputs comparable across systems.

Acknowledgements & references
- The experimental design and evaluation metrics follow common ASR benchmarking practices (WER/CER) and augment them with semantic and linguistic metrics to better reflect usefulness for downstream tasks.

License
- Add your preferred license here (e.g. MIT) or keep as proprietary for private use.

If you want, I can also: add example commands for running single-file transcription, create a minimal `README-SUMMARY.md` for the repo front page, or update `requirements.txt` with exact versions used for the experiments.
