"""
Evaluate and compare Whisper and Voxtral transcription outputs.
"""

import argparse
import io
import json
import os
import re
import sys
import unicodedata
import warnings
import subprocess
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── Force UTF-8 output on Windows (cp1252 can't handle Slovenian chars) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────── helpers ────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _first_numeric(payload: dict, keys: list[str]):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return np.nan


def _first_text(payload: dict, keys: list[str]):
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _find_audio_path(file_index: int) -> Path | None:
    """Best-effort lookup for the source audio file used by both ASR models."""
    candidates = []
    for base_dir in [Path("posnetki"), Path("posnetki_preprocessed")]:
        for path in base_dir.glob(f"posnetek{file_index}.*"):
            if path.suffix.lower() in {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".webm", ".aac"}:
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.parent.name == "posnetki" else 1, p.suffix.lower()))
    return candidates[0]


def _extract_audio_duration(audio_path: Path | None) -> float:
    if audio_path is None or not audio_path.exists():
        return np.nan

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return round(float(result.stdout.strip()), 3)
    except Exception:
        if audio_path.suffix.lower() == ".wav":
            try:
                import wave

                with wave.open(str(audio_path), "rb") as wav_file:
                    return round(wav_file.getnframes() / float(wav_file.getframerate()), 3)
            except Exception:
                return np.nan
        return np.nan


def _load_sidecar_metadata(file_index: int, model_name: str) -> dict:
    """Load optional JSON sidecar data saved by the transcription scripts."""
    transcript_dir = Path("transkripcije")
    model_slug = model_name.lower()
    candidates = [
        transcript_dir / f"transkripcija{file_index}_{model_slug}.json",
        transcript_dir / f"transkripcija{file_index}_{model_slug}.txt.json",
        transcript_dir / f"transkripcija{file_index}_{model_slug}.meta.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue

        metadata = {
            "audio_duration_s": _first_numeric(payload, ["audio_duration", "audio_duration_s", "duration", "duration_s", "audio_len", "audio_length"]),
            "condition": _first_text(payload, ["condition", "dataset", "noise_level", "accent", "domain", "category"]),
        }

        if metadata["condition"] is None:
            metadata["condition"] = "Overall"
        return metadata

    return {
        "audio_duration_s": np.nan,
        "condition": "Overall",
    }


# ─────────────────────────── WER / CER ─────────────────────────

def compute_wer(reference: str, hypothesis: str) -> dict:
    """Returns WER, MER, WIL, WIP, Word Accuracy and edit counts via jiwer."""
    from jiwer import process_words
    ref_n = normalise(reference)
    hyp_n = normalise(hypothesis)
    out = process_words(ref_n, hyp_n)
    total = out.hits + out.substitutions + out.deletions
    wer = out.wer if total > 0 else 0.0
    return {
        "wer":           round(wer * 100, 2),
        "word_accuracy": round((1 - wer) * 100, 2),
        "mer":           round(out.mer * 100, 2),
        "wil":           round(out.wil * 100, 2),
        "wip":           round(out.wip * 100, 2),
        "hits":          out.hits,
        "substitutions": out.substitutions,
        "deletions":     out.deletions,
        "insertions":    out.insertions,
    }


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    from jiwer import cer  # cha_error_rate was removed in jiwer 4.x
    ref_n = normalise(reference)
    hyp_n = normalise(hypothesis)
    return round(cer(ref_n, hyp_n) * 100, 2)


# ──────────────────── Semantic Error Rate (SER) ─────────────────

_sem_model = None

def _get_sem_model():
    global _sem_model
    if _sem_model is None:
        print("  Loading sentence-transformer model (first run may download ~90 MB)…")
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sem_model


def compute_ser(reference: str, hypothesis: str, threshold: float = 0.85) -> dict:
    """
    Semantic Error Rate at sentence level.
    A sentence is 'semantically wrong' when cosine similarity < threshold.
    Returns SER (%) and mean similarity.
    """
    import torch
    model = _get_sem_model()

    ref_sents = [s.strip() for s in re.split(r"[.!?]+", reference) if s.strip()]
    hyp_sents = [s.strip() for s in re.split(r"[.!?]+", hypothesis) if s.strip()]

    # Pad shorter list with empty strings
    max_len = max(len(ref_sents), len(hyp_sents))
    ref_sents += [""] * (max_len - len(ref_sents))
    hyp_sents += [""] * (max_len - len(hyp_sents))

    if max_len == 0:
        return {"ser": 0.0, "mean_similarity": 1.0}

    emb_ref = model.encode(ref_sents, convert_to_tensor=True, show_progress_bar=False)
    emb_hyp = model.encode(hyp_sents, convert_to_tensor=True, show_progress_bar=False)

    from sentence_transformers.util import cos_sim
    sims = cos_sim(emb_ref, emb_hyp).diagonal().cpu().numpy()
    wrong = np.sum(sims < threshold)
    ser = round(float(wrong) / max_len * 100, 2)
    mean_sim = round(float(np.mean(sims)), 4)
    return {"ser": ser, "mean_similarity": mean_sim}


# ──────────────────── Punctuation accuracy ──────────────────────

PUNCTUATION = set(".,!?;:–—-\"'()")

def punct_accuracy(reference: str, hypothesis: str) -> float:
    """
    % of reference punctuation marks that are correctly reproduced
    at the same relative position (token-based, after alignment).
    Falls back to simple presence ratio when alignment is impractical.
    """
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)

    ref_punct = [c for c in ref_tokens if c in PUNCTUATION]
    if not ref_punct:
        return 100.0

    # Count how many reference punctuation marks appear in hypothesis
    # (order-insensitive approximation)
    from collections import Counter
    ref_cnt = Counter(ref_punct)
    hyp_cnt = Counter(c for c in hyp_tokens if c in PUNCTUATION)

    correct = sum(min(ref_cnt[p], hyp_cnt[p]) for p in ref_cnt)
    total   = sum(ref_cnt.values())
    return round(correct / total * 100, 2)


# ──────────────────── NER – proper names ────────────────────────

_nlp_ner = None

def _get_ner():
    global _nlp_ner
    if _nlp_ner is None:
        try:
            import spacy
            _nlp_ner = spacy.load("xx_ent_wiki_sm")
            print("  NER: using xx_ent_wiki_sm")
        except OSError:
            try:
                import classla
                classla.download("sl")
                _nlp_ner = classla.Pipeline("sl", processors="tokenize,ner")
                print("  NER: using classla Slovenian pipeline")
            except Exception:
                print("  NER: no model available – NER metric will be skipped.")
                _nlp_ner = "NONE"
    return _nlp_ner


def ner_accuracy(reference: str, hypothesis: str) -> float | None:
    """% of named entities from reference found (case-insensitive) in hypothesis."""
    nlp = _get_ner()
    if nlp == "NONE":
        return None

    try:
        import spacy
        doc_ref = nlp(reference)
        doc_hyp = nlp(hypothesis)
        ref_ents = {e.text.lower() for e in doc_ref.ents}
        hyp_ents = {e.text.lower() for e in doc_hyp.ents}
    except Exception:
        return None

    if not ref_ents:
        return None  # nothing to judge

    found = ref_ents & hyp_ents
    return round(len(found) / len(ref_ents) * 100, 2)


# ──────────────────── Morphology accuracy ───────────────────────

_nlp_morph = None

def _get_morph():
    global _nlp_morph
    if _nlp_morph is None:
        try:
            import classla
            classla.download("sl")
            _nlp_morph = classla.Pipeline("sl", processors="tokenize,lemma,pos")
            print("  Morphology: using classla Slovenian pipeline")
        except Exception:
            try:
                import spacy
                _nlp_morph = spacy.load("xx_ent_wiki_sm")
                print("  Morphology: using spacy xx_ent_wiki_sm (limited lemma support)")
            except Exception:
                print("  Morphology: no model available – morphology metric will be skipped.")
                _nlp_morph = "NONE"
    return _nlp_morph


def morph_accuracy(reference: str, hypothesis: str) -> float | None:
    """
    % of reference lemmas found in hypothesis lemmas.
    Approximates morphological correctness via lemma overlap.
    """
    nlp = _get_morph()
    if nlp == "NONE":
        return None

    try:
        doc_ref = nlp(reference)
        doc_hyp = nlp(hypothesis)
        ref_lemmas = [t.lemma_.lower() for t in doc_ref if not t.is_punct and not t.is_space]
        hyp_lemmas = [t.lemma_.lower() for t in doc_hyp if not t.is_punct and not t.is_space]
    except Exception:
        return None

    if not ref_lemmas:
        return None

    from collections import Counter
    ref_cnt = Counter(ref_lemmas)
    hyp_cnt = Counter(hyp_lemmas)
    correct = sum(min(ref_cnt[l], hyp_cnt[l]) for l in ref_cnt)
    return round(correct / sum(ref_cnt.values()) * 100, 2)


# ──────────────────── Error type analysis ───────────────────────

def error_type_analysis(reference: str, hypothesis: str) -> dict:
    """
    Uses jiwer alignment to categorise errors into:
      substitutions, insertions (dodajanja), deletions (izpuščanja).
    Additionally estimates morphological errors (same lemma, diff surface form)
    and proper-name errors.
    """
    from jiwer import process_words, visualize_alignment
    ref_n = normalise(reference)
    hyp_n = normalise(hypothesis)
    out = process_words(ref_n, hyp_n)

    result = {
        "substitutions": out.substitutions,
        "insertions":    out.insertions,
        "deletions":     out.deletions,
        "substitution_pairs": [],
    }

    # Morphological errors: substituted pairs sharing same first 4 chars (heuristic)
    morph_errors = 0
    for chunk in out.alignments:
        for op in chunk:
            if op.type == "substitute":
                ref_word = ref_n.split()[op.ref_start_idx] if op.ref_start_idx < len(ref_n.split()) else ""
                hyp_word = hyp_n.split()[op.hyp_start_idx] if op.hyp_start_idx < len(hyp_n.split()) else ""
                if ref_word and hyp_word and ref_word[:4] == hyp_word[:4] and ref_word != hyp_word:
                    morph_errors += 1
                if ref_word and hyp_word:
                    result["substitution_pairs"].append((ref_word, hyp_word))

    result["morphological_errors"] = morph_errors

    # Proper-name errors (capitalised words in original that were wrong)
    ref_words = reference.split()
    hyp_lower = hypothesis.lower()
    name_errors = 0
    for word in ref_words:
        cleaned = re.sub(r"[^\w]", "", word)
        if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
            if cleaned.lower() not in hyp_lower:
                name_errors += 1
    result["proper_name_errors"] = name_errors

    return result


# ──────────────────── BLEU metric ───────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU score (0–100)."""
    ref_tokens = normalise(reference).split()
    hyp_tokens = normalise(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        return round(
            float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)) * 100,
            2,
        )
    except Exception:
        # Offline-safe fallback: lightweight BLEU-4 approximation with add-one smoothing.
        from collections import Counter

        def ngrams(tokens, n):
            return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        precisions = []
        for n in range(1, 5):
            ref_counts = Counter(ngrams(ref_tokens, n))
            hyp_counts = Counter(ngrams(hyp_tokens, n))
            if not hyp_counts:
                precisions.append(0.0)
                continue
            overlap = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            precisions.append((overlap + 1.0) / (sum(hyp_counts.values()) + 1.0))

        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        brevity_penalty = 1.0 if hyp_len > ref_len else np.exp(1.0 - (ref_len / max(hyp_len, 1)))
        score = brevity_penalty * np.exp(np.mean(np.log(np.clip(precisions, 1e-12, 1.0))))
        return round(float(score) * 100, 2)


# ──────────────────── File discovery ────────────────────────────

def find_files(gold_dir: Path, whisper_dir: Path, voxtral_dir: Path):
    """
    Returns list of (index, gold_path, whisper_path, voxtral_path).
    Matches by number extracted from filename.
    """
    def index_of(path: Path) -> int:
        m = re.search(r"(\d+)", path.stem)
        return int(m.group(1)) if m else -1

    gold_files    = {index_of(p): p for p in gold_dir.glob("*.txt") if not p.name.startswith(".")}
    whisper_files = {index_of(p): p for p in whisper_dir.glob("*_whisper.txt")}
    voxtral_files = {index_of(p): p for p in voxtral_dir.glob("*_voxtral.txt")}

    common = sorted(set(gold_files) & set(whisper_files) & set(voxtral_files))
    if not common:
        sys.exit("ERROR: No matching transcript triplets found. "
                 "Check directory names and file naming conventions.")

    return [(i, gold_files[i], whisper_files[i], voxtral_files[i]) for i in common]


# ──────────────────── Core evaluation loop ──────────────────────

def evaluate_all(triplets):
    rows = []
    confusion_counter = Counter()
    confusion_type_counter = Counter()
    print(f"\nEvaluating {len(triplets)} transcript pair(s)…\n")

    for idx, gold_path, wh_path, vx_path in triplets:
        print(f"  [{idx}] {gold_path.name}")
        gold = load_text(gold_path)
        wh   = load_text(wh_path)
        vx   = load_text(vx_path)
        audio_path = _find_audio_path(idx)
        audio_duration = _extract_audio_duration(audio_path)

        for model_name, hyp in [("Whisper", wh), ("Voxtral", vx)]:
            meta    = _load_sidecar_metadata(idx, model_name)
            wer_d   = compute_wer(gold, hyp)
            cer     = compute_cer(gold, hyp)
            ser_d   = compute_ser(gold, hyp)
            punct   = punct_accuracy(gold, hyp)
            ner     = ner_accuracy(gold, hyp)
            morph   = morph_accuracy(gold, hyp)
            bleu    = compute_bleu(gold, hyp)
            err_d   = error_type_analysis(gold, hyp)
            subs = err_d.get("substitution_pairs", [])
            for pair in subs:
                try:
                    a, b = pair
                except Exception:
                    continue
                confusion_counter[(a, b)] += 1
                cat = _classify_confusion_pair(a, b)
                confusion_type_counter[(a, b, cat)] += 1

            row_audio_duration = meta.get("audio_duration_s", np.nan)
            if np.isnan(row_audio_duration):
                row_audio_duration = audio_duration

            row = {
                "file_index":        idx,
                "model":             model_name,
                "condition":         meta.get("condition", "Overall"),
                "audio_duration_s":  row_audio_duration,
                # ── error-rate metrics (lower = better) ──
                "WER (%)":           wer_d["wer"],
                "CER (%)":           cer,
                "MER (%)":           wer_d["mer"],
                "WIL (%)":           wer_d["wil"],
                # ── accuracy / quality metrics (higher = better) ──
                "Word Accuracy (%)": wer_d["word_accuracy"],
                "WIP (%)":           wer_d["wip"],
                "SER (%)":           ser_d["ser"],
                "Sem. similarity":   ser_d["mean_similarity"],
                "BLEU":              bleu,
                "Punct (%)":         punct,
                "NER (%)":           ner,
                "Morph (%)":         morph,
                # ── error-type counts ──
                "Substitutions":     err_d["substitutions"],
                "Insertions":        err_d["insertions"],
                "Deletions":         err_d["deletions"],
                "Morph errors":      err_d["morphological_errors"],
                "Name errors":       err_d["proper_name_errors"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["confusion_pairs"] = confusion_counter
    df.attrs["confusion_pairs_by_type"] = confusion_type_counter
    return df


# ──────────────────── Plotting ───────────────────────────────────

PALETTE = {"Whisper": "#4C72B0", "Voxtral": "#DD8452"}

def _set_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def _classify_confusion_pair(a: str, b: str) -> str:
    """Heuristic classification for substitution confusion pairs.

    Categories: Case, Punctuation, Morphological, ProperName, Other
    """
    if not a or not b:
        return "Other"
    a_str = str(a)
    b_str = str(b)
    # Case-only
    if a_str.lower() == b_str.lower() and a_str != b_str:
        return "Case"
    # Punctuation-only
    a_nopunct = re.sub(r"[\W_]+", "", a_str)
    b_nopunct = re.sub(r"[\W_]+", "", b_str)
    if a_nopunct == b_nopunct and a_str != b_str:
        return "Punctuation"
    # Proper name (heuristic: capitalized token in gold or hypothesis)
    if any(tok[0].isupper() for tok in a_str.split() if tok) or any(tok[0].isupper() for tok in b_str.split() if tok):
        return "ProperName"
    # Morphological heuristic: simple suffix differences
    morph_suffixes = ("s", "es", "ed", "ing", "ly", "ment", "ion")
    for suf in morph_suffixes:
        if (a_str.endswith(suf) and a_str[:-len(suf)] == b_str) or (b_str.endswith(suf) and b_str[:-len(suf)] == a_str):
            return "Morphological"
    return "Other"


def plot_metric_comparison(df: pd.DataFrame, metric: str, title: str,
                           ylabel: str, output_dir: Path, filename: str):
    _set_style()
    fig, ax = plt.subplots(figsize=(max(8, len(df["file_index"].unique()) * 0.6 + 2), 4.5))

    df_plot = df[df[metric].notna()].copy()
    x = np.arange(df_plot["file_index"].nunique())
    indices = sorted(df_plot["file_index"].unique())
    width = 0.35

    for i, (model, color) in enumerate(PALETTE.items()):
        vals = [df_plot[(df_plot["file_index"] == idx) & (df_plot["model"] == model)][metric].values
                for idx in indices]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        ax.bar(x + (i - 0.5) * width, vals, width, label=model, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" for i in indices], fontsize=9)
    ax.set_xlabel("Recording index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_radar(summary: pd.DataFrame, output_dir: Path):
    """Radar / spider chart comparing the two models on key metrics."""
    metrics = ["WER (%)", "CER (%)", "SER (%)"]
    inverted = ["Word Accuracy (%)", "WIP (%)", "BLEU", "Sem. similarity", "Punct (%)"]

    # Use only numeric, non-NaN columns
    all_metrics = metrics + [m for m in inverted if m in summary.columns]
    summary_clean = summary[["model"] + all_metrics].dropna(axis=1)
    all_metrics = [m for m in all_metrics if m in summary_clean.columns]

    N = len(all_metrics)
    if N < 3:
        return

    # Normalise each metric 0-1 (0 = worst, 1 = best)
    def norm(col, invert=False):
        mn, mx = summary_clean[col].min(), summary_clean[col].max()
        if mx == mn:
            return pd.Series([0.5] * len(summary_clean), index=summary_clean.index)
        v = (summary_clean[col] - mn) / (mx - mn)
        return 1 - v if invert else v

    normed = pd.DataFrame(index=summary_clean.index)
    for m in all_metrics:
        normed[m] = norm(m, invert=(m in metrics))  # lower error = better

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for _, row in summary_clean.iterrows():
        model = row["model"]
        vals = normed.loc[row.name].tolist() + [normed.loc[row.name].iloc[0]]
        ax.plot(angles, vals, "o-", linewidth=2, label=model,
                color=PALETTE.get(model, "grey"))
        ax.fill(angles, vals, alpha=0.15, color=PALETTE.get(model, "grey"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_metrics, size=9)
    ax.set_yticklabels([])
    ax.set_title("Model comparison radar chart (normalized)", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
    fig.tight_layout()
    fig.savefig(output_dir / "radar_comparison.png")
    plt.close(fig)


def plot_error_types(df: pd.DataFrame, output_dir: Path):
    """Stacked bar of error type distribution per model."""
    _set_style()
    err_cols = ["Substitutions", "Insertions", "Deletions", "Morph errors", "Name errors"]
    labels_en = ["Substitutions", "Insertions", "Deletions", "Morph. errors", "Name errors"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = df["model"].unique()
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    colors = sns.color_palette("Set2", len(err_cols))

    for col, label, color in zip(err_cols, labels_en, colors):
        vals = [df[df["model"] == m][col].sum() for m in models]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.9)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Total error count")
    ax.set_title("Error types by model")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "error_types_stacked.png")
    plt.close(fig)


def plot_metric_boxplot(df: pd.DataFrame, metrics: list, titles: list, output_dir: Path):
    """Box-plots for selected metrics side-by-side."""
    _set_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, title in zip(axes, metrics, titles):
        data = [df[df["model"] == m][metric].dropna().values for m in PALETTE]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], PALETTE.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(list(PALETTE.keys()))
        ax.set_title(title)
        ax.set_ylabel(metric)

    fig.tight_layout()
    fig.savefig(output_dir / "metric_boxplots.png")
    plt.close(fig)


def plot_summary_heatmap(summary: pd.DataFrame, output_dir: Path):
    """Heatmap of mean metrics per model."""
    _set_style()
    cols = [
        "WER (%)", "CER (%)", "MER (%)", "WIL (%)",
        "BLEU", "Word Accuracy (%)", "WIP (%)", "SER (%)",
        "Sem. similarity", "Punct (%)", "NER (%)", "Morph (%)",
    ]
    cols = [c for c in cols if c in summary.columns and summary[c].notna().any()]
    if not cols:
        return

    pivot = summary.set_index("model")[cols].astype(float)

    # Construct a score matrix where higher == better for consistent colouring.
    # For error-type metrics (lower is better) we invert: score = 100 - value.
    lower_is_better = {"WER (%)", "CER (%)", "MER (%)", "WIL (%)", "SER (%)"}
    score = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for c in pivot.columns:
        col = pivot[c].copy()
        # Sem. similarity is 0..1; scale to 0..100 for colouring but keep annotation original
        if c == "Sem. similarity":
            col_for_scale = col * 100
        else:
            col_for_scale = col

        if c in lower_is_better:
            score[c] = 100.0 - col_for_scale
        else:
            score[c] = col_for_scale

    # Prepare annotations from the original values (keep original scale for readability)
    annot = pivot.copy()
    annot = annot.apply(lambda col: col.map(lambda x: f"{x:.2f}" if pd.notna(x) else ""))

    fig, ax = plt.subplots(figsize=(len(cols) * 1.4 + 1, 2.8))
    sns.heatmap(score, annot=annot, fmt="", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Average metrics by model")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_heatmap.png")
    plt.close(fig)


def plot_wer_per_file(df: pd.DataFrame, output_dir: Path):
    """Line plot of WER per file."""
    _set_style()
    fig, ax = plt.subplots(figsize=(max(8, df["file_index"].nunique() * 0.5 + 2), 4))

    for model, color in PALETTE.items():
        sub = df[df["model"] == model].sort_values("file_index")
        ax.plot(sub["file_index"], sub["WER (%)"], "o-", label=model,
                color=color, linewidth=1.8, markersize=5)

    ax.set_xlabel("Recording index")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER by recording")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wer_per_file.png")
    plt.close(fig)


# ──────────────────── Statistical tests ─────────────────────────

def statistical_tests(df: pd.DataFrame) -> dict:
    """Paired Wilcoxon signed-rank test + bootstrap 95 % CI for key metrics."""
    results = {}
    for metric in ["WER (%)", "CER (%)", "BLEU", "Sem. similarity"]:
        wh = df[df["model"] == "Whisper"].sort_values("file_index")[metric].dropna().values
        vx = df[df["model"] == "Voxtral"].sort_values("file_index")[metric].dropna().values
        n  = min(len(wh), len(vx))
        if n < 3:
            results[metric] = {"skipped": True}
            continue
        wh, vx = wh[:n], vx[:n]
        diffs   = wh - vx

        p_wilcox = p_ttest = None
        try:
            from scipy.stats import wilcoxon, ttest_rel
            _, p_wilcox = wilcoxon(wh, vx)
            _, p_ttest  = ttest_rel(wh, vx)
            p_wilcox = round(float(p_wilcox), 4)
            p_ttest  = round(float(p_ttest),  4)
        except ImportError:
            pass

        rng        = np.random.default_rng(42)
        boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        pooled_std   = np.std(np.concatenate([wh, vx]), ddof=1)
        cohens_d     = float(np.mean(diffs)) / pooled_std if pooled_std > 0 else 0.0

        results[metric] = {
            "whisper_mean": round(float(np.mean(wh)),    3),
            "voxtral_mean": round(float(np.mean(vx)),    3),
            "mean_diff":    round(float(np.mean(diffs)), 3),
            "ci_lo":        round(float(ci_lo),          3),
            "ci_hi":        round(float(ci_hi),          3),
            "cohens_d":     round(float(cohens_d),       3),
            "wilcoxon_p":   p_wilcox,
            "ttest_p":      p_ttest,
            "n":            n,
        }
    return results

# ──────────────────── Requested visualizations ──────────────────

def _sorted_conditions(df: pd.DataFrame) -> list[str]:
    values = df.get("condition", pd.Series(dtype=str)).fillna("Overall").astype(str)
    unique_values = list(dict.fromkeys(values.tolist()))
    if not unique_values:
        return ["Overall"]
    if "Overall" in unique_values:
        unique_values = ["Overall"] + [value for value in unique_values if value != "Overall"]
    return unique_values


def _mean_by_condition(df: pd.DataFrame, model: str, metric: str) -> list[float]:
    values = []
    for condition in _sorted_conditions(df):
        subset = df[(df["model"] == model) & (df["condition"].fillna("Overall").astype(str) == condition)]
        values.append(pd.to_numeric(subset[metric], errors="coerce").mean())
    return values


def plot_accuracy_by_condition(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart for WER and CER per model, broken down by condition."""
    _set_style()
    conditions = _sorted_conditions(df)
    x = np.arange(len(conditions))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(max(9, len(conditions) * 1.8), 5.2))
    bar_specs = [
        ("Whisper", "WER (%)", 0, PALETTE["Whisper"], "Whisper WER"),
        ("Whisper", "CER (%)", 1, sns.light_palette(PALETTE["Whisper"], 3)[2], "Whisper CER"),
        ("Voxtral", "WER (%)", 2, PALETTE["Voxtral"], "Voxtral WER"),
        ("Voxtral", "CER (%)", 3, sns.light_palette(PALETTE["Voxtral"], 3)[2], "Voxtral CER"),
    ]

    for model, metric, offset_index, color, label in bar_specs:
        vals = _mean_by_condition(df, model, metric)
        offsets = (offset_index - 1.5) * bar_width
        bars = ax.bar(x + offsets, vals, bar_width, color=color, label=label, alpha=0.9, edgecolor="white")
        for bar, value in zip(bars, vals):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                        f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("WER and CER by Model and Condition", fontsize=12, fontweight="bold")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_condition.png")
    plt.close(fig)


def plot_wer_across_conditions(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart showing WER across conditions for each model."""
    _set_style()
    conditions = _sorted_conditions(df)
    x = np.arange(len(conditions))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.6), 4.8))
    for offset_index, (model, color) in enumerate(PALETTE.items()):
        vals = _mean_by_condition(df, model, "WER (%)")
        bars = ax.bar(x + (offset_index - 0.5) * bar_width, vals, bar_width,
                      color=color, label=model, alpha=0.9, edgecolor="white")
        for bar, value in zip(bars, vals):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                        f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER Across Conditions", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_across_conditions.png")
    plt.close(fig)





def plot_error_breakdown(df: pd.DataFrame, output_dir: Path):
    """Stacked error-count bar chart for substitutions, deletions, and insertions."""
    _set_style()
    models = list(PALETTE.keys())
    error_cols = [("Substitutions", "Substitutions", "#4C78A8"),
                  ("Deletions", "Deletions", "#F58518"),
                  ("Insertions", "Insertions", "#54A24B")]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(models))
    bottom = np.zeros(len(models))

    for col, label, color in error_cols:
        values = [df[df["model"] == model][col].sum() if col in df.columns else 0 for model in models]
        ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.9, edgecolor="white")
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Total error count")
    ax.set_title("Error Type Breakdown by Model", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_breakdown_stacked.png")
    plt.close(fig)


def plot_confusion_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of the most common substitution confusions."""
    _set_style()
    confusion_counter = df.attrs.get("confusion_pairs", Counter())
    if not confusion_counter:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Confusion metadata unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_heatmap.png")
        plt.close(fig)
        return

    top_pairs = confusion_counter.most_common(36)
    ref_labels = sorted({ref for (ref, _), _count in top_pairs})
    hyp_labels = sorted({hyp for (_, hyp), _count in top_pairs})
    heatmap = pd.DataFrame(0, index=ref_labels, columns=hyp_labels, dtype=int)
    for (ref, hyp), count in top_pairs:
        heatmap.loc[ref, hyp] = count

    fig, ax = plt.subplots(figsize=(max(7, len(hyp_labels) * 0.65), max(4, len(ref_labels) * 0.45)))
    sns.heatmap(heatmap, annot=True, fmt="d", cmap="Reds", linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Top substitution confusions", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted word")
    ax.set_ylabel("Reference word")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_heatmap.png")
    plt.close(fig)


def plot_confusion_heatmap_top10(df: pd.DataFrame, output_dir: Path):
    """Compact heatmap showing the 10 most common substitution pairs."""
    _set_style()
    confusion_counter = df.attrs.get("confusion_pairs", Counter())
    if not confusion_counter:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Confusion metadata unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_heatmap_top10.png")
        plt.close(fig)
        return

    top_pairs = confusion_counter.most_common(10)
    ref_labels = sorted({ref for (ref, _), _count in top_pairs})
    hyp_labels = sorted({hyp for (_, hyp), _count in top_pairs})
    heatmap = pd.DataFrame(0, index=ref_labels, columns=hyp_labels, dtype=int)
    for (ref, hyp), count in top_pairs:
        heatmap.loc[ref, hyp] = count

    fig, ax = plt.subplots(figsize=(max(3.6, len(hyp_labels) * 0.6), max(2.2, len(ref_labels) * 0.5)))
    sns.heatmap(heatmap, annot=True, fmt="d", cmap="Reds", linewidths=0.4, ax=ax, cbar=False)
    ax.set_title("Top substitution confusions (top-10)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted word")
    ax.set_ylabel("Reference word")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_heatmap_top10.png")
    plt.close(fig)


def plot_distribution_views(df: pd.DataFrame, output_dir: Path):
    """WER box plot and overlaid histogram on a single figure."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    wer_data = [pd.to_numeric(df[df["model"] == model]["WER (%)"], errors="coerce").dropna().values for model in PALETTE]
    if any(len(values) > 0 for values in wer_data):
        bp = axes[0].boxplot(wer_data, patch_artist=True, widths=0.55,
                             medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], PALETTE.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[0].set_xticklabels(list(PALETTE.keys()))
        axes[0].set_ylabel("WER (%)")
        axes[0].set_title("WER Distribution by Model", fontsize=11, fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "WER samples unavailable", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()

    for model, color in PALETTE.items():
        values = pd.to_numeric(df[df["model"] == model]["WER (%)"], errors="coerce").dropna().values
        if len(values) == 0:
            continue
        sns.histplot(values, bins=min(12, max(4, len(values) // 2)), stat="density", element="step",
                     fill=False, common_norm=False, alpha=0.9, color=color, ax=axes[1], label=model)
    axes[1].set_xlabel("WER (%)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("WER Histogram Overlay", fontsize=11, fontweight="bold")
    if axes[1].has_data():
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "wer_distribution_views.png")
    plt.close(fig)


def plot_bleu_views(df: pd.DataFrame, output_dir: Path):
    """Dedicated BLEU views: distribution, condition comparison and relation to WER."""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    bleu_data = [pd.to_numeric(df[df["model"] == model]["BLEU"], errors="coerce").dropna().values for model in PALETTE]
    if any(len(values) > 0 for values in bleu_data):
        bp = axes[0].boxplot(bleu_data, patch_artist=True, widths=0.55,
                             medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], PALETTE.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        axes[0].set_xticklabels(list(PALETTE.keys()))
        axes[0].set_ylabel("BLEU")
        axes[0].set_title("BLEU distribution", fontsize=11, fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "BLEU samples unavailable", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()

    conditions = _sorted_conditions(df)
    x = np.arange(len(conditions))
    bar_width = 0.35
    for offset_index, (model, color) in enumerate(PALETTE.items()):
        vals = []
        for condition in conditions:
            subset = df[(df["model"] == model) & (df["condition"].fillna("Overall").astype(str) == condition)]
            vals.append(pd.to_numeric(subset["BLEU"], errors="coerce").mean())
        bars = axes[1].bar(x + (offset_index - 0.5) * bar_width, vals, bar_width,
                           color=color, label=model, alpha=0.9, edgecolor="white")
        for bar, value in zip(bars, vals):
            if not np.isnan(value):
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                             f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=20, ha="right")
    axes[1].set_ylabel("BLEU")
    axes[1].set_title("BLEU by condition", fontsize=11, fontweight="bold")
    axes[1].legend()

    for model, color in PALETTE.items():
        sub = df[df["model"] == model].copy()
        bleu_vals = pd.to_numeric(sub["BLEU"], errors="coerce")
        wer_vals = pd.to_numeric(sub["WER (%)"], errors="coerce")
        mask = bleu_vals.notna() & wer_vals.notna()
        if mask.any():
            axes[2].scatter(bleu_vals[mask], wer_vals[mask], label=model, color=color, alpha=0.8, s=40,
                            edgecolor="white", linewidth=0.5)
    axes[2].set_xlabel("BLEU")
    axes[2].set_ylabel("WER (%)")
    axes[2].set_title("BLEU vs WER", fontsize=11, fontweight="bold")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "bleu_views.png")
    plt.close(fig)


def plot_error_type_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of core error types by model."""
    _set_style()
    cols = ["Substitutions", "Deletions", "Insertions", "Morph errors", "Name errors"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return

    pivot = df.groupby("model")[available].sum().astype(float)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), 2.8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Error type heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Error type")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(output_dir / "error_type_heatmap.png")
    plt.close(fig)


def plot_overall_metric_heatmap(summary: pd.DataFrame, output_dir: Path):
    """Compact heatmap for the most informative overall metrics."""
    _set_style()
    cols = ["WER (%)", "CER (%)", "MER (%)", "BLEU", "Word Accuracy (%)", "Sem. similarity", "Punct (%)", "NER (%)"]
    cols = [c for c in cols if c in summary.columns and summary[c].notna().any()]
    if not cols:
        return
    pivot = summary.set_index("model")[cols].astype(float)

    lower_is_better = {"WER (%)", "CER (%)", "MER (%)"}
    score = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for c in pivot.columns:
        col = pivot[c].copy()
        if c == "Sem. similarity":
            col_for_scale = col * 100
        else:
            col_for_scale = col
        if c in lower_is_better:
            score[c] = 100.0 - col_for_scale
        else:
            score[c] = col_for_scale

    annot = pivot.copy()
    annot = annot.apply(lambda col: col.map(lambda x: f"{x:.2f}" if pd.notna(x) else ""))

    fig, ax = plt.subplots(figsize=(len(cols) * 1.35 + 1, 2.8))
    sns.heatmap(score, annot=annot, fmt="", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Overall metric heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_metric_heatmap.png")
    plt.close(fig)


def plot_requested_visualizations(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    """Generate the requested visualization suite in one place."""
    plot_accuracy_by_condition(df, output_dir)
    plot_wer_across_conditions(df, output_dir)
    plot_overall_accuracy(summary, output_dir)
    plot_overall_metric_heatmap(summary, output_dir)
    plot_error_breakdown(df, output_dir)
    plot_error_type_heatmap(df, output_dir)
    plot_confusion_heatmap(df, output_dir)
    plot_confusion_heatmap_top10(df, output_dir)
    plot_top_confusion_pairs(df, output_dir)
    plot_distribution_views(df, output_dir)
    plot_violin_distributions(df, output_dir)
    plot_bleu_views(df, output_dir)
    plot_wer_per_file(df, output_dir)
    plot_per_file_winner(df, output_dir)
    plot_delta_per_file(df, output_dir)
    plot_summary_heatmap(summary, output_dir)
    plot_radar(summary, output_dir)

def plot_overall_accuracy(summary: pd.DataFrame, output_dir: Path):
    """Grouped bar chart for overall accuracy metrics."""
    _set_style()
    metrics  = ["WER (%)", "CER (%)", "BLEU"]
    labels   = ["WER (%)", "CER (%)", "BLEU"]
    models   = list(PALETTE.keys())
    x        = np.arange(len(metrics))
    width    = 0.34

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model, color) in enumerate(PALETTE.items()):
        row  = summary[summary["model"] == model].iloc[0]
        vals = [row.get(m, np.nan) for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=model, color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Overall model comparison", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overall_accuracy.png")
    plt.close(fig)


def plot_violin_distributions(df: pd.DataFrame, output_dir: Path):
    """Violin plots for WER, CER and Semantic Similarity."""
    _set_style()
    metrics = ["WER (%)", "CER (%)", "Sem. similarity"]
    titles  = ["Word Error Rate (%)", "Character Error Rate (%)", "Semantic Similarity"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, metric, title in zip(axes, metrics, titles):
        data  = [df[df["model"] == m][metric].dropna().values for m in PALETTE]
        # Skip violin if any model has fewer than 2 data points
        if any(len(d) < 2 for d in data):
            ax.set_title(title + "\n(not enough data)", fontsize=10)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(list(PALETTE.keys()))
            continue
        parts = ax.violinplot(data, positions=[1, 2], showmedians=True,
                              showextrema=True, widths=0.6)
        for pc, color in zip(parts["bodies"], PALETTE.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part].set_color("black")
            parts[part].set_linewidth(1.2)
        # overlay individual points
        for j, (m, color) in enumerate(PALETTE.items(), start=1):
            vals = df[df["model"] == m][metric].dropna().values
            ax.scatter(np.random.normal(j, 0.05, len(vals)), vals,
                       color=color, alpha=0.5, s=14, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(list(PALETTE.keys()))
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.suptitle("Metric distributions", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "violin_distributions.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_confusion_pairs(df: pd.DataFrame, output_dir: Path, top_n: int = 24):
    """Horizontal bar chart of the most frequent substitution confusions grouped by category."""
    _set_style()
    conf_by_type = df.attrs.get("confusion_pairs_by_type")
    if not conf_by_type:
        # fall back to raw pairs
        raw = df.attrs.get("confusion_pairs", Counter())
        items = [((a, b, "Other"), cnt) for (a, b), cnt in raw.items()]
    else:
        items = [((a, b, cat), cnt) for (a, b, cat), cnt in conf_by_type.items()]

    if not items:
        return

    # aggregate counts per (pair, category)
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
    pairs = [f"{a} → {b}" for (a, b, _), _ in items_sorted]
    counts = [cnt for _, cnt in items_sorted]
    cats = [cat for (a, b, cat), _ in items_sorted]

    # color mapping for categories
    cat_colors = {
        "Case": "#8dd3c7",
        "Punctuation": "#ffffb3",
        "Morphological": "#bebada",
        "ProperName": "#fb8072",
        "Other": "#80b1d3",
    }
    colors = [cat_colors.get(c, "#d9d9d9") for c in cats]

    fig, ax = plt.subplots(figsize=(10, max(3, len(pairs) * 0.45)))
    y_pos = np.arange(len(pairs))[::-1]
    ax.barh(y_pos, counts[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs[::-1])
    ax.set_xlabel("Count")
    ax.set_title("Top substitution confusions", fontsize=12, fontweight="bold")
    # legend
    unique_cats = list(dict.fromkeys(cats))
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors.get(c, "#d9d9d9")) for c in unique_cats]
    ax.legend(handles, unique_cats, title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "top_confusions.png", bbox_inches="tight")
    plt.close(fig)


def save_worst_examples(df: pd.DataFrame, output_dir: Path, n: int = 5):
    """Export a simple, user-friendly Markdown file with the worst N examples per model.

    Includes file index, WER, gold and hypothesis snippets.
    """
    lines = ["# Worst transcription examples\n"]
    for model in df["model"].unique():
        lines.append(f"## {model}\n")
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values("WER (%)", ascending=False).head(n)
        for _, row in sub.iterrows():
            lines.append(f"- **File**: {int(row['file_index'])} — **WER**: {row.get('WER (%)', np.nan):.2f}%")
            gold_text = str(row.get('gold_text', '')) if 'gold_text' in row else ''
            hyp_text = str(row.get('hypothesis', '')) if 'hypothesis' in row else ''
            # fallback: the DataFrame may not contain raw texts; so keep short
            if gold_text:
                lines.append(f"  - Gold: {gold_text}")
            if hyp_text:
                lines.append(f"  - Hypothesis: {hyp_text}")
        lines.append("\n")

    path = output_dir / "worst_examples.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_wer_cdf(df: pd.DataFrame, output_dir: Path):
    """Empirical CDF of WER values per model – useful for threshold analysis."""
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in PALETTE.items():
        vals = np.sort(df[df["model"] == model]["WER (%)"].dropna().values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, where="post", label=model, color=color, linewidth=2)
        ax.scatter(vals, cdf, color=color, s=20, alpha=0.6, zorder=3)
    ax.set_xlabel("WER (%)")
    ax.set_ylabel("Cumulative proportion of recordings")
    ax.set_title("Empirical CDF of Word Error Rate", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_cdf.png")
    plt.close(fig)


def plot_per_file_winner(df: pd.DataFrame, output_dir: Path):
    """Colour-coded grid: green = model wins on WER, red = loses, grey = tie."""
    _set_style()
    indices = sorted(df["file_index"].unique())
    wh_wer  = df[df["model"] == "Whisper"].set_index("file_index")["WER (%)"]
    vx_wer  = df[df["model"] == "Voxtral"].set_index("file_index")["WER (%)"]

    fig, ax = plt.subplots(figsize=(max(10, len(indices) * 0.45 + 2), 3))
    for xi, idx in enumerate(indices):
        w, v = wh_wer.get(idx, np.nan), vx_wer.get(idx, np.nan)
        if np.isnan(w) or np.isnan(v):
            colors = ["#cccccc", "#cccccc"]
        elif w < v:
            colors = ["#4CAF50", "#EF5350"]   # Whisper wins
        elif v < w:
            colors = ["#EF5350", "#4CAF50"]   # Voxtral wins
        else:
            colors = ["#FFC107", "#FFC107"]   # tie

        for yi, (color, label) in enumerate(zip(colors, ["Whisper", "Voxtral"])):
            rect = plt.Rectangle((xi, yi), 0.9, 0.85, color=color, alpha=0.85)
            ax.add_patch(rect)
            val = w if label == "Whisper" else v
            ax.text(xi + 0.45, yi + 0.42, f"{val:.0f}%",
                    ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    ax.set_xlim(0, len(indices))
    ax.set_ylim(-0.1, 2.1)
    ax.set_xticks([i + 0.45 for i in range(len(indices))])
    ax.set_xticklabels([f"#{i}" for i in indices], fontsize=8, rotation=45)
    ax.set_yticks([0.42, 1.42])
    ax.set_yticklabels(["Voxtral", "Whisper"], fontsize=10)
    ax.set_title("Per-Recording Winner (WER) – Green = lower WER", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recording index")
    ax.axis("on")
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(output_dir / "per_file_winner.png")
    plt.close(fig)


def plot_delta_per_file(df: pd.DataFrame, output_dir: Path):
    """Bar chart of (WER_Whisper − WER_Voxtral) per recording.
    Positive = Whisper is worse; negative = Voxtral is worse."""
    _set_style()
    indices = sorted(df["file_index"].unique())
    wh_wer  = df[df["model"] == "Whisper"].set_index("file_index")["WER (%)"]
    vx_wer  = df[df["model"] == "Voxtral"].set_index("file_index")["WER (%)"]
    deltas  = [wh_wer.get(i, np.nan) - vx_wer.get(i, np.nan) for i in indices]

    colors = ["#4CAF50" if d > 0 else "#EF5350" if d < 0 else "#FFC107" for d in deltas]

    fig, ax = plt.subplots(figsize=(max(10, len(indices) * 0.5 + 2), 4))
    ax.bar(range(len(indices)), deltas, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"#{i}" for i in indices], fontsize=8, rotation=45)
    ax.set_ylabel("WER delta (Whisper − Voxtral, %)")
    ax.set_title("Per-Recording WER Delta\n(green = Whisper worse, red = Voxtral worse)",
                 fontsize=12, fontweight="bold")
    legend_elements = [
        mpatches.Patch(color="#4CAF50", label="Voxtral wins"),
        mpatches.Patch(color="#EF5350", label="Whisper wins"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_delta_per_file.png")
    plt.close(fig)


def plot_stat_significance(stats: dict, output_dir: Path):
    """Forest plot of mean WER/CER differences with 95 % bootstrap CI."""
    _set_style()
    items = [(m, v) for m, v in stats.items() if not v.get("skipped")]
    if not items:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(items) * 0.8 + 1)))
    colors_ci = []
    for yi, (metric, v) in enumerate(reversed(items)):
        diff   = v["mean_diff"]
        ci_lo  = v["ci_lo"]
        ci_hi  = v["ci_hi"]
        color  = "#E74C3C" if ci_lo > 0 else "#2ECC71" if ci_hi < 0 else "#95A5A6"
        colors_ci.append(color)
        ax.plot([ci_lo, ci_hi], [yi, yi], color=color, linewidth=3, solid_capstyle="round")
        ax.scatter(diff, yi, color=color, s=60, zorder=5)
        p_str = f"p={v['wilcoxon_p']}" if v.get("wilcoxon_p") is not None else "p=N/A"
        ax.text(ci_hi + 0.2, yi, f"  d={v['cohens_d']:.2f}, {p_str}", va="center", fontsize=8)

    ax.axvline(0, color="black", linewidth=1, ls="--")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([m for m, _ in reversed(items)], fontsize=9)
    ax.set_xlabel("Mean difference (Whisper − Voxtral)\nPositive = Whisper is worse")
    ax.set_title("Statistical Comparison (95 % Bootstrap CI)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "stat_significance.png")
    plt.close(fig)


# ──────────────────── Report generation ─────────────────────────

def generate_report(df: pd.DataFrame, summary: pd.DataFrame, stats: dict, output_dir: Path):
    """Generates a detailed Markdown report for researchers and prints a clean summary to stdout."""
    # 1. Console Output
    print("\n" + "=" * 70)
    print("  EVALVACIJA TRANSKRIPTOV – PRIMERJAVA WHISPER vs VOXTRAL")
    print("=" * 70)
    print(f"\nŠtevilo posnetkov: {df['file_index'].nunique()}\n")

    print("── POVPREČNE METRIKE ─────────────────────────────────────────────\n")
    metric_cols = [c for c in [
        "WER (%)", "CER (%)", "Word Accuracy (%)", "SER (%)", "Sem. similarity",
        "Punct (%)", "NER (%)", "Morph (%)", "BLEU",
    ] if c in summary.columns]
    print(summary.set_index("model")[metric_cols].to_string())
    print("\n")

    print("── SKUPNE NAPAKE PO TIPU ─────────────────────────────────────────\n")
    err_cols = ["Substitutions", "Insertions", "Deletions", "Morph errors", "Name errors"]
    err_labels = {
        "Substitutions": "Substitucije",
        "Insertions":    "Dodajanja",
        "Deletions":     "Izpuščanja",
        "Morph errors":  "Morfološke napake",
        "Name errors":   "Napake lastnih imen",
    }
    for model in ["Whisper", "Voxtral"]:
        sub = df[df["model"] == model]
        print(f"  {model}:")
        for col in err_cols:
            if col in sub.columns:
                print(f"    {err_labels[col]:<28} {int(sub[col].sum())}")
    print("\n")

    # 2. Detailed Markdown Report
    md = []
    md.append(f"# Speech Recognition Evaluation Report")
    md.append(f"**Date generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Total recordings evaluated:** {df['file_index'].nunique()}\n")

    md.append("## 1. Overall Performance Summary\n")
    md.append(summary.set_index("model")[metric_cols].round(2).to_markdown())
    md.append("\n")

    # Add concise numeric highlights and comparisons
    try:
        if set(["Whisper", "Voxtral"]).issubset(set(summary["model"])):
            wh = summary[summary["model"] == "Whisper"].iloc[0]
            vx = summary[summary["model"] == "Voxtral"].iloc[0]
            wer_diff = wh.get("WER (%)", np.nan) - vx.get("WER (%)", np.nan)
            cer_diff = wh.get("CER (%)", np.nan) - vx.get("CER (%)", np.nan)
            wer_winner = "Whisper" if wh.get("WER (%)", np.nan) < vx.get("WER (%)", np.nan) else "Voxtral"

            wh_std = df[df["model"] == "Whisper"]["WER (%)"].std()
            vx_std = df[df["model"] == "Voxtral"]["WER (%)"].std()

            # per-record wins
            wins = {"Whisper": 0, "Voxtral": 0, "Tie": 0}
            for idx in sorted(df["file_index"].unique()):
                sub = df[df["file_index"] == idx].set_index("model")["WER (%)"].to_dict()
                w = sub.get("Whisper", np.nan)
                v = sub.get("Voxtral", np.nan)
                if np.isnan(w) or np.isnan(v):
                    continue
                if w < v:
                    wins["Whisper"] += 1
                elif v < w:
                    wins["Voxtral"] += 1
                else:
                    wins["Tie"] += 1

            md.append("### Highlights\n")
            md.append(f"- Mean WER — Whisper: {wh.get('WER (%)', np.nan):.2f}%, Voxtral: {vx.get('WER (%)', np.nan):.2f}%; winner: {wer_winner} (difference {abs(wer_diff):.2f} percentage points).\n")
            md.append(f"- Mean CER — Whisper: {wh.get('CER (%)', np.nan):.2f}%, Voxtral: {vx.get('CER (%)', np.nan):.2f}%; difference {abs(cer_diff):.2f}.\n")
            md.append(f"- WER variability (std): Whisper {wh_std:.2f}%, Voxtral {vx_std:.2f}%.\n")
            md.append(f"- Record-level lower-WER wins: Whisper {wins['Whisper']}, Voxtral {wins['Voxtral']}, ties {wins['Tie']}.\n")

            # Top confusion pairs
            conf = df.attrs.get('confusion_pairs', None)
            if conf and hasattr(conf, 'most_common'):
                top = conf.most_common(8)
                md.append("### Top substitution confusion pairs (aggregated)\n")
                md.append("| Pair | Count |\n|------:|------:|")
                for (a, b), cnt in top:
                    md.append(f"| {a} → {b} | {cnt} |")
                md.append("\n")
    except Exception:
        # Keep report generation robust — fall back to the basic table
        pass

    md.append("## 2. Statistical Significance Tests (Paired Analysis)\n")
    md.append("Paired Wilcoxon signed-rank test and bootstrap 95% Confidence Intervals for the difference (Whisper − Voxtral).\n")
    md.append("| Metric | Whisper Mean | Voxtral Mean | Diff (W-V) | 95% CI | Wilcoxon p-value | Cohen's d |")
    md.append("|--------|--------------|--------------|------------|--------|------------------|-----------|")
    for m, v in stats.items():
        if v.get("skipped"): continue
        ci_str = f"[{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]"
        pval = f"{v['wilcoxon_p']:.4f}" if v.get('wilcoxon_p') else "N/A"
        md.append(f"| {m} | {v['whisper_mean']:.2f} | {v['voxtral_mean']:.2f} | {v['mean_diff']:.2f} | {ci_str} | {pval} | {v['cohens_d']:.2f} |")
    md.append("\n")

    md.append("## 3. Error Type Distribution\n")
    md.append("| Model | Substitutions | Insertions | Deletions | Morphological | Proper Names |")
    md.append("|-------|---------------|------------|-----------|---------------|--------------|")
    for model in ["Whisper", "Voxtral"]:
        sub = df[df["model"] == model]
        md.append(f"| {model} | {int(sub['Substitutions'].sum())} | {int(sub['Insertions'].sum())} | "
                  f"{int(sub['Deletions'].sum())} | {int(sub.get('Morph errors', 0).sum())} | "
                  f"{int(sub.get('Name errors', 0).sum())} |")
    md.append("\n")

    md.append("## 4. Per-Recording Detailed Results\n")
    detail_cols = [
        "file_index", "model", "condition",
        "WER (%)", "CER (%)", "Word Accuracy (%)", "SER (%)"
    ]
    detail_cols = [col for col in detail_cols if col in df.columns]
    md.append(df[detail_cols].round(2).to_markdown(index=False))
    md.append("\n")

    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    
    print(f"✓ Vsi rezultati shranjeni v mapo: '{output_dir}/'")
    print(f"✓ Podrobno Markdown poročilo: {report_path}")


# ──────────────────── Main ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evalvacija transkriptov Whisper / Voxtral vs gold standard"
    )
    parser.add_argument("--gold",    default="original-transkript",
                        help="Mapa z referenčnimi transkripcijami (gold standard)")
    parser.add_argument("--whisper", default="transkripcije",
                        help="Mapa z Whisper transkripcijami (*_whisper.txt)")
    parser.add_argument("--voxtral", default="transkripcije",
                        help="Mapa z Voxtral transkripcijami (*_voxtral.txt)")
    parser.add_argument("--output",  default="rezultati",
                        help="Izhodna mapa za poročila in grafe")
    parser.add_argument("--ser-threshold", type=float, default=0.85,
                        help="Prag podobnosti za SER (privzeto 0.85)")
    args = parser.parse_args()

    gold_dir    = Path(args.gold)
    whisper_dir = Path(args.whisper)
    voxtral_dir = Path(args.voxtral)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in [gold_dir, whisper_dir, voxtral_dir]:
        if not d.exists():
            sys.exit(f"ERROR: Directory not found: {d}")

    triplets = find_files(gold_dir, whisper_dir, voxtral_dir)
    print(f"Najdeno {len(triplets)} trojic (gold + whisper + voxtral).")

    df = evaluate_all(triplets)

    # Save raw CSV
    csv_path = output_dir / "rezultati_vsi.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Summary
    numeric_cols = df.select_dtypes("number").columns.difference(["file_index"])
    summary = df.groupby("model")[numeric_cols].mean().round(3).reset_index()
    summary_path = output_dir / "povzetek.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    
    # Statistical significance
    stats = statistical_tests(df)

    # ── Plots ──────────────────────────────────────────────────
    print("\nGeneriranje grafov…")

    plot_requested_visualizations(df, summary, output_dir)
    plot_stat_significance(stats, output_dir)

    # Generate reports and terminal output
    generate_report(df, summary, stats, output_dir)


if __name__ == "__main__":
    main()
