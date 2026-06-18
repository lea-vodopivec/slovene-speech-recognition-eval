# Speech Recognition Evaluation Report
**Date generated:** 2026-06-01 18:31:43
**Total recordings evaluated:** 27

## 1. Overall Performance Summary

| model   |   WER (%) |   CER (%) |   Word Accuracy (%) |   SER (%) |   Sem. similarity |   Punct (%) |   NER (%) |   BLEU |
|:--------|----------:|----------:|--------------------:|----------:|------------------:|------------:|----------:|-------:|
| Voxtral |     11.86 |      3.43 |               88.14 |     52.3  |              0.66 |       91.08 |     78.38 |  77.84 |
| Whisper |     12.59 |      4.39 |               87.41 |     43.18 |              0.73 |       92.04 |     74.39 |  78.18 |


### Highlights

- Mean WER — Whisper: 12.59%, Voxtral: 11.86%; winner: Voxtral (difference 0.74 percentage points).

- Mean CER — Whisper: 4.39%, Voxtral: 3.43%; difference 0.96.

- WER variability (std): Whisper 7.09%, Voxtral 8.03%.

- Record-level lower-WER wins: Whisper 13, Voxtral 11, ties 3.

### Top substitution confusion pairs (aggregated)

| Pair | Count |
|------:|------:|
| sm → sem | 8 |
| po → ponavadi | 6 |
| predsodkov → predsotkov | 6 |
| predsodki → predsotki | 6 |
| stanj → stan | 5 |
| fužine → fuzine | 4 |
| napoved → poved | 4 |
| avtocesti → autocesti | 4 |


## 2. Statistical Significance Tests (Paired Analysis)

Paired Wilcoxon signed-rank test and bootstrap 95% Confidence Intervals for the difference (Whisper − Voxtral).

| Metric | Whisper Mean | Voxtral Mean | Diff (W-V) | 95% CI | Wilcoxon p-value | Cohen's d |
|--------|--------------|--------------|------------|--------|------------------|-----------|
| WER (%) | 12.59 | 11.86 | 0.74 | [-1.91, 3.43] | 0.7317 | 0.10 |
| CER (%) | 4.39 | 3.43 | 0.96 | [-0.73, 2.86] | 0.8489 | 0.27 |
| BLEU | 78.19 | 77.84 | 0.34 | [-2.87, 3.64] | 0.6617 | 0.03 |
| Sem. similarity | 0.73 | 0.66 | 0.07 | [0.01, 0.15] | 0.1551 | 0.32 |


## 3. Error Type Distribution

| Model | Substitutions | Insertions | Deletions | Morphological | Proper Names |
|-------|---------------|------------|-----------|---------------|--------------|
| Whisper | 279 | 115 | 31 | 65 | 43 |
| Voxtral | 338 | 45 | 23 | 99 | 40 |


## 4. Per-Recording Detailed Results

|   file_index | model   | condition   |   WER (%) |   CER (%) |   Word Accuracy (%) |   SER (%) |
|-------------:|:--------|:------------|----------:|----------:|--------------------:|----------:|
|            1 | Whisper | Overall     |     24.49 |     13.68 |               75.51 |     81.82 |
|            1 | Voxtral | Overall     |      9.52 |      2.74 |               90.48 |     77.78 |
|            2 | Whisper | Overall     |     12.12 |      4.71 |               87.88 |    100    |
|            2 | Voxtral | Overall     |     18.94 |      7.42 |               81.06 |    100    |
|            3 | Whisper | Overall     |     12.77 |      3.65 |               87.23 |     33.33 |
|            3 | Voxtral | Overall     |     12.77 |      2.64 |               87.23 |      0    |
|            4 | Whisper | Overall     |      6.45 |      1.06 |               93.55 |     12.5  |
|            4 | Voxtral | Overall     |     14.52 |      6.15 |               85.48 |      0    |
|            5 | Whisper | Overall     |      8.2  |      1.39 |               91.8  |      0    |
|            5 | Voxtral | Overall     |      8.2  |      1.51 |               91.8  |      0    |
|            6 | Whisper | Overall     |     16.81 |      9.5  |               83.19 |     16.67 |
|            6 | Voxtral | Overall     |      9.24 |      3.17 |               90.76 |     80    |
|            7 | Whisper | Overall     |     25.89 |      7.76 |               74.11 |     73.33 |
|            7 | Voxtral | Overall     |     15.18 |      8.07 |               84.82 |     73.33 |
|            8 | Whisper | Overall     |     14.88 |      4.79 |               85.12 |     82.35 |
|            8 | Voxtral | Overall     |      4.13 |      0.68 |               95.87 |     82.35 |
|            9 | Whisper | Overall     |     23.85 |     13.85 |               76.15 |     25    |
|            9 | Voxtral | Overall     |      9.17 |      1.1  |               90.83 |      0    |
|           10 | Whisper | Overall     |     15.15 |      1.39 |               84.85 |     50    |
|           10 | Voxtral | Overall     |     23.23 |      8.07 |               76.77 |     75    |
|           11 | Whisper | Overall     |      7.03 |      1.39 |               92.97 |      0    |
|           11 | Voxtral | Overall     |      8.59 |      4.05 |               91.41 |      0    |
|           12 | Whisper | Overall     |      6.9  |      1.01 |               93.1  |     50    |
|           12 | Voxtral | Overall     |     10.34 |      1.52 |               89.66 |     50    |
|           13 | Whisper | Overall     |     11.67 |      3.01 |               88.33 |     33.33 |
|           13 | Voxtral | Overall     |      9.17 |      3.51 |               90.83 |      8.33 |
|           14 | Whisper | Overall     |      9.92 |      1.35 |               90.08 |      0    |
|           14 | Voxtral | Overall     |     11.57 |      3.07 |               88.43 |     62.5  |
|           15 | Whisper | Overall     |      3.74 |      0.93 |               96.26 |      0    |
|           15 | Voxtral | Overall     |      2.8  |      0.93 |               97.2  |     88.89 |
|           16 | Whisper | Overall     |     10.66 |      4.76 |               89.34 |     69.23 |
|           16 | Voxtral | Overall     |      7.38 |      1.16 |               92.62 |     58.33 |
|           17 | Whisper | Overall     |      7.3  |      1.56 |               92.7  |      0    |
|           17 | Voxtral | Overall     |     12.41 |      4.09 |               87.59 |     27.27 |
|           18 | Whisper | Overall     |     12.59 |      3.25 |               87.41 |     75    |
|           18 | Voxtral | Overall     |     13.29 |      3.14 |               86.71 |     71.43 |
|           19 | Whisper | Overall     |     10.53 |      2.66 |               89.47 |    100    |
|           19 | Voxtral | Overall     |      7.89 |      1.78 |               92.11 |    100    |
|           20 | Whisper | Overall     |      7.41 |      0.88 |               92.59 |      0    |
|           20 | Voxtral | Overall     |     11.11 |      1.91 |               88.89 |      0    |
|           21 | Whisper | Overall     |     13.33 |      4.44 |               86.67 |     14.29 |
|           21 | Voxtral | Overall     |      6.67 |      2.22 |               93.33 |     28.57 |
|           22 | Whisper | Overall     |     30.41 |      8.55 |               69.59 |    100    |
|           22 | Voxtral | Overall     |     45.27 |     13.3  |               54.73 |    100    |
|           23 | Whisper | Overall     |      8.33 |      2.74 |               91.67 |     75    |
|           23 | Voxtral | Overall     |     14.1  |      3.24 |               85.9  |     69.23 |
|           24 | Whisper | Overall     |      8.47 |      1.41 |               91.53 |     72.73 |
|           24 | Voxtral | Overall     |     11.02 |      2.04 |               88.98 |     81.82 |
|           25 | Whisper | Overall     |      6.14 |      2.5  |               93.86 |     20    |
|           25 | Voxtral | Overall     |      6.14 |      2.37 |               93.86 |     33.33 |
|           26 | Whisper | Overall     |     21.67 |     13.77 |               78.33 |     72.22 |
|           26 | Voxtral | Overall     |     13.33 |      1.75 |               86.67 |     53.85 |
|           27 | Whisper | Overall     |      3.33 |      2.5  |               96.67 |      9.09 |
|           27 | Voxtral | Overall     |      4.17 |      1.05 |               95.83 |     90    |

