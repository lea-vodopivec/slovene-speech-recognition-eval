# Primerjalna evalvacija sistemov ASR za slovenski govor

Ta repozitorij vsebuje govorni korpus, skripte in evalvacijsko kodo za primerjavo dveh sistemov za samodejno razpoznavanje govora na slovenskih govornih posnetkih:

- OpenAI Whisper (large)
- MistralAI Voxtral Small 24B

## Cilj projekta

Cilj projekta je ovrednotiti kakovost samodejnih transkripcij slovenskega govora kot jezika z manj razpoložljivimi jezikovnimi viri ter primerjati klasični transformer model ASR (Whisper) z multimodalnim jezikovnim modelom za pretvorbo govora v besedilo (Voxtral Small 24B).

Evalvacija vključuje primerjavo na leksikalni, semantični in jezikoslovni ravni.

---

# Vsebina repozitorija

## Datoteke in skripte

- `preprocess_audio.py` — pretvorba in priprava zvočnih datotek v format WAV (16 kHz, mono).
- `transcribe_whisper.py` — transkripcija z modelom Whisper (large).
- `transcribe_voxtral.py` — transkripcija z modelom Voxtral Small 24B (Transformers/PyTorch).
- `evaluate_transcriptions.py` — izračun evalvacijskih metrik in primerjava samodejnih transkriptov z referenčnimi transkripti.
- `requirements.txt` — seznam potrebnih Python knjižnic.

---

## Struktura podatkov

- `posnetki/` — originalne zvočne datoteke (M4A AAC, stereo, 48 kHz). Zaradi varovanja zasebnosti niso javno dostopne v repozitoriju. Za raziskovalne namene so dostopne na zahtevo: **prevajanje.lv@gmail.com**.

- `posnetki_preprocessed/` — predobdelane zvočne datoteke WAV (PCM, 16 kHz, mono), uporabljene kot vhod za modele.

- `original-transkript/` — referenčni človeški transkripti (UTF-8 TXT datoteke: `besedilo1.txt` … `besedilo27.txt`).

- `transkripcije/` — generirane transkripcije modelov (npr. `transkripcija1_whisper.txt`, `transkripcija1_voxtral.txt`).

- `rezultati/` — izhodne evalvacijske datoteke, vizualizacije, povzetki v CSV-formatu in poročilo.

---

## Namestitev

1. Ustvari in aktiviraj virtualno okolje Python 3.10 ali novejše

```bash
python -m venv .venv
source .venv/Scripts/activate 
pip install -r requirements.txt
```

2. Prepričaj se, da je `ffmpeg` nameščen in dostopen v sistemski poti PATH. (uporablja se za predobdelavo zvočnih datotek).

Uporaba:
1. Predobdelava zvoka (ustvari WAV datoteke v mapi `posnetki_preprocessed/`):

```bash
python preprocess_audio.py
```

2. Generiranje transkripcij z Whisper:

```bash
python transcribe_whisper.py
```

3. Generiranje transkripcij z Voxtral:

```bash
python transcribe_voxtral.py
```

4. Evalvacija transkripcij (ustvari CSV rezultate in poročilo `evaluation_report.md` v mapi `rezultati/`):

```bash
python evaluate_transcriptions.py
```

# Evalvacijske metrike
- WER (Word Error Rate) — stopnja napak na ravni besed.
- CER (Character Error Rate) — stopnja napak na ravni znakov.
- SER (Semantic Error Rate) — semantična podobnost na ravni povedi z uporabo večjezičnega semantičnega modela in kosinusne podobnosti.
- BLEU — mera prekrivanja n-gramov.
- WIP / WIL (Word Information Preserved / Lost) — delež ohranjenih oziroma izgubljenih informacij.
- NER (Named Entity Recognition) — natančnost prepoznave imenovanih entitet z uporabo večjezičnega modela spaCy.
- Morfološka natančnost — primerjava lem z uporabo slovenskega jezikovnega cevovoda `classla`.
- Natančnost ločil — primerjava pravilnosti vstavljanja ločil.


# Podatkovni korpus in pogoji snemanja
- Korpus vsebuje 27 govornih posnetkov enega govorca.
- Značilnosti posnetkov: snemanje je potekalo v tihem domačem okolju, uporabljen je bil običajen vgrajeni mikrofon, izvorni format je M4A AAC, stereo zvok, frekvenca vzorčenja 48 kHz, dolžina posameznega posnetka je približno 51–66 sekund.
- Korpus vključuje različne jezikovne registre: literarni, novinarski, obvestilni, strokovni, pravni, kuharski, turistični, spletni interakcijski in poljudnoznanstveni.
- Referenčni transkripti so ortografski in ne vsebujejo anotacije paralingvističnih pojavov, kot so mašila, premori in neverbalni zvoki.


# Povzetek rezultatov
- Oba sistema dosegata primerljivo splošno uspešnost.
- Noben model ne prevlada pri vseh evalvacijskih merah.
- Voxtral dosega nekoliko boljšo leksikalno natančnost (nižji WER in CER ter višjo uspešnost pri prepoznavi imenovanih entitet).
- Whisper kaže boljše ohranjanje semantičnega pomena in nekoliko boljšo natančnost ločil.
- Razlike med modeloma so večinoma majhne in so odvisne od registra besedila ter fonetične in leksikalne zahtevnosti posnetkov.


# Reproducibilnost
- Skripte predvidevajo izvajanje modelov z uporabo grafičnega procesorja (GPU).
- Pri izvajanju na procesorju (CPU) je izvajanje počasnejše in lahko pride do omejitev pomnilnika.
- Predobdelava vključuje samo spremembo frekvence vzorčenja in pretvorbo stereo → mono.
- Odstranjevanje šuma in normalizacija glasnosti nista uporabljena, da ostanejo vhodni podatki primerljivi med sistemi.


# Licenca

Izvorna koda v tem repozitoriju je namenjena raziskovalni in izobraževalni uporabi. 

Govorni posnetki niso javno objavljeni zaradi varovanja zasebnosti. Dostop do zvočnih podatkov je mogoč izključno za raziskovalne namene in na podlagi predhodnega dogovora.

Pri uporabi tega korpusa ali rezultatov raziskave je potrebno ustrezno navesti vir in namen uporabe podatkov.

Za dostop do zvočnih posnetkov kontaktirajte:
**prevajanje.lv@gmail.com**
