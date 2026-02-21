# Chapter 08 — Data Preparation & Quality: Building a Vietnamese TTS Corpus

> **Audience**: ML engineers new to audio dataset curation and TTS-specific data requirements.
> **Goal**: Understand the mathematical and practical criteria for high-quality Vietnamese TTS data, and build a complete preprocessing pipeline from raw recordings to encoded training data.

---

## Table of Contents

1. [What Makes a Good TTS Corpus](#1-what-makes-a-good-tts-corpus)
2. [Vietnamese-Specific Challenges](#2-vietnamese-specific-challenges)
3. [Audio Quality Metrics](#3-audio-quality-metrics)
4. [The Filter Pipeline (filter_data.py)](#4-the-filter-pipeline-filter_datapy)
5. [Audio Preprocessing](#5-audio-preprocessing)
6. [Encoding Pipeline (encode_data.py)](#6-encoding-pipeline-encode_datapy)
7. [Dataset Statistics to Check](#7-dataset-statistics-to-check)

---

## 1. What Makes a Good TTS Corpus

### 1.1 Phoneme and Tone Coverage

A TTS model learns a mapping from linguistic units (phonemes, tones, words) to acoustic patterns. If the training corpus does not contain examples of certain phonemes or tones, the model cannot learn to produce them correctly — it will either interpolate poorly or collapse to the nearest seen example.

**Phoneme coverage** is measured as:

$$\text{Coverage} = \frac{|\mathcal{P}_{\text{dataset}} \cap \mathcal{P}_{\text{language}}|}{|\mathcal{P}_{\text{language}}|}$$

For standard Vietnamese (Northern dialect), the phoneme inventory $\mathcal{P}_{\text{language}}$ consists of approximately:
- 22 initial consonants: /b, m, f, v, t, t', d, n, z, ʐ, s, ʂ, c, ɲ, trʈ, x, ŋ, h, l, ɣ, χ/
- 14 vowel nuclei (including diphthongs): /a, ă, â, e, ê, i, o, ô, ơ, u, ư, y, ia, ua/
- 8 final consonants: /m, n, ŋ, p, t, k, j, w/
- **6 tones**: ngang (flat), huyền (falling), sắc (rising), hỏi (dipping-rising), ngã (creaky-rising), nặng (heavy-falling)

**Minimum recommended coverage: ≥ 95%** of the phoneme inventory, **100% of all 6 tones**.

**Practical check:** Use a Vietnamese g2p (grapheme-to-phoneme) tool to convert all training texts to phoneme sequences, then count unique phonemes and compare to the full inventory.

### 1.2 Duration Distribution

The duration of individual audio clips affects training in several ways:

**Why 3–15 seconds is optimal:**

- **Too short (< 3 s):** Insufficient context for the model to establish prosodic patterns. Very short clips also have a disproportionate amount of silence (onset/offset silence accounts for ~0.5 s regardless of total duration), reducing effective speech content ratio.

- **Too long (> 15 s):** Exceeds the codec sequence length that the model was pretrained on. A 15-second clip at 24 kHz with NeuCodec produces approximately $15 \times 50 = 750$ codec frames, which is within the model's context window but at its upper limit. Longer clips also produce longer token sequences that consume disproportionate GPU memory.

- **Optimal (3–15 s):** Provides enough context for sentence-level prosody while keeping sequence lengths manageable. Natural sentence boundaries at this duration also correspond to grammatically complete units, which helps the model learn natural prosodic phrasing.

**Duration distribution target:**

$$p(\text{duration}) \sim \text{Log-Normal}(\mu_{\ln} = \ln(6), \sigma_{\ln}^2 = 0.5)$$

A log-normal distribution with peak around 6 seconds and a long tail to 15 seconds is typical of natural Vietnamese speech (declarative sentences average 6–8 seconds at natural speaking rate).

### 1.3 Speaker Consistency

TTS fine-tuning for a single voice requires **strict speaker consistency**. Even a few samples from a different speaker can cause the model to learn a blend, resulting in voice instability:

- All recordings must be from the **same physical microphone** (or at least the same microphone model)
- **Same room** or same acoustic environment (consistent reverb)
- **Same time period** (people's voices change over weeks, especially if sick)
- **No post-processing differences** between clips (no EQ, no reverb on some clips)

### 1.4 Recording Environment

The target acoustic environment for training data:

**Noise floor:** $\leq -60 \text{ dBFS}$ (decibels relative to full scale)

This means the RMS level of background noise (measured during silence segments) should be at most $10^{-60/20} = 0.001$ of the maximum possible signal amplitude. A typical quiet home studio achieves $-65$ to $-70$ dBFS; a treated recording booth achieves $-75$ to $-80$ dBFS.

**Signal level:** Record at $-12$ to $-6$ dBFS peak (leave headroom above the noise floor but below clipping). This corresponds to RMS levels around $-18$ to $-12$ dBFS for speech.

**Reverberation time ($T_{60}$):** Ideally < 150 ms. High reverb blurs phoneme boundaries and reduces intelligibility of stop consonants, which are perceptually critical in Vietnamese.

### 1.5 Sample Rate

**Record at:** 44.1 kHz or 48 kHz (professional audio standard)

**Downsample to:** 16 kHz for NeuCodec encoding

The 16 kHz downsampling is applied by the encoding pipeline. Recording at higher sample rate preserves the option to use higher-quality codecs in the future and avoids aliasing artifacts from field-captured audio.

**Why 16 kHz for the codec?** NeuCodec was trained on 16 kHz speech. Operating outside its training distribution degrades codec reconstruction quality. The speech frequency range critical for intelligibility (300–3400 Hz for telephone bandwidth, 100–8000 Hz for wideband speech) is fully preserved at 16 kHz (Nyquist frequency = 8000 Hz).

---

## 2. Vietnamese-Specific Challenges

### 2.1 The Six-Tone System

Vietnamese is a **tonal language** — the pitch contour of a syllable is a lexeme-distinguishing feature (not merely a prosodic modifier as in English). The six tones of Northern Vietnamese are:

| Tone | Name | Diacritic example | Pitch contour | Register |
|------|------|-------------------|---------------|----------|
| 1 | Ngang (level) | a | Mid-level, flat | Mid |
| 2 | Huyền (falling) | à | Low-falling, breathy | Low |
| 3 | Sắc (rising) | á | High-rising | High |
| 4 | Hỏi (dipping-rising) | ả | Mid-dipping then rising | Mid-low |
| 5 | Ngã (creaky-rising) | ã | Mid-rising with creakiness/glottalization | High |
| 6 | Nặng (heavy-falling) | ạ | Low-falling, heavy/constricted | Low |

**The tone bias problem:** In natural Vietnamese text, tones are not equally distributed. Studies show that ngang (1) and huyền (2) account for ~50% of syllables in typical prose, while ngã (5) and nặng (6) are less frequent. A naively collected corpus will have this imbalance, causing the model to:
- Generate flat/falling tones too often on ambiguous contexts
- Mispronounce ngã as sắc (both are rising but different register)

**Mitigation:** Curate text specifically designed for tone coverage. Include poetry (Vietnamese poetry explicitly cycles through tones), tongue twisters, and minimal tone-pair examples.

### 2.2 Dialect Considerations

Vietnamese has three major dialect groups:

| Dialect | Region | Key phonetic differences |
|---------|--------|--------------------------|
| Northern (Hà Nội) | North Vietnam | Preserves all 6 tones; nh, ng distinct |
| Central (Huế, Đà Nẵng) | Central Vietnam | 5 or 6 tones with different contours; distinct finals |
| Southern (TP. HCM) | South Vietnam | 5 tones (ngã → nặng merger); initial consonant mergers |

**Critical rule:** Train on exactly **one dialect at a time**. Mixing dialects in training data creates a hybrid model that:
- Has unstable tone realization (switching between dialects mid-sentence)
- Produces inconsistent vowel quality
- Confuses the model's prosodic patterns (sentence-final intonation differs significantly between dialects)

**Southern Vietnamese specific:** The ngã (5) and nặng (6) tones merge in Southern dialect — they are produced identically (falling-constricted). This means:
1. Southern speakers' training data will not give the model any signal to distinguish these tones
2. If you fine-tune on Southern data and then ask the model to produce ngã explicitly, it may produce nặng

**Solution for Southern dialect models:** Accept the merger — train with both ngã and nặng words and let the model learn the regional variant.

### 2.3 Numbers, Dates, and Abbreviations

**The fundamental problem:** Written Vietnamese numbers and abbreviations are morphologically ambiguous — they have multiple valid readings depending on context.

Examples:
- "123" → "một trăm hai mươi ba" (cardinal), "một hai ba" (digit-by-digit), "thứ một trăm hai mươi ba" (ordinal)
- "2/9" → "ngày hai tháng chín" (date, September 2) or "hai phần chín" (fraction 2/9)
- "TP.HCM" → "Thành phố Hồ Chí Minh" (full expansion) or "tê-pê Hồ Chí Minh" (partial)

If the training corpus contains these ambiguous forms, the model cannot learn a consistent pronunciation — it will be exposed to the same written form with different acoustic realizations, creating contradictory training signal.

**Solution in filter_data.py:**
```python
import re

def has_digits(text):
    return bool(re.search(r'\d', text))

def has_acronym(text):
    # Acronyms: 2+ consecutive uppercase letters
    return bool(re.search(r'[A-Z]{2,}', text))
```

Filter out any sample containing digits or acronyms. The preferred approach is **text normalization** (convert "123" → "một trăm hai mươi ba" using a normalization library) before including the sample in the dataset.

### 2.4 Code-Switching

Vietnamese speakers, particularly in technology and business contexts, frequently switch between Vietnamese and English within a sentence:

> "Chúng ta cần update cái database này trước khi meeting."

VieNeu-TTS handles code-switching through its tokenizer (which includes Latin characters) and the pretrained model's exposure to multilingual data. Including code-switching examples in fine-tuning data **improves the model's ability to handle mixed-language inputs** in deployment.

Recommended: Include 10–15% code-switching samples in the training set, with English words appearing naturally in Vietnamese sentences.

### 2.5 Common Recording Errors

For quality control, listen specifically for these Vietnamese-specific recording artifacts:

1. **Glottal stops:** Some speakers insert glottal stops before initial vowels (especially in stressed syllables). This is natural in some dialects but may confuse the codec's boundary detection.

2. **Tone sandhi:** In fast speech, adjacent tones can assimilate (e.g., hỏi + hỏi → hỏi + ngã). This is normal but should be consistent across the corpus.

3. **Final consonant dropping (Southern dialect):** /k/, /t/, /p/ finals are unreleased or dropped in fast speech. If training on Southern data, ensure this is consistent — either all dropped or all present.

4. **Diphthong gliding:** The diphthong /ia/ may be pronounced as [iə] or [ie] depending on speaker and dialect. This is acceptable if consistent.

---

## 3. Audio Quality Metrics

### 3.1 Signal-to-Noise Ratio (SNR)

SNR measures the ratio of useful signal power to background noise power:

$$\text{SNR} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right) \text{ dB}$$

where $P_{\text{signal}} = \frac{1}{N_s}\sum_{n \in \text{speech}} x[n]^2$ and $P_{\text{noise}} = \frac{1}{N_n}\sum_{n \in \text{noise}} x[n]^2$.

In practice, speech and noise are separated using Voice Activity Detection (VAD) — silence segments approximate the noise floor.

**Interpretation for TTS data:**

| SNR range | Quality | Action |
|-----------|---------|--------|
| > 40 dB | Excellent | Use directly |
| 30–40 dB | Good | Acceptable |
| 20–30 dB | Fair | Denoise before use |
| 10–20 dB | Poor | Likely to harm training |
| < 10 dB | Unusable | Discard |

**Why SNR matters for TTS training:** Low-SNR samples train the model to reproduce noise artifacts. At test time on clean text, the model may generate speech with background noise characteristics ("hallucinated noise"). A threshold of **SNR > 25 dB** is used in the VieNeu-TTS filter pipeline.

### 3.2 PESQ (Perceptual Evaluation of Speech Quality)

PESQ (ITU-T P.862) is a full-reference metric that compares a degraded signal to a clean reference:

$$\text{PESQ}_{\text{WB}} \in [-0.5, 4.5]$$

- $\geq 4.0$: Excellent (indistinguishable from clean reference)
- $3.0$ to $4.0$: Good (slight, acceptable degradation)
- $2.0$ to $3.0$: Fair (noticeable degradation)
- $< 2.0$: Poor (significant quality loss)

PESQ requires a clean reference, so it is used for **evaluating the quality of synthesized speech** (compare synthesis to ground-truth recording) rather than for data filtering. It is particularly useful for measuring codec reconstruction quality.

### 3.3 MOS (Mean Opinion Score)

MOS is the gold standard for speech quality evaluation:

$$\text{MOS} = \frac{1}{N_E} \sum_{e=1}^{N_E} \frac{1}{N_S} \sum_{s=1}^{N_S} \text{rating}_{e,s}$$

where $N_E$ is the number of evaluators and $N_S$ is the number of samples evaluated.

The ACR (Absolute Category Rating) scale:

| Score | Label | Meaning |
|-------|-------|---------|
| 5 | Excellent | No perceptible degradation |
| 4 | Good | Slightly annoying |
| 3 | Fair | Annoying but not objectionable |
| 2 | Poor | Fairly objectionable |
| 1 | Bad | Very objectionable |

**Confidence interval for MOS:** With $N_E$ raters, the 95% confidence interval is approximately $\pm 1.96 \cdot \sigma / \sqrt{N_E}$. For $\sigma \approx 0.8$ (typical inter-rater variance) and $N_E = 10$ raters, this gives $\pm 0.50$ — a fairly wide interval. Reliable MOS requires at least $N_E \geq 20$ raters.

### 3.4 Spectral Flatness

Spectral flatness (Wiener entropy) measures how "noise-like" a spectrum is:

$$\text{SF} = \frac{\exp\left(\frac{1}{N}\sum_{k=0}^{N-1} \ln X[k]\right)}{\frac{1}{N}\sum_{k=0}^{N-1} X[k]}$$

where $X[k]$ is the power spectral density at bin $k$.

- $\text{SF} \approx 1.0$: White noise (flat spectrum)
- $\text{SF} \approx 0.0$: Pure tone (all energy at one frequency)
- Speech: $\text{SF} \approx 0.1$ to $0.4$ depending on phoneme

**Application:** High spectral flatness in silence segments indicates background noise (not pure electrical hum). Low spectral flatness during speech segments indicates a clean, tonal signal — desirable.

### 3.5 Voice Activity Detection (VAD)

VAD determines which frames contain speech vs. silence/noise. The speech ratio $r_s$ is:

$$r_s = \frac{N_{\text{speech frames}}}{N_{\text{total frames}}}$$

For TTS training data:
- **Target:** $r_s > 0.7$ (at least 70% of the clip is active speech)
- **Too low $r_s$** (< 0.5): Clip has too much silence — either record more densely or trim more aggressively
- **Too high $r_s$** (> 0.95): Clip may have been over-trimmed — missing natural phrase boundaries

---

## 4. The Filter Pipeline (filter_data.py)

### 4.1 Pipeline Overview

The filter pipeline processes a `metadata.csv` file with columns `[filename, text]` and produces a filtered `metadata_filtered.csv`. Each sample passes through a series of checks:

```
metadata.csv
    │
    ▼
[Duration check]          → discard if < 3s or > 15s
    │
    ▼
[Sample rate check]       → discard if sr < 16000
    │
    ▼
[Text: digit check]       → discard if contains digits
    │
    ▼
[Text: acronym check]     → discard if contains 2+ uppercase consecutive
    │
    ▼
[Text: punctuation check] → discard if does not end with . ! ?
    │
    ▼
[Text: length check]      → discard if text too short (< 5 chars)
    │
    ▼
metadata_filtered.csv
```

### 4.2 Duration Check Implementation

```python
import soundfile as sf

def check_duration(filepath, min_s=3.0, max_s=15.0):
    info = sf.info(filepath)
    duration = info.duration
    return min_s <= duration <= max_s, duration
```

**Why `sf.info()` instead of loading the full audio?**

`soundfile.info()` reads only the file header (metadata), not the audio samples. This is approximately 1000× faster for large files:

- `sf.info("file.wav")` on a 10-second file: ~0.1 ms
- `librosa.load("file.wav", sr=16000)` on a 10-second file: ~150 ms

For a dataset of 10,000 files, this difference is 1 second vs 25 minutes.

### 4.3 Text Rules and Rationale

**Digit filter:**
```python
import re
has_digit = bool(re.search(r'\d', text))
```

Rationale: Numbers have ambiguous Vietnamese pronunciation (cardinal vs. ordinal vs. digit-by-digit). Including them creates contradictory training signal.

**Acronym filter:**
```python
has_acronym = bool(re.search(r'[A-Z]{2,}', text))
```

Rationale: Acronyms like "TP.HCM", "GDP", "AI" have multiple valid expansions and non-standard phonology. The pretrained model handles some acronyms via its pretraining distribution, but fine-tuning data should be unambiguous.

**Punctuation filter:**
```python
ends_correctly = text.strip().endswith(('.', '!', '?', ','))
```

Rationale: Sentence-final punctuation signals prosodic completion. A sentence without terminal punctuation may represent an incomplete utterance, which creates inconsistent prosodic patterns at the end of clips.

**Length filter:**
```python
min_text_length = 5  # characters
valid_length = len(text.strip()) >= min_text_length
```

Rationale: Very short texts (< 5 chars) are often isolated words with unusual prosody that does not match full-sentence speech.

### 4.4 Expected Pass Rate

For well-prepared studio recordings from a single speaker:

| Filter | Expected pass rate |
|--------|-------------------|
| Duration (3–15 s) | 90–95% |
| Sample rate ≥ 16 kHz | 99%+ |
| No digits | 70–80% (natural text has many numbers) |
| No acronyms | 85–90% |
| Ends with punctuation | 90–95% |
| All filters combined | **60–80%** |

**What to do with rejected samples:**
- Digit-containing: run through Vietnamese text normalization (e.g., `underthesea.num2words()`) and re-evaluate
- Duration-too-long: split at natural pause points using VAD
- Duration-too-short: merge with adjacent sentence if they are semantically connected

### 4.5 Handling Filtered Samples

It is a mistake to simply discard all rejected samples, especially when the dataset is small. For each rejection reason:

1. **Numbers/digits:** Use a normalizer to convert `"Năm 2024 là năm thú vị."` → `"Năm hai nghìn không trăm hai mươi tư là năm thú vị."` then pass through the filter again.

2. **Too long:** Use `librosa.effects.split()` to find natural silence boundaries, split at silences > 0.3 seconds, creating multiple shorter clips.

3. **Acronyms:** Manually expand the most common ones; skip rare ones.

---

## 5. Audio Preprocessing

### 5.1 Resampling

VieNeu-TTS's NeuCodec operates at 16 kHz. Resampling from the recording sample rate (typically 44.1 kHz or 48 kHz) uses a **polyphase filterbank**:

$$x_{16k}[n] = \sum_{k=-\infty}^{\infty} x_{\text{orig}}[k] \cdot h_{\text{lp}}[n \cdot D/U - k]$$

where:
- $D$ is the downsampling factor
- $U$ is the upsampling factor
- $h_{\text{lp}}$ is a low-pass anti-aliasing filter with cutoff at $f_{\text{Nyquist}} = 8000$ Hz

The `librosa.resample()` function uses `soxr` (Sound eXchange Resampler) by default, which applies a Kaiser-windowed sinc filter with > 100 dB stopband attenuation. This is important to avoid **aliasing** — frequencies above 8 kHz folding back into the audible range.

```python
import librosa
wav_16k = librosa.resample(wav_orig, orig_sr=sr_orig, target_sr=16000, res_type='soxr_hq')
```

### 5.2 Amplitude Normalization

**Peak normalization** scales the signal so the maximum amplitude equals a target level:

$$x_{\text{norm}}[n] = \frac{x[n]}{\max_n |x[n]|} \cdot 10^{L_{\text{target}}/20}$$

For $L_{\text{target}} = -3 \text{ dBFS}$: $10^{-3/20} \approx 0.708$

Peak normalization ensures no clipping and that all samples have consistent peak level. It is preferred over RMS normalization for TTS because RMS normalization can artificially amplify soft or whispered speech.

**DC offset removal:**
```python
x_dc_free = x - np.mean(x)
```

DC offset (a constant positive or negative bias) can occur from microphone circuits or recording interface impedance mismatches. It does not affect the codec directly but can cause audible clicks at clip boundaries.

### 5.3 Silence Trimming

Librosa's `trim()` function removes leading and trailing silence:

```python
wav_trimmed, trim_indices = librosa.effects.trim(wav, top_db=25, frame_length=2048, hop_length=512)
```

**Algorithm:**
1. Compute short-time RMS energy with `frame_length=2048` samples (128 ms at 16 kHz)
2. Convert to dB: $E_{\text{dB}}[f] = 20 \log_{10}(E_{\text{rms}}[f])$
3. Mark frame as silence if $E_{\text{dB}}[f] < E_{\text{max}} - \text{top\_db}$
4. Trim from both ends until non-silence is found

**top_db=25** means: trim frames more than 25 dB below the loudest frame. This is a balanced threshold — 20 dB would trim too aggressively (clipping consonant onsets), 30 dB would leave too much silence.

### 5.4 Pre-emphasis Filter

Some preprocessing pipelines apply a pre-emphasis filter to boost high-frequency energy (which naturally falls off in speech):

$$x_{\text{pe}}[n] = x[n] - \alpha x[n-1], \quad \alpha \approx 0.97$$

**VieNeu-TTS does NOT apply pre-emphasis** to the time-domain audio before NeuCodec encoding, because NeuCodec was trained on unfiltered audio. Applying pre-emphasis would shift the signal distribution outside the codec's training distribution.

Pre-emphasis is only relevant for **feature extraction** (MFCC computation), not for neural codec encoding.

---

## 6. Encoding Pipeline (encode_data.py)

### 6.1 NeuCodec vs DistillNeuCodec for Training

VieNeu-TTS uses two codecs for different purposes:

| Codec | Use case | Quality | Speed |
|-------|----------|---------|-------|
| NeuCodec (full RVQ) | Training data encoding | Highest | Slow |
| DistillNeuCodec | Inference decoding | Slightly lower | 4× faster |

**Why use NeuCodec (full RVQ) for training?**

NeuCodec's full Residual Vector Quantization (RVQ) uses multiple codebook levels, capturing fine-grained spectral details. Training on these richer codes gives the backbone model more information to learn from. The DistillNeuCodec is a distilled, faster version that trades a small amount of quality for inference speed — acceptable at inference time but would reduce information density during training.

### 6.2 Encoding Process

The encoding process converts each audio clip to a sequence of integer codebook indices:

```
audio.wav (16 kHz, float32)
    │
    ▼
NeuCodec.encode_code(audio_tensor)
    │
    ▼
codes: shape [num_reps, num_frames]
    │
    ▼
flatten → 1D integer sequence
    │
    ▼
JSON string: "[1234, 5678, ...]"
```

The output format for `metadata_encoded.csv`:

```
filename|text|codes
audio_001.wav|Xin chào Việt Nam.|[1234, 5678, 9012, ...]
audio_002.wav|Hôm nay trời đẹp quá.|[2345, 6789, ...]
```

The `|` separator is used instead of `,` because the codes list itself is comma-separated JSON.

### 6.3 Code Range Validation

NeuCodec codes are unsigned 16-bit integers in range $[0, 65535]$. After encoding, validate:

```python
codes_list = json.loads(codes_str)
assert all(0 <= c < 65536 for c in codes_list), "Code out of range!"
assert len(codes_list) > 0, "Empty codes!"
```

Common causes of invalid codes:
- Audio is all zeros (silent file) → codec may produce 0-valued codes or error
- Corrupted audio file with NaN samples → codec produces undefined output
- Incorrect sample rate passed to codec → produces garbled codes

### 6.4 Shuffling and max_samples Cap

The encoded dataset is **randomly shuffled** before training:

```python
import random
random.shuffle(samples)
if max_samples is not None:
    samples = samples[:max_samples]
```

**Why shuffle matters:**
- Avoids model overfitting to recording session ordering (typically, session 1 is often "warmer" speech than session 3)
- Prevents gradient bias toward the most recent speaker style if recordings are grouped by time
- Ensures each batch has diverse text content (without shuffling, adjacent samples may be similar sentences)

**max_samples cap:** Useful during debugging to train on a small subset quickly. Set to `None` for full training.

### 6.5 Sequence Length Considerations

The maximum sequence length in the training configuration limits how many codec tokens a single training sample can contain:

$$N_{\text{max tokens}} = N_{\text{text tokens}} + N_{\text{voice prompt tokens}} + N_{\text{speech tokens}}$$

For a 10-second audio clip:
- Speech tokens: $10 \text{ s} \times 50 \text{ Hz} = 500 \text{ frames} \times K_{\text{codebooks}} \approx 500$
- Text tokens: typically 20–50 for a Vietnamese sentence
- Voice prompt tokens: typically 100–200

Total: ~620–750 tokens. The model's `max_seq_len` should be set to ≥ 1024 to accommodate these.

---

## 7. Dataset Statistics to Check

### 7.1 Duration Distribution

Plot a histogram of clip durations after filtering:

```python
import matplotlib.pyplot as plt
durations = df['duration_s'].values
plt.hist(durations, bins=30, edgecolor='black')
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.axvline(3, color='r', linestyle='--', label='Min (3s)')
plt.axvline(15, color='r', linestyle='--', label='Max (15s)')
plt.title("Duration Distribution")
```

**Red flags:**
- Spike at exactly 3 s or 15 s (clips being cut at the boundary — may indicate a chunking error)
- Very long tail beyond 15 s (filter didn't apply correctly)
- All clips very close to same duration (unnatural recording setup)

**Target total dataset size for single-voice fine-tuning:**
- Minimum: 15 minutes (900 seconds total)
- Recommended: 30–60 minutes
- Diminishing returns above: 120 minutes

### 7.2 Tone Distribution

For each text sample, count syllables by tone mark using Unicode analysis. Vietnamese tone marks are realized as diacritics on vowels:

| Tone | Unicode combining marks | Example |
|------|------------------------|---------|
| Ngang (1) | None | a, e, i |
| Huyền (2) | U+0300 (combining grave) | à, è, ì |
| Sắc (3) | U+0301 (combining acute) | á, é, í |
| Hỏi (4) | U+0309 (combining hook above) | ả, ẻ, ỉ |
| Ngã (5) | U+0303 (combining tilde) | ã, ẽ, ĩ |
| Nặng (6) | U+0323 (combining dot below) | ạ, ẹ, ị |

**Target distribution:** Within a factor of 2 across all tones. For example, if ngang has 40% of syllables, the rarest tone should have at least 20%.

### 7.3 Sentence Length Distribution

Plot the distribution of text lengths (in characters and in words):

```python
df['word_count'] = df['text'].apply(lambda t: len(t.split()))
df['char_count'] = df['text'].apply(len)
```

**Targets:**
- Word count: most sentences 5–25 words (Vietnamese sentences)
- Char count: most sentences 20–120 characters

Long-tail sentences (> 30 words) should be clipped or split — they risk exceeding the model's context window.

### 7.4 Phoneme Coverage Analysis

Using a Vietnamese grapheme-to-phoneme converter:

```python
from underthesea import word_tokenize
# Or use a custom g2p for Vietnamese
```

After converting all training texts to phoneme sequences, check:

$$\text{Coverage}_{i} = \frac{\text{count}(\text{phoneme}_i \text{ in dataset})}{\text{total phonemes in dataset}}$$

A balanced phoneme distribution ensures the model has seen each sound in many different contexts. Flag any phoneme with coverage < 0.5% of total phonemes for targeted augmentation.

### 7.5 Pre-Training Checklist Summary

Before running `encode_data.py`, verify:

```
[ ] All audio files exist and are readable
[ ] Duration range: 3–15 seconds for all files
[ ] Sample rate: ≥ 16 kHz
[ ] Single channel (mono)
[ ] No clipping (max absolute value < 1.0)
[ ] SNR > 25 dB (estimated)
[ ] No digits in any text
[ ] No consecutive uppercase letters (acronyms)
[ ] All texts end with . ! or ?
[ ] At least one example of each of the 6 Vietnamese tones
[ ] Speaker consistency (all from same speaker)
[ ] Total duration ≥ 15 minutes (900 seconds)
```

Passing all checks does not guarantee a perfect dataset, but it eliminates the most common causes of training failure and degraded voice quality.

---

## Summary

| Aspect | Recommendation | Minimum |
|--------|---------------|---------|
| Total duration | 30–60 min | 15 min |
| Clip duration | 4–12 s | 3–15 s |
| Sample rate | Record at 44.1/48 kHz → encode at 16 kHz | 16 kHz |
| SNR | > 40 dB | > 25 dB |
| Tone coverage | All 6 tones > 5% each | All 6 present |
| Dialect | Single dialect only | Consistent |
| Numbers/acronyms | None (normalize first) | None |
| Shuffling | Always before training | Required |

The next chapter covers how to use this prepared dataset to fine-tune VieNeu-TTS, monitor the training process, and evaluate the results.
