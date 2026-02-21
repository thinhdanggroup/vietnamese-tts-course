# Chapter 06 — Zero-Shot Voice Cloning

## Overview

Voice cloning is the task of synthesizing speech in a target speaker's voice — their pitch, timbre, speaking style, and accent — given only a short audio sample. **Zero-shot** cloning means the system was never trained on that specific speaker: it generalizes from the reference audio alone, with no fine-tuning at inference time.

This chapter covers the complete theory and practice of zero-shot voice cloning in VieNeu-TTS, from classical speaker embedding approaches to modern in-context learning mechanisms. We also address the unique challenges posed by Vietnamese, a tonal language with significant regional dialect variation.

By the end of this chapter you will understand:
- What "voice identity" is in acoustic and representational terms
- Why traditional speaker embedding methods fail in the zero-shot setting
- How in-context learning via codec tokens achieves zero-shot cloning
- What makes a good reference clip for Vietnamese voices
- How to handle code-switching between Vietnamese and English
- How to measure speaker similarity quantitatively

---

## 1. What Is Voice Cloning?

### 1.1 Definitions

**Voice cloning** refers to synthesizing speech in a target speaker's voice, characterized by:
- **Fundamental frequency (F0) range:** The speaker's typical pitch range and intonation patterns
- **Vocal tract resonances (formants):** Determined by physical anatomy — unique per speaker
- **Timbre:** The spectral envelope shape, including the harmonic structure of the voice
- **Prosodic patterns:** Rhythm, pacing, pause structure, emphasis patterns
- **Phonetic realization:** Regional accent, vowel quality, consonant articulation

**Zero-shot cloning** (what VieNeu-TTS does):
- Requires: 1 reference audio clip + its transcript
- No fine-tuning, no optimization at inference time
- Voice conditioning is done purely through the prompt
- Target speakers may be completely unseen during training

**Few-shot cloning:**
- Requires: 1-5 minutes of reference audio
- Optional fine-tuning of the last few transformer layers (LoRA adaptation)
- Better speaker similarity than zero-shot, especially for unusual voices

**Full clone:**
- Requires: hours of audio + full fine-tuning
- Used in production voice assistant systems
- Covered in Chapters 7-8 (fine-tuning section)

### 1.2 The Fundamental Challenge

Voice identity is a high-dimensional, speaker-specific manifold embedded in acoustic space. The challenge of zero-shot cloning is:

1. **Disentanglement:** Separate voice identity from linguistic content in the reference
2. **Transfer:** Apply the extracted voice to arbitrary new content
3. **Generalization:** Do this for speakers never seen during training

Traditional TTS systems addressed (1) and (2) but failed at (3). The breakthrough of LLM-based TTS is that in-context learning naturally solves all three simultaneously.

---

## 2. Traditional Approaches: Speaker Embeddings

### 2.1 d-Vector (Deep Speaker Embedding)

The d-vector approach trains a deep neural network for **speaker classification**. Given a segment of speech, the network predicts which of $N$ training speakers produced it.

**Architecture:**
1. Feature extraction: MFCC or log-filterbank features → $(T, F)$ matrix
2. Time aggregation: average pooling or LSTM → single vector $h \in \mathbb{R}^D$
3. Classification head: $\hat{y} = \text{softmax}(h W + b) \in \mathbb{R}^N$
4. Loss: cross-entropy over speaker identities

After training, the classification head is removed. The penultimate-layer activation $h$ is the **d-vector** — a compact representation of speaker identity.

**Why it works:** The network must compress all speaker-discriminative information into $h$ to classify speakers. It therefore learns to encode precisely the characteristics that distinguish speakers (vocal tract shape, pitch, speaking style).

**Limitation:** The d-vector space is learned from the training speakers. For a new, unseen speaker, the d-vector may fall outside the distribution the decoder was trained to handle — causing the synthesis to default toward a "generic" voice or produce artifacts.

### 2.2 x-Vector (TDNN-Based Embedding)

The x-vector replaces the LSTM/DNN with a **Time Delay Neural Network (TDNN)**:

$$h_t^{(l)} = f\!\left(W^{(l)} \begin{bmatrix} h_{t-d}^{(l-1)} \\ \vdots \\ h_{t+d}^{(l-1)} \end{bmatrix} + b^{(l)}\right)$$

Each layer accesses a temporal context of $2d+1$ frames. Stacking TDNNs with increasing delays gives exponentially growing receptive fields without recurrence.

After a statistics-pooling layer (mean and standard deviation over time) and two fully-connected layers, the system produces an x-vector. Trained with additive margin softmax (AM-Softmax) or AAM-Softmax for better class separation:

$$\mathcal{L}_\text{AAM} = -\log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos\theta_j}}$$

where $m$ is the additive angular margin that pushes class boundaries apart.

**x-Vectors in VieNeu-TTS evaluation:** x-vectors (or similar speaker encoders) are used as a **metric** for speaker similarity — not for voice conditioning. We compute cosine similarity between the x-vector of the reference audio and the x-vector of the generated audio to measure how well the voice was cloned.

### 2.3 i-Vector (Total Variability Space)

The i-vector is a classical (pre-deep-learning) approach based on a Gaussian Mixture Model Universal Background Model (GMM-UBM).

**Method:**
1. Train a GMM-UBM on all speakers: $\lambda_\text{UBM} = \{w_c, \mu_c, \Sigma_c\}_{c=1}^{C}$ ($C$ Gaussian components)
2. For a new utterance, compute Baum-Welch statistics: $N_c = \sum_t \gamma_c(t)$, $F_c = \sum_t \gamma_c(t) x_t$
3. Adapt: use a low-rank **total variability matrix** $T \in \mathbb{R}^{(C \cdot D) \times K}$ to project statistics to a $K$-dimensional i-vector

The i-vector represents both speaker and channel variability in a shared low-dimensional space. Channel (session) effects are removed with PLDA (Probabilistic Linear Discriminant Analysis).

I-vectors were the state of the art before 2017. They are still used in some forensic speaker recognition applications. For TTS, they have been largely superseded by neural approaches.

### 2.4 Conditioning on Speaker Embeddings

Traditional speaker-conditioned TTS (e.g., Tacotron 2 with speaker embeddings) concatenates or adds the speaker embedding to the decoder input at each step:

$$\text{decoder input}_t = [c_t; e_\text{speaker}]$$

where $c_t$ is the text context and $e_\text{speaker}$ is the d-vector or x-vector.

**Fundamental limitation:** At inference for a new speaker, $e_\text{speaker}$ is computed from the reference audio — but the decoder was never trained to handle this specific speaker's embedding. The embedding may be in a region of embedding space that the decoder maps to an incorrect voice or artifacts.

This is the core reason why traditional approaches fail at zero-shot cloning: the model has a fixed "voice space" defined by its training speakers. In-context learning eliminates this constraint entirely.

---

## 3. In-Context Voice Cloning (LLM Approach)

### 3.1 The Mechanism: Reference Tokens as Context

In VieNeu-TTS, voice conditioning is achieved by inserting the reference speaker's codec tokens directly into the prompt:

```
... <|SPEECH_GENERATION_START|> [ref_code_1][ref_code_2]...[ref_code_T] [generate]
```

The model is trained on data where this prompt format always corresponds to: "continue generating speech in the same voice as the reference codes." During training, the model sees thousands of (reference segment, continuation segment) pairs for each speaker — learning that the acoustic style of the continuation should match the reference.

### 3.2 What the Model Implicitly Learns

Consider the model's computation from a mechanistic view. The self-attention over the reference codes computes, at each layer, query-key-value operations that aggregate information across the reference token sequence.

Specifically, let $H_\text{ref} \in \mathbb{R}^{T_\text{ref} \times d}$ be the hidden states over the reference tokens. During generation of token $t$:

$$\alpha_i = \text{softmax}\left(\frac{q_t \cdot k_i}{\sqrt{d_k}}\right), \quad i = 1, \ldots, T_\text{ref}$$

$$\text{context from ref} = \sum_{i=1}^{T_\text{ref}} \alpha_i v_i$$

The model aggregates the reference token representations, weighted by how relevant each reference token is to the current generation step. Over training, the model learns:
- Early attention heads: extract speaker-specific timbre features from reference
- Later attention heads: condition the prosody of generated tokens on reference patterns
- Cross-layer propagation: voice information from reference flows into the generation stream

### 3.3 Formal Probabilistic Framework

The model parameterizes:

$$P_\theta(s_t \mid s_{<t}, \text{ref\_codes}, \mathbf{x})$$

where:
- $s_t \in \{0, \ldots, V_s - 1\}$ — generated speech token at step $t$
- $s_{<t}$ — all previously generated speech tokens
- $\text{ref\_codes} = [r_1, r_2, \ldots, r_{T_r}]$ — reference audio tokens
- $\mathbf{x} = [x_1, \ldots, x_L]$ — input text phoneme tokens

The joint distribution over the full generated sequence:

$$P_\theta(\mathbf{s} \mid \text{ref\_codes}, \mathbf{x}) = \prod_{t=1}^{T} P_\theta(s_t \mid s_{<t}, \text{ref\_codes}, \mathbf{x})$$

Training maximizes log-likelihood over a corpus of (ref, target) audio pairs with corresponding text:

$$\mathcal{L}(\theta) = -\sum_{(\text{ref}, \text{target}) \in \mathcal{D}} \sum_{t=1}^{T} \log P_\theta(s_t \mid s_{<t}, \text{ref\_codes}^\text{ref}, \mathbf{x}^\text{target})$$

The model is never explicitly told to "clone the voice" — it learns this association from the structure of the training data.

### 3.4 Generalization to Unseen Speakers

Why does this generalize to speakers not in the training set? The key insight is that the model learns **voice as a concept**, not as a set of speaker identities.

Consider an analogy with text LLMs: GPT-4 was never trained on the text "Translate this to Klingon:", yet it can attempt the task because it learned the concept of translation from many (language A, language B) pairs. Similarly, VieNeu-TTS learns the concept of "voice conditioning" from thousands of speakers — and this concept generalizes to new speakers at inference.

Formally, the model learns a function $f: \text{ref\_codes} \mapsto \text{voice distribution modulation}$ that is not tied to specific speaker identities but to the acoustic properties of the reference tokens themselves.

---

## 4. What Makes a Good Reference Clip

### 4.1 Duration Requirements

**Minimum:** 1.5 seconds (75 tokens at 50 tok/sec)
- Below this, the model cannot extract enough voice information
- Results in "generic" voice with only partial cloning

**Optimal:** 3-5 seconds (150-250 tokens)
- Sufficient for full voice fingerprint
- Covers enough phoneme variety for timbre and prosody estimation
- Matches the training distribution (most training clips are 3-7 seconds)

**Maximum useful:** ~10 seconds
- Beyond this, voice cloning quality plateaus
- Additional tokens consume context window without benefit

**Why 3 seconds is enough:** Phoneme diversity. Vietnamese at normal speaking rate (5-6 syllables/sec) covers 15-18 distinct syllables in 3 seconds. A syllable contains a tone, an initial consonant, and a vowel nucleus — the three dimensions that define voice identity in Vietnamese. 3 seconds provides sufficient coverage.

### 4.2 Signal Quality Requirements

**Minimum SNR:** 20 dB
- Below 20 dB, background noise interferes with codec tokenization
- The codec will encode noise patterns into the reference tokens → noise bleeds into synthesis

**Recommended SNR:** > 30 dB
- Clean, studio-quality or close-microphone recording
- Far-field recordings (phone, webcam) often have SNR 20-25 dB — acceptable but not ideal

**Formula for SNR measurement:**

$$\text{SNR} = 10 \log_{10} \frac{P_\text{signal}}{P_\text{noise}} = 10 \log_{10} \frac{\frac{1}{T_\text{speech}} \sum_{t \in \text{speech}} x_t^2}{\frac{1}{T_\text{silence}} \sum_{t \in \text{silence}} x_t^2}$$

Use a Voice Activity Detector (VAD) to identify speech vs silence regions.

### 4.3 Phoneme Coverage for Vietnamese

Vietnamese phonemes:
- **Initials (consonants):** b, m, f, v, t, th, đ, n, l, ch, nh, kh, g, ng, h, x, s, r, k (19 initials)
- **Finals:** a, ă, â, e, ê, i, o, ô, ơ, u, ư (11 main vowels, plus diphthongs/triphthongs)
- **Tones:** flat (ngang), falling-sharp (sắc), dipping (hỏi), asking (ngã), drop (nặng), falling (huyền)

A good reference clip should ideally contain:
- Multiple different tones (not just one tone class)
- Variety of vowels (open and closed)
- Both initial consonants and final consonants

**Rule of thumb:** A natural sentence in Vietnamese naturally covers sufficient phoneme diversity. The reference clip does not need to be specially constructed — any natural spoken sentence works.

### 4.4 What to Avoid

| Problem | Effect | Solution |
|---|---|---|
| Background music | Music tokens corrupt voice tokens | Remove with source separation |
| Overlapping speech | Multiple voice patterns in tokens | Use single-speaker clean audio |
| Breathy/whispered voice | Unusual acoustic pattern | Use modal voice reference |
| Very fast speech | Less phoneme diversity per second | Normal speaking rate |
| Strong reverb | Room acoustics encoded → coloration | Dry, close-mic recording |
| Heavy accent switching | Unstable voice pattern | Single dialect consistent clip |
| Clipping/saturation | Codec fails on distorted signal | Peak amplitude < 0.95 |

### 4.5 Vietnamese-Specific Considerations

**Dialect awareness:** Vietnamese has three main dialect groups — Northern (Hà Nội), Central (Huế), Southern (Hồ Chí Minh City). Each has different:
- Tonal realization: Northern has 6 distinct tones; Southern merges some tones
- Initial consonant sounds: Northern /v/ vs Southern /j/ (for "v-" words)
- Vowel quality: slight differences in /ă/, /â/

**Best practice:** The reference clip should match the dialect of the intended output. Mixing a Northern reference with text that contains Southern-dialect vocabulary creates inconsistency.

---

## 5. Code-Switching: Vietnamese + English

### 5.1 Why Code-Switching Matters in Vietnamese TTS

Vietnamese technical and professional speech frequently mixes Vietnamese and English:
- Brand names: "Google", "Apple", "Microsoft"
- Technical terms: "machine learning", "GPU", "API"
- Academic vocabulary: "data", "batch size", "loss function"
- Proper nouns: "ChatGPT", "Python"

Failing to handle code-switching correctly produces two failure modes:
1. **Vietnamese-ized English:** "machine" → /maɕin/ (with Vietnamese accent)
2. **English-ized Vietnamese:** Vietnamese surrounding words mispronounced

VieNeu-TTS handles both languages naturally because:
- The tokenizer includes English subword tokens
- Training data contains code-switched Vietnamese-English speech
- `espeak-ng` phonemizes English words with IPA, which the model has seen

### 5.2 Phonemization of Mixed Language

The VieNeu-TTS phonemizer uses a cascaded approach:
1. Language detection per word (Vietnamese dictionary lookup, then fallback to English)
2. Vietnamese words: dictionary-based G2P (Grapheme-to-Phoneme)
3. English words: espeak-ng IPA phonemization with English rules
4. Numbers: language-specific verbalization

Example:

```
Input:  "Machine learning model chạy trên GPU."
Phones: /məˈʃiːn ˈlɜːnɪŋ/ /mɔdəl/ /tɕǎj/ /tɕɛn/ /dʒiː piː juː/
         ↑ English IPA           ↑ Vietnamese ↑ English letter-by-letter
```

### 5.3 Language-Specific Codec Token Distributions

English and Vietnamese have different phoneme inventories. This means their codec token distributions are distinct:
- Vietnamese: rich in tonal variation, many open vowels, distinctive aspiration
- English: more consonant clusters, different vowel inventory, stress-based prosody

VieNeu-TTS handles this because the codec is language-agnostic — it encodes acoustic features directly. The LLM then learns which codec token patterns correspond to which phonetic contexts (Vietnamese vs English) from the training data.

---

## 6. Speaker Similarity Metrics

### 6.1 Cosine Similarity on Speaker Embeddings

Given a reference audio clip and a synthesized audio clip, compute their speaker embeddings $e_\text{ref}, e_\text{gen} \in \mathbb{R}^D$ using a pretrained speaker encoder (e.g., WeSpeaker, SpeechBrain).

**Cosine similarity:**

$$\text{SECS} = \cos(e_\text{ref}, e_\text{gen}) = \frac{e_\text{ref} \cdot e_\text{gen}}{\|e_\text{ref}\| \cdot \|e_\text{gen}\|}$$

$$\text{SECS} \in [-1, 1]$$

Higher is better. For a good zero-shot clone: SECS $\geq 0.85$. For a poor clone: SECS $< 0.70$.

**Why cosine similarity (not Euclidean):** Speaker embeddings live on a hypersphere (often L2-normalized). The angle between vectors is the natural similarity measure. Euclidean distance conflates angle with magnitude.

### 6.2 Equal Error Rate (EER)

EER is a speaker verification metric. A speaker verification system assigns a score to each (reference, test) pair, and classifies "same speaker" if the score exceeds a threshold $\theta$.

$$\text{FAR}(\theta) = P(\text{score} > \theta \mid \text{different speaker})$$
$$\text{FRR}(\theta) = P(\text{score} \leq \theta \mid \text{same speaker})$$

The EER is the threshold where FAR = FRR:

$$\text{EER}: \text{FAR}(\theta^*) = \text{FRR}(\theta^*)$$

**Interpretation:** Lower EER = better speaker discriminability. For evaluating voice cloning: compute EER between the reference speaker and generated audio from that reference vs generated audio from different references. A lower EER of the cloning system means the synthesized voice is more reliably identified as the reference speaker.

Typical values:
| System | EER |
|---|---|
| Ground truth (reference speaker vs others) | 1-5% |
| Good zero-shot clone | 10-20% |
| Poor zero-shot clone | 30-45% |
| No cloning (generic voice) | 45-50% |

### 6.3 SECS — Speaker Encoder Cosine Similarity

SECS is the standard metric in recent TTS papers (used in VALL-E, NaturalSpeech 2). It uses a pretrained speaker encoder (typically WavLM-Large fine-tuned on speaker verification) to extract speaker embeddings, then computes cosine similarity.

**SECS protocol for VieNeu-TTS evaluation:**
1. Take reference clip → extract embedding $e_\text{ref}$
2. Generate speech with VieNeu-TTS using that reference → extract embedding $e_\text{gen}$
3. Compute $\cos(e_\text{ref}, e_\text{gen})$
4. Repeat for 100+ (reference, generated) pairs, report mean

Reported SECS on Vietnamese test set: ~0.87 for VieNeu-TTS-0.3B.

---

## 7. Limitations and Failure Modes

### 7.1 Accent Transfer

The model conditions on all acoustic properties of the reference — including accent. A reference clip with a heavy Hanoi accent will produce Hanoi-accented synthesis, even when the input text is written in a Southern dialect style.

**Failure mode:** User provides a strongly accented reference and expects "neutral" output. This is not possible in zero-shot mode — the model faithfully reproduces the accent.

**Workaround:** Use a "neutral" reference clip from a broadcaster or news reader.

### 7.2 Emotional Mismatch

Reference clips carry emotional tone (neutral, happy, sad, urgent). The model conditions on this emotional signature along with voice identity.

A neutral reference → the model cannot generate emotional speech, even if the input text contains highly emotional content. The emotional prosody (F0 patterns, speech rate, energy dynamics) is learned from the reference.

**Formal view:** The model's output distribution $P(s_t | \ldots, \text{ref\_codes})$ is shifted toward speech tokens consistent with the emotional pattern of the reference. This is a feature for coherent synthesis but a limitation for emotional flexibility.

### 7.3 Out-of-Domain Text

Unusual phoneme sequences degrade voice consistency. If the input text contains:
- Foreign proper names with complex consonant clusters
- Technical abbreviations (e.g., "/LSTM/", "/BERT/")
- Non-Vietnamese numerals or symbols

...the model may produce glitches or brief departures from the reference voice on those tokens, because these sequences appear rarely in training data.

**Mitigation:** Preprocess text to normalize abbreviations, expand numerals to words, and spell out acronyms phonetically.

### 7.4 Voice Drift in Long Outputs

Autoregressive generation means each generated token depends on all previous tokens. For very long utterances (>15 seconds), the cumulative probability over hundreds of tokens can cause **voice drift**: the model's conditioning on the reference codes weakens relative to the conditioning from the most recent generated codes.

**Symptom:** First 5 seconds sound like the reference voice; by second 20, the voice has shifted toward a more generic pattern.

**Mitigation strategies:**
1. Split long text into shorter segments (each max 8-10 seconds)
2. Repeat the reference codes every N generated tokens (sliding window attention)
3. Use longer reference clips so the reference codes occupy more of the context

### 7.5 Vietnamese-Specific Failure: Regional Dialect Mixing

Vietnamese has significant acoustic differences between dialects. A reference clip with strong Central dialect phonemes (Huế accent: distinctive vowel nasalization, different tonal contours) combined with Northern-dialect text creates a mismatch that can produce unstable synthesis.

**Why this happens:** The model's training data has speakers labeled by voice, not by dialect. If Speaker A has a Huế accent and the training always pairs their reference with their own speech, the model learns to associate those acoustic patterns with that speaker. But at inference, if you ask for Northern-dialect phonemes in a Huế-accented voice, the model encounters a distribution it rarely saw during training.

**Best practice:** Match the dialect of the reference audio to the expected phonemization of the input text.

---

## Summary

| Aspect | Traditional (d-vector) | VieNeu-TTS (In-Context) |
|---|---|---|
| Seen speakers | Excellent cloning | Excellent cloning |
| Unseen speakers | Fails / artifacts | Good cloning (zero-shot) |
| Required fine-tuning | Yes (speaker adaptation) | No |
| Reference duration | 30+ seconds | 3-5 seconds |
| Mechanism | Fixed speaker embedding | Reference tokens in prompt |
| Generalization | Bounded by training set | Unlimited (in-context) |

**Core equation:**

$$P_\theta(\mathbf{s} \mid \text{ref\_codes}, \mathbf{x}) = \prod_{t=1}^{T} P_\theta(s_t \mid s_{<t}, \text{ref\_codes}, \mathbf{x})$$

**Voice is in the tokens:** The reference speaker's voice identity is encoded in the distribution of their codec tokens. By prepending these tokens to the prompt, the model learns to continue the voice — no explicit speaker embedding required.

**Vietnamese-specific checklist for good cloning:**
- [x] Reference: 3-5 seconds, clean, SNR > 25 dB
- [x] Reference contains multiple tones
- [x] Reference dialect matches intended output dialect
- [x] Text is preprocessed (expanded abbreviations, normalized)
- [x] Temperature τ = 1.0 (preserve tone accuracy)
- [x] Segment text into chunks < 10 seconds each

---

## Further Reading

- Wang et al. (2023). *VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*. [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
- Shen et al. (2023). *NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech Synthesizers*. ICLR 2024.
- Snyder et al. (2018). *X-Vectors: Robust DNN Embeddings for Speaker Recognition*. ICASSP 2018.
- Saeki et al. (2022). *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022*. Interspeech 2022.
- Phan et al. (2020). *Vietnamese Speech Recognition Survey*. (Background on Vietnamese phonology for TTS.)
