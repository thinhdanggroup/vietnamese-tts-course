# Chapter 03 — TTS Architecture Evolution

> **Audience**: ML engineers who understand neural networks and want to understand how TTS systems evolved from hand-crafted signal processing to LLM-based synthesis.
> **Goal**: Build a principled understanding of each TTS paradigm, including its mathematical foundations, so you can reason about why VieNeu-TTS makes the architectural choices it does.

---

## Table of Contents

1. [Era 1: Concatenative TTS](#1-era-1-concatenative-tts)
2. [Era 2: Parametric TTS — HMM-Based](#2-era-2-parametric-tts--hmm-based)
3. [Era 3a: Tacotron 2](#3-era-3a-tacotron-2)
4. [Era 3b: FastSpeech 2](#4-era-3b-fastspeech-2)
5. [Era 3c: VITS (End-to-End)](#5-era-3c-vits-end-to-end)
6. [Era 4: LLM-Based TTS](#6-era-4-llm-based-tts)
7. [Architecture Comparison](#7-architecture-comparison)

---

## 1. Era 1: Concatenative TTS

### 1.1 Core Idea

**Concatenative TTS** synthesizes speech by selecting and joining segments of pre-recorded speech. The voice database consists of recordings of a single speaker saying many sentences covering a large variety of phoneme sequences. To synthesize a new sentence, the system:

1. Determines the target phoneme sequence (via G2P)
2. Searches the database for recorded units that match each target phoneme in context
3. Selects the best sequence of units using dynamic programming
4. Concatenates the selected units, applying signal processing to smooth joins

This approach produces **highly natural speech** (because it uses real human recordings) but has severe limitations, especially for Vietnamese.

### 1.2 Unit Selection — The Formal Specification

**Unit types** (in order of increasing size and naturalness):

| Unit Type | Description | Pros | Cons |
|-----------|-------------|------|------|
| Phoneme | Single phoneme | Small database | Many joins, unnatural |
| Diphone | From center of phoneme to center of next | Fewer joins | Database still large |
| Triphone | 3-phoneme context | Better coarticulation | Very large database |
| Syllable | Full syllable unit | Natural coarticulation | Huge database for tonal languages |
| Word | Full word | Most natural | Impractical corpus size |

For Vietnamese, **syllable-level units** are optimal because:
- Vietnamese is monosyllabic — each word is one syllable
- Tone is a property of the whole syllable — splitting within a syllable corrupts tone
- Most coarticulation happens at syllable boundaries

### 1.3 The Unit Selection Cost Function

Given target phoneme sequence $T = [t_1, t_2, \ldots, t_n]$ and candidate unit $u_{i,j}$ (the $j$-th candidate for target $t_i$):

**Target cost** $C_T(t_i, u_{i,j})$: Measures how well unit $u_{i,j}$ matches the target specification $t_i$. Typically a weighted sum of feature differences:

$$C_T(t_i, u_{i,j}) = \sum_k w_k \cdot d_k(f_k(t_i), f_k(u_{i,j}))$$

where $f_k$ are acoustic features (F0, duration, spectral features) and $w_k$ are weights.

**Join cost** $C_J(u_{i,j}, u_{i+1,j'})$: Measures how smoothly unit $u_{i,j}$ can be concatenated with the next unit $u_{i+1,j'}$. Typically measured at the join boundary using:

$$C_J(u_{i,j}, u_{i+1,j'}) = d_{\text{spectral}}\!\left(f_{\text{end}}(u_{i,j}),\ f_{\text{start}}(u_{i+1,j'})\right)$$

Common spectral distance measures: cepstral distance, log spectral distance, or MFCC Euclidean distance.

**Dynamic programming selection**: The total cost of a path $U = [u_{1,j_1}, u_{2,j_2}, \ldots, u_{n,j_n}]$ through the unit candidates is:

$$C_{\text{total}}(U) = \sum_{i=1}^{n} C_T(t_i, u_{i,j_i}) + \sum_{i=1}^{n-1} C_J(u_{i,j_i}, u_{i+1,j_{i+1}})$$

The optimal path is found by the **Viterbi algorithm** in $O(n \cdot |U|^2)$ time, where $|U|$ is the maximum number of candidates per target.

### 1.4 Why Vietnamese Tones Make Concatenative TTS Expensive

**Tonal inventory analysis**:

For a diphone inventory (transitions between phoneme pairs), the number of units required scales with the number of tonal contrasts. In Vietnamese:

- Each syllable carries one of 6 tones
- Coarticulation at syllable boundaries depends on BOTH the tone of the current syllable and the adjacent syllable
- The full diphone space for tone-aware Vietnamese: each diphone unit (X→Y) must be recorded for each of the 6×6 = 36 tone pair combinations

Concretely:

```
Tones:              6
Vowels (nuclei):   ~12
Onset consonants:   23
Final consonants:    8

Tone-aware syllables: 6 × 12 × 23 × (8+1) ≈ 14,904 distinct syllable types
Diphones (boundaries): ~14,904 × 14,904 ≈ 222 million possible diphone types
Practical diphone inventory (with pruning): ~10,000–50,000 units
Recording time @ 0.5s per unit: 5,000–25,000 seconds ≈ 1.4–7 hours
```

Compare to English (no tones), where a typical diphone inventory has ~2,000 units and requires less than 20 minutes of recording.

The large required corpus is expensive to record and curate. Furthermore, finding perfect unit matches for rare tone sequence combinations is difficult, leading to either:
- **Missing units**: fallback to suboptimal units with high join cost → audible artifacts
- **Gaps in prosody coverage**: some F0 contours cannot be faithfully reproduced by any available unit

### 1.5 Limitations of Concatenative TTS

1. **Fixed voice**: The system is completely locked to one recorded speaker. No voice cloning or style transfer.
2. **Unnatural joins**: Signal processing at concatenation points (cross-fading, PSOLA pitch modification) introduces audible artifacts.
3. **Large database**: High storage cost. A 5-hour voice database = ~5 GB of audio files + indexes.
4. **Prosody inflexibility**: Cannot freely modify pitch or duration without signal processing artifacts.
5. **OOV handling**: Rare words and proper nouns that aren't in the database require fallback to smaller units, degrading quality.
6. **Scalability**: Cannot easily scale to multiple languages or accents.

---

## 2. Era 2: Parametric TTS — HMM-Based

### 2.1 Core Idea

Instead of storing recordings, **parametric TTS** learns a **statistical model** of the acoustic features of speech. At synthesis time, the model **generates** acoustic features frame-by-frame, then converts them to a waveform using a **vocoder**.

The dominant parametric TTS framework (2000s–2014) used **Hidden Markov Models (HMMs)** to model the joint distribution of acoustic features conditioned on the phoneme sequence.

### 2.2 Acoustic Features in HMM-TTS

HMM-TTS represents speech using three sets of acoustic features extracted from a vocoder (WORLD, STRAIGHT):

| Feature | Symbol | Dimension | Represents |
|---------|--------|-----------|------------|
| F0 (log pitch) | $f_0[t]$ | 1 per frame | Fundamental frequency (tone contour) |
| Mel-cepstral coefficients | $\mathbf{c}[t]$ | 25 per frame | Spectral envelope (vowel quality, timbre) |
| Band aperiodicity | $\mathbf{b}[t]$ | 5 per frame | Voicing, breathiness |

Plus their delta and delta-delta features: total ~100 features per frame.

### 2.3 HMM Structure

A phone (phoneme segment) is modeled as a **left-to-right HMM with $S$ states** (typically $S = 5$). Each state $s$ has:

- **Emission distribution** $p(\mathbf{o}_t | s)$: typically a single multivariate Gaussian (or Gaussian Mixture Model) over the acoustic features
- **Transition probabilities** $a_{s,s}$ (self-loop) and $a_{s,s+1}$ (advance)

A complete utterance is modeled by concatenating the HMMs for each phoneme in the phoneme sequence.

### 2.4 The HMM Forward-Backward Algorithm

**Notation**:
- $O = [\mathbf{o}_1, \mathbf{o}_2, \ldots, \mathbf{o}_T]$: observation sequence (acoustic feature frames)
- $Q = [q_1, q_2, \ldots, q_T]$: hidden state sequence
- $\lambda = (A, B, \pi)$: HMM parameters (transition matrix, emission distributions, initial state distribution)

**Problem**: Compute $P(O | \lambda) = \sum_Q P(O, Q | \lambda)$ — the total likelihood of observations summed over all possible state sequences.

**Naive computation** is $O(|Q|^T)$ — exponential in sequence length. The **forward algorithm** solves this in $O(T \cdot S^2)$ time.

**Forward variable** $\alpha_t(s)$: probability of observing $\mathbf{o}_1, \ldots, \mathbf{o}_t$ and being in state $s$ at time $t$:

$$\alpha_t(s) = P(\mathbf{o}_1, \ldots, \mathbf{o}_t, q_t = s | \lambda)$$

**Initialization**:

$$\alpha_1(s) = \pi_s \cdot b_s(\mathbf{o}_1)$$

where $\pi_s$ is the initial probability of state $s$ and $b_s(\mathbf{o}) = \mathcal{N}(\mathbf{o}; \boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s)$ is the emission probability.

**Recursion**:

$$\alpha_t(s) = \left[\sum_{s'=1}^{S} \alpha_{t-1}(s') \cdot a_{s',s}\right] \cdot b_s(\mathbf{o}_t)$$

**Termination**:

$$P(O | \lambda) = \sum_{s=1}^{S} \alpha_T(s)$$

**Backward variable** $\beta_t(s)$: probability of observing $\mathbf{o}_{t+1}, \ldots, \mathbf{o}_T$ given state $s$ at time $t$:

$$\beta_t(s) = P(\mathbf{o}_{t+1}, \ldots, \mathbf{o}_T | q_t = s, \lambda)$$

The backward algorithm mirrors the forward algorithm, running from $T$ back to $1$.

**E-step** (Baum-Welch): Use $\alpha_t(s) \cdot \beta_t(s)$ to compute the posterior probability of being in state $s$ at time $t$, then re-estimate parameters $\lambda$ in the **M-step**.

### 2.5 Viterbi Decoding for State Sequence

At synthesis time, we need the **most likely state sequence** given the observation sequence (or, in synthesis, we generate the most likely observation sequence). The **Viterbi algorithm** finds:

$$Q^* = \arg\max_Q P(O, Q | \lambda)$$

**Viterbi variable** $\delta_t(s) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t = s, \mathbf{o}_1, \ldots, \mathbf{o}_t | \lambda)$

$$\delta_t(s) = \max_{s'} \left[\delta_{t-1}(s') \cdot a_{s',s}\right] \cdot b_s(\mathbf{o}_t)$$

The optimal path is recovered by backtracking through stored argmax values.

### 2.6 Synthesis: Generating Acoustic Features

At synthesis time:
1. The phoneme sequence determines which HMMs to concatenate
2. The state duration is determined by the most likely state sequence length (controlled by transition probability $a_{s,s}$ vs $a_{s,s+1}$)
3. The acoustic features for each state are generated from the emission distribution mean $\boldsymbol{\mu}_s$
4. Delta and delta-delta constraints are applied (maximum-likelihood parameter generation, MLPG) to ensure smooth temporal trajectories

**MLPG — Maximum Likelihood Parameter Generation**:

The generated feature sequence $\hat{\mathbf{c}}_{1:T}$ must satisfy both the static and dynamic constraints. This is formulated as maximizing:

$$\hat{\mathbf{c}} = \arg\max_{\mathbf{c}} \sum_t \log \mathcal{N}(\mathbf{o}_t; W_t \mathbf{c}, \Sigma)$$

where $W_t$ is a transformation matrix that computes static + delta + delta-delta features from the static sequence $\mathbf{c}$.

The solution is:

$$\hat{\mathbf{c}} = (W^T \Sigma^{-1} W)^{-1} W^T \Sigma^{-1} \boldsymbol{\mu}$$

This global optimization introduces **over-smoothing** — the generated trajectory is the MLE mean, which tends to be excessively smooth compared to natural speech.

### 2.7 WORLD Vocoder

The **WORLD vocoder** (Morise et al., 2016) synthesizes a waveform from the acoustic features:

- **F0 analysis (DIO + StoneMask)**: Extract fundamental frequency
- **Spectral envelope (CheapTrick)**: Extract smoothed spectral envelope
- **Aperiodicity estimation (D4C)**: Separate voiced and unvoiced components

Synthesis: Excitation signal (periodic + noise mixed according to aperiodicity) convolved with the spectral envelope → waveform.

### 2.8 Limitation: The Over-Smoothing Problem

HMM-TTS has one notorious flaw: **over-smoothing** of acoustic features. The MLPG solution is a global least-squares estimate, which averages over variability. The result:

- **Spectral over-smoothing**: Formant transitions are blurred → "muffled" vowel quality
- **F0 over-smoothing**: Tone contours are rounded off → tones sound less distinct
- **Perceptual effect**: "Buzzy", robotic quality characteristic of HMM-TTS

For Vietnamese, this is particularly damaging because the 6-tone system requires precise F0 contour shape. HMM-TTS's tendency to smooth F0 causes tone confusion (especially between hỏi and ngã, which have similar F0 shapes but different phonation types — phonation type is not well modeled).

---

## 3. Era 3a: Tacotron 2

### 3.1 Architecture Overview

**Tacotron 2** (Shen et al., 2018) is a neural sequence-to-sequence model that directly maps a phoneme sequence to a mel spectrogram, bypassing explicit feature engineering. A separate neural vocoder (WaveNet or HiFi-GAN) then converts the mel spectrogram to a waveform.

The architecture consists of:
1. **Encoder**: Converts phoneme sequence → encoder hidden states
2. **Attention**: Learns a soft alignment between encoder states and decoder frames
3. **Decoder**: Autoregressively predicts mel spectrogram frames
4. **Post-net**: Residual CNN that refines the mel spectrogram

### 3.2 Encoder

**Input embedding**: Each phoneme is mapped to a 512-dimensional embedding vector.

**CBHG module** (Convolution Bank + Highway network + GRU):
1. **Convolution bank**: Apply $K = 16$ sets of 1D convolutions with filter sizes $k = 1, 2, \ldots, K$ in parallel. This captures local patterns at multiple scales.
2. **Max pooling**: Reduces sequence length, provides local shift invariance.
3. **Projection convolutions**: 1D Conv → 1D Conv → residual connection to original input.
4. **Highway network**: $F$ layers of gated networks: $y = H(x) \cdot T(x) + x \cdot (1 - T(x))$, where $T$ is the "transform gate".
5. **Bidirectional GRU**: Produces final encoder hidden states $\mathbf{h} = [\overrightarrow{\mathbf{h}}, \overleftarrow{\mathbf{h}}]$.

### 3.3 Location-Sensitive Attention

The attention mechanism in Tacotron 2 is **location-sensitive attention** — a variant that uses cumulative attention weights to maintain monotonic progress through the input sequence.

**Attention energy computation** at decoder step $i$, for encoder step $j$:

$$e_{i,j} = v^T \tanh\!\left(W_{\text{query}} \mathbf{s}_i + V_{\text{key}} \mathbf{h}_j + U_{\text{loc}} \mathbf{f}_{i,j} + \mathbf{b}\right)$$

where:
- $\mathbf{s}_i$: decoder state at step $i$ (the "query")
- $\mathbf{h}_j$: encoder state at position $j$ (the "key")
- $\mathbf{f}_{i,j}$: **location feature** computed from previous attention weights
- $W_{\text{query}}, V_{\text{key}}, U_{\text{loc}}, \mathbf{b}$: learned parameters
- $v^T$: learned scoring vector

**Location features**: The previous attention weight vector $\boldsymbol{\alpha}_{i-1}$ is convolved with a bank of $F = 32$ 1D filters of width $31$:

$$\mathbf{F}_i = \text{Conv1D}(\boldsymbol{\alpha}_{i-1}; \text{filters} \in \mathbb{R}^{32 \times 31})$$

This gives $\mathbf{f}_{i,j} \in \mathbb{R}^{32}$ — information about where attention was focused at the previous step.

**Attention weights** (softmax normalization):

$$\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{j'} \exp(e_{i,j'})}$$

**Context vector** (weighted sum of encoder states):

$$\mathbf{c}_i = \sum_j \alpha_{i,j} \mathbf{h}_j$$

**Why location-sensitive attention for TTS**:
- Standard content-based attention (Bahdanau) can get "stuck" repeating a position or skipping ahead too fast
- The location features provide a "where have I been" signal that encourages **monotonic, left-to-right progress** through the input — essential for reading text aloud in order
- For Vietnamese, this is especially important because tones must align with the correct syllable

### 3.4 Decoder

The decoder predicts $r$ mel spectrogram frames at each step (frame stacking, $r = 2$ in the original paper).

**Input**: Previous $r$ mel frames concatenated with the context vector $\mathbf{c}_{i-1}$

**Architecture**:
1. Pre-net: 2 FC layers with dropout (acts as information bottleneck, forces attention to learn meaningful alignment)
2. 2 unidirectional LSTM layers (1024 units each)
3. Projection: predicts mel spectrogram frames + **stop token** (binary: continue vs. stop)

**Stop token**: A scalar sigmoid output that predicts when to stop generating. Loss:

$$L_{\text{stop}} = -\frac{1}{T} \sum_t [y_t \log \hat{y}_t + (1-y_t) \log(1-\hat{y}_t)]$$

where $y_t = 1$ at the final frame.

### 3.5 Post-Net

A 5-layer 1D CNN that takes the mel spectrogram and predicts a residual correction:

$$\hat{\mathbf{M}}_{\text{final}} = \hat{\mathbf{M}}_{\text{decoder}} + \text{PostNet}(\hat{\mathbf{M}}_{\text{decoder}})$$

Each Conv layer: 512 filters, kernel size 5, Batch Norm, tanh (except last layer which is linear).

The post-net improves fine spectral details without affecting the coarse temporal structure established by the decoder.

### 3.6 Loss Function

Tacotron 2 uses a simple combination of L1 and L2 losses on the mel spectrogram:

$$L_{\text{total}} = L_{\text{mel,before}} + L_{\text{mel,after}} + L_{\text{stop}}$$

$$L_{\text{mel,before}} = \frac{1}{T \cdot M} \sum_{t,m} |\hat{M}^{\text{decoder}}_{t,m} - M_{t,m}|$$

$$L_{\text{mel,after}} = \frac{1}{T \cdot M} \sum_{t,m} |\hat{M}^{\text{postnet}}_{t,m} - M_{t,m}|^2$$

where $M \in \mathbb{R}^{T \times 80}$ is the target mel spectrogram and $\hat{M}$ is the predicted spectrogram.

Note: both L1 and L2 are used — L2 focuses optimization on large errors, L1 is more robust to outliers.

### 3.7 WaveNet Vocoder

A separate **WaveNet** model converts the mel spectrogram to a waveform. WaveNet is an autoregressive dilated causal convolution network that models:

$$p(\mathbf{x} | \mathbf{M}) = \prod_{t=1}^{T} p(x_t | x_{<t}, \mathbf{M})$$

Each audio sample $x_t$ is quantized to 256 levels (mu-law companding) and predicted by a categorical distribution. This produces extremely high quality audio but is **very slow** at inference (real-time factor ~1000x for the original WaveNet).

### 3.8 Tacotron 2 for Vietnamese

Tacotron 2 was successfully applied to Vietnamese by several researchers and commercial systems. Key findings:

- **Tone modeling**: The attention alignment captures tone implicitly — the decoder must spend more frames on syllables with longer tones (huyền, hỏi), and the mel spectrogram encodes the F0 contour
- **Attention failures**: A known failure mode is attention getting "stuck" or "jumping" on tone boundaries — the prosodic boundary after a tone syllable can confuse the attention mechanism
- **Data requirement**: ~20-30 hours of single-speaker studio-quality Vietnamese speech needed for good quality

---

## 4. Era 3b: FastSpeech 2

### 4.1 Motivation — Problems with Autoregressive TTS

Tacotron 2's autoregressive decoder has two critical weaknesses:

1. **Slow inference**: Generating $T$ mel frames requires $T$ sequential decoder steps. For a 5-second utterance at 24 kHz with hop length 256: $T = 5 \times 24000 / 256 = 468$ frames → 468 sequential LSTMs forward passes.

2. **Attention errors**: The learned attention can fail, causing:
   - **Repetition**: Attention gets stuck on one phoneme → repeated syllable
   - **Skipping**: Attention jumps ahead → missing words
   - **Degradation**: Performance deteriorates for very long or unusual sentences

**FastSpeech 2** (Ren et al., 2021) eliminates the autoregressive decoder with a **non-autoregressive, parallel** architecture that generates all mel frames simultaneously.

### 4.2 Architecture Overview

```
Phoneme sequence
       ↓
[Phoneme Embedding]
       ↓
[Feed-Forward Transformer (FFT) × N_encoder]
       ↓
[Variance Adaptor]
   - Duration Predictor → [Length Regulator] → expands hidden states
   - Pitch Predictor → add to hidden states
   - Energy Predictor → add to hidden states
       ↓
[Feed-Forward Transformer (FFT) × N_decoder]
       ↓
[Linear Projection]
       ↓
Mel spectrogram (all frames in parallel)
```

### 4.3 Feed-Forward Transformer (FFT) Block

Each FFT block replaces the traditional Transformer's attention+FFN with:

1. **Multi-Head Self-Attention**: Standard scaled dot-product attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

2. **1D Convolutional FFN**: Instead of standard FFN (which is position-wise), uses 1D convolution to capture local dependencies:

$$\text{FFN}(\mathbf{h}) = \text{Conv1D}(\text{ReLU}(\text{Conv1D}(\mathbf{h}, W_1)), W_2)$$

with kernel size 9 (capturing ±4 frames of context), which is better for speech than point-wise FFN.

3. **Layer Normalization + Residual Connections**: Standard.

### 4.4 Variance Adaptor

The **variance adaptor** is the key innovation that enables non-autoregressive synthesis. It predicts three variance-related quantities from the encoder output:

#### 4.4.1 Duration Predictor

The duration predictor is a 2-layer 1D CNN that maps each phoneme hidden state to a **duration** (number of mel frames for that phoneme):

$$\hat{d}_i = \text{DurationPredictor}(\mathbf{h}_i)$$

**Training**: Durations are extracted from a **forced alignment** system (Montreal Forced Aligner or a separate alignment model). The loss is MSE on log durations:

$$L_{\text{duration}} = \text{MSE}(\log \hat{d}, \log d)$$

Log durations are used because duration distributions are log-normal (most phonemes are short, few are very long).

#### 4.4.2 Length Regulator

The length regulator **expands** the phoneme sequence hidden states according to durations:

$$\text{LR}(\mathbf{h}, \mathbf{d}) = [\underbrace{\mathbf{h}_1, \ldots, \mathbf{h}_1}_{d_1}, \underbrace{\mathbf{h}_2, \ldots, \mathbf{h}_2}_{d_2}, \ldots, \underbrace{\mathbf{h}_n, \ldots, \mathbf{h}_n}_{d_n}]$$

Each phoneme hidden state is repeated $d_i$ times. The output has length $T = \sum_i d_i$ — the same as the target mel spectrogram.

This length regulator also enables **speech rate control** at inference time: multiply all durations by a scalar $\alpha$ to speed up ($\alpha < 1$) or slow down ($\alpha > 1$) the speech.

#### 4.4.3 Pitch Predictor

Predicts the F0 contour (after length regulation, so one F0 value per mel frame):

$$\hat{F0}[t] = \text{PitchPredictor}(\mathbf{h}_{\text{expanded}}[t])$$

**For Vietnamese**: The pitch predictor must learn the 6 distinct tone contours. This is a significant modeling challenge because:
- The same phoneme sequence can have different F0 if the tone is different
- The tone is specified in the phoneme embedding (encoded as part of the phoneme symbol), so the pitch predictor has access to tone information
- In practice, FastSpeech 2 learns convincing Vietnamese F0 contours given enough training data

Loss: MSE on F0 values (or log F0).

#### 4.4.4 Energy Predictor

Predicts the frame-level energy (sum of squared mel spectrogram values per frame):

$$E[t] = \sqrt{\frac{1}{M} \sum_{m=1}^{M} \mathbf{S}[t, m]^2}$$

Loss: MSE on energy values.

### 4.5 FastSpeech 2 Loss Function

Total loss:

$$L_{\text{total}} = L_{\text{mel}} + \lambda_d L_{\text{duration}} + \lambda_p L_{\text{pitch}} + \lambda_e L_{\text{energy}}$$

where:

$$L_{\text{mel}} = \frac{1}{T \cdot M} \sum_{t,m} |\hat{\mathbf{S}}[t,m] - \mathbf{S}[t,m]|$$

(MAE loss on mel spectrogram, found more stable than MSE for mel prediction)

### 4.6 Inference Speed Advantage

**Tacotron 2 inference**: $T$ sequential decoder steps, each requiring attention computation over all encoder states. Complexity: $O(T^2)$ for attention, $O(T)$ for LSTM.

**FastSpeech 2 inference**:
1. Encoder: $O(n^2)$ for self-attention over $n$ phonemes (fast, $n$ is small)
2. Duration predictor: $O(n)$
3. Length regulator: $O(T)$ (copy operation)
4. Decoder: $O(T^2)$ for self-attention (can be parallelized on GPU)

In practice, FastSpeech 2 is **5–10× faster** than Tacotron 2 on GPU (all mel frames generated in one forward pass) and more stable (no attention failures).

---

## 5. Era 3c: VITS (End-to-End)

### 5.1 Motivation — Eliminating the Two-Stage Pipeline

All previous systems (Tacotron 2, FastSpeech 2) require a **two-stage pipeline**:
1. **Acoustic model**: text → mel spectrogram
2. **Vocoder**: mel spectrogram → waveform

This separation has drawbacks:
- Training the two stages independently → suboptimal overall quality
- The vocoder must handle all reconstructions errors from the acoustic model
- Mel spectrogram is a lossy intermediate — the vocoder cannot recover phase information lost in the log-mel computation

**VITS** (Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS, Kim et al., 2021) integrates both stages into a single end-to-end model.

### 5.2 Variational Autoencoder Foundation

**VITS** is built on a **Conditional Variational Autoencoder (CVAE)**.

**VAE Setup**:
- Encoder (posterior): $q_\phi(z | x)$ — encodes waveform $x$ (or mel spectrogram) into latent $z$
- Decoder (prior): $p_\theta(x | z, c)$ — decodes latent $z$ into waveform $x$ conditioned on text $c$

**ELBO (Evidence Lower BOund)**:

We want to maximize the log-likelihood $\log p_\theta(x | c)$. Using Jensen's inequality with any distribution $q_\phi(z|x)$:

$$\log p_\theta(x | c) = \log \int p_\theta(x, z | c) dz$$

$$= \log \int \frac{q_\phi(z|x)}{q_\phi(z|x)} p_\theta(x, z | c) dz$$

$$= \log \mathbb{E}_{q_\phi(z|x)} \left[\frac{p_\theta(x, z | c)}{q_\phi(z|x)}\right]$$

$$\geq \mathbb{E}_{q_\phi(z|x)} \left[\log \frac{p_\theta(x, z | c)}{q_\phi(z|x)}\right] \quad \text{(Jensen's inequality)}$$

$$= \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x | z, c)] - \text{KL}(q_\phi(z|x) \| p_\theta(z|c))$$

$$= \underbrace{\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x | z, c)]}_{\text{reconstruction term}} - \underbrace{\text{KL}(q_\phi(z|x) \| p_\theta(z|c))}_{\text{regularization term}}$$

This is the **ELBO** — a lower bound on the true log-likelihood. Maximizing the ELBO simultaneously:
- **Reconstruction**: forces the decoder to reconstruct $x$ from $z$ → good audio quality
- **KL regularization**: forces the posterior $q_\phi(z|x)$ to be close to the prior $p_\theta(z|c)$ → the prior (conditioned on text $c$) learns to generate realistic speech from text alone

### 5.3 Normalizing Flows for Expressive Prior

A key challenge: the prior $p_\theta(z|c)$ must be expressive enough to match the complex distribution of speech. A simple Gaussian prior is insufficient.

**VITS** uses **normalizing flows** to transform a simple base distribution $p_0$ into an expressive prior:

$$z = f_\phi(u), \quad u \sim p_0(u) = \mathcal{N}(0, I)$$

where $f_\phi$ is an invertible neural network (flow). The log-likelihood under the flow is:

$$\log p_z(z) = \log p_0(f_\phi^{-1}(z)) + \log \left|\det \frac{\partial f_\phi^{-1}(z)}{\partial z}\right|$$

The Jacobian determinant term accounts for the volume change in the transformation. VITS uses **affine coupling layers** (Real-NVP style), which have tractable Jacobians.

### 5.4 GAN Discriminator for Waveform Quality

The reconstruction term in the ELBO, $\mathbb{E}[\log p_\theta(x | z, c)]$, with a Gaussian likelihood is equivalent to MSE on the waveform — which produces over-smoothed audio.

VITS augments the ELBO with an **adversarial loss** using a multi-scale, multi-period discriminator (from HiFi-GAN):

$$L_{\text{adv}}(G, D) = \mathbb{E}_{(z,c)}[(D(G(z, c)) - 1)^2] + \mathbb{E}_x[(D(x))^2]$$

The generator $G$ (decoder) is trained to fool the discriminator $D$, which is trained to distinguish real from generated waveforms. This drives the generated waveform toward the **manifold of real speech** rather than minimizing MSE.

**Feature matching loss**: Additionally, intermediate discriminator features are matched:

$$L_{\text{fm}}(G, D) = \mathbb{E} \sum_l \frac{1}{N_l} \|D^l(x) - D^l(G(z,c))\|_1$$

This provides additional gradient signal without the instability of the raw GAN loss.

### 5.5 Total VITS Loss

$$L_{\text{total}} = L_{\text{recon}} + L_{\text{KL}} + L_{\text{dur}} + L_{\text{adv}} + L_{\text{fm}}$$

where $L_{\text{dur}}$ is the duration predictor loss (VITS also has a stochastic duration predictor that models the duration distribution, not just the mean).

### 5.6 Vietnamese VITS Challenges

**Tone modeling in latent space**:

VITS encodes all prosodic variation into the latent variable $z$. For Vietnamese:
- Tone information comes from the phoneme embedding (tones are encoded as separate phoneme symbols)
- The decoder must learn to generate the correct F0 contour AND the correct phonation type (modal, breathy, creaky) from the latent code + phoneme embedding
- This works well in practice but requires sufficient training data to learn the 6 distinct phonation patterns

**Phonation type (breathiness, creakiness)**:
- Huyền's breathy voice requires the decoder to generate a "noisy" harmonic structure in the waveform
- Ngã's creaky voice requires irregular period and glottal constriction in the generated waveform
- These are complex acoustic patterns that the GAN discriminator effectively enforces — discriminator trained on real Vietnamese speech learns to penalize incorrectly phonated outputs

---

## 6. Era 4: LLM-Based TTS

### 6.1 The Key Insight — Speech as a Language Modeling Problem

The LLM revolution (GPT-3, 2020; ChatGPT, 2022) demonstrated that large-scale language model pretraining creates powerful representations that transfer to many downstream tasks. The key insight for TTS is:

> **If speech can be represented as a sequence of discrete tokens, then TTS becomes a language modeling problem: predict the next speech token given text tokens.**

This reframing has profound implications:
1. **Unified architecture**: The same transformer that processes text now also processes speech
2. **Transfer learning**: Speech tokens benefit from the LLM's already-learned representations of language, meaning, and context
3. **Zero-shot capability**: The LLM can leverage in-context learning to clone a voice from just a few seconds of reference audio — no fine-tuning required
4. **Code-switching**: The LLM naturally handles mixed Vietnamese/English because it has seen multilingual text during pretraining

### 6.2 Neural Audio Codecs — Discretizing Speech

The first requirement for LLM-TTS is a way to convert continuous speech waveforms to discrete token sequences. This is done by a **neural audio codec**.

**Architecture of a neural codec** (Encodec / NeuCodec style):

```
Waveform → [Encoder (CNN+TCN)] → Continuous embedding
                                           ↓
                              [Residual Vector Quantization (RVQ)]
                                           ↓
                               Discrete token IDs (multiple codebooks)
```

**Residual Vector Quantization (RVQ)**:

Given continuous embedding $\mathbf{e} \in \mathbb{R}^d$:
1. Find nearest code in codebook 1: $\mathbf{q}_1 = \arg\min_{\mathbf{c} \in \mathcal{C}_1} \|\mathbf{e} - \mathbf{c}\|^2$
2. Compute residual: $\mathbf{r}_1 = \mathbf{e} - \mathbf{q}_1$
3. Quantize residual: $\mathbf{q}_2 = \arg\min_{\mathbf{c} \in \mathcal{C}_2} \|\mathbf{r}_1 - \mathbf{c}\|^2$
4. Repeat for $K$ codebooks.

The total discrete representation is the sequence of codebook indices $[k_1, k_2, \ldots, k_K]$ per frame.

**VieNeu-TTS NeuCodec settings**:
- Sample rate: 24,000 Hz
- Frame rate: ~75 tokens/second (hop size ≈ 320 samples)
- Codebook size: 1024 codes per codebook
- Number of codebooks: 8 (RVQ with 8 levels)
- Total bits: $8 \times \log_2(1024) = 80$ bits per frame → 6,000 bits/second

For LLM-TTS, typically only the **first codebook** (the coarse semantic layer) is used for language modeling, then the remaining codebooks are predicted in a secondary pass (or a cascaded model). NeuCodec's first codebook captures the semantic/prosodic content — including tone.

### 6.3 The LLM-TTS Training and Inference Format

**Training objective**: Given a pair (text, speech), the model is trained to predict speech tokens autoregressively:

$$L = -\sum_t \log P(\text{speech\_token}_t | \text{speech\_token}_{<t}, \text{text\_tokens}, \text{ref\_speech\_tokens})$$

This is standard cross-entropy language modeling loss — identical to next-token prediction in an LLM.

**Prompt format** at inference (zero-shot voice cloning):

```
[TEXT_START] phoneme_1 phoneme_2 ... phoneme_n [TEXT_END]
[SPEECH_START] ref_1 ref_2 ... ref_m [SPEECH_END]
[SPEECH_GEN_START] → model generates → gen_1 gen_2 ... gen_T [SPEECH_GEN_END]
```

The model has seen this format during training with many (text, reference speech, target speech) triples. At inference, providing a new reference speaker's tokens causes the model to clone that speaker's voice via in-context learning — **no fine-tuning needed**.

### 6.4 The VieNeu-TTS Lineage

VieNeu-TTS follows a lineage of LLM-TTS systems:

| System | Key Innovation | Year |
|--------|---------------|------|
| AudioLM (Google) | Speech as tokens in LM | 2022 |
| VALL-E (Microsoft) | Zero-shot cloning from 3s | 2023 |
| NeuTTS | Vietnamese-specific LLM-TTS | 2023 |
| **VieNeu-TTS** | Open-source Vietnamese LLM-TTS with NeuCodec | 2024 |

VieNeu-TTS adapts the NeuTTS/VALL-E architecture with:
- **NeuCodec**: A Vietnamese-optimized neural codec with high speech quality at low bitrate
- **Vietnamese phonemizer**: `phonemize_with_dict` for accurate phoneme input
- **VieNeu language model**: Trained on Vietnamese speech data with the LLM-TTS prompt format

### 6.5 Why LLM-TTS Outperforms Previous Approaches for Vietnamese

**1. Tone accuracy**:

Previous systems (Tacotron 2, FastSpeech 2, VITS) must learn F0 contours and phonation types from audio alone. The LLM approach additionally leverages:
- The LLM's pretraining knowledge of Vietnamese tone marks (the LLM has seen millions of Vietnamese text tokens and understands that "à" carries huyền tone)
- Contextual understanding: the LLM knows that exclamatory sentences have different prosody than questions
- The codec's quantized representation implicitly captures tone in the first codebook's distribution

**2. Code-switching**:

When the input is "Mô hình AI sử dụng GPU NVIDIA để training", previous systems must handle the English words through fragile rule-based fallbacks. The LLM has seen this type of mixed text and handles it natively — it knows "AI" is likely pronounced as letter names in Vietnamese tech contexts.

**3. Zero-shot cloning**:

VITS requires fine-tuning on the target speaker's data for multi-speaker adaptation. VieNeu-TTS achieves zero-shot cloning through in-context learning — provide 3–10 seconds of reference audio, and the model clones the voice without any parameter updates.

**4. Naturalness**:

The LLM's language understanding enables more natural prosody:
- Emphasis on important words (LLM knows which words are content words vs function words)
- Appropriate pause insertion at clause boundaries
- Natural variation in speaking rate and energy

### 6.6 Trade-off: Latency

The main cost of LLM-TTS vs FastSpeech 2 is **inference latency**:

**FastSpeech 2**: All mel frames generated in one forward pass → ~50ms RTF on CPU for a 5-second sentence.

**VieNeu-TTS**: Autoregressive generation of T speech tokens → ~2–5× real-time on CPU (needs GPU acceleration for real-time performance).

**Mitigation strategies**:
- **Streaming**: Generate and play speech tokens in chunks as they are produced
- **Speculative decoding**: Use a small draft model to propose tokens, verify with large model
- **Distillation**: Distill the LLM-TTS into a smaller, non-autoregressive model (ongoing research)

---

## 7. Architecture Comparison

| Dimension | Concatenative | HMM Parametric | Tacotron 2 | FastSpeech 2 | VITS | VieNeu-TTS |
|-----------|--------------|----------------|------------|--------------|------|------------|
| **Input** | Phonemes | Phonemes | Phonemes | Phonemes | Phonemes | Phonemes + Ref Audio |
| **Intermediate** | Audio units | Acoustic features | Mel spectrogram | Mel spectrogram | Latent $z$ | Speech tokens |
| **Output** | Waveform | Waveform (vocoded) | Waveform (vocoded) | Waveform (vocoded) | Waveform | Waveform (decoded) |
| **Vocoder** | None (direct concat) | WORLD/STRAIGHT | WaveNet/HiFi-GAN | HiFi-GAN | None (end-to-end) | NeuCodec decoder |
| **Autoregressive?** | No | No | Yes (decoder) | No | No | Yes (LLM) |
| **Training data** | 5–20h recordings | 5–20h recordings | 20–50h single speaker | 20–50h + alignments | 20–50h single speaker | 100h+ multi-speaker |
| **Vietnamese tones** | Poor (DB coverage) | OK (F0 modeled) | Good | Good | Good | Excellent |
| **Phonation type** | Good (real recording) | Poor | Fair | Fair | Good (GAN) | Good (codec) |
| **Multi-speaker** | No | Yes (limited) | No (single) | Yes (with embed) | Yes (with embed) | Yes (zero-shot) |
| **Zero-shot cloning** | No | No | No | No | Limited | Yes |
| **Approx RTF (CPU)** | <1× (fast) | ~1× | ~10× | ~3× | ~5× | ~2–5× |
| **VRAM needed** | Minimal | Minimal | ~2 GB | ~1 GB | ~2 GB | ~4–8 GB |
| **Main failure mode** | DB coverage gaps | Over-smoothing | Attention errors | Monotone prosody | Training instability | Slow inference |
| **Vietnamese support** | Limited (large DB) | Good | Good | Good | Good | Best |

### 7.1 Choosing the Right System

| Use Case | Recommended System | Reason |
|----------|-------------------|--------|
| Single speaker, offline, quality paramount | VITS or VieNeu-TTS | Best quality |
| Low-latency streaming application | FastSpeech 2 + HiFi-GAN | Fastest inference |
| Limited data (<5h) | Pre-trained VieNeu-TTS (zero-shot) | No fine-tuning needed |
| Voice cloning | VieNeu-TTS | Only system with zero-shot capability |
| Research platform | VITS or FastSpeech 2 | Better understood, easier to modify |
| Edge deployment | FastSpeech 2 | Smallest footprint |
| Maximum Vietnamese tone accuracy | VieNeu-TTS | LLM understands tones semantically |

---

## Further Reading

- Hunt, A., & Black, A. (1996). Unit selection in a concatenative speech synthesis system. *ICASSP*.
- Tokuda, K., et al. (2013). Speech synthesis based on hidden Markov models. *Proceedings of the IEEE*, 101(5).
- Shen, J., et al. (2018). Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. *ICASSP*.
- Ren, Y., et al. (2021). FastSpeech 2: Fast and high-quality end-to-end text to speech. *ICLR*.
- Kim, J., et al. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. *ICML*.
- Wang, C., et al. (2023). VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. *arXiv*.
- Défossez, A., et al. (2022). High Fidelity Neural Audio Compression. *arXiv* (EnCodec).
- Peng, S., et al. (2023). Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale. *NeurIPS*.
