# Chapter 04 — Neural Audio Codecs: Turning Sound into Tokens

## Overview

This chapter addresses a fundamental bottleneck in building LLM-based TTS systems: language models operate on discrete tokens, but audio is a continuous waveform sampled at tens of thousands of values per second. To bridge this gap, we need a system that can compress a continuous audio signal into a compact sequence of discrete integer tokens — and reconstruct high-quality audio from those tokens. This system is called a **neural audio codec**.

By the end of this chapter you will understand:
- Why naive tokenization of audio fails
- The mathematics of Vector Quantization (VQ) and Residual Vector Quantization (RVQ)
- How NeuCodec and DistillNeuCodec are architected
- Why DistillNeuCodec is the right choice for VieNeu-TTS
- How to compute token rates and context window constraints for Vietnamese TTS

---

## 1. The Codec Problem

### 1.1 Goal: Discrete Tokens for an LLM

A large language model operates over a vocabulary of discrete tokens. For text, a tokenizer (BPE, SentencePiece) maps strings to integers. For speech, we need an analogous map:

$$\text{audio waveform} \xrightarrow{\text{codec encoder}} \text{sequence of integers}$$

$$\text{sequence of integers} \xrightarrow{\text{codec decoder}} \text{audio waveform}$$

This is not optional — it is the architectural prerequisite for treating TTS as a language modeling task.

### 1.2 Why You Cannot Just Use Mel Spectrogram Tokens

A naive idea: discretize the mel spectrogram. Why does this fail?

**Problem 1 — Too many frames.** A mel spectrogram at a 10 ms hop length produces 100 frames per second. A 5-second sentence yields 500 frames. Each frame is a 80-dimensional vector. Even if you quantize each value to 256 bins, one frame becomes 80 integers — giving 40,000 integers for 5 seconds. An LLM context of 2048 tokens can hold only ~0.25 seconds. This is completely unworkable.

**Problem 2 — Continuous values.** Each mel bin is a real-valued log-energy. You cannot feed continuous values into a discrete vocabulary. Naively binning each coefficient independently destroys inter-coefficient correlations.

**Problem 3 — Reconstruction.** A mel spectrogram is lossy by design — it discards phase information. Converting mel → waveform requires a vocoder (Griffin-Lim, HiFi-GAN), which introduces its own errors. We want an end-to-end differentiable system.

### 1.3 Desired Properties of a Neural Codec

| Property | Why It Matters |
|---|---|
| **Compact** | Fewer tokens per second → longer sentences in context window |
| **Discrete** | Enables LLM cross-entropy training |
| **Reconstructable** | High-quality audio output (not just compression) |
| **Semantic** | Nearby codes → acoustically similar sounds |
| **Differentiable (training)** | End-to-end gradient flow during codec training |

Neural codecs — specifically those based on Residual Vector Quantization — satisfy all five properties.

---

## 2. Vector Quantization (VQ)

### 2.1 The Codebook

Vector Quantization maps a continuous vector to the nearest element in a finite set called a **codebook**.

Formally, let:
- $z_e \in \mathbb{R}^D$ — the encoder output (a continuous latent vector)
- $\mathcal{C} = \{e_1, e_2, \ldots, e_K\} \subset \mathbb{R}^D$ — the codebook, $K$ vectors each of dimension $D$

The **quantization function** maps $z_e$ to the nearest codebook entry:

$$z_q = e_{k^*}, \quad k^* = \arg\min_{k \in \{1,\ldots,K\}} \|z_e - e_k\|_2^2$$

The index $k^*$ is the discrete token. Given $K$ codebook entries, we need $\lceil \log_2 K \rceil$ bits to represent one token. For example, $K = 1024$ requires 10 bits per frame.

### 2.2 The Straight-Through Estimator

The quantization operation $z_q = e_{k^*}$ has zero gradient almost everywhere (the $\arg\min$ is not differentiable). Without a workaround, gradients cannot flow from the decoder back to the encoder.

The **straight-through estimator (STE)** solves this by defining the gradient in the backward pass as:

$$\frac{\partial \mathcal{L}}{\partial z_e} \approx \frac{\partial \mathcal{L}}{\partial z_q}$$

That is, we pretend the quantization step does not exist during the backward pass, and pass gradients through unchanged. In PyTorch notation:

```python
z_q = z_e + (z_q - z_e).detach()
# Forward: z_q is quantized
# Backward: gradient flows to z_e unchanged
```

This is a biased gradient estimator, but works extremely well in practice.

### 2.3 Commitment Loss

With STE, the encoder gradient exists, but training is still unstable. Two issues arise:

1. **Codebook entries may be far from encoder outputs** — encoding is random, codebook never moves toward data
2. **Encoder outputs may stray from codebook** — encoder maps to arbitrary regions, codebook can't track them

The **commitment loss** addresses both by adding an explicit penalty:

$$\mathcal{L}_\text{commit} = \|\text{sg}[z_e] - e_{k^*}\|_2^2 + \beta \|z_e - \text{sg}[e_{k^*}]\|_2^2$$

where $\text{sg}[\cdot]$ denotes the **stop-gradient** operator (treat as constant during backprop):
- First term: moves codebook vector $e_{k^*}$ toward encoder output $z_e$ (codebook learning)
- Second term: moves encoder output toward codebook (commitment), weighted by $\beta$
- Typical value: $\beta = 0.25$

In practice, the first term (codebook gradient) is replaced by EMA updates (see §2.5), leaving only:

$$\mathcal{L}_\text{commit} = \beta \|z_e - \text{sg}[e_{k^*}]\|_2^2$$

### 2.4 Codebook Collapse

**Codebook collapse** is the most common failure mode of VQ: most codebook entries are never used, and the model degrades to a tiny effective vocabulary.

**Why it happens:** During early training, some entries are assigned many vectors by chance. Those entries get large gradient updates and drift toward data clusters. Other entries get no updates and stay where they were initialized — increasingly far from the data manifold. As training continues, these unused entries are never assigned, their gradients remain zero, and they contribute nothing.

**Detection:** Track codebook utilization — the fraction of entries used in a given batch. Healthy training: >80% utilization. Collapsed: <20%.

**Solution 1 — EMA updates** (see §2.5): decouple codebook updates from gradient descent.

**Solution 2 — Random restarts:** At each training step, identify dead entries (usage below a threshold) and reinitialize them to randomly sampled encoder outputs from the current batch.

```python
# Pseudocode: random restart
for k in range(K):
    if usage_count[k] < threshold:
        # Reinitialize from a random encoder output in current batch
        random_idx = torch.randint(0, z_e_batch.shape[0], (1,))
        codebook[k] = z_e_batch[random_idx].detach()
```

### 2.5 EMA Codebook Updates

Instead of using gradient descent to update codebook vectors (which is slow and unstable), **Exponential Moving Average (EMA) updates** maintain running statistics:

Let:
- $n_k$ — number of vectors assigned to entry $k$ in the current batch
- $m_k$ — sum of encoder outputs assigned to entry $k$ in the current batch
- $N_k, M_k$ — EMA accumulators
- $\gamma \in (0,1)$ — decay factor (typically 0.99)

EMA update rules:

$$N_k \leftarrow \gamma N_k + (1-\gamma) n_k$$

$$M_k \leftarrow \gamma M_k + (1-\gamma) \sum_{z_e: k^*(z_e)=k} z_e$$

$$e_k \leftarrow \frac{M_k}{N_k}$$

Interpretation: the codebook vector $e_k$ is updated to track the exponential moving average of the encoder outputs assigned to it. This is equivalent to online k-means clustering. The EMA update bypasses gradient descent entirely for the codebook, making training far more stable.

The gradient of $\mathcal{L}_\text{commit}$ with respect to $e_k$ is then zero (we use EMA instead), leaving:

$$\mathcal{L}_\text{commit} = \beta \|z_e - \text{sg}[e_{k^*}]\|_2^2$$

This only trains the encoder to commit to codebook entries — the codebook itself is updated via EMA.

---

## 3. Residual Vector Quantization (RVQ)

### 3.1 Motivation: Single VQ Limits Quality

A single VQ codebook with $K$ entries can represent at most $K$ distinct latent vectors. Even with $K = 8192$, the reconstruction quality is limited because the encoder output $z_e$ is compressed into a single nearest neighbor. The quantization error (residual) can be large.

For audio, this translates to artifacts — blurring of fine spectral detail, loss of high-frequency content, muffled voice quality.

### 3.2 RVQ: Quantizing Residuals Iteratively

**Residual Vector Quantization (RVQ)** applies VQ multiple times, each time quantizing the residual left by the previous stage.

Formally, with $Q$ quantization stages:

$$z_0 = z_e$$

For $i = 1, 2, \ldots, Q$:

$$k_i^* = \arg\min_k \|z_{i-1} - e_k^{(i)}\|_2^2$$

$$q_i = e_{k_i^*}^{(i)}$$

$$z_i = z_{i-1} - q_i$$

Final quantized representation:

$$z_q = \sum_{i=1}^{Q} q_i$$

Each stage $i$ has its own independent codebook $\mathcal{C}^{(i)}$ with $K$ entries.

The discrete token for one frame is the tuple $(k_1^*, k_2^*, \ldots, k_Q^*)$ — a vector of $Q$ integers.

### 3.3 Why RVQ Works

Each successive stage refines the approximation. The reconstruction error after $Q$ stages is:

$$\|z_e - z_q\|_2^2 = \|z_Q\|_2^2$$

which is the residual after all quantization stages. With $Q$ stages each of $K = 1024$ entries, the effective codebook size is $K^Q = 1024^Q$ — exponentially large — but using only $Q \cdot K$ actual storage.

Empirically:
- $Q=1$: rough voice, noticeable artifacts
- $Q=4$: good intelligibility
- $Q=8$: near-transparent quality (EnCodec)

### 3.4 Token Rate Analysis

For RVQ, each frame of audio produces $Q$ integer tokens (one per stage). If the codec operates at $F$ frames per second:

$$\text{tokens/sec} = Q \times F$$

**EnCodec (Meta, 2022):**
- Sample rate: 24,000 Hz
- Hop size: 320 samples → $F = 24000/320 = 75$ frames/sec
- $Q = 8$ codebook levels
- Token rate: $8 \times 75 = \mathbf{600}$ tokens/sec

This is far too high for LLM modeling. A 5-second Vietnamese sentence would consume 3,000 tokens — exceeding most context windows entirely.

| Model | Sample Rate | Hop | Frames/sec | Q | Tokens/sec |
|---|---|---|---|---|---|
| EnCodec 24k | 24,000 Hz | 320 | 75 | 8 | **600** |
| SoundStream | 16,000 Hz | 320 | 50 | 8 | 400 |
| NeuCodec | 24,000 Hz | 320 | 75 | 8 | 600 |
| DistillNeuCodec | 24,000 Hz | 480 | 50 | 1 | **50** |

---

## 4. NeuCodec Architecture

NeuCodec is the full-quality neural codec developed by Neuphonic, using RVQ. It consists of three components: an encoder, an RVQ quantizer, and a decoder.

### 4.1 Encoder: Causal CNN Stack

The encoder maps a raw waveform $x \in \mathbb{R}^T$ to a sequence of latent vectors $z_e \in \mathbb{R}^{T' \times D}$, where $T' \ll T$.

Architecture (causal = no future context, required for streaming):
1. Input: waveform, shape $(B, 1, T)$
2. Causal Conv1d layer — maps to $C$ channels
3. Stack of Causal EncoderBlocks, each doubling the stride:
   - Stride-2 Conv1d (downsample)
   - Residual LSTM or dilated causal convolution layers
4. Final Conv1d → latent $z_e \in \mathbb{R}^{B \times T' \times D}$

The total downsampling factor (hop size) is the product of all strides. For NeuCodec: strides $[2, 4, 5, 8]$ → $2 \times 4 \times 5 \times 8 = 320$ → 24,000/320 = 75 frames/sec.

### 4.2 Quantizer: RVQ with Q Codebooks

Each codebook has $K = 1024$ entries of dimension $D$. For each frame:
1. Find nearest entry in codebook 1 → token $k_1$, quantized vector $q_1$
2. Compute residual $r_1 = z_e - q_1$
3. Find nearest entry in codebook 2 for $r_1$ → token $k_2$, quantized vector $q_2$
4. ... repeat for $Q$ stages
5. $z_q = q_1 + q_2 + \cdots + q_Q$

**Codebook update:** EMA with $\gamma = 0.99$, random restarts for dead entries.

### 4.3 Decoder: Mirror CNN Stack

The decoder maps $z_q \in \mathbb{R}^{B \times T' \times D}$ back to the waveform $\hat{x} \in \mathbb{R}^{B \times 1 \times T}$:

1. Initial Conv1d — maps $D$ → $C$ channels
2. Stack of Causal DecoderBlocks with upsampling (transposed convolutions)
3. Final Conv1d → output waveform, $\tanh$ activation to bound values to $[-1, 1]$

The decoder mirrors the encoder: if the encoder had strides $[2, 4, 5, 8]$, the decoder has upsampling factors $[8, 5, 4, 2]$.

### 4.4 Training Loss

NeuCodec is trained end-to-end with a composite loss:

$$\mathcal{L} = \lambda_r \mathcal{L}_\text{recon} + \lambda_s \mathcal{L}_\text{STFT} + \lambda_c \mathcal{L}_\text{commit} + \lambda_g \mathcal{L}_\text{GAN}$$

**Reconstruction loss** — waveform-level L1:
$$\mathcal{L}_\text{recon} = \|x - \hat{x}\|_1$$

**STFT perceptual loss** — multi-resolution STFT:
$$\mathcal{L}_\text{STFT} = \sum_{s \in \text{scales}} \left( \||\text{STFT}_s(x)| - |\text{STFT}_s(\hat{x})|\|_F + \|\log|\text{STFT}_s(x)| - \log|\text{STFT}_s(\hat{x})|\|_F \right)$$

Multiple window sizes $s \in \{512, 1024, 2048\}$ capture both fine-grained and coarse spectral structure.

**Commitment loss:**
$$\mathcal{L}_\text{commit} = \beta \sum_{i=1}^{Q} \|z_{i-1} - \text{sg}[q_i]\|_2^2$$

**GAN discriminator loss** — using multi-period and multi-scale discriminators (as in HiFi-GAN):
$$\mathcal{L}_\text{GAN} = \mathcal{L}_\text{adv} + \lambda_\text{fm} \mathcal{L}_\text{feat}$$

where $\mathcal{L}_\text{adv}$ is the adversarial loss and $\mathcal{L}_\text{feat}$ is the feature matching loss (L1 distance between discriminator internal features of real vs generated audio).

Typical weights: $\lambda_r = 1.0$, $\lambda_s = 1.0$, $\lambda_c = 1.0$, $\lambda_g = 3.0$ (GAN needs higher weight to avoid blurry reconstructions).

---

## 5. DistillNeuCodec

### 5.1 Motivation: The Context Length Problem

NeuCodec produces $Q$ tokens per frame. At 75 frames/sec with $Q=8$, that is 600 tokens/sec. A single Vietnamese sentence of 4 seconds needs 2,400 tokens — already exceeding a 2,048-token context window before any text tokens are added.

DistillNeuCodec addresses this through two modifications:

1. **Single codebook ($Q=1$):** one integer per frame instead of $Q$
2. **Larger hop size:** 480 samples instead of 320 → 50 frames/sec instead of 75

Combined: $1 \times 50 = \mathbf{50}$ tokens/sec.

### 5.2 Knowledge Distillation from RVQ Teacher

Training a single-codebook codec from scratch with only a reconstruction loss fails — the single codebook cannot capture enough detail. Instead, DistillNeuCodec uses **knowledge distillation** from the NeuCodec RVQ teacher.

**Teacher:** NeuCodec (full RVQ, $Q=8$) — produces rich, detailed latents $z_q^\text{teacher} = \sum_{i=1}^{Q} q_i^{(i)}$

**Student:** DistillNeuCodec (single VQ, $Q=1$) — produces $z_q^\text{student} = q_1^{(1)}$

**Distillation loss:**

$$\mathcal{L}_\text{distill} = \|z_q^\text{student} - z_q^\text{teacher}\|_2^2$$

The student is trained to minimize the distance between its quantized latent and the teacher's sum-of-residuals latent. The student's codebook must compress what the teacher captures across 8 codebooks into one.

Full student training loss:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda_s \mathcal{L}_\text{STFT} + \lambda_c \mathcal{L}_\text{commit} + \lambda_d \mathcal{L}_\text{distill}$$

### 5.3 Quality Trade-off

What is lost and what is kept when moving from NeuCodec to DistillNeuCodec:

| Aspect | NeuCodec (RVQ, Q=8) | DistillNeuCodec (Q=1) |
|---|---|---|
| Token rate | 600 tokens/sec | 50 tokens/sec |
| Fine spectral detail | Preserved | Some loss |
| Voice identity | Excellent | Good |
| Prosody (pitch, rhythm) | Excellent | Good |
| Consonant sharpness | Sharp | Slightly softer |
| Vietnamese tones | Full fidelity | Well preserved |
| Reconstruction PESQ | ~4.0 | ~3.5 |
| LLM context (5s sentence) | 3,000 tokens | 250 tokens |

The critical insight: for **TTS**, the LLM needs to model the distribution of speech tokens. A simpler token sequence (50/sec) is much easier to model than a complex multi-stream sequence (600/sec). The small quality reduction from DistillNeuCodec is vastly outweighed by the improved LLM context utilization and reduced modeling complexity.

### 5.4 Why DistillNeuCodec is Right for VieNeu-TTS

VieNeu-TTS is a generative LLM. It must predict speech tokens autoregressively. The longer the token sequence, the harder the LLM's task and the more GPU memory is consumed.

With DistillNeuCodec at 50 tokens/sec:
- A 3-second Vietnamese sentence: ~150 tokens (easy to model, fits in any context)
- Reference audio (3 seconds): ~150 tokens
- Text phonemes: ~50 tokens
- Total prompt: ~350 tokens → context utilization ~17% of 2,048

This leaves ample room for the model to generate the output speech tokens without hitting the context limit.

---

## 6. Token Rate Mathematics for VieNeu-TTS

### 6.1 Core Parameters

| Parameter | Value |
|---|---|
| Sample rate | 24,000 Hz |
| Codec hop length | 480 samples |
| Frames per second | $24{,}000 / 480 = 50$ frames/sec |
| Codebook levels (Q) | 1 |
| Tokens per frame | 1 |
| Token rate | 50 tokens/sec |

### 6.2 Sentence-Level Token Budget

For a sentence of duration $d$ seconds:

$$\text{speech tokens} = 50 \times d$$

Typical Vietnamese sentence statistics:
- Short sentence (2s): $50 \times 2 = 100$ tokens
- Medium sentence (4s): $50 \times 4 = 200$ tokens
- Long sentence (8s): $50 \times 8 = 400$ tokens

Vietnamese phoneme rate: approximately 5-7 syllables/second. At an average 4 syllables, with tones, a typical news sentence runs 3-4 seconds.

### 6.3 Context Window Analysis

VieNeu-TTS uses a 2,048-token context window. The prompt structure:

```
[text tokens: ~50-100] + [ref_codes: 50×d_ref] + [speech_start token] + [generated tokens: 50×d_gen]
```

Solving for maximum generation length:

$$50 + 50 \times d_\text{ref} + 1 + 50 \times d_\text{gen} \leq 2048$$

For a 3-second reference clip ($d_\text{ref} = 3$):

$$50 \times d_\text{gen} \leq 2048 - 50 - 150 - 1 = 1847$$

$$d_\text{gen} \leq 36.9 \text{ seconds}$$

This is more than enough for any single utterance in Vietnamese TTS.

### 6.4 Derivation of Hop Length from Encoder Strides

The encoder downsamples with strides $[s_1, s_2, \ldots, s_n]$. The total hop length is:

$$H = \prod_{i=1}^{n} s_i$$

For DistillNeuCodec strides $[2, 4, 6, 10]$:

$$H = 2 \times 4 \times 6 \times 10 = 480$$

This gives exactly 50 frames/sec at 24 kHz.

---

## 7. Codec Quality Metrics

### 7.1 PESQ — Perceptual Evaluation of Speech Quality

PESQ (ITU-T P.862) is an intrusive metric: it requires both the reference (original) and degraded (reconstructed) signal. It models the human auditory system to predict Mean Opinion Score (MOS).

$$\text{PESQ} \in [-0.5, 4.5]$$

Interpretation:
| PESQ | Quality |
|---|---|
| 4.0 – 4.5 | Excellent (transparent) |
| 3.5 – 4.0 | Good |
| 3.0 – 3.5 | Fair |
| 2.0 – 3.0 | Poor |
| < 2.0 | Bad |

### 7.2 SI-SNR — Scale-Invariant Signal-to-Noise Ratio

SI-SNR removes the scale ambiguity from SNR. Given reference $s$ and estimate $\hat{s}$, both zero-mean:

**Target component** (projection of estimate onto reference):

$$s_\text{target} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2} s$$

**Noise component:**

$$e_\text{noise} = \hat{s} - s_\text{target}$$

**SI-SNR:**

$$\text{SI-SNR} = 10 \log_{10} \frac{\|s_\text{target}\|^2}{\|e_\text{noise}\|^2}$$

SI-SNR is scale-invariant: multiplying $\hat{s}$ by any scalar does not change the score. A score above 20 dB is generally considered good reconstruction quality.

### 7.3 UTMOS — Neural MOS Predictor

UTMOS (Saeki et al., 2022) is a non-intrusive neural predictor of MOS. It takes only the synthesized audio and outputs a predicted MOS score. Trained on the BVCC dataset of human MOS annotations.

$$\text{UTMOS} \in [1.0, 5.0]$$

UTMOS does not require the original reference, making it suitable for evaluating TTS systems where the "ground truth" is the natural voice — but no original recording of that exact utterance exists.

### 7.4 MCD — Mel Cepstral Distortion

MCD measures spectral distortion in the mel-cepstral domain:

$$\text{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{k=1}^{K} (c_k^\text{ref} - c_k^\text{syn})^2} \quad \text{[dB]}$$

where $c_k$ are mel-frequency cepstral coefficients (MFCCs). Lower MCD is better. A MCD below 5 dB is generally considered acceptable; below 3 dB is very good.

### 7.5 Comprehensive Metric Summary

| Metric | Type | Range | Better | Requires Reference? |
|---|---|---|---|---|
| PESQ | Intrusive | [-0.5, 4.5] | Higher | Yes |
| SI-SNR | Intrusive | (−∞, +∞) dB | Higher | Yes |
| MCD | Intrusive | [0, ∞) dB | Lower | Yes |
| UTMOS | Non-intrusive | [1.0, 5.0] | Higher | No |
| CER | ASR-based | [0%, 100%] | Lower | No |

---

## Summary

Neural audio codecs are the critical bridge between continuous audio and discrete language modeling. The key ideas:

1. **VQ** maps encoder outputs to the nearest codebook vector — enabling discrete tokens but with limited quality.
2. **Commitment loss + EMA updates** ensure stable VQ training and prevent codebook collapse.
3. **RVQ** iteratively quantizes residuals across $Q$ codebooks, exponentially increasing effective vocabulary while keeping storage linear.
4. **NeuCodec** uses RVQ ($Q=8$) at 75 frames/sec → 600 tokens/sec, producing near-transparent audio quality.
5. **DistillNeuCodec** uses knowledge distillation to produce a single codebook at 50 frames/sec → 50 tokens/sec, enabling practical LLM-based TTS.
6. For Vietnamese TTS with a 2,048-token context: a 3-second reference clip uses 150 tokens, leaving room for long generated utterances.

In the next chapter, we will see how these discrete speech tokens are fed to a causal language model to generate synthesized speech autoregressively.

---

## Further Reading

- Défossez et al. (2022). *High Fidelity Neural Audio Compression* (EnCodec). [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
- van den Oord et al. (2017). *Neural Discrete Representation Learning* (VQ-VAE). [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Kumar et al. (2023). *High-Fidelity Audio Compression with Improved RVQGAN*. NeurIPS 2023.
- Zeghidour et al. (2021). *SoundStream: An End-to-End Neural Audio Codec*. IEEE/ACM TASLP.
