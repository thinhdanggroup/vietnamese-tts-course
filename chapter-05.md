# Chapter 05 — LLM-Based TTS: The VieNeu-TTS Architecture

## Overview

This chapter bridges the gap between the audio codec of Chapter 4 and a full speech synthesis system. The central idea — and the breakthrough that defines modern neural TTS — is surprisingly clean: **if audio is a sequence of discrete tokens, then text-to-speech is a next-token prediction problem**. We train a causal language model to predict speech tokens conditioned on text tokens, exactly as a text LLM predicts words.

By the end of this chapter you will understand:
- The exact mathematical objective that VieNeu-TTS optimizes
- The full prompt format and the role of every component
- The Transformer decoder architecture and its Vietnamese-relevant design choices
- Zero-shot voice cloning as in-context learning
- Sampling strategies and their effect on Vietnamese tonal speech
- Why autoregressive generation is practical on consumer hardware

---

## 1. Framing TTS as Language Modeling

### 1.1 The Core Insight

Consider a text language model. It learns a probability distribution over sequences of word tokens:

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})$$

Now substitute "speech tokens" for "word tokens." A TTS language model learns:

$$P(s_1, s_2, \ldots, s_T \mid \text{text}) = \prod_{t=1}^{T} P(s_t \mid s_1, \ldots, s_{t-1}, \text{text})$$

where $s_t \in \{0, 1, \ldots, V_s - 1\}$ is the $t$-th speech token from the DistillNeuCodec vocabulary ($V_s = 65{,}536$ in VieNeu-TTS).

This is the entire theoretical foundation of VieNeu-TTS. Everything else — the architecture, the prompt format, the training procedure — is an engineering realization of this simple idea.

### 1.2 Training Objective: Cross-Entropy Loss

Training uses the standard autoregressive cross-entropy loss. Given a training pair (text $\mathbf{x}$, speech token sequence $\mathbf{s}$):

$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(s_t \mid s_{<t}, \mathbf{x})$$

where $\theta$ are the model parameters and $s_{<t} = (s_1, \ldots, s_{t-1})$.

This is identical to next-token prediction in GPT-style models, with the difference that the "vocabulary" now includes both text tokens and speech tokens, and the conditioning includes both text and preceding speech tokens.

**Why cross-entropy works for speech:** The codec discretizes the continuous audio manifold. Even though there are 65,536 possible token values, the actual distribution of speech tokens is highly structured — neighboring phonemes constrain which tokens are likely. The LLM learns this structure from data, exactly as text LLMs learn grammar and semantics.

### 1.3 Unified Vocabulary

VieNeu-TTS uses a single combined vocabulary:

$$\mathcal{V} = \mathcal{V}_\text{text} \cup \mathcal{V}_\text{speech} \cup \mathcal{V}_\text{special}$$

| Component | Size | Description |
|---|---|---|
| $\mathcal{V}_\text{text}$ | ~32,000 | Standard LLM text vocabulary (IPA phonemes + characters) |
| $\mathcal{V}_\text{speech}$ | 65,536 | Codec tokens: `<\|speech_0\|>` to `<\|speech_65535\|>` |
| $\mathcal{V}_\text{special}$ | ~20 | Control tokens (see §2) |
| **Total** | **~97,556** | |

The model learns a single embedding matrix for this entire vocabulary. Speech tokens and text tokens are represented in the same space — the model must learn to associate acoustic patterns in speech tokens with phonetic patterns in text tokens.

---

## 2. The Prompt Format

### 2.1 Full Structure

VieNeu-TTS uses a structured prompt that encodes both the voice reference and the synthesis task:

```
user: Convert the text to speech:
<|TEXT_PROMPT_START|>{ref_text_phonemes} {input_text_phonemes}<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>{ref_codes}{generated_codes}<|SPEECH_GENERATION_END|>
```

At inference time, the model is given everything up to and including `{ref_codes}`, and it autoregressively generates `{generated_codes}` until it emits `<|SPEECH_GENERATION_END|>`.

### 2.2 Component Analysis

**`ref_text_phonemes` + `input_text_phonemes` — Joint Phonemization:**

Both the reference text and input text are phonemized and concatenated. This is not an arbitrary choice — joint phonemization serves a critical purpose:

1. **Consistency:** The model sees a single uninterrupted phoneme sequence corresponding to the entire audio span (reference + target). This matches the training format, where the audio is a continuous recording.

2. **Duration conditioning:** By having the full phoneme sequence, the model can implicitly reason about how long the generated speech should be, based on the phoneme count and the rate established by the reference segment.

3. **Pronunciation context:** Vietnamese phonemization is ambiguous without context (e.g., homographs with different tones). Joint context helps resolve ambiguity.

**`ref_codes` — Voice Fingerprint:**

The reference speech tokens $\{s_1^\text{ref}, s_2^\text{ref}, \ldots, s_{T_\text{ref}}^\text{ref}\}$ are inserted verbatim into the prompt. These are the actual codec tokens extracted from the reference audio.

This acts as a "voice fingerprint" in the prompt. When the model generates `{generated_codes}`, it conditions on these reference tokens and learns to produce speech in the same voice style. The voice characteristics are implicitly encoded in the distribution of reference tokens — timbre, pitch range, rhythm, and speaking style all affect which tokens appear.

**Special tokens and their roles:**

| Token | Role |
|---|---|
| `<\|TEXT_PROMPT_START\|>` | Begin phoneme region |
| `<\|TEXT_PROMPT_END\|>` | End phoneme region |
| `<\|SPEECH_GENERATION_START\|>` | Begin speech token region |
| `<\|SPEECH_GENERATION_END\|>` | End-of-speech marker (stop condition) |
| `<\|speech_i\|>` | Codec token $i$ (for $i = 0, \ldots, 65535$) |

### 2.3 Why Phonemes, Not Raw Text?

Vietnamese text has highly ambiguous pronunciation rules. Consider:
- Diacritics encode tone (there are 6 tones: flat, falling-sharp, dipping, asking, drop, heavy)
- Some characters have different pronunciations in northern vs southern dialects
- Loanwords and abbreviations need special handling

Using the IPA (International Phonetic Alphabet) or a Vietnamese phoneme system as the intermediate representation removes most ambiguity. The model sees a near-unambiguous pronunciation representation, making training more consistent.

Example: "xin chào" → `/ɕɪn̟ tɕàw/` (simplified Vietnamese IPA)

---

## 3. Causal Language Model Architecture

### 3.1 Transformer Decoder Overview

VieNeu-TTS-0.3B uses a **Transformer decoder** — a stack of $L$ identical layers, where each layer applies:
1. Causal (masked) self-attention
2. Feed-forward network
3. Layer normalization (before each sublayer — pre-norm)

**Causal masking** ensures that when computing the representation of position $t$, only tokens at positions $1, \ldots, t$ are visible. This enforces the autoregressive property: $P(s_t | s_{<t}, \mathbf{x})$.

The causal attention mask $M \in \{0, -\infty\}^{T \times T}$:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

This mask is added to the attention scores before softmax, making future positions contribute zero probability.

### 3.2 Scaled Dot-Product Attention

For a sequence of $T$ tokens, first compute query, key, and value matrices:

$$Q = X W^Q \in \mathbb{R}^{T \times d_k}, \quad K = X W^K \in \mathbb{R}^{T \times d_k}, \quad V = X W^V \in \mathbb{R}^{T \times d_v}$$

Attention output:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

The division by $\sqrt{d_k}$ (where $d_k$ is the key dimension) prevents the dot products from growing large with embedding dimension, which would saturate the softmax and lead to vanishing gradients.

**Why $\sqrt{d_k}$?** Assume $q, k \sim \mathcal{N}(0, 1)$, both $d_k$-dimensional. Then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. The standard deviation is $\sqrt{d_k}$, so dividing by $\sqrt{d_k}$ normalizes the scale back to unit variance.

### 3.3 Multi-Head Attention

Multi-head attention runs $h$ attention functions in parallel, each with smaller dimension $d_k = d_\text{model} / h$:

$$\text{head}_j = \text{Attention}(Q W_j^Q, K W_j^K, V W_j^V), \quad j = 1, \ldots, h$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Each head learns to attend to different aspects of the input — some heads may focus on local acoustic patterns, others on prosodic structure, others on speaker identity. This multi-aspect attention is critical for handling the diversity of Vietnamese speech patterns.

### 3.4 RoPE Positional Encoding

Standard sinusoidal or learned positional encodings add position information to token embeddings. **Rotary Position Embedding (RoPE)** instead rotates the query and key vectors by position-dependent angles before computing attention:

For position $m$, token dimension $i$, and base $\theta = 10000$:

$$\theta_i = \frac{1}{\theta^{2i/d}} \quad \text{(frequency for dimension pair } i\text{)}$$

The rotation applied to query $q_m$ at position $m$ and key $k_n$ at position $n$:

$$q_m^{(2i)}, q_m^{(2i+1)} \leftarrow q_m^{(2i)} \cos(m\theta_i) - q_m^{(2i+1)} \sin(m\theta_i), \quad q_m^{(2i)} \sin(m\theta_i) + q_m^{(2i+1)} \cos(m\theta_i)$$

The key property of RoPE: the dot product $\langle q_m, k_n \rangle$ depends only on the **relative position** $m - n$, not absolute positions. This is ideal for TTS because:
- The model learns local acoustic dependencies regardless of where in the sentence they appear
- RoPE generalizes to sequences longer than those seen during training (critical for long Vietnamese compound sentences)

### 3.5 RMSNorm

Standard Layer Norm computes:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$$

**RMSNorm** (used in LLaMA-style models) removes the mean-centering step:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$$

where $\gamma$ is a learned per-dimension scale. RMSNorm is:
- Simpler (no bias term $\beta$, no mean computation)
- Equally effective in practice
- Slightly faster to compute

Pre-norm architecture (apply norm before each sublayer, not after) improves training stability, especially for deeper models.

### 3.6 SwiGLU Feed-Forward Network

The standard Transformer FFN is:

$$\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2$$

LLaMA-style models use **SwiGLU**:

$$\text{SwiGLU}(x) = \left(xW_1 \odot \sigma(xW_2)\right) W_3$$

where $\sigma$ is the sigmoid function, $\odot$ is element-wise multiplication, and $W_1, W_2, W_3$ are learned weight matrices.

The gating mechanism ($\sigma(xW_2)$) selectively amplifies or suppresses components of $xW_1$. This gives the network more expressive capacity with the same parameter count, and empirically improves perplexity across language modeling benchmarks.

For TTS, SwiGLU FFN is important because it must capture both:
- **Phonetic transitions** (local, fast-changing patterns)
- **Prosodic structure** (global, slow-changing patterns over the sentence)

The gating mechanism helps the FFN modulate between these two time scales.

---

## 4. VieNeu-TTS-0.3B Model Details

### 4.1 Architecture Specifications

| Parameter | Value |
|---|---|
| Parameters | ~300 million |
| Layers ($L$) | 28 |
| Hidden dimension ($d_\text{model}$) | 1024 |
| Attention heads ($h$) | 16 |
| Key/value dimension ($d_k$) | 64 |
| FFN intermediate dimension | 4096 |
| Max context length | 2048 tokens |
| Vocabulary size | ~97,556 |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |

### 4.2 Vocabulary Split

The vocabulary is carefully partitioned:
- **Text tokens:** Standard multilingual BPE tokenizer covering Vietnamese, English, and IPA symbols
- **Speech tokens 0–65535:** Each corresponds to one entry in the DistillNeuCodec codebook (size $K = 65{,}536$)
- **Special tokens:** As listed in §2.2

Why 65,536 codebook entries? A larger codebook captures finer acoustic detail per token. With $V_s = 65{,}536 = 2^{16}$, each token carries 16 bits of acoustic information. In contrast, standard text vocabularies of ~32,000 carry ~15 bits.

### 4.3 Training on Vietnamese Data

VieNeu-TTS-0.3B is fine-tuned from the NeuTTS Air base model using:
- Vietnamese speech data from multiple speakers and dialects
- Code-switched (Vietnamese + English) data
- Data augmentation: speed perturbation, pitch shift, room impulse response convolution

Training uses the prompt format from §2. The model sees thousands of (reference, target) pairs per speaker, learning to generalize voice conditioning to unseen speakers at inference.

---

## 5. Zero-Shot In-Context Voice Cloning

### 5.1 The Mechanism

At inference, voice cloning requires no fine-tuning. The reference speaker's voice is communicated entirely through the prompt:

```
P(s_t | s_{<t}, ref_codes, text_tokens)
```

The `ref_codes` are the DistillNeuCodec tokens of the reference audio — a compact representation of the speaker's acoustic characteristics.

### 5.2 Why In-Context Learning Works for Voice

During training, the model sees a vast number of (ref_codes, ref_text, target_codes) triples. It learns that:
- Speech tokens following `ref_codes` should match the acoustic style established by `ref_codes`
- This conditioning applies to pitch range, speaking rate, timbre, vowel formants, and prosodic patterns

This is the speech analog of few-shot in-context learning in text LLMs: just as GPT-3 learns a new task from a few examples in the prompt, VieNeu-TTS learns a new voice from reference tokens in the prompt.

### 5.3 Formal Description

Let $e_v$ denote the latent "voice embedding" that the model implicitly computes from the reference codes. The model approximates:

$$P(s_t | s_{<t}, \text{ref\_codes}, \mathbf{x}) \approx P(s_t | s_{<t}, e_v, \mathbf{x})$$

The reference codes act as a soft conditioning vector — the model never explicitly computes $e_v$, but the attention mechanism effectively distills it from the reference token sequence.

### 5.4 Why 3-5 Seconds of Reference Is Sufficient

A 3-second reference clip at 50 tokens/sec = 150 speech tokens. This is enough because:

1. **Phoneme coverage:** 3 seconds of natural Vietnamese speech contains ~15-20 distinct phonemes, sufficient to establish pitch range, vowel formants, and consonant articulation style.

2. **Prosodic signature:** Speaking rate, pause patterns, and rhythm are established within the first 2-3 seconds.

3. **Tonal pattern:** Vietnamese tones are established at the syllable level; even 2-3 tones are enough to infer the speaker's tonal register.

4. **Diminishing returns:** More than 10 seconds of reference does not significantly improve voice similarity, but does consume more context.

**The minimum:** Empirically, below 1.5 seconds the voice cloning quality degrades noticeably. Below 1 second, the model often defaults to a "generic" voice.

---

## 6. Sampling Strategies

### 6.1 The Sampling Problem for TTS

For text generation, some randomness in sampling is desirable — it produces varied, creative output. For TTS, randomness has a different trade-off:
- **Too little randomness:** Monotone prosody, robotic rhythm, unnatural pitch
- **Too much randomness:** Tone errors (catastrophic for Vietnamese), phoneme substitutions, unintelligible output

### 6.2 Greedy Decoding

The simplest decoding: always pick the most probable token:

$$s_t = \arg\max_{v \in \mathcal{V}} P_\theta(v \mid s_{<t}, \mathbf{x})$$

**Problem for TTS:** Speech has genuine local ambiguity — at any moment, there are multiple equally natural continuations. Greedy decoding systematically chooses the "safe" continuation, producing monotone, unnatural prosody. Human speech has prosodic variation that greedy decoding cannot reproduce.

### 6.3 Temperature Sampling

Temperature $\tau > 0$ reshapes the logit distribution before sampling:

$$P'(s_t = v) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}$$

where $z_v$ are the raw logits from the model. This is equivalent to:

$$P'(s_t = v) \propto P_\theta(s_t = v)^{1/\tau}$$

**Effect of temperature:**
- $\tau \to 0$: $P'$ degenerates to a point mass at the argmax → greedy decoding
- $\tau = 1$: $P' = P_\theta$ → sample from the model's learned distribution
- $\tau > 1$: distribution is flattened → more random, less coherent

**Vietnamese-specific considerations:** Vietnamese has 6 lexical tones. A wrong tone = wrong word. At high temperature ($\tau > 1.3$), the model may sample tokens corresponding to the wrong tone for a syllable. The default $\tau = 1.0$ in VieNeu-TTS is carefully chosen to balance prosodic naturalness with tonal accuracy.

**Entropy analysis:** The entropy of the distribution under temperature $\tau$:

$$H_\tau = -\sum_v P'(v) \log P'(v)$$

As $\tau$ increases, $H_\tau$ increases — the model explores more of the distribution. For typical Vietnamese speech tokens, the natural entropy is around $H \approx 5$-$7$ nats, corresponding to $\tau \approx 0.9$-$1.1$.

### 6.4 Top-k Sampling

After temperature scaling, restrict sampling to the $k$ most probable tokens:

$$P''(s_t = v) \propto P'(v) \cdot \mathbb{1}[v \in \text{top-}k]$$

This prevents the model from sampling very unlikely tokens (acoustic glitches, wrong phoneme classes) while preserving prosodic variation.

Typical setting: $k = 50$ for TTS. The top 50 tokens at each step usually span a coherent set of acoustically similar continuations.

**Combined (Temperature + Top-k):**
1. Compute logits $z_v$
2. Apply temperature: $z'_v = z_v / \tau$
3. Apply softmax to get $P'$
4. Zero out all but top-k entries, renormalize
5. Sample from the resulting distribution

This is the standard decoding strategy used in VieNeu-TTS.

### 6.5 Effect on Vietnamese Tones: A Worked Example

Vietnamese tone errors occur when the model samples a speech token corresponding to the wrong tone on a syllable. Consider generating the syllable "ma":
- Flat tone (ma): token IDs cluster around, e.g., $\{234, 235, 251, \ldots\}$
- Rising tone (má): token IDs cluster around, e.g., $\{1024, 1031, 1055, \ldots\}$

At $\tau = 0.8$: the probability mass is concentrated on the correct cluster (very unlikely to sample the wrong tone cluster)

At $\tau = 1.5$: probability spreads across clusters → finite probability of sampling "má" when "ma" was intended

This motivates the choice of $\tau = 1.0$ (or slightly below) for Vietnamese TTS.

---

## 7. Autoregressive Generation Speed

### 7.1 The KV-Cache

Naive autoregressive generation is $O(T^2)$ in compute: for each new token, we recompute attention over all previous tokens. The **key-value (KV) cache** reduces this to $O(T)$ amortized.

**Without KV-cache:** To generate token $t$, compute attention over all $t$ tokens. Then to generate token $t+1$, recompute attention over $t+1$ tokens — recomputing all past keys and values.

**With KV-cache:** Store the key and value matrices for all past tokens. For each new token, only compute the query for the new token, then attend to the cached keys and values. The incremental cost is $O(T)$ per new token (dot product of one query vector with $T$ key vectors).

**Memory cost:** KV-cache requires storing $2 \times L \times T \times d_k \times h$ values (2 for K and V, $L$ layers, $T$ past tokens, $h$ heads). For VieNeu-TTS-0.3B generating 200 tokens: $2 \times 28 \times 200 \times 64 \times 16 = 11.5M$ floats ≈ 46 MB at float32. This is negligible.

### 7.2 Generation Speed on Different Hardware

| Hardware | Format | Speed | Real-Time Factor |
|---|---|---|---|
| Apple M2 Pro CPU | GGUF Q4_K | ~60-80 tok/sec | ~1.4x real-time |
| NVIDIA RTX 3090 | float16 | ~200-400 tok/sec | ~5-8x real-time |
| NVIDIA A100 | float16 | ~600-1000 tok/sec | ~15x real-time |
| CPU (x86, no SIMD) | GGUF Q4_K | ~15-25 tok/sec | ~0.4x real-time |

Real-time factor = (generated audio duration) / (wall-clock inference time). RTF > 1.0 means faster than real-time.

At 50 tokens/sec (audio token rate), an LLM generating 60 tokens/sec on M2 CPU runs at 1.2x real-time — sufficient for interactive TTS.

### 7.3 GGUF Quantization

VieNeu-TTS distributes a GGUF-format model for CPU inference. GGUF is a binary format for quantized models used by `llama.cpp`.

**Q4_K_M quantization:** Most weight matrices are quantized to 4 bits per weight (from 32-bit float). This gives:
- Model size reduction: ~8x (32-bit → 4-bit)
- Quality degradation: minimal (<1% perplexity increase for TTS use cases)
- Speed improvement: ~2-4x on CPU (fewer memory bandwidth requirements, SIMD-optimized int8/int4 kernels)

For VieNeu-TTS-0.3B (~300M parameters):
- Float32: ~1.2 GB
- Q4_K_M: ~175 MB

The Q4_K_M model fits entirely in L3 cache on modern CPUs, dramatically improving throughput.

### 7.4 Streaming Generation

VieNeu-TTS implements **streaming**: it begins decoding the first audio chunk before the full speech token sequence is complete.

**Process:**
1. Generate first $N$ speech tokens (e.g., $N = 50$ = 1 second of audio)
2. Pass tokens to codec decoder → first audio chunk
3. Stream audio chunk to playback system
4. Concurrently continue generating tokens $N+1, N+2, \ldots$
5. Repeat until `<|SPEECH_GENERATION_END|>` is emitted

This reduces **time to first audio (TTFA)** from the full generation time to approximately 1 second — making VieNeu-TTS feel responsive for interactive applications.

---

## Summary

The VieNeu-TTS architecture reduces TTS to a single unified problem: next-token prediction over a combined text-and-speech vocabulary. Every architectural choice has a clear motivation:

| Component | Choice | Why |
|---|---|---|
| Model type | Causal LM | Autoregressive generation |
| Attention | Causal masked multi-head | Prevent future information leakage |
| Positional encoding | RoPE | Relative positions, length generalization |
| Normalization | RMSNorm pre-norm | Stability, simplicity |
| FFN | SwiGLU | Gated expressivity |
| Sampling | Temperature=1.0, Top-k=50 | Balance naturalness vs tone correctness |
| Decoding | KV-cache + GGUF Q4 | Real-time on consumer hardware |
| Voice cloning | In-context (ref_codes) | Zero-shot, no fine-tuning required |

**Training objective:** $\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(s_t \mid s_{<t}, \mathbf{x})$

**Inference:** Autoregressively sample $s_t$ until `SPEECH_GENERATION_END`, decode with DistillNeuCodec, stream audio.

---

## Further Reading

- Touvron et al. (2023). *LLaMA 2: Open Foundation and Fine-Tuned Chat Models*. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Wang et al. (2023). *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (VALL-E). [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- Frantar et al. (2022). *GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers*. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
