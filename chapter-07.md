# Chapter 07 — LoRA Fine-tuning Theory: Adapting VieNeu-TTS to Your Voice

> **Audience**: ML engineers who understand training loops, loss functions, and backpropagation, but are new to parameter-efficient fine-tuning and its application to TTS.
> **Goal**: Understand the mathematical foundations of LoRA, derive why it works, and apply it correctly to fine-tune VieNeu-TTS on a new Vietnamese voice.

---

## Table of Contents

1. [Why Full Fine-tuning is Expensive](#1-why-full-fine-tuning-is-expensive)
2. [LoRA: Low-Rank Adaptation](#2-lora-low-rank-adaptation)
3. [Parameter Count Comparison](#3-parameter-count-comparison)
4. [Which Layers to Target](#4-which-layers-to-target)
5. [Training Dynamics](#5-training-dynamics)
6. [Rank Hyperparameter Analysis](#6-rank-hyperparameter-analysis)
7. [Overfitting in TTS](#7-overfitting-in-tts)
8. [LoRA vs QLoRA vs Full Fine-tune](#8-lora-vs-qlora-vs-full-fine-tune)

---

## 1. Why Full Fine-tuning is Expensive

### 1.1 The Memory Arithmetic

To understand why parameter-efficient methods exist, you need to understand exactly what occupies GPU memory during training. Full fine-tuning of a language model requires storing five distinct categories of tensors simultaneously:

**1. Model weights:** Every parameter in the network stored at training precision.

$$M_{\text{weights}} = N_{\text{params}} \times B_{\text{dtype}}$$

For VieNeu-TTS-0.3B with $N = 300 \times 10^6$ parameters in fp32 ($B = 4$ bytes):

$$M_{\text{weights}} = 300 \times 10^6 \times 4 = 1.2 \text{ GB}$$

**2. Gradients:** The backward pass computes $\partial \mathcal{L} / \partial \theta_i$ for every parameter. These have the same shape and dtype as the weights:

$$M_{\text{gradients}} = M_{\text{weights}} = 1.2 \text{ GB}$$

**3. Optimizer states:** The Adam optimizer (Kingma & Ba, 2015) maintains two moving average vectors per parameter — the first moment $m_t$ and the second moment $v_t$:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Both $m_t$ and $v_t$ have the same shape as $\theta$. In practice, Adam is often kept in fp32 even when training in mixed precision. This doubles the optimizer footprint:

$$M_{\text{optimizer}} = 2 \times M_{\text{weights}} = 2.4 \text{ GB}$$

**4. Activations:** During the forward pass, intermediate activations must be stored for the backward pass (unless gradient checkpointing is used). For a Transformer with batch size $B$, sequence length $L$, hidden dimension $d$, and $N_L$ layers:

$$M_{\text{activations}} \approx B \times L \times d \times N_L \times B_{\text{dtype}}$$

For VieNeu-TTS-0.3B with $B=2$, $L=2048$, $d=1024$, $N_L=24$, fp32:

$$M_{\text{activations}} = 2 \times 2048 \times 1024 \times 24 \times 4 = 402 \text{ MB} \approx 0.4 \text{ GB}$$

This is a lower bound — actual activation memory is higher because of attention matrices ($B \times N_H \times L \times L$) and intermediate MLP states.

**5. Input data buffers and miscellaneous:** Typically ~200–500 MB.

**Total GPU memory for full fine-tuning:**

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}}$$
$$= 1.2 + 1.2 + 2.4 + 0.4 = 5.2 \text{ GB (minimum)}$$

In practice, with attention matrices, activation spikes during backward pass, and PyTorch memory fragmentation, the real requirement is **6–8 GB** for VieNeu-TTS-0.3B. This fits on an RTX 3090 (24 GB) but not on an RTX 3060 (12 GB) at batch size > 1.

### 1.2 Catastrophic Forgetting

Memory is not the only problem. When you update all 300M parameters to learn a new voice, the model can **forget** its previously learned capabilities:

- Phonetic coverage learned during pretraining degrades
- Prosody for uncommon words collapses
- The model overspecializes to the training speaker

This phenomenon, first described by McCloskey & Cohen (1989) as **catastrophic interference**, occurs because gradient updates that improve performance on the new task destructively interfere with the weight configurations that encoded previous knowledge.

Mathematically, if the loss landscape for the original task has its minimum at $\theta^*$ and the new task at $\theta^{\dagger}$, the gradient of the new task loss $\nabla_\theta \mathcal{L}_{\text{new}}$ points generally away from $\theta^*$. Full gradient descent follows this gradient without constraint, moving the model far from $\theta^*$.

### 1.3 The Minimal Fine-tuning Hypothesis

Hu et al. (2021) observed a critical empirical finding: **when large language models adapt to new tasks, the weight changes $\Delta W$ have a surprisingly low intrinsic rank**. That is, the "meaningful" changes can be captured by a low-dimensional subspace of the full parameter space.

This finding — combined with the theoretical result that neural networks are over-parameterized — motivates Low-Rank Adaptation.

---

## 2. LoRA: Low-Rank Adaptation

### 2.1 Core Mathematical Formulation

Consider a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in a Transformer layer. During standard fine-tuning, the weight update is unconstrained:

$$W' = W_0 + \Delta W, \quad \Delta W \in \mathbb{R}^{d \times k}$$

LoRA constrains $\Delta W$ to a **low-rank decomposition**:

$$\Delta W = B A$$

where:
- $B \in \mathbb{R}^{d \times r}$ is the left factor
- $A \in \mathbb{R}^{r \times k}$ is the right factor
- $r \ll \min(d, k)$ is the **rank** hyperparameter

The forward pass with LoRA applied becomes:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

Since $W_0$ is frozen, only $B$ and $A$ receive gradient updates. The modified forward pass adds a small computational overhead but dramatically reduces trainable parameters.

### 2.2 Initialization Strategy

The initialization is carefully designed so that **LoRA has zero effect at the start of training** (i.e., the pretrained model behavior is preserved at step 0):

- $A \sim \mathcal{N}(0, \sigma^2)$ — initialized randomly (Gaussian noise)
- $B = \mathbf{0}$ — initialized to all zeros

This ensures $\Delta W = BA = \mathbf{0} \cdot A = \mathbf{0}$ at initialization, so the model starts exactly at the pretrained weights. As training proceeds, $B$ learns non-zero values that combine with $A$ to form a useful low-rank update.

### 2.3 The Scaling Factor

The actual weight update applied during inference is:

$$W' = W_0 + \frac{\alpha}{r} \cdot BA$$

where $\alpha$ is a **scaling hyperparameter** (called `lora_alpha` in the PEFT library). The factor $\alpha/r$ serves two purposes:

1. **Rank-invariance:** When you change $r$, you do not need to re-tune the effective learning rate if you keep $\alpha$ constant. Doubling $r$ without the $1/r$ factor would double the magnitude of the LoRA update at initialization, destabilizing training.

2. **Magnitude control:** $\alpha$ allows you to control the overall scale of the adaptation independently from the rank. The standard practice is $\alpha = 2r$ or $\alpha = r$, giving an effective scale of 2 or 1.

### 2.4 Why Low-Rank Works: The SVD Justification

**Theorem (Eckart-Young-Mirsky):** The best rank-$r$ approximation of a matrix $M \in \mathbb{R}^{d \times k}$ in Frobenius norm is given by its truncated SVD:

$$M_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top = U_r \Sigma_r V_r^\top$$

where $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_{\min(d,k)}$ are the singular values, and $U_r, V_r$ are the first $r$ left/right singular vectors.

The approximation error is:

$$\|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(d,k)} \sigma_i^2}$$

When the ideal weight update $\Delta W^*$ has fast-decaying singular values (most energy concentrated in the first $r$ components), the rank-$r$ approximation is excellent.

**Why should $\Delta W^*$ be low-rank?** Empirical evidence from multiple studies shows:

1. **Task manifold hypothesis**: Natural language tasks and speech tasks live on low-dimensional manifolds in the space of possible behaviors. The gradient updates that implement task adaptation therefore span a low-dimensional subspace.

2. **Overparameterization**: A 300M parameter model has vastly more capacity than needed to represent a single new voice. The actual "information" added (speaker identity, style) is much lower-dimensional.

3. **Fine-tuning trajectory analysis**: Intrinsic dimension measurements (Aghajanyan et al., 2021) show that 90% of the loss reduction during fine-tuning of LLMs occurs within a subspace of dimension ~100–1000, far below the full parameter dimension.

### 2.5 Full Derivation of LoRA Optimization

The training objective with LoRA is:

$$\min_{A, B} \mathcal{L}(\theta_0 + BA)$$

where $\theta_0 = \text{vec}(W_0)$ is fixed. The gradient with respect to $B$ and $A$ is:

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial \Delta W} \cdot A^\top$$
$$\frac{\partial \mathcal{L}}{\partial A} = B^\top \cdot \frac{\partial \mathcal{L}}{\partial \Delta W}$$

where $\frac{\partial \mathcal{L}}{\partial \Delta W}$ is the gradient of the loss with respect to $\Delta W$ — the same quantity computed during standard backpropagation.

Note that $\frac{\partial \mathcal{L}}{\partial \Delta W} \in \mathbb{R}^{d \times k}$ is the "full gradient matrix" which would be required for standard fine-tuning. LoRA uses this to update only the low-rank factors $B$ and $A$, achieving the parameter count reduction.

The Adam update for $A$ and $B$ is then standard:

$$B_{t+1} = B_t - \alpha_{\text{lr}} \cdot \frac{\hat{m}_t^B}{\sqrt{\hat{v}_t^B} + \epsilon}$$
$$A_{t+1} = A_t - \alpha_{\text{lr}} \cdot \frac{\hat{m}_t^A}{\sqrt{\hat{v}_t^A} + \epsilon}$$

---

## 3. Parameter Count Comparison

### 3.1 Per-Layer Parameter Count

For a weight matrix $W \in \mathbb{R}^{d \times k}$:

| Method | Trainable parameters |
|--------|---------------------|
| Full fine-tuning | $d \times k$ |
| LoRA (rank $r$) | $d \times r + r \times k = r(d + k)$ |

The savings ratio is:

$$\text{savings} = \frac{d \times k}{r(d + k)}$$

For square matrices ($d = k$):

$$\text{savings} = \frac{d^2}{2rd} = \frac{d}{2r}$$

**Example for VieNeu-TTS attention projections ($d = k = 1024$, $r = 16$):**

$$\text{Full: } 1024 \times 1024 = 1{,}048{,}576 \text{ params}$$
$$\text{LoRA: } 16 \times (1024 + 1024) = 16 \times 2048 = 32{,}768 \text{ params}$$
$$\text{Savings: } \frac{1{,}048{,}576}{32{,}768} = 32\times$$

### 3.2 Total Model Parameter Count

VieNeu-TTS-0.3B has 24 Transformer layers, each containing:

| Layer | Shape | Full params | LoRA params (r=16) | Savings |
|-------|-------|-------------|-------------------|---------|
| q_proj | 1024→1024 | 1,048,576 | 32,768 | 32× |
| k_proj | 1024→1024 | 1,048,576 | 32,768 | 32× |
| v_proj | 1024→1024 | 1,048,576 | 32,768 | 32× |
| o_proj | 1024→1024 | 1,048,576 | 32,768 | 32× |
| gate_proj | 1024→4096 | 4,194,304 | 81,920 | 51× |
| up_proj | 1024→4096 | 4,194,304 | 81,920 | 51× |
| down_proj | 4096→1024 | 4,194,304 | 81,920 | 51× |
| **Per layer total** | | **16,778,240** | **376,832** | **44.5×** |

Multiply by 24 layers:
- Full fine-tuning: $16{,}778{,}240 \times 24 \approx 402.7 \text{ M params}$
- LoRA ($r=16$): $376{,}832 \times 24 \approx 9.0 \text{ M params}$

In practice, VieNeu-TTS fine-tuning with $r=16$ results in approximately **4–5 M trainable parameters** (roughly 1.3–1.7% of the total 300M), because not all layers are targeted and embedding/output layers are excluded.

### 3.3 Memory Savings

Only the trainable parameters (LoRA $A$ and $B$ matrices) require:
1. Gradient storage
2. Adam optimizer states ($m$ and $v$)

The frozen base model can be loaded in fp16, cutting its footprint in half:

| Component | Full fine-tune | LoRA fine-tune |
|-----------|---------------|----------------|
| Frozen model (fp16) | — | 0.6 GB |
| Trainable weights (fp32) | 1.2 GB | ~0.02 GB |
| Gradients | 1.2 GB | ~0.02 GB |
| Adam states | 2.4 GB | ~0.04 GB |
| Activations | 0.4 GB | 0.4 GB |
| **Total** | **~5.2 GB** | **~1.1 GB** |

LoRA requires approximately **5× less GPU memory** for VieNeu-TTS fine-tuning, bringing it well within reach of consumer GPUs.

---

## 4. Which Layers to Target

### 4.1 The Anatomy of a Transformer Block

Each Transformer block in VieNeu-TTS consists of two main components:

**Self-attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
$$Q = W_Q x, \quad K = W_K x, \quad V = W_V x$$
$$\text{output} = W_O \cdot \text{Attention}(Q, K, V)$$

**Feed-forward MLP (SwiGLU variant used in VieNeu-TTS):**
$$\text{FFN}(x) = \left(\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x\right) W_{\text{down}}$$

### 4.2 Why Both Attention and MLP Matter for TTS

The choice of which weight matrices to adapt is not arbitrary. Different components encode different aspects of the model's behavior:

**Attention weight matrices ($W_Q, W_K, W_V, W_O$):**
- Capture **speaker identity** and **prosodic style** through attention patterns over the voice prompt
- $W_Q$ and $W_K$ determine which parts of the context the model attends to — voice cloning relies heavily on learning to attend to the reference speaker's characteristics
- Empirical evidence: LoRA on only $W_Q, W_K, W_V$ produces a model that learns the speaker's overall voice quality but has poor tonal control for Vietnamese

**MLP weight matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$):**
- Store **phonetic and lexical knowledge** — how to pronounce specific words
- Vietnamese has context-dependent tone realization (especially Southern dialect); this is encoded in MLP weights
- Fine-tuning MLP layers helps the model produce correct tone realization for the new speaker's dialect

**Practical targeting for VieNeu-TTS:**

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj"        # MLP
]
```

This targets all 7 weight matrices per layer across all 24 layers.

### 4.3 Layer Depth and Specialization

Research on BERT and GPT-style models (Jawahar et al., 2019; Tenney et al., 2019) shows a consistent pattern of layer specialization:

- **Early layers (1–8):** Low-level phonetic and surface features
- **Middle layers (9–16):** Syntactic and prosodic structure
- **Late layers (17–24):** Semantic content and speaker-level identity

For voice adaptation (changing speaker identity while preserving language capability), **the later layers are most important**. However, for adapting to a new dialect (e.g., Southern Vietnamese tones), **middle layers also matter**.

For simplicity and robustness, VieNeu-TTS fine-tuning targets **all layers equally** — this was found empirically to give the most stable results for new speakers.

### 4.4 What NOT to Target

The following components are intentionally excluded from LoRA:

- **Embedding layer** (`embed_tokens`): Changing embeddings destabilizes the vocabulary distribution. The speech token embeddings (0–65535) learned during pretraining should be preserved.
- **Output head** (`lm_head`): The output projection maps hidden states back to logits over the vocabulary. Changing this alters the model's calibration across all tokens.
- **LayerNorm weights**: Small parameters, adapting them provides little benefit but risks destabilizing training dynamics.

---

## 5. Training Dynamics

### 5.1 The Loss Function

During LoRA fine-tuning of VieNeu-TTS, the training objective is **cross-entropy loss masked to speech token positions only**:

$$\mathcal{L} = -\frac{1}{|S|} \sum_{t \in S} \log P(s_t \mid s_{<t}, \text{context})$$

where:
- $S$ is the set of time steps corresponding to **speech token positions**
- $s_t$ is the target speech token at step $t$
- $\text{context}$ includes the full prompt: `[SPKR]` + voice codes + `[END_SPKR]` + `[TXT]` + text tokens + `[END_TXT]` + `[SPCH]`
- Text token positions have their labels set to $-100$ (ignored by PyTorch's `CrossEntropyLoss`)

The masking is critical: **we only supervise the model on what it generates (speech tokens), not on its reading of the input (text tokens)**. Including text token loss would incorrectly penalize the model for not "reproducing" the input text as an output.

### 5.2 Learning Rate

LoRA supports higher learning rates than full fine-tuning because:

1. The LoRA subspace is small — large steps in this space correspond to small perturbations of the effective weight $W_0 + \frac{\alpha}{r} BA$
2. The frozen base weights provide a strong prior — even aggressive LoRA updates cannot destroy the base model's capabilities

**VieNeu-TTS recommended:** `lr = 2e-4`

Comparison:
- Full fine-tuning typical LR: `1e-5` to `5e-5`
- LoRA typical LR: `1e-4` to `3e-4`

### 5.3 Warmup Schedule

Linear warmup is used for the first 5% of training steps to avoid instability from cold-started Adam optimizer states:

$$\alpha(t) = \alpha_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}}, \quad t < T_{\text{warmup}}$$

For 3000 training steps with 5% warmup: $T_{\text{warmup}} = 150$ steps.

After warmup, a **cosine decay** schedule is applied:

$$\alpha(t) = \alpha_{\text{max}} \cdot \frac{1}{2}\left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}}\right)\right)$$

### 5.4 Gradient Accumulation

Gradient accumulation allows simulating a larger effective batch size without increasing GPU memory:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}_k$$

where $K$ is the accumulation steps. The optimizer step is only taken after accumulating $K$ mini-batch gradients.

**Memory implication:** Gradient accumulation does NOT reduce peak activation memory (activations are still computed for each mini-batch). It only reduces the frequency of optimizer state updates.

**VieNeu-TTS effective batch calculation:**
```
per_device_batch_size    = 2
gradient_accumulation    = 8
effective_batch_size     = 2 × 8 = 16
```

An effective batch of 16 sequences provides stable gradient estimates for learning a new voice.

### 5.5 Gradient Clipping

To prevent gradient explosion (common in early LoRA training when $B$ has just learned non-zero values and the product $BA$ starts contributing large updates):

$$g_t \leftarrow g_t \cdot \min\left(1, \frac{c}{\|g_t\|_2}\right)$$

VieNeu-TTS uses `max_grad_norm = 1.0` (clip if gradient norm exceeds 1.0).

---

## 6. Rank Hyperparameter Analysis

### 6.1 What Rank Controls

The rank $r$ determines the **capacity of the LoRA adaptation**. Higher $r$ means:
- More trainable parameters → more expressive adaptation
- Larger memory overhead
- Higher risk of overfitting with small datasets

The number of trainable parameters scales linearly with $r$:

$$N_{\text{trainable}} = r \cdot \sum_{\text{targeted layers}} (d_i + k_i)$$

### 6.2 Rank Selection Guide for VieNeu-TTS

| Rank | Trainable params | Use case | Risk |
|------|-----------------|----------|------|
| $r=4$ | ~2M | Quick style probe, accent test | Underfitting for new voice |
| $r=8$ | ~4.5M | Accent adaptation with 30+ min data | Good balance for dialect shift |
| $r=16$ | ~9M | **Default: single new voice** | Low with proper early stopping |
| $r=32$ | ~18M | Voice with unusual prosody, large dataset | Overfitting with < 1 hour data |
| $r=64$ | ~36M | Multi-voice adapter, research | Significant overfit risk |

### 6.3 The Alpha Convention

The standard practice is:

$$\alpha = 2r \quad \text{(e.g., } r=16, \alpha=32\text{)}$$

This means the effective LoRA scale is $\alpha/r = 2$, which has been found empirically to work well across many tasks. The scale factor of 2 means LoRA updates contribute approximately twice the magnitude of a unit-scale initialization.

Why not just set $r$ higher and $\alpha/r$ smaller? The rank determines the **dimensionality** of the adaptation subspace, while $\alpha/r$ determines the **magnitude**. These are orthogonal hyperparameters — you cannot compensate for low rank with higher alpha.

### 6.4 Quality vs. Parameters Trade-off

Empirically, the marginal quality improvement diminishes rapidly beyond $r=16$ for typical single-voice TTS fine-tuning:

```
r=4:  MOS ~ 3.6  (measurably worse, voice sounds thin)
r=8:  MOS ~ 3.9  (acceptable, slight accent drift)
r=16: MOS ~ 4.1  (good quality, recommended default)
r=32: MOS ~ 4.1  (no improvement vs r=16 with 30-min dataset)
r=64: MOS ~ 3.8  (overfit — degrades on out-of-training text)
```

This is the classic **bias-variance tradeoff** applied to LoRA rank selection.

---

## 7. Overfitting in TTS

### 7.1 What Overfitting Looks Like in TTS

Overfitting in TTS is more subtle than in classification tasks because there is no discrete "accuracy" metric on training data. Instead, overfitting manifests as:

1. **Training loss continues decreasing** while synthesized speech on unseen text degrades
2. **Perceptual signs:**
   - Robotic, over-regularized rhythm on sentences not in training data
   - Tone collapse: the model defaults to the most common tones seen in training
   - Strange prosody at unknown words (often words with diacritics not well-represented)
   - Voice sounds "correct" on training sentences but "stiff" on novel sentences

### 7.2 The Memorization-Generalization Tension

In TTS, a fine-tuned model needs to:
- **Memorize** the target speaker's voice characteristics (timbre, prosody style)
- **Generalize** to new text combinations that were not in the training set

With small datasets (< 30 minutes), there are not enough examples of all Vietnamese phonemes and tones in all possible contexts. The model memorizes the specific phoneme-tone combinations it saw, and when it encounters novel combinations at inference time, it "hallucinates" incorrect prosody.

Mathematically, consider the training distribution $p_{\text{train}}$ over (text, audio) pairs versus the inference distribution $p_{\text{infer}}$. Overfitting occurs when the LoRA weights minimize $\mathbb{E}_{p_{\text{train}}}[\mathcal{L}]$ at the cost of increasing $\mathbb{E}_{p_{\text{infer}}}[\mathcal{L}]$.

### 7.3 Mitigation Strategies

**Early stopping:**
- Monitor synthesis quality on 10 held-out test sentences every 500 steps
- Stop training when UTMOS score on test sentences peaks (typically 2500–4000 steps for 30-min datasets)
- Do not train to convergence

**LoRA dropout:**
```python
lora_dropout = 0.05  # 5% dropout on LoRA activations
```
During training, each LoRA activation $BAx$ is randomly zeroed with probability 0.05. This prevents co-adaptation between specific rows of $B$ and columns of $A$.

**Dataset augmentation:**
- Speed perturbation: resample audio at 0.9× and 1.1× speed
- Pitch shifting: ±2 semitones (be careful — this changes Vietnamese tones!)
- Room impulse response convolution: simulate slight room acoustics

**Lower rank:** If overfitting is observed, reducing $r$ from 16 to 8 often resolves it.

### 7.4 The Validation Protocol

Every 500 steps, synthesize the same 10 held-out sentences with the current checkpoint. These should cover:
- All 6 Vietnamese tones explicitly
- Both short (< 5 words) and longer (15–20 words) sentences
- Technical vocabulary not likely in training data
- A code-switching sentence (Vietnamese + English)

Log the UTMOS score for each checkpoint and select the peak.

---

## 8. LoRA vs QLoRA vs Full Fine-tune

### 8.1 Full Fine-tuning

**When to use:** You have a large dataset (> 10 hours), access to multi-GPU setup, and want maximum quality with no capacity constraints.

**VieNeu-TTS context:** Generally impractical for single-voice adaptation. Reserved for training a new dialect-specific base model.

### 8.2 Standard LoRA

**Definition:** Freeze base model weights; train low-rank adapter matrices in fp32 while the frozen model is loaded in fp16.

**Memory profile:**
- Base model: loaded in fp16 → 0.6 GB for 0.3B
- LoRA params + grads + Adam: ~0.1 GB for r=16
- Total: ~0.7–1.5 GB (plus activations)

**Quality:** Essentially identical to full fine-tuning when $r$ is chosen correctly, because the adaptation subspace is sufficient for single-voice tasks.

**VieNeu-TTS verdict: Recommended for all standard fine-tuning scenarios.**

### 8.3 QLoRA

**Definition:** Quantize the base model to 4-bit NF4 (Normal Float 4) format, then train LoRA adapters on top of the quantized model.

The NF4 quantization maps weights to one of $2^4 = 16$ quantization levels that are **equally spaced in terms of quantiles** of a standard normal distribution (not linearly spaced). This is optimal for normally-distributed weights.

**Memory reduction vs standard LoRA:**
- NF4 uses 4 bits per parameter instead of 16 → 4× reduction in base model size
- Base model: $300 \times 10^6 \times 0.5 / 10^9 = 0.15 \text{ GB}$ (vs 0.6 GB for fp16)
- Total QLoRA memory: ~0.4 GB (vs 1.5 GB for LoRA)

**Quality degradation:**
Quantization introduces rounding errors at every forward pass through the frozen model. The error is approximately:
$$\|h - h_q\|_2 \approx O\left(\frac{\sigma_W \cdot \|x\|_2}{\sqrt{2^b - 1}}\right)$$

For 4-bit ($b=4$): error is proportional to $\sigma_W / \sqrt{15} \approx 0.258 \sigma_W$. This accumulates through 24 layers, resulting in a slight but perceptible quality drop on fine-grained prosodic tasks like Vietnamese tone rendering.

**VieNeu-TTS verdict:** Use QLoRA only if running on hardware with < 4 GB VRAM and quality slightly below standard LoRA is acceptable.

### 8.4 Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| Single GPU ≥ 8 GB VRAM, 30-min Vietnamese corpus | Standard LoRA, r=16 |
| Single GPU 4–8 GB VRAM | QLoRA, r=16 |
| Multi-GPU, large corpus (> 5 hours) | Full fine-tuning or LoRA r=64 |
| CPU-only fine-tuning | Not recommended (too slow) |
| Quick quality test (< 5 min training) | LoRA r=4, 500 steps |

---

## Summary

| Concept | Key Formula | VieNeu-TTS Default |
|---------|-------------|-------------------|
| LoRA weight update | $\Delta W = \frac{\alpha}{r} BA$ | — |
| Parameter savings (r=16, d=k=1024) | $1024^2 / (16 \times 2048) = 32\times$ | 4M / 300M = 1.3% |
| Initialization | $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ | Stable start |
| Training loss | $-\frac{1}{\|S\|}\sum_{t \in S} \log P(s_t \mid s_{<t})$ | Speech tokens only |
| Learning rate | 2e-4 | Higher than full fine-tune |
| Rank recommendation | r=16, alpha=32 | Single new voice |
| Early stopping | Peak UTMOS on 10 test sentences | ~2500–4000 steps |

### LoRA Hyperparameter Cheat Sheet for VieNeu-TTS

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                           # rank
    lora_alpha=32,                  # scaling (= 2*r)
    target_modules=[                # all attention + MLP
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,              # slight regularization
    bias="none",                    # do not adapt bias terms
    task_type="CAUSAL_LM"          # causal language modeling
)
```

```python
training_args = TrainingArguments(
    num_train_epochs=3,
    max_steps=3000,                 # typically overrides epochs
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,                      # mixed precision
    logging_steps=50,
    save_steps=500,
    max_grad_norm=1.0,
)
```

The next chapter covers how to prepare the Vietnamese audio dataset that feeds this fine-tuning pipeline — ensuring the data quality matches the model's requirements.
