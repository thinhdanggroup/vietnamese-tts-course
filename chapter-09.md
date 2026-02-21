# Chapter 09 — Training, Monitoring & Evaluation: From Loss Curves to MOS Scores

> **Audience**: ML engineers who understand gradient descent and cross-entropy loss, but are new to TTS-specific training loops and evaluation protocols.
> **Goal**: Understand the complete training loop for VieNeu-TTS LoRA fine-tuning, learn to interpret loss curves and quality metrics, and design a robust evaluation protocol for Vietnamese TTS.

---

## Table of Contents

1. [The Training Loop for TTS](#1-the-training-loop-for-tts)
2. [Loss Curves — Reading the Training Signal](#2-loss-curves--reading-the-training-signal)
3. [Objective Metrics](#3-objective-metrics)
4. [UTMOS — Neural MOS Predictor](#4-utmos--neural-mos-predictor)
5. [Subjective Evaluation — MOS Test Design](#5-subjective-evaluation--mos-test-design)
6. [Checkpoint Selection](#6-checkpoint-selection)
7. [Common Failure Modes in Vietnamese TTS](#7-common-failure-modes-in-vietnamese-tts)

---

## 1. The Training Loop for TTS

### 1.1 Data Flow and Tokenization

The VieNeu-TTS training loop operates on sequences of mixed text and speech tokens. Understanding the exact token structure is essential for understanding the loss function.

A single training sample has the following token layout:

```
[SPKR] v_1 v_2 ... v_M [END_SPKR] [TXT] t_1 t_2 ... t_P [END_TXT] [SPCH] s_1 s_2 ... s_N
                                                                               └──────────┘
                                                                            LOSS COMPUTED HERE
```

Where:
- `[SPKR]`, `[END_SPKR]`, `[TXT]`, `[END_TXT]`, `[SPCH]`: Special tokens
- $v_1, \ldots, v_M$: NeuCodec tokens of the reference voice clip ($M \approx 100$–$300$)
- $t_1, \ldots, t_P$: Text tokens ($P \approx 10$–$50$ for a Vietnamese sentence)
- $s_1, \ldots, s_N$: NeuCodec tokens for the target speech ($N \approx 50 \times \text{duration\_s}$)

The target sequence for the language model is shifted by one position (next-token prediction):

$$\text{input:  } [\text{SPKR}] \; v_1 \; \ldots \; v_M \; [\text{END\_SPKR}] \; \ldots \; [\text{SPCH}] \; s_1 \; s_2 \; \ldots \; s_{N-1}$$
$$\text{target: }  v_1 \; \ldots \; v_M \; [\text{END\_SPKR}] \; \ldots \; [\text{SPCH}] \; s_1 \; s_2 \; \ldots \; s_N$$

The labels for all positions except the speech tokens are set to $-100$:

```python
labels = sequence.clone()
labels[:speech_start_idx] = -100  # mask everything before [SPCH]
```

PyTorch's `CrossEntropyLoss` ignores positions with label $= -100$.

### 1.2 Forward Pass

The forward pass through the Transformer computes logits over the full vocabulary:

$$\ell_t = W_{\text{head}} \cdot h_t + b_{\text{head}}$$

where $h_t \in \mathbb{R}^{d_{\text{model}}}$ is the hidden state at step $t$ and $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$ maps to vocabulary size $V$.

For VieNeu-TTS:
- Text vocabulary: ~32,000 tokens (BPE subwords for Vietnamese and English)
- Speech token vocabulary: 65,536 (NeuCodec codebook entries)
- Total $V \approx 32,000 + 65,536 + \text{special tokens} \approx 97,544$

The probabilities are:

$$P(s_t = c \mid s_{<t}, \text{context}) = \text{softmax}(\ell_t)_c = \frac{e^{\ell_{t,c}}}{\sum_{j=1}^{V} e^{\ell_{t,j}}}$$

**LoRA modification to the forward pass:**

With LoRA applied to a linear layer $h = W_0 x$, the modified forward pass is:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

Since $W_0$ is frozen, only the $BAx$ term requires gradient computation. The computational overhead is:
- $Ax$: multiply $r \times k$ matrix by $k$-vector → $O(rk)$ FLOPs
- $BAx$: multiply $d \times r$ matrix by $r$-vector → $O(dr)$ FLOPs
- Total LoRA overhead per layer: $O(r(d + k))$ vs. full forward $O(dk)$ → negligible for $r \ll d$

### 1.3 Loss Computation

The cross-entropy loss for a single training sequence:

$$\mathcal{L}_{\text{seq}} = -\frac{1}{|S|} \sum_{t \in S} \log P(s_t \mid s_{<t}, \text{context})$$

$$= -\frac{1}{N} \sum_{n=1}^{N} \log \text{softmax}(\ell_{\text{speech\_start} + n})_{s_n}$$

where $S$ is the set of speech token positions and $N = |S|$ is the number of speech tokens.

The denominator $|S|$ normalizes by the number of speech tokens, not the total sequence length. This ensures that shorter clips (fewer speech tokens) have the same magnitude loss as longer clips, preventing the optimizer from preferring shorter sequences.

Over a batch of $B$ sequences:

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}_{\text{seq}}^{(b)}$$

### 1.4 Backward Pass Through LoRA

The backward pass computes gradients only for the LoRA parameters $A$ and $B$. The chain rule through the LoRA forward pass:

For a layer with input $x$, output $h = W_0 x + \frac{\alpha}{r} BAx$:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top \frac{\partial \mathcal{L}}{\partial h} x^\top$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^\top$$

The gradient with respect to the frozen weights $W_0$ is computed (necessary for backpropagating through the frozen layers to reach the trainable LoRA layers below), but the parameter update step is skipped for frozen parameters:

```python
for param in model.parameters():
    if param.requires_grad:
        optimizer.step_for(param)  # update LoRA params
    # Frozen params: gradient computed but not updated
```

### 1.5 Optimizer Step with Gradient Accumulation

With gradient accumulation factor $K$:

```python
for step, batch in enumerate(dataloader):
    loss = model(batch) / K          # scale loss
    loss.backward()                   # accumulate gradients

    if (step + 1) % K == 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

The loss scaling by $1/K$ is critical — without it, the accumulated gradient magnitude would be $K$ times larger than a single step, requiring adjustment of the learning rate.

**Mathematical equivalence:** Gradient accumulation over $K$ steps with batch size $B$ produces the same parameter update (in expectation) as a single step with batch size $KB$, because:

$$\sum_{k=1}^{K} \nabla_\theta \mathcal{L}(\theta; x_k) \approx K \cdot \mathbb{E}_{x \sim p_{\text{data}}}[\nabla_\theta \mathcal{L}(\theta; x)]$$

The stochastic nature means it is not exactly equivalent, but for $K \leq 16$, the approximation holds well.

---

## 2. Loss Curves — Reading the Training Signal

### 2.1 Expected Loss Range for VieNeu-TTS

The cross-entropy loss is measured in nats when using the natural logarithm (or bits if using $\log_2$). For VieNeu-TTS:

**Initial loss (untrained LoRA, step 0):**
The base model already understands the task from pretraining. With LoRA initialized to zero, the base model generates a probability distribution over the 97,544-token vocabulary. The initial cross-entropy should be around:

$$\mathcal{L}_0 \approx -\log\left(\frac{1}{V_{\text{speech}}}\right) = \log(65536) \approx 11.1 \text{ nats}$$

But because the base model has learned some priors about speech token sequences, the actual initial loss is much lower — approximately **2.2–2.8 nats**.

**Converged loss (after 3000 steps):** Approximately **1.7–1.9 nats**.

Why can't the loss reach 0? Because speech synthesis is stochastic — the model must predict one specific codec token from among many plausible choices. The entropy of the ground-truth codec token distribution is irreducibly positive.

### 2.2 Diagnostic Loss Curves

**Healthy training (decreasing, smooth):**
$$\mathcal{L}(t) \approx \mathcal{L}_0 \cdot e^{-t/\tau} + \mathcal{L}_{\infty}$$

where $\tau \approx 2000$ steps is the characteristic decay time and $\mathcal{L}_\infty \approx 1.8$ is the asymptotic loss. Small noise fluctuations ($\pm 0.05$) are normal.

**Overfitting:** Training loss continues decreasing below 1.7, but:
- Synthesis quality on new text degrades (evaluate every 500 steps)
- The model memorizes specific codec sequences for training sentences
- Loss decrease rate slows but never truly plateaus (model keeps memorizing)

**Underfitting (learning rate too low, or data too noisy):**
- Loss decreases very slowly or not at all
- After 1000 steps, loss still > 2.2
- **Actions:** Check learning rate (should be 2e-4), check data quality (run through filter pipeline again), verify LoRA config is applied correctly

**Gradient explosion (unstable training):**
- Loss has large spikes (increase by > 0.3 nats in one step)
- Sometimes recovers, sometimes diverges to NaN
- **Cause:** Too large learning rate, incorrectly scaled loss, corrupted training sample
- **Fix:** Reduce LR to 5e-5, ensure `max_grad_norm=1.0` is applied, remove corrupted samples

**Loss spike at specific steps:** Often caused by a single corrupted training sample (e.g., a clip with a sudden loud noise). Track which sample was in the batch at the spike step and inspect it.

### 2.3 Interpreting Loss Numerically

The cross-entropy loss relates to **perplexity**:

$$\text{PPL} = e^{\mathcal{L}}$$

For VieNeu-TTS training:
- $\mathcal{L} = 2.5$: $\text{PPL} = 12.2$ (each token has ~12 plausible choices)
- $\mathcal{L} = 2.0$: $\text{PPL} = 7.4$
- $\mathcal{L} = 1.8$: $\text{PPL} = 6.0$ (well-converged)
- $\mathcal{L} = 1.5$: $\text{PPL} = 4.5$ (may indicate overfitting)

A perplexity of 6 means the model has narrowed down the next speech token to approximately 6 equally likely choices — a reasonable amount of uncertainty given that multiple valid pronunciations exist for any given phoneme.

### 2.4 Training vs. Validation Loss Split

Because VieNeu-TTS fine-tuning typically uses small datasets, holding out a validation set is important. Recommended split:
- Training: 90% of samples
- Validation: 10% (at least 10 different sentences not in training)

Compute validation loss every 500 steps:

```python
model.eval()
with torch.no_grad():
    val_loss = sum(model(batch).loss.item() for batch in val_loader) / len(val_loader)
model.train()
```

The key signal is the **gap between train and validation loss**. A gap > 0.3 nats indicates overfitting.

---

## 3. Objective Metrics

### 3.1 CER (Character Error Rate)

CER measures **intelligibility** — does the synthesized speech contain the right characters?

**Measurement protocol:**
1. Synthesize a test sentence with the fine-tuned model
2. Transcribe the synthesized audio with an ASR model (e.g., Whisper-large-v3 for Vietnamese)
3. Compare transcription to the reference text using the Levenshtein edit distance

**Levenshtein distance:**
Given reference $R = r_1 r_2 \ldots r_n$ and hypothesis $H = h_1 h_2 \ldots h_m$:

$$d(R, H) = \min_{\text{edit script}} |\text{substitutions}| + |\text{deletions}| + |\text{insertions}|$$

Computed via dynamic programming in $O(nm)$ time:

$$\text{dp}[i][j] = \begin{cases}
i & \text{if } j = 0 \\
j & \text{if } i = 0 \\
\text{dp}[i-1][j-1] & \text{if } r_i = h_j \\
1 + \min(\text{dp}[i-1][j], \text{dp}[i][j-1], \text{dp}[i-1][j-1]) & \text{otherwise}
\end{cases}$$

**CER formula:**

$$\text{CER} = \frac{S + D + I}{N} \times 100\%$$

where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = number of reference characters.

**CER benchmarks for VieNeu-TTS:**

| System | CER (%) |
|--------|---------|
| Human speech (reference) | 2–5% (ASR errors) |
| VieNeu-TTS base (pretrained) | 5–8% |
| Fine-tuned (good quality) | 6–9% |
| Fine-tuned (overfitting) | 15–30% on unseen text |

**Important caveat:** CER measures whether the correct sounds are present, not whether the tones are correct. A word pronounced with wrong tone may still have correct consonants and vowels, resulting in low CER but high perceptual error. For Vietnamese, supplement CER with a tone-specific evaluation.

### 3.2 WER (Word Error Rate)

WER operates at the word level:

$$\text{WER} = \frac{S_w + D_w + I_w}{N_w} \times 100\%$$

where subscript $w$ indicates word-level counts. For Vietnamese:

**Complication:** Vietnamese word segmentation is non-trivial. Vietnamese "words" are often compound units (e.g., "học sinh" is one lexical unit but two syllables). Using syllable-level WER is more consistent for Vietnamese:

$$\text{SWER} = \frac{S_s + D_s + I_s}{N_s} \times 100\%$$

### 3.3 MCD (Mel Cepstral Distortion)

MCD measures **spectral similarity** between reference and synthesized speech. It is computed from the Mel-Frequency Cepstral Coefficients (MFCCs):

$$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \cdot \frac{1}{T} \sum_{t=1}^{T} \sqrt{\sum_{d=1}^{D} (mc_d^{(t)} - \hat{mc}_d^{(t)})^2} \text{ dB}$$

where:
- $mc_d^{(t)}$ is the $d$-th MFCC coefficient of the reference at frame $t$
- $\hat{mc}_d^{(t)}$ is the $d$-th MFCC coefficient of the synthesis at frame $t$
- $D = 13$ is the number of cepstral coefficients (excluding $c_0$)
- $T$ is the number of frames after Dynamic Time Warping (DTW) alignment

**Note on DTW alignment:** Reference and synthesis will not have the same duration. Before computing MCD, align frames using DTW to find the optimal monotonic alignment path.

**MCD interpretation:**

| MCD (dB) | Quality assessment |
|-----------|--------------------|
| < 4 dB | Excellent (near-identical spectra) |
| 4–6 dB | Good (slight timbre difference) |
| 6–9 dB | Fair (noticeable spectral difference) |
| > 9 dB | Poor (significant quality loss) |

**MCD limitation:** MCD measures spectral similarity but not prosodic accuracy (tone, rhythm). A synthesis with correct spectrum but wrong tones will have low MCD if the spectrum is similar. For Vietnamese, MCD must be supplemented with tone evaluation.

### 3.4 F0 Error (Tone Evaluation)

For Vietnamese, the fundamental frequency ($F_0$, pitch) trajectory is the direct carrier of tonal information. Compare $F_0$ of reference vs. synthesis:

$$\text{RMSE}_{F_0} = \sqrt{\frac{1}{T}\sum_{t=1}^T (F_0^{(t)} - \hat{F}_0^{(t)})^2} \text{ Hz}$$

Or in semitones (which is perceptually more meaningful):

$$\text{RMSE}_{F_0}^{\text{semi}} = \sqrt{\frac{1}{T}\sum_{t=1}^T \left(12 \log_2\frac{F_0^{(t)}}{\hat{F}_0^{(t)}}\right)^2} \text{ semitones}$$

**Target:** RMSE < 20 Hz or < 2 semitones for acceptable Vietnamese tone accuracy.

---

## 4. UTMOS — Neural MOS Predictor

### 4.1 Architecture

UTMOS (UTokyo-SaruLab MOS prediction system) is a neural network trained to predict human MOS scores from audio waveforms. It consists of:

1. **Feature extraction:** A pre-trained self-supervised learning (SSL) model (specifically, wav2vec 2.0 or HuBERT) that extracts rich acoustic representations from raw waveforms. These representations have been trained on thousands of hours of speech and capture both acoustic quality and linguistic content.

2. **Prediction head:** A lightweight MLP that takes the averaged SSL features and predicts a scalar MOS score in range [1.0, 5.0]:

$$\text{UTMOS} = \text{MLP}(\text{mean\_pool}(\text{SSL}(x_{\text{wav}})))$$

The SSL backbone is frozen during UTMOS training; only the prediction head is trained on human MOS labels.

### 4.2 Training Data and Correlation

UTMOS was trained on the VoiceMOS Challenge 2022 dataset, which contains:
- ~7000 speech samples from various TTS systems
- Human MOS ratings from trained evaluators (5-point ACR scale)

Reported performance: **Pearson correlation of 0.95** with human MOS on the challenge test set. This makes UTMOS highly reliable as a proxy for human evaluation in automated pipelines.

### 4.3 Usage in VieNeu-TTS Evaluation

```python
import utmos
predictor = utmos.Score(sampling_rate=16000)

import librosa
wav, _ = librosa.load("synthesized.wav", sr=16000)
score = predictor.score(wav, sampling_rate=16000)
print(f"UTMOS: {score:.3f}")
```

**Expected scores for VieNeu-TTS:**

| Scenario | UTMOS score |
|----------|-------------|
| Human speech (clean recording) | 4.2–4.5 |
| VieNeu-TTS base (no fine-tuning) | 3.8–4.2 |
| Good fine-tuned checkpoint | 4.0–4.3 |
| Overfit checkpoint (on test text) | 3.0–3.5 |
| Low-quality fine-tuning data | 3.2–3.8 |

**Limitation:** UTMOS was validated primarily on English TTS. For Vietnamese TTS, the absolute score may be slightly lower (~0.1–0.2 UTMOS units) due to domain shift, but the **relative ranking** between checkpoints remains reliable for checkpoint selection.

### 4.4 Automated Evaluation Pipeline

For systematic checkpoint evaluation:

```python
def evaluate_checkpoint(checkpoint_path, test_sentences, reference_voice):
    """Evaluate a LoRA checkpoint with UTMOS on test sentences."""
    # Load model with LoRA adapter
    model = load_model_with_lora(checkpoint_path)
    tts = Vieneu(model=model)

    scores = []
    for text in test_sentences:
        audio = tts.infer(text, voice=reference_voice)
        wav = audio.astype(np.float32) / 32768.0  # normalize to [-1, 1]
        score = predictor.score(wav, sampling_rate=24000)
        scores.append(score)

    return np.mean(scores), np.std(scores)
```

---

## 5. Subjective Evaluation — MOS Test Design

### 5.1 Evaluation Methodologies

**Absolute Category Rating (ACR) — the standard MOS test:**
- Each listener hears each sample once, in isolation
- Rates on 5-point scale (1=Bad, 2=Poor, 3=Fair, 4=Good, 5=Excellent)
- Advantage: Simple, widely used, comparable to published benchmarks
- Disadvantage: Susceptible to individual calibration differences

**AB test (preference test):**
- Listener hears two samples (A and B) for the same text
- Chooses which they prefer (or "no preference")
- Advantage: More sensitive for small quality differences between two systems
- Disadvantage: Only compares two systems; does not give absolute quality

**MUSHRA (Multiple Stimuli with Hidden Reference and Anchor):**
- Listener hears several systems + a hidden reference + a known degraded anchor
- Rates each on 0–100 scale
- Required: identify the hidden reference (anchors calibration)
- Advantage: Highly sensitive; accounts for calibration differences
- Disadvantage: More complex to set up and run

**For VieNeu-TTS fine-tuning evaluation, recommended protocol:**
1. Use ACR MOS for absolute quality measurement
2. Use AB preference to compare your fine-tuned model vs. a reference system (base model or ground truth)

### 5.2 Test Design for Vietnamese TTS

**Sentence selection criteria:**

The 30+ test sentences should cover:

| Category | Count | Example |
|----------|-------|---------|
| Ngang tone dominant | 3 | "Ba ông ba ba bao ba ba ông ba." |
| Huyền tone dominant | 3 | "Bà già già bà bà già bà." |
| Sắc tone dominant | 3 | "Sáng sáng, Sơn sắc sắc sóng sắc." |
| Hỏi/ngã tones | 3 | "Cô ấy hỏi mãi mà không trả lời." |
| Nặng tone dominant | 3 | "Ông nội nhớ nhà, nặng lòng." |
| Mixed tones | 5 | "Hôm nay tôi học tiếng Việt rất thú vị." |
| Long sentence (> 20 words) | 3 | Technical or news-style sentence |
| Code-switching | 3 | "Chúng ta cần meeting online về project này." |
| Proper nouns | 3 | "Hà Nội, TP. Hồ Chí Minh, Đà Nẵng." (pre-normalized) |
| Numbers (pre-normalized) | 3 | "Một trăm năm mươi hai ngàn đồng." |

**Total: ~30 sentences**

### 5.3 Evaluator Selection for Vietnamese

For Vietnamese TTS evaluation:
- **Native speakers only:** Non-native speakers cannot reliably judge Vietnamese tones
- **Regional consistency:** For a Northern-dialect model, use Northern-dialect listeners; they are more sensitive to tone errors in their own dialect
- **Trained evaluators preferred:** Even brief training (5 minutes explaining the scale with examples) significantly improves inter-rater reliability
- **Minimum 10 evaluators** for 95% confidence interval width < ±0.5 MOS

### 5.4 Statistical Analysis

**Inter-rater reliability:** Cronbach's alpha measures consistency among evaluators:

$$\alpha = \frac{k}{k-1}\left(1 - \frac{\sum_{i=1}^k \sigma_{y_i}^2}{\sigma_y^2}\right)$$

Target: $\alpha > 0.7$ (acceptable), $\alpha > 0.8$ (good). If $\alpha < 0.7$, evaluators disagree significantly and results are not reliable.

**Confidence interval for MOS:**

$$\text{MOS} \pm z_{0.025} \cdot \frac{\sigma}{\sqrt{N_E \cdot N_S}}$$

For 95% CI with 20 evaluators and 30 sentences: $z_{0.025} = 1.96$, so CI width $\approx 2 \times 1.96 \times 0.8 / \sqrt{600} \approx 0.064$ — a very tight interval that reliably detects differences of ≥ 0.1 MOS.

---

## 6. Checkpoint Selection

### 6.1 Why the Final Checkpoint is Often Not the Best

Training loss continues to decrease as long as there are optimization steps, even past the point of useful learning. In TTS:
- The model memorizes training-set prosody patterns
- Training loss: decreases monotonically
- Test-set UTMOS: peaks around 2500–4000 steps, then degrades

The optimal checkpoint is typically at **60–80% of the maximum planned steps** for a 30-minute Vietnamese dataset.

### 6.2 The Checkpoint Selection Protocol

```
Every 500 steps:
    1. Save LoRA adapter checkpoint
    2. Synthesize 10 fixed test sentences (held out from training)
    3. Compute UTMOS for each synthesis
    4. Log mean UTMOS + standard deviation

After training:
    5. Plot UTMOS vs. step (should peak and then decrease)
    6. Select checkpoint at UTMOS peak
    7. Verify subjectively: listen to all 10 test syntheses from that checkpoint
    8. If subjective check reveals issues: try ±500 steps
```

### 6.3 The 10 Fixed Test Sentences

These 10 sentences should be selected before training begins and never included in training:

```python
TEST_SENTENCES_VI = [
    # Basic tones
    "Ba con bò đang gặm cỏ xanh bên bờ sông.",      # mostly ngang
    "Bầu trời mùa thu thường có màu xanh nhạt.",      # huyền + ngang
    "Ánh nắng buổi sáng chiếu vào căn phòng nhỏ.",    # sắc
    "Hỏi han nhau mãi mà chẳng biết phải làm gì.",    # hỏi/ngã
    "Đất đai màu mỡ bồi đắp bởi dòng sông lớn.",     # nặng + sắc

    # Complex sentences
    "Trí tuệ nhân tạo đang thay đổi mọi lĩnh vực.",  # mixed
    "Hệ thống tổng hợp tiếng nói hoạt động ổn định.", # mixed, technical

    # Long sentences
    "Trong những năm gần đây, công nghệ trí tuệ nhân tạo phát triển mạnh mẽ và ảnh hưởng sâu sắc đến cuộc sống hàng ngày của người dân Việt Nam.",

    # Code-switching
    "Chúng ta cần deploy hệ thống lên server trước deadline.",

    # Proper nouns
    "Hà Nội và thành phố Hồ Chí Minh là hai trung tâm kinh tế lớn nhất Việt Nam.",
]
```

### 6.4 Early Stopping Criteria

Implement automatic early stopping based on validation UTMOS:

```python
best_utmos = 0.0
patience = 3  # stop if no improvement for 3 consecutive evaluations (1500 steps)
no_improvement_count = 0

for step in range(max_steps):
    train_step()

    if step % 500 == 0:
        utmos = evaluate_utmos(test_sentences)
        if utmos > best_utmos:
            best_utmos = utmos
            save_checkpoint(f"checkpoint_best")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at step {step}")
            break
```

---

## 7. Common Failure Modes in Vietnamese TTS

### 7.1 Tone Collapse

**Symptom:** The model generates speech with correct consonants and vowels but monotone or incorrect tones on words not seen in training.

**Root cause:** The training data has insufficient tone diversity. The model learns a "default tone" (usually ngang or huyền, the most common) and applies it when uncertain.

**Diagnostic:** Synthesize a set of minimal tone pairs:
- "ma" (ghost), "mà" (but), "má" (mother), "mả" (grave), "mã" (code), "mạ" (rice seedling)

If tone collapse is occurring, these will sound similar or all the same tone.

**Fix:**
1. Audit tone distribution in training data (see Chapter 8, Section 7.2)
2. Add targeted recordings with underrepresented tones (hỏi and ngã are most commonly underrepresented)
3. Reduce learning rate to 1e-4 (less aggressive adaptation, preserves pretrained tone knowledge)
4. Use a lower rank (r=8) to limit the model's ability to overwrite pretrained tone patterns

### 7.2 Rhythm Instability

**Symptom:** Pauses appear at incorrect positions — often within a compound word or after a particle. The rhythm sounds choppy or unnatural.

**Root cause:** Text normalization issues in training data. If some training samples have unusual punctuation (em-dashes, ellipses, non-standard spacing), the model learns incorrect pause patterns.

**Diagnostic:** Synthesize a long sentence with clear compound words:
> "Trường học, bệnh viện, và nhà máy đều đang hoạt động tốt."

Listen for inappropriate pauses within "trường học" or "bệnh viện".

**Fix:**
1. Run all training texts through a Vietnamese text normalizer before encoding
2. Remove samples with unusual punctuation patterns
3. Verify that the training data text exactly matches the spoken content (no added/missing words)

### 7.3 Code-switching Errors

**Symptom:** English words within Vietnamese sentences receive Vietnamese pronunciation (e.g., "AI" pronounced /a-i/ in Vietnamese style instead of /eɪ-aɪ/).

**Root cause:** The base model's code-switching capability is overwritten during fine-tuning if the training data is exclusively Vietnamese.

**Diagnostic:** Synthesize:
> "Chúng ta sử dụng machine learning để phân tích data."

Listen to "machine learning" and "data" — they should sound English-accented, not Vietnamese-accented.

**Fix:**
1. Include 10–15% code-switching samples in fine-tuning data
2. Use phonemizer fallback for English words (see Chapter 6)
3. Reduce LoRA rank to preserve more pretrained code-switching knowledge

### 7.4 Speaker Drift on Long Sequences

**Symptom:** Voice quality or timbre changes partway through a long synthesis (> 200 characters). The first part sounds like the target speaker, but later parts sound more generic.

**Root cause:** The attention mechanism's ability to maintain speaker conditioning weakens over very long sequences. The model loses the "signal" from the voice prompt tokens as the generated sequence grows.

**Root cause (mathematical):** In causal attention, the attention weight on the voice prompt tokens decays as:

$$\alpha_{t \to \text{prompt}} \propto e^{\text{score}(h_t, h_{\text{prompt}}) / \sqrt{d_k}}$$

For large $t$ (far from the prompt tokens), the positional encoding difference grows, and attention scores to distant past positions can weaken for some architectures.

**Fix:**
1. Set `max_chars=256` for a single inference call (split longer text)
2. Use a longer reference clip (15 s instead of 5 s) — more voice prompt tokens provide a stronger signal
3. Chunk long text at natural sentence boundaries, synthesize each chunk separately, concatenate audio

### 7.5 Hallucinated Silence or Repetition

**Symptom:** The model generates excessive silence, or repeats a phoneme/word multiple times.

**Root cause:** The autoregressive generation can enter loops — a sequence prefix that has high probability of generating silence tokens (especially end-of-speech tokens) prematurely.

**Fix:**
1. Check that end-of-speech special token handling is correct — the generation loop must stop when `[END_SPCH]` is generated
2. Reduce temperature or adjust `repetition_penalty` parameter
3. Verify that the input text ends with proper punctuation (missing terminal punctuation leaves the model uncertain about where speech should end)

### 7.6 Debugging Strategy — Ablation by Dataset Subset

When facing unexplained quality issues, use a systematic ablation:

1. Train on 5 minutes of data (50–100 clips) — verify the pipeline is working
2. Train on 15 minutes — is the voice quality improving?
3. Train on 30 minutes — does quality continue to improve or plateau?
4. If 30 minutes gives worse quality than 15 minutes → overfitting, reduce steps or rank

This binary search approach helps localize whether the issue is the pipeline, the data amount, or the hyperparameters.

---

## Summary

| Metric | Formula | Target for fine-tuned VieNeu-TTS |
|--------|---------|----------------------------------|
| Training loss | $-\frac{1}{N}\sum \log P(s_t)$ | 1.8–2.0 nats |
| CER | $(S+D+I)/N \times 100\%$ | < 10% |
| MCD | $\frac{10\sqrt{2}}{\ln 10}\sqrt{\sum (mc_d - mc'_d)^2}$ | < 7 dB |
| UTMOS | Neural MOS predictor | > 3.8 |
| Human MOS | 5-point ACR scale | > 3.5 |

### Evaluation Pipeline Code Template

```python
# Complete evaluation pipeline for one checkpoint
def evaluate_checkpoint_full(checkpoint_step, test_sentences,
                              reference_voice, predictor):
    results = []
    for text in test_sentences:
        audio = tts.infer(text, voice=reference_voice)

        # UTMOS
        wav_16k = librosa.resample(audio.astype(np.float32),
                                    orig_sr=24000, target_sr=16000)
        utmos = predictor.score(wav_16k / np.max(np.abs(wav_16k) + 1e-8),
                                sampling_rate=16000)

        # ASR for CER
        asr_result = whisper_model.transcribe(audio)
        cer_score = cer(text, asr_result["text"])

        results.append({"text": text, "utmos": utmos, "cer": cer_score})

    df = pd.DataFrame(results)
    print(f"Step {checkpoint_step}: "
          f"UTMOS={df.utmos.mean():.3f}±{df.utmos.std():.3f}, "
          f"CER={df.cer.mean():.1f}%")
    return df
```

The final chapter covers how to deploy the best checkpoint — whether as a GGUF quantized model, a streaming API, or a packaged `voices.json` for easy distribution.
