# Chapter 10 — Deployment & Optimization: Running VieNeu-TTS in Production

> **Audience**: ML engineers who have trained a model and now need to serve it efficiently in real-world applications.
> **Goal**: Understand the full deployment stack — from quantization math to streaming latency to production serving patterns — for VieNeu-TTS.

---

## Table of Contents

1. [Inference Modes Overview](#1-inference-modes-overview)
2. [GGUF Quantization Deep Dive](#2-gguf-quantization-deep-dive)
3. [Memory Footprint Comparison](#3-memory-footprint-comparison)
4. [Latency Benchmarks and RTF](#4-latency-benchmarks-and-rtf)
5. [LoRA Merge for Production](#5-lora-merge-for-production)
6. [Serving Patterns](#6-serving-patterns)
7. [Streaming for Low Latency](#7-streaming-for-low-latency)
8. [voices.json — Portable Voice Packaging](#8-voicesjson--portable-voice-packaging)

---

## 1. Inference Modes Overview

VieNeu-TTS supports four inference modes, each with different quality-memory-speed trade-offs:

| Mode | Backend | Memory | Quality | Use case |
|------|---------|--------|---------|----------|
| Full PyTorch (fp32) | CUDA/CPU | Highest | Reference quality | Research, GT comparison |
| Full PyTorch (fp16) | CUDA | ~50% of fp32 | Negligible difference | GPU production |
| GGUF Q8 | llama.cpp (CPU/Metal) | ~50% of fp16 | Near-identical | CPU server with good RAM |
| GGUF Q4 | llama.cpp (CPU/Metal) | ~25% of fp16 | Slight perceptual difference | Edge devices, real-time CPU |

The **backbone model** (the language model generating speech tokens) and the **codec** (decoding tokens to audio) are separate components with separate optimization paths:

```
[Text] → [Backbone] → [Speech tokens] → [Codec] → [Audio waveform]
           ↑                                ↑
    GGUF Q4 on CPU              DistillNeuCodec
    (bottleneck)                  ONNX on CPU
```

The backbone is the primary latency bottleneck (it generates tokens autoregressively); the codec is fast and can be CPU-optimized independently.

### 1.1 Mode Selection Guide

**Full PyTorch:** Use when you have a GPU with sufficient VRAM and want maximum quality for research comparisons or high-value production where quality is paramount.

**GGUF Q8:** Best choice for CPU servers with ≥ 8 GB RAM where quality must be close to PyTorch baseline. The 8-bit quantization has virtually no perceptible quality difference for TTS.

**GGUF Q4:** The default mode for VieNeu-TTS (`Vieneu()` uses Q4 by default). Real-time TTS on any modern CPU (2020+). Slight quality reduction versus Q8, imperceptible on most sentences, occasional tone softening on complex tonal sequences.

---

## 2. GGUF Quantization Deep Dive

### 2.1 What GGUF Is

GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp for storing quantized language models. It is a binary format that stores:
- Model architecture metadata (header)
- Tensor data (quantized)
- Tokenizer vocabulary

The key advantage: GGUF allows **mixed quantization** — different tensors within the same model use different bit widths based on their sensitivity.

### 2.2 Linear Quantization Mathematics

The simplest quantization scheme maps floating-point weights to $b$-bit integers:

Given a block of floating-point weights $\mathbf{w} \in \mathbb{R}^{N}$:

**Step 1: Compute scale and zero-point**
$$w_{\min} = \min(\mathbf{w}), \quad w_{\max} = \max(\mathbf{w})$$
$$s = \frac{w_{\max} - w_{\min}}{2^b - 1}, \quad z = -\text{round}\left(\frac{w_{\min}}{s}\right)$$

**Step 2: Quantize**
$$q_i = \text{clip}\left(\text{round}\left(\frac{w_i}{s}\right) + z, \; 0, \; 2^b - 1\right)$$

**Step 3: Dequantize (at inference time)**
$$\hat{w}_i = s \cdot (q_i - z)$$

**Quantization error:**
$$\epsilon_i = w_i - \hat{w}_i \approx w_i - s \cdot \text{round}(w_i / s)$$

The maximum quantization error for any weight is bounded by $s/2$ (half a quantization step):

$$|\epsilon_i| \leq \frac{s}{2} = \frac{w_{\max} - w_{\min}}{2(2^b - 1)}$$

For $b=4$ with $w_{\max} - w_{\min} = 0.3$ (typical for post-attention projection weights):
$$|\epsilon_{\max}| = \frac{0.3}{2 \times 15} = 0.01$$

For $b=8$: $|\epsilon_{\max}| = 0.3 / 510 \approx 0.0006$ — much smaller.

### 2.3 Block Quantization

Naive global quantization (one scale for the entire weight matrix) performs poorly when weights have outliers — a few very large values force the scale to be large, degrading precision for the majority of normal-range values.

GGUF solves this with **block quantization**: the weight matrix is divided into small blocks (e.g., 32 values), and each block has its own scale factor:

$$s_j = \frac{\max_{i \in \text{block}_j} |w_i|}{2^{b-1} - 1}$$

This allows the scale to adapt to local weight statistics, significantly reducing quantization error in practice.

### 2.4 Q4_K_M Format — Mixed Precision

The `Q4_K_M` format used for VieNeu-TTS GGUF files is more sophisticated than simple 4-bit quantization:

- **"K"**: K-quant method — uses a two-level quantization scheme where the block scales themselves are quantized (using 6 bits for the scale of each 32-weight block)
- **"M"**: Mixed precision — some weight matrices (specifically attention Q and K projections) are quantized at 5 or 6 bits instead of 4 bits, because these are more sensitive to quantization error

The "M" (medium) variant is a pragmatic compromise:
- Most weights (MLP): Q4_K (4-bit with K-quant)
- Attention Q and K: Q5_K (5-bit with K-quant)
- Embeddings: Q8_0 (8-bit, nearly lossless)

**Why attention Q and K are more sensitive:** The dot-product $\frac{QK^\top}{\sqrt{d_k}}$ is very sensitive to precision because small errors in Q or K values produce multiplicative errors in attention weights through the softmax. For Vietnamese tone generation, correct attention pattern is critical — the model must attend precisely to tone-bearing vowels in the voice prompt.

### 2.5 Quantization Error and Its Effect on TTS

For the output $h = Wx$ of a quantized linear layer:

$$\hat{h} = \hat{W}x = (W + \Delta W_q) x = Wx + \Delta W_q x$$

The error in the output is:
$$\|h - \hat{h}\|_2 = \|\Delta W_q x\|_2 \leq \|\Delta W_q\|_F \|x\|_2$$

For Q4_K_M with typical VieNeu-TTS weight matrices:
$$\|\Delta W_q\|_F / \|W\|_F \approx 0.3\%$$

This 0.3% relative error per layer compounds through 24 layers. Assuming independent errors:
$$\text{Total relative error} \approx \sqrt{24} \times 0.3\% \approx 1.5\%$$

A 1.5% relative error in the final hidden states is generally below perceptual thresholds for TTS quality, especially after softmax normalization normalizes the logit magnitudes.

### 2.6 Apple Silicon Metal Backend

On Apple Silicon (M1/M2/M3), llama.cpp uses the Metal GPU compute API:

- Weights remain in RAM (unified memory architecture)
- Metal shaders perform the quantized matrix multiplications on the GPU cores
- No PCIe transfer bottleneck (CPU and GPU share the same memory)

This is why VieNeu-TTS Q4 achieves real-time RTF on M2 MacBooks despite using "CPU" inference — the Metal backend provides GPU acceleration within the unified memory architecture.

---

## 3. Memory Footprint Comparison

### 3.1 Backbone Model Size

For a model with $N$ parameters:

| Format | Bytes per parameter | 0.3B model | 0.5B model |
|--------|--------------------|---------:|----------:|
| fp32 | 4 | 1.2 GB | 2.0 GB |
| fp16 | 2 | 0.6 GB | 1.0 GB |
| Q8_0 | ~1.0 | ~0.3 GB | ~0.5 GB |
| Q4_K_M | ~0.5 | ~0.15 GB | ~0.25 GB |

Note: The effective size per parameter for Q4_K_M is slightly above 0.5 bytes/param due to the overhead of storing block scales.

### 3.2 Codec Memory

DistillNeuCodec is a compact model:
- Model weights: ~45 MB
- Runtime buffers: ~5 MB
- **Total: ~50 MB** regardless of backbone size

The codec runs efficiently on CPU using ONNX Runtime.

### 3.3 Runtime vs. Model File Size

At runtime, additional memory is needed for:
- **KV cache:** $2 \times N_L \times N_H \times L \times d_H \times \text{bytes}$

For VieNeu-TTS-0.3B (24 layers, 16 heads, $d_H=64$) with sequence length $L=750$:
$$\text{KV cache} = 2 \times 24 \times 16 \times 750 \times 64 \times 2 = 55 \text{ MB (fp16)}$$

- **Activation buffers:** ~100–200 MB during forward pass
- **Output logits buffer:** $V \times 4 \text{ bytes} = 97544 \times 4 \approx 0.37 \text{ MB}$ (negligible)

**Practical RAM requirements:**

| Configuration | Model file | Runtime overhead | Total RAM needed |
|--------------|-----------|-----------------|-----------------|
| 0.3B Q4_K_M + DistillNeuCodec | 0.15 + 0.05 GB | ~0.5 GB | **~1.0 GB** |
| 0.3B Q8 + DistillNeuCodec | 0.30 + 0.05 GB | ~0.5 GB | **~1.5 GB** |
| 0.5B Q4_K_M + DistillNeuCodec | 0.25 + 0.05 GB | ~0.7 GB | **~1.5 GB** |
| 0.3B fp16 (GPU) | 0.60 GB VRAM | ~0.3 GB | **~1.0 GB VRAM** |

The Q4_K_M configuration fits easily in 4 GB RAM — enabling deployment on low-end devices.

---

## 4. Latency Benchmarks and RTF

### 4.1 Real-Time Factor Definition

Real-Time Factor (RTF) measures inference speed relative to the duration of generated audio:

$$\text{RTF} = \frac{t_{\text{inference}}}{d_{\text{audio}}}$$

- RTF < 1.0: Faster than real-time (system can stream audio as it generates)
- RTF = 1.0: Exactly real-time
- RTF > 1.0: Slower than real-time (not suitable for streaming)

**For streaming TTS, RTF must be sustainably < 1.0** — not just on average but also for individual chunks.

### 4.2 Latency Components

VieNeu-TTS inference has three latency components:

**1. Prefill latency:** Processing the prompt tokens (voice codes + text) in one forward pass. This is a one-time cost at the start.
$$t_{\text{prefill}} = \frac{N_{\text{prompt}}^2 \cdot d_{\text{model}} \cdot N_L}{C_{\text{throughput}}}$$

Typical: 50–200 ms for a prompt of 200–400 tokens.

**2. Token generation latency:** Autoregressive generation of each speech token. This is the dominant cost for long outputs.
$$t_{\text{gen}} = N_{\text{speech tokens}} \times t_{\text{token}}$$

where $t_{\text{token}} \approx 8$–$15$ ms per token on a modern CPU with GGUF Q4.

**3. Codec decoding latency:** Converting speech tokens to waveform.
$$t_{\text{codec}} = N_{\text{frames}} \times t_{\text{frame}}$$

DistillNeuCodec processes approximately 500 frames/second on CPU — typically 10–50 ms for a full utterance.

**Total latency:**
$$t_{\text{total}} = t_{\text{prefill}} + t_{\text{gen}} + t_{\text{codec}}$$

**For a 5-second sentence (250 speech tokens at 50 Hz):**
- Prefill: ~100 ms
- Generation: ~250 × 10 ms = 2500 ms
- Codec: ~50 ms
- Total: ~2650 ms for 5 s audio → RTF ≈ 0.53

### 4.3 Hardware Benchmark Targets

| Hardware | Model | RTF | Notes |
|----------|-------|-----|-------|
| Apple M2 (CPU, Metal) | 0.3B Q4_K_M | ~0.5–0.7 | Metal GPU via llama.cpp |
| Apple M1 (CPU, Metal) | 0.3B Q4_K_M | ~0.7–0.9 | |
| Intel Core i7-12th gen | 0.3B Q4_K_M | ~0.8–1.2 | Varies with core count |
| NVIDIA RTX 3060 (GPU) | 0.3B fp16 | ~0.1–0.2 | GPU very fast |
| NVIDIA RTX 3090 (GPU) | 0.5B fp16 | ~0.1–0.15 | |
| Raspberry Pi 4 | 0.3B Q4_K_M | ~4.0–6.0 | Not real-time |

### 4.4 Bottleneck Analysis: Backbone vs. Codec

For a typical 5-second synthesis on Apple M2:

```
Prefill:       100 ms  (4%)
Backbone gen: 2200 ms (83%)  ← PRIMARY BOTTLENECK
Codec decode:   50 ms  (2%)
Overhead:      300 ms (11%)
─────────────────────────────
Total:        2650 ms
```

The backbone (autoregressive token generation) accounts for 83% of total latency. This is the standard bottleneck for LLM-based TTS — the codec decoding is fast because it is a single forward pass (not autoregressive).

**Optimization priority:** Reduce backbone latency first. Options:
1. Smaller model (0.3B vs 0.5B): ~40% latency reduction
2. Lower quantization (Q4 vs Q8): ~20% latency reduction
3. Speculative decoding: ~30–50% latency reduction (experimental)
4. Distillation: train a smaller student model (research)

### 4.5 Impact of Text Length

Latency scales approximately linearly with text length (more speech tokens to generate):

```python
# Approximate RTF as function of text length
chars = [50, 100, 150, 200, 250, 300]
rtf_estimates = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
```

The RTF increases slightly with text length because longer KV caches make each attention step more expensive. For very long texts (> 300 characters), chunking at sentence boundaries is recommended.

---

## 5. LoRA Merge for Production

### 5.1 Why Merge?

When you train LoRA, you have two files:
1. The base model ($\sim$600 MB for 0.3B fp16)
2. The LoRA adapter ($\sim$20 MB)

Loading both requires loading the base model and applying LoRA dynamically at each layer. This adds:
- Initialization time: ~1 second extra for adapter loading and merging at init
- Inference overhead: essentially zero (LoRA merge is applied before inference starts in PEFT's default mode)

**Primary reason to merge:** Enable GGUF conversion. The `convert_hf_to_gguf.py` script from llama.cpp only handles standard HuggingFace model formats — it cannot process a base model + adapter pair directly. Merging produces a single standard model file that can be quantized.

### 5.2 The Merge Operation

LoRA merge computes the effective weight matrix for each adapted layer:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$

This is a simple matrix operation. In PyTorch / PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "pnnbao-ump/VieNeu-TTS-0.3B",
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Load and merge LoRA
merged_model = PeftModel.from_pretrained(base_model, "finetune/output/VieNeu-TTS-0.3B-LoRA")
merged_model = merged_model.merge_and_unload()

# Save as standard HuggingFace model
merged_model.save_pretrained("finetune/output/VieNeu-TTS-0.3B-merged")
```

The `merge_and_unload()` call:
1. For each LoRA-adapted layer, computes $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$ in fp32 (for numerical precision)
2. Stores the merged weight in the model
3. Removes the LoRA adapter bookkeeping structures

The result is a standard PyTorch model indistinguishable from a fully-fine-tuned model — it has no knowledge of LoRA internally.

### 5.3 GGUF Conversion Pipeline

After merging:

```bash
# Step 1: Convert merged HF model to GGUF (fp16)
python llama.cpp/convert_hf_to_gguf.py \
    finetune/output/VieNeu-TTS-0.3B-merged \
    --outtype f16 \
    --outfile VieNeu-TTS-0.3B-custom.f16.gguf

# Step 2: Quantize to Q4_K_M
./llama.cpp/llama-quantize \
    VieNeu-TTS-0.3B-custom.f16.gguf \
    VieNeu-TTS-0.3B-custom.Q4_K_M.gguf \
    Q4_K_M

# Step 3: Verify
./llama.cpp/llama-cli \
    -m VieNeu-TTS-0.3B-custom.Q4_K_M.gguf \
    --vocab-only
```

**File sizes produced:**
- `VieNeu-TTS-0.3B-merged/` (HF format): ~600 MB
- `VieNeu-TTS-0.3B-custom.f16.gguf`: ~600 MB
- `VieNeu-TTS-0.3B-custom.Q4_K_M.gguf`: ~150 MB

The Q4_K_M GGUF is the deployment artifact.

### 5.4 Quality Verification After Merge

After conversion, verify that merge introduced no errors:

```python
from vieneu import Vieneu

# Test with original (LoRA adapter)
tts_lora = Vieneu(model_path="base_model", lora_path="lora_adapter")

# Test with merged GGUF
tts_gguf = Vieneu(model_path="custom_model.Q4_K_M.gguf")

test_text = "Xin chào, đây là bài kiểm tra chất lượng mô hình."
audio_lora = tts_lora.infer(test_text)
audio_gguf = tts_gguf.infer(test_text)

# They should sound nearly identical
```

---

## 6. Serving Patterns

### 6.1 Local CLI (scripts/generate.py)

For batch processing — converting a list of texts to audio files:

```python
# scripts/generate.py usage
python scripts/generate.py \
    --input texts.txt \
    --output_dir output_audio/ \
    --voice Binh \
    --model pnnbao-ump/VieNeu-TTS-0.3B \
    --format wav
```

**Architecture:** Single-process, sequential processing. Suitable for offline batch jobs. No concurrency or memory optimization needed — just throughput.

**Output:** One `.wav` file per line of `texts.txt`, named `001.wav`, `002.wav`, etc.

### 6.2 Gradio UI (gradio_app.py)

For demos, testing, and sharing with non-technical users:

```python
import gradio as gr
from vieneu import Vieneu

tts = Vieneu()

def synthesize(text, voice_name, speed):
    voice_data = tts.get_preset_voice(voice_name)
    audio = tts.infer(text, voice=voice_data, speed=speed)
    return (24000, audio)

demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label="Vietnamese text", lines=3),
        gr.Dropdown(["Binh", "Tuyen", "Ly"], label="Voice"),
        gr.Slider(0.7, 1.3, value=1.0, label="Speed"),
    ],
    outputs=gr.Audio(label="Synthesized speech"),
    title="VieNeu-TTS Demo",
)
demo.launch(share=True)
```

**Architecture:** Single-process Gradio server. Suitable for demos. Not suitable for production (no request queuing, no concurrency, no authentication).

### 6.3 REST API with LMDeploy

For production serving:

```python
# Start the API server
lmdeploy serve api_server pnnbao-ump/VieNeu-TTS-0.3B \
    --server-port 8080 \
    --tp 1  # tensor parallelism

# FastAPI endpoint on top of LMDeploy
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    audio = tts.infer(request.text, voice=request.voice)
    return StreamingResponse(
        io.BytesIO(audio_to_wav_bytes(audio)),
        media_type="audio/wav"
    )
```

**Production considerations:**
- Use `uvicorn` with multiple workers for concurrent requests
- Implement request queuing (one TTS inference at a time per GPU/CPU)
- Add authentication (API keys)
- Monitor latency with Prometheus metrics

### 6.4 Docker Deployment

```dockerfile
# Dockerfile for VieNeu-TTS API
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download models at build time
RUN python -c "from vieneu import Vieneu; Vieneu()"

EXPOSE 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
```

**GPU passthrough:**
```bash
docker run --gpus all -p 8080:8080 vieneu-tts:latest
```

**CPU-only deployment (for GGUF Q4):**
```bash
docker run -p 8080:8080 -e USE_GGUF=1 vieneu-tts:latest
```

### 6.5 Concurrency Model

VieNeu-TTS inference is **not thread-safe** when using a shared model object — two concurrent requests will produce corrupted outputs if they access the same model simultaneously.

**Pattern A (recommended for low concurrency):** One model instance per worker process, no concurrency within a process.

**Pattern B (for high concurrency):** Model pool — maintain $N$ model instances, serve requests from the pool.

```python
from asyncio import Queue

class TtsPool:
    def __init__(self, n=4):
        self.pool = Queue()
        for _ in range(n):
            self.pool.put_nowait(Vieneu())

    async def infer(self, text, voice):
        tts = await self.pool.get()
        try:
            return tts.infer(text, voice=voice)
        finally:
            await self.pool.put(tts)
```

---

## 7. Streaming for Low Latency

### 7.1 First-Token Latency vs. Total Latency

In a non-streaming response, the user waits until the entire audio is generated before hearing anything:

$$t_{\text{user wait}} = t_{\text{total}} = t_{\text{prefill}} + t_{\text{gen}} + t_{\text{codec}}$$

In streaming, audio is delivered as soon as the first chunk is decoded:

$$t_{\text{first audio}} = t_{\text{prefill}} + t_{\text{gen, chunk\_1}} + t_{\text{codec, chunk\_1}}$$

For a 50-frame chunk (1 second at 50 Hz):
- Generation: 50 × 10 ms = 500 ms
- Codec: ~10 ms
- First chunk: ~100 + 500 + 10 = 610 ms

Compare to non-streaming wait for a 5-second utterance: 2650 ms.

**Streaming reduces first-audio latency by ~4× for typical Vietnamese sentences.**

### 7.2 VieNeu-TTS Streaming API

```python
from vieneu import Vieneu
import numpy as np

tts = Vieneu()
voice_data = tts.get_preset_voice("Binh")

audio_buffer = []
for chunk in tts.infer_stream(text, voice=voice_data, chunk_frames=25):
    # chunk is a numpy array of audio samples
    # chunk_frames=25 means 25 codec frames = 25 × 480 samples = 12000 samples
    # at 24 kHz: 12000/24000 = 0.5 seconds per chunk
    audio_buffer.append(chunk)
    # → send chunk to audio output immediately

full_audio = np.concatenate(audio_buffer)
```

**Chunk size calculation:**
$$\text{chunk duration} = \frac{C_{\text{frames}} \times F_{\text{samples}}}{\text{sample rate}}$$

For `chunk_frames=25`, `frame_size=480`, `sample_rate=24000`:
$$\text{chunk duration} = \frac{25 \times 480}{24000} = 0.5 \text{ s}$$

### 7.3 Overlap-Add for Smooth Transitions

Naive chunked generation can produce clicking artifacts at chunk boundaries because:
- Codec decodes each chunk independently
- The boundary frames do not have context from adjacent chunks
- Boundary audio may have a slight discontinuity in amplitude or phase

**Overlap-Add (OLA) method:**

Each chunk includes $N_{\text{overlap}}$ overlap frames with the previous chunk. The output at the boundary is a linear crossfade:

$$y[n] = w_1[n] \cdot x_{\text{prev}}[n] + w_2[n] \cdot x_{\text{curr}}[n]$$

where:
- $w_1[n] = 1 - n/N_{\text{overlap}}$ (linear fade-out of previous chunk)
- $w_2[n] = n/N_{\text{overlap}}$ (linear fade-in of current chunk)
- $n = 0, 1, \ldots, N_{\text{overlap}} - 1$

For `overlap_frames=5` (5 codec frames = 2400 samples = 100 ms at 24 kHz), the crossfade is inaudible.

### 7.4 Use Cases for Streaming

**Real-time voice assistant (Vietnamese chatbot):**
```
User speaks → ASR (Whisper) → LLM response → VieNeu-TTS stream → Speaker
                                                    ↑
                            First chunk played while rest is generating
```

Target end-to-end latency: < 1 second from LLM response start to first audio output.

**Text-to-audio podcast generation:**
- Long article → split into sentences → stream synthesis → write to audio file
- Streaming enables writing audio before full synthesis completes, useful for very long texts

**WebSocket API for browser TTS:**
```python
@app.websocket("/stream_tts")
async def stream_tts(websocket: WebSocket, text: str):
    await websocket.accept()
    for chunk in tts.infer_stream(text):
        wav_bytes = numpy_to_wav_bytes(chunk)
        await websocket.send_bytes(wav_bytes)
    await websocket.close()
```

---

## 8. voices.json — Portable Voice Packaging

### 8.1 Why voices.json Exists

The in-context voice cloning paradigm (Chapter 6) requires the user to provide a reference audio clip when calling `tts.infer()`. This creates deployment friction:

- Users must have reference audio files
- Audio files must be pre-processed (16 kHz, mono)
- Loading audio and encoding to codec tokens adds latency at runtime

`voices.json` solves this by **pre-encoding the reference voice clips** into codec tokens and storing them in a JSON file distributed with (or alongside) the model. Users can synthesize with a preset voice using just a name string.

### 8.2 File Structure

```json
{
    "default_voice": "Binh",
    "presets": {
        "Binh": {
            "codes": [1234, 5678, 9012, ...],
            "text": "Xin chào, tôi là Bình, đây là giọng nói mẫu.",
            "description": "Male voice, Northern Vietnamese, formal register"
        },
        "Tuyen": {
            "codes": [2345, 6789, ...],
            "text": "Chào bạn, mình là Tuyên nhé.",
            "description": "Male voice, Northern Vietnamese, casual register"
        },
        "Ly": {
            "codes": [3456, 7890, ...],
            "text": "Xin chào các bạn, tôi là Lý.",
            "description": "Female voice, Northern Vietnamese"
        }
    }
}
```

**Key fields:**
- `codes`: Flat list of NeuCodec integer tokens encoding the reference voice clip
- `text`: The transcription of the reference voice clip (used to format the voice prompt)
- `description`: Human-readable description for UI display

### 8.3 Creating a voices.json Entry

To add a new voice to `voices.json`:

```python
import torch
import librosa
import json
from neucodec import DistillNeuCodec

device = "cuda" if torch.cuda.is_available() else "cpu"
codec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").to(device).eval()

def encode_voice_reference(audio_path, ref_text, voice_name, description):
    # Load and preprocess audio
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Encode to codec tokens
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = codec.encode_code(wav_tensor)

    # Flatten to 1D list
    codes_list = codes.squeeze().cpu().numpy().flatten().tolist()
    codes_int = [int(c) for c in codes_list]

    # Validate
    assert all(0 <= c < 65536 for c in codes_int), "Invalid code range"
    assert len(codes_int) > 0, "Empty codes"

    return {
        voice_name: {
            "codes": codes_int,
            "text": ref_text,
            "description": description
        }
    }

# Usage
new_entry = encode_voice_reference(
    "my_voice.wav",
    "Xin chào, đây là giọng nói tùy chỉnh của tôi.",
    "my_voice",
    "Custom voice, Southern Vietnamese dialect"
)

# Load existing voices.json and add new entry
with open("voices.json") as f:
    voices = json.load(f)

voices["presets"].update(new_entry)

with open("voices.json", "w", encoding="utf-8") as f:
    json.dump(voices, f, ensure_ascii=False, indent=2)
```

### 8.4 Runtime Loading

At initialization, `Vieneu()` loads `voices.json` from the HuggingFace cache:

```python
class Vieneu:
    def __init__(self, model_id="pnnbao-ump/VieNeu-TTS-0.3B"):
        # Download voices.json from HuggingFace Hub
        voices_path = hf_hub_download(model_id, "voices.json")

        with open(voices_path) as f:
            self._voices = json.load(f)

        self._default_voice = self._voices["default_voice"]

    def get_preset_voice(self, voice_name=None):
        name = voice_name or self._default_voice
        preset = self._voices["presets"][name]
        return {
            "codes": torch.tensor(preset["codes"]),
            "text": preset["text"]
        }
```

**Latency impact:** `voices.json` for three preset voices is approximately 5 MB (each 10-second voice clip produces ~500 × 4 bytes = ~2 KB of code integers; for 3 voices: 6 KB, plus JSON formatting). Initialization time to load and parse is < 10 ms.

### 8.5 Distributing a Fine-Tuned Model with Custom Voice

When you fine-tune VieNeu-TTS for a custom voice and want to share it:

```
HuggingFace repository structure:
├── config.json                    # Model architecture config
├── tokenizer.json                 # Tokenizer vocab
├── model.Q4_K_M.gguf              # Quantized model (GGUF)
├── voices.json                    # Pre-encoded voice presets
│   └── "my_custom_voice": {      # Your fine-tuned speaker
│         "codes": [...],
│         "text": "...",
│         "description": "..."
│       }
└── README.md                      # Model card
```

A user can then use your model with one line:
```python
tts = Vieneu(model_id="your-username/VieNeu-TTS-custom-voice")
audio = tts.infer("Xin chào Việt Nam!")  # Uses your custom voice by default
```

---

## Summary

| Deployment scenario | Model format | Serving pattern | Target RTF |
|--------------------|-------------|-----------------|-----------|
| Research / GT comparison | PyTorch fp16 (GPU) | scripts/generate.py | 0.1× |
| Production server (GPU) | PyTorch fp16 (GPU) | REST API (LMDeploy) | 0.15× |
| CPU server | GGUF Q8 | REST API (FastAPI) | 0.8× |
| Edge device / laptop | GGUF Q4_K_M | CLI / local app | 0.6× (M2) |
| Demo / testing | GGUF Q4_K_M | Gradio | N/A |
| Low-latency streaming | GGUF Q4_K_M | WebSocket | < 1s first chunk |

### Production Deployment Checklist

**Model preparation:**
- [ ] Merge LoRA adapter into base model
- [ ] Convert to GGUF Q4_K_M format
- [ ] Create `voices.json` with pre-encoded reference voice
- [ ] Upload merged model + `voices.json` to HuggingFace Hub

**Quality verification:**
- [ ] Test all 6 Vietnamese tones
- [ ] Test code-switching (Vietnamese + English)
- [ ] Measure UTMOS on 10 test sentences (target > 3.8)
- [ ] Compare RTF on target hardware

**Serving:**
- [ ] Choose serving pattern appropriate to traffic volume
- [ ] Implement request queuing for concurrency
- [ ] Add health check endpoint
- [ ] Monitor latency and error rates

**Distribution:**
- [ ] Write model card with voice description
- [ ] Document usage example
- [ ] Test `Vieneu(model_id="your-repo")` from fresh environment

---

## Course Complete — What You've Learned

This 10-chapter course has taken you from raw audio signals to deploying a production Vietnamese TTS system:

| Chapter | Topic | Key takeaway |
|---------|-------|-------------|
| 01 | Audio Fundamentals | Waveforms, STFT, Mel spectrograms, MFCCs |
| 02 | Text Processing | Unicode, phonemization, Vietnamese tones |
| 03 | TTS Architecture Evolution | Concatenative → Neural → LLM-based TTS |
| 04 | Neural Codecs | VQ, RVQ, NeuCodec, DistillNeuCodec |
| 05 | LLM-Based TTS | VieNeu-TTS architecture, prompt format, sampling |
| 06 | Voice Cloning | In-context cloning, speaker similarity |
| 07 | LoRA Theory | Low-rank adaptation, parameter savings |
| 08 | Data Preparation | Quality metrics, filter pipeline, encoding |
| 09 | Training & Evaluation | Loss curves, CER, UTMOS, MOS |
| 10 | Deployment | GGUF quantization, streaming, voices.json |

**Next steps:**
1. Collect your Vietnamese audio dataset (Chapter 8 quality standards)
2. Run the filter and encode pipeline (Chapters 8)
3. Fine-tune with LoRA (Chapter 7 config + Chapter 9 monitoring)
4. Evaluate with UTMOS and human MOS (Chapter 9)
5. Deploy your custom model (this chapter)
6. Contribute your fine-tuned voice back to the VieNeu-TTS community!
