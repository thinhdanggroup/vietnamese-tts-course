# Chapter 01 — Audio Fundamentals

> **Audience**: ML engineers who understand training loops, loss functions, and backpropagation, but are new to audio signal processing and TTS.
> **Goal**: Build a rigorous mathematical foundation for every audio representation used in VieNeu-TTS.

---

## Table of Contents

1. [What is Sound?](#1-what-is-sound)
2. [Digital Audio](#2-digital-audio)
3. [Waveform — The Time Domain](#3-waveform--the-time-domain)
4. [Fourier Transform and STFT](#4-fourier-transform-and-stft)
5. [The Mel Scale](#5-the-mel-scale)
6. [MFCCs](#6-mfccs)
7. [Summary: Representations in VieNeu-TTS](#7-summary-representations-in-vieneu-tts)

---

## 1. What is Sound?

### 1.1 Pressure Waves

Sound is a longitudinal mechanical wave — a disturbance that propagates through an elastic medium (air, water, solid) by compressing and rarefying the medium's molecules. Unlike electromagnetic waves, sound requires a physical medium to travel.

When a speaker diaphragm (or a vocal cord) vibrates, it periodically compresses the air in front of it, creating alternating regions of **high pressure** (compression) and **low pressure** (rarefaction). These pressure variations travel outward at the speed of sound (~343 m/s in air at 20°C) and are captured by a microphone as a continuous voltage signal.

Mathematically, a pure tone is modeled as a sinusoid:

$$x(t) = A \cos(2\pi f t + \phi)$$

where:

| Symbol | Name | Unit |
|--------|------|------|
| $A$ | Amplitude | Pascal (Pa) or normalized |
| $f$ | Frequency | Hertz (Hz) |
| $t$ | Time | Seconds (s) |
| $\phi$ | Phase | Radians |

### 1.2 Frequency

Frequency $f$ is the number of complete oscillation cycles per second. It is the primary correlate of **pitch** — the perceptual quality that tells us how high or low a sound is. A higher frequency means more cycles per second, which we perceive as a higher pitch.

- **Low frequency** → deep, bass sound (e.g., 80 Hz male fundamental pitch)
- **High frequency** → bright, treble sound (e.g., 8 kHz fricative consonants like /s/)

### 1.3 Amplitude

Amplitude $A$ is the peak pressure deviation from equilibrium. It directly correlates with **loudness**. However, human loudness perception is logarithmic, which is why audio engineers use **decibels (dB)**:

$$L_{dB} = 20 \log_{10}\left(\frac{A}{A_{\text{ref}}}\right)$$

where $A_{\text{ref}}$ is a reference amplitude (often $20 \mu$Pa, the threshold of human hearing).

### 1.4 Phase

Phase $\phi$ determines the position of the waveform within its cycle at time $t = 0$. Phase is critical for:
- Wave interference (constructive when in-phase, destructive when out-of-phase)
- Signal reconstruction from the STFT (you need both magnitude AND phase)
- Vocoders that model or discard phase information

### 1.5 Human Hearing Range

The human auditory system responds to frequencies between **20 Hz and 20,000 Hz (20 kHz)**, though this range narrows significantly with age (especially above 12 kHz for adults over 40).

Within this range, different frequency regions carry different types of speech information:

| Frequency Band | Speech Content |
|----------------|----------------|
| 80–300 Hz | Fundamental pitch (F0), voicing |
| 300–3,400 Hz | Vowel formants F1, F2 (most intelligibility here) |
| 3,400–8,000 Hz | Fricatives, sibilants, clarity |
| 8,000–20,000 Hz | Breathiness, naturalness, "air" |

> **Vietnamese note**: Vietnamese tone is carried primarily in the **F0 contour** (the pitch trajectory over a syllable) and **phonation type** (modal voice vs. creaky voice for tone ngã and nặng). The F0 range for a typical Vietnamese female speaker spans approximately 150–400 Hz.

---

## 2. Digital Audio

### 2.1 Analog-to-Digital Conversion (ADC)

Real-world audio is a **continuous** signal — it exists at every instant in time and can take any real value. Computers can only store **discrete** sequences of numbers. The **Analog-to-Digital Converter (ADC)** bridges this gap through two operations:

1. **Sampling**: Measure the signal's amplitude at regular time intervals
2. **Quantization**: Round each measured amplitude to the nearest representable digital value

### 2.2 Sampling Rate

The **sampling rate** (or sample rate) $f_s$ is the number of samples taken per second, measured in Hz. Common values:

| Sample Rate | Application |
|-------------|-------------|
| 8,000 Hz | Telephone (narrowband) |
| 16,000 Hz | Speech recognition (wideband) |
| 22,050 Hz | Audio ML (half CD quality) |
| 24,000 Hz | **VieNeu-TTS** (covers full speech range) |
| 44,100 Hz | CD quality, music production |
| 48,000 Hz | Professional broadcast, video |

### 2.3 The Nyquist-Shannon Sampling Theorem

This is the foundational theorem of digital signal processing.

**Theorem (Nyquist-Shannon, 1949)**: A bandlimited continuous-time signal $x(t)$ with maximum frequency component $B$ Hz can be **perfectly reconstructed** from its samples if and only if the sampling rate satisfies:

$$f_s \geq 2B$$

The minimum acceptable rate $f_s = 2B$ is called the **Nyquist rate**.

#### Proof Sketch

Consider a continuous signal $x(t)$ whose Fourier spectrum $X(f)$ is zero for $|f| > B$ (it is bandlimited to $B$ Hz).

**Step 1 — Sampling in time domain is multiplication by an impulse train.**

The sampled signal is:

$$x_s(t) = x(t) \cdot \text{III}_{T_s}(t) = x(t) \cdot \sum_{n=-\infty}^{\infty} \delta(t - nT_s)$$

where $T_s = 1/f_s$ is the sampling interval and $\text{III}_{T_s}$ is the Dirac comb (Shah function).

**Step 2 — In the frequency domain, sampling causes periodic replication.**

The Fourier transform of the Dirac comb is also a Dirac comb:

$$\mathcal{F}\left\{\sum_n \delta(t - nT_s)\right\} = f_s \sum_k \delta(f - k f_s)$$

Therefore, multiplication in time becomes convolution in frequency:

$$X_s(f) = X(f) * \left(f_s \sum_k \delta(f - k f_s)\right) = f_s \sum_k X(f - k f_s)$$

The sampled spectrum $X_s(f)$ is a **sum of shifted copies** of the original spectrum $X(f)$, shifted by integer multiples of $f_s$.

**Step 3 — Reconstruction requires no overlap between copies.**

The copies of $X(f)$ each span $[-B, B]$. After shifting by $kf_s$, adjacent copies span $[kf_s - B, kf_s + B]$. For the copies not to overlap (alias), we need the gap between them to be non-negative:

$$f_s - B \geq B \implies f_s \geq 2B \quad \square$$

If $f_s \geq 2B$, we can recover $X(f)$ perfectly by applying a **low-pass filter** with cutoff at $B$ Hz, then scaling. In practice this is the **reconstruction filter** in the DAC (Digital-to-Analog Converter).

#### Aliasing — What Goes Wrong Below Nyquist

If $f_s < 2B$, the spectral copies **overlap**. A high-frequency component at frequency $f$ appears as a "ghost" at the alias frequency:

$$f_{\text{alias}} = |f - k f_s|$$

for the integer $k$ that brings $f_{\text{alias}}$ into $[0, f_s/2]$.

**Concrete example**: Suppose $f_s = 8000$ Hz (telephone). The Nyquist limit is 4000 Hz. If a sound contains a 5000 Hz component (a sibilant /s/):

$$f_{\text{alias}} = |5000 - 1 \times 8000| = 3000 \text{ Hz}$$

The 5000 Hz energy folds back to appear at 3000 Hz — distorting the signal. This is why telephone speech sounds "muffled" and has reduced intelligibility for fricatives.

#### Anti-Aliasing Filter

Modern ADCs apply a **low-pass anti-aliasing filter** before sampling to remove all frequency content above $f_s / 2$. This prevents aliasing at the cost of discarding frequencies above the Nyquist limit.

### 2.4 Bit Depth and Dynamic Range

**Bit depth** is the number of bits used to represent each sample's amplitude. The key formula for dynamic range is:

$$\text{Dynamic Range} = 6.02 \times N_{\text{bits}} \text{ dB}$$

**Derivation**: With $N$ bits, there are $2^N$ quantization levels. The smallest representable signal has amplitude $\approx 1/2^N$ relative to full scale. The signal-to-quantization-noise ratio (SQNR) is:

$$\text{SQNR} = 6.02 N + 1.76 \text{ dB} \approx 6.02 N \text{ dB}$$

This $6.02 \approx 20 \log_{10}(2)$ factor comes directly from the doubling of representable levels per bit.

| Bit Depth | Dynamic Range | Audio Use Case |
|-----------|---------------|----------------|
| 8-bit | ~48 dB | Old games, telephone (mu-law) |
| 16-bit | ~96 dB | CD audio, most ML datasets |
| 24-bit | ~144 dB | Professional recording |
| 32-bit float | ~1528 dB (theoretical) | In-memory processing, no clipping |

**Why VieNeu-TTS uses 24 kHz**:

Vietnamese speech contains:
- Fundamental pitch F0: 80–400 Hz (well within 24 kHz)
- Formants F1–F4: up to ~4 kHz (well within 12 kHz Nyquist)
- Aspirated consonants /th/, /ph/, /kh/: energy up to ~8 kHz (within 12 kHz Nyquist)
- Fricatives /s/, /x/: energy up to ~10 kHz (within 12 kHz Nyquist)
- Naturalness / breathiness: energy up to ~12 kHz (right at the Nyquist limit)

At 24 kHz, the Nyquist limit is exactly 12 kHz — covering all speech-critical frequencies while using half the storage and compute of 44.1 kHz. This is the optimal trade-off for a speech synthesis system.

---

## 3. Waveform — The Time Domain

### 3.1 The Amplitude-Time Plot

The raw waveform is the most basic audio representation: a 1D array of amplitude values, one per sample:

$$\mathbf{x} = [x[0], x[1], x[2], \ldots, x[N-1]]$$

where $x[n] = x(nT_s)$ is the sample at time $nT_s$.

This representation is:
- **Lossless** — contains all information in the digital signal
- **Difficult to process directly** — frequency information is entangled across all samples
- **Used in neural vocoders** — WaveNet, HiFi-GAN, NeuCodec operate on raw waveforms

### 3.2 Vietnamese Tones in the Time Domain

Vietnamese has 6 tones (thanh điệu), each with a distinctive pitch contour that is visible in the waveform's envelope (amplitude modulation) and, indirectly, in the periodicity of the waveform (period $= 1/F0$).

| Tone | Vietnamese Name | Diacritic | F0 Contour | Phonation |
|------|----------------|-----------|------------|-----------|
| Ngang | Level/Flat | (none) | High, flat | Modal |
| Huyền | Falling | `` ` `` (grave) | Mid-low, falling | Breathy |
| Sắc | Rising | ´ (acute) | High, rising | Modal |
| Hỏi | Dipping | ̉ (hook above, e.g. ả) | Mid, dipping then rising | Modal |
| Ngã | Creaky Rising | ˜ (tilde) | High rising, creaky | Creaky |
| Nặng | Low Falling | . (dot below) | Low, short, falling | Constricted |

In the raw waveform:
- **Ngang**: Uniformly spaced cycles of similar amplitude throughout the vowel
- **Huyền**: Longer period cycles (lower pitch) with slightly irregular amplitude (breathiness)
- **Sắc**: Cycles get shorter (higher pitch) toward the end — you can see the period decreasing
- **Nặng**: Abruptly terminated waveform — shorter duration, lower amplitude at end

While tone information is present in the waveform, it is much easier to analyze in the **frequency domain** (spectrogram), where F0 appears as the fundamental frequency and its harmonics at $2F0, 3F0, \ldots$

---

## 4. Fourier Transform and STFT

### 4.1 The Continuous Fourier Transform

The **Continuous Fourier Transform (CFT)** decomposes a time-domain signal $x(t)$ into its constituent frequencies:

$$X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-j2\pi ft} \, dt$$

The **inverse transform** recovers the original signal:

$$x(t) = \int_{-\infty}^{\infty} X(f) \, e^{j2\pi ft} \, df$$

**Full derivation of the inverse formula**:

Substitute $X(f)$ into the inverse integral and assume we can exchange the order of integration:

$$\int_{-\infty}^{\infty} X(f) e^{j2\pi ft} df = \int_{-\infty}^{\infty} \left[\int_{-\infty}^{\infty} x(\tau) e^{-j2\pi f\tau} d\tau\right] e^{j2\pi ft} df$$

$$= \int_{-\infty}^{\infty} x(\tau) \left[\int_{-\infty}^{\infty} e^{j2\pi f(t-\tau)} df\right] d\tau$$

The inner integral is the **Dirac delta function**: $\int e^{j2\pi f(t-\tau)} df = \delta(t - \tau)$.

Therefore:

$$= \int_{-\infty}^{\infty} x(\tau) \delta(t - \tau) d\tau = x(t) \quad \square$$

**Interpretation**: $X(f)$ is a **complex-valued** function of frequency. Its magnitude $|X(f)|$ tells us how much of frequency $f$ is present in the signal. Its phase $\angle X(f)$ tells us the phase offset of that frequency component.

### 4.2 The Discrete Fourier Transform (DFT)

For a finite sequence of $N$ samples $x[0], x[1], \ldots, x[N-1]$, the **Discrete Fourier Transform** is:

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

**Key properties**:
- $N$ input samples → $N$ complex output coefficients
- $X[k]$ corresponds to frequency $f_k = k \cdot f_s / N$
- Frequency resolution: $\Delta f = f_s / N$ Hz per bin
- Only bins $k = 0, \ldots, N/2$ are unique (the rest are conjugate symmetric for real inputs)

**The Fast Fourier Transform (FFT)** computes the DFT in $O(N \log N)$ time instead of $O(N^2)$ by exploiting the symmetry $e^{-j2\pi k(n+N/2)/N} = -e^{-j2\pi kn/N}$ to recursively split the computation (Cooley-Tukey algorithm, 1965).

### 4.3 Limitation of Global Fourier Transform for Speech

The standard DFT treats the entire signal as **stationary** — it finds the frequencies present across the whole recording. But speech is **non-stationary**: the frequency content changes over time as different phonemes are produced.

**Example**: The Vietnamese word "chào" contains:
1. A palatal affricate /tɕ/ (broadband noise, high frequencies)
2. A mid-low vowel /aː/ (clear F1, F2 formants)
3. A glide /w/ (shifting formants)
4. The **huyền tone** (falling F0)

The global DFT would smear all of these together into one uninterpretable spectrum. We need a **time-frequency representation**.

### 4.4 The Short-Time Fourier Transform (STFT)

The **Short-Time Fourier Transform** computes the DFT on short, overlapping windows of the signal:

$$X[m, k] = \sum_{n=-\infty}^{\infty} x[n] \, w[n - mH] \, e^{-j2\pi kn/N}$$

where:
- $m$ is the **frame index** (time axis)
- $k$ is the **frequency bin** index (frequency axis)
- $w[n]$ is the **window function** (e.g., Hann window)
- $H$ is the **hop length** (step size between frames, in samples)
- $N$ is the **FFT size** (DFT length per frame)

**The Hann Window** is the most common choice:

$$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right), \quad n = 0, \ldots, N-1$$

The window serves two purposes:
1. **Tapering**: smoothly reduces the signal to zero at the edges, preventing **spectral leakage** (the Gibbs phenomenon when the DFT assumes periodic extension of the finite window)
2. **Localization**: focuses the DFT on a short time segment, giving us time resolution

### 4.5 The Time-Frequency Resolution Trade-off (Heisenberg-Gabor Limit)

The STFT faces a fundamental trade-off: you cannot have **both** high time resolution and high frequency resolution simultaneously.

**The Heisenberg-Gabor Uncertainty Principle for Signals**:

For any windowed signal, the time spread $\Delta t$ and frequency spread $\Delta f$ satisfy:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

**Practical implications for VieNeu-TTS at $f_s = 24000$ Hz**:

If we choose $N_{\text{FFT}} = 1024$ and $H = 256$:
- Frequency resolution: $\Delta f = 24000 / 1024 \approx 23.4$ Hz per bin
- Time resolution: $H / f_s = 256 / 24000 \approx 10.7$ ms per frame
- Window duration: $N / f_s = 1024 / 24000 \approx 42.7$ ms

The window must be long enough to contain several cycles of the lowest frequency of interest. For F0 = 80 Hz, one period = 12.5 ms, so we need $N \gg 12.5$ ms, hence $N = 1024$ (42.7 ms) comfortably captures 3–4 F0 cycles.

**The trade-off in extremes**:

| $N_{\text{FFT}}$ | $\Delta f$ | $\Delta t$ | Good for |
|---------|---------|---------|---------|
| 128 | 187.5 Hz | ~5 ms | Transients, stops |
| 512 | 46.9 Hz | ~21 ms | Compromise |
| 1024 | 23.4 Hz | ~43 ms | **VieNeu-TTS default** |
| 2048 | 11.7 Hz | ~85 ms | Precise pitch analysis |

### 4.6 Power Spectrogram

From the STFT, we compute the **power spectrogram** by taking the squared magnitude:

$$P[m, k] = |X[m, k]|^2 = \text{Re}(X[m,k])^2 + \text{Im}(X[m,k])^2$$

This discards the phase information and represents the **energy** at each time-frequency cell. The power spectrogram is:
- Always non-negative (can take log)
- Symmetric: $P[m, k] = P[m, N-k]$ for real signals
- In practice we keep only bins $k = 0, \ldots, N/2$ (the one-sided spectrogram)

The **log power spectrogram** (in dB) compresses the dynamic range:

$$P_{\text{dB}}[m, k] = 10 \log_{10}(P[m, k] + \epsilon)$$

where $\epsilon$ is a small constant for numerical stability.

---

## 5. The Mel Scale

### 5.1 Why Linear Frequency Fails

The human auditory system does not perceive frequency linearly. Equal ratios of frequency (octaves) are perceived as equal steps of pitch. This is called **logarithmic frequency perception**. Specifically:
- The difference between 100 Hz and 200 Hz (a ratio of 2) sounds like the same pitch interval as 1000 Hz and 2000 Hz
- But on a linear scale, 1000 Hz and 2000 Hz are 10× farther apart than 100 Hz and 200 Hz

Furthermore, the **cochlea** (inner ear) acts as a biological frequency analyzer with **non-uniform resolution**: very fine resolution at low frequencies, coarser resolution at high frequencies. The basilar membrane is physically longer at the apex (low frequency end) than the base (high frequency end).

A linear spectrogram wastes most of its frequency bins on high frequencies that humans barely distinguish. We need a **perceptually motivated frequency scale**.

### 5.2 The Mel Scale — Definition and Formula

The **mel scale** maps physical frequency (Hz) to perceptual pitch units (mels), derived empirically from psychoacoustic experiments where listeners were asked to identify the "halfway" pitch between two tones.

The standard formula (O'Shaughnessy, 1987):

$$m = 2595 \times \log_{10}\!\left(1 + \frac{f}{700}\right)$$

**Inverse (mel to Hz)**:

$$f = 700 \times \left(10^{m/2595} - 1\right)$$

**Derivation intuition**: At low frequencies ($f \ll 700$), the mel formula is approximately linear:

$$m \approx 2595 \times \frac{f}{700 \ln 10} \approx 1.61 f$$

At high frequencies ($f \gg 700$), it is approximately logarithmic:

$$m \approx 2595 \times \log_{10}(f/700)$$

The "break point" is at $f = 700$ Hz (approximately where the cochlear map transitions from nearly linear to logarithmic behavior).

**Example mel values for Vietnamese-relevant frequencies**:

| Hz | Mel | Notes |
|----|-----|-------|
| 80 | 122 | Minimum F0 (low male) |
| 150 | 212 | Typical male F0 |
| 250 | 331 | Typical female F0 |
| 400 | 499 | High female F0 (tone sắc peak) |
| 700 | 840 | Mel "break point" |
| 1000 | 1000 | First formant F1 (vowel /a/) |
| 2000 | 1540 | Second formant F2 (vowel /i/) |
| 4000 | 2146 | Fricative consonants |
| 8000 | 2840 | Breathiness, naturalness |
| 12000 | 3358 | Nyquist at 24 kHz |

### 5.3 Mel Filterbank

The **mel filterbank** is a set of $M$ triangular bandpass filters spaced uniformly on the mel scale. Each filter $H_m(k)$ selects a range of frequency bins from the power spectrogram:

The filter center frequencies in mel are:

$$m_i = m_{\min} + i \times \frac{m_{\max} - m_{\min}}{M + 1}, \quad i = 0, 1, \ldots, M+1$$

Converting back to Hz:

$$f_i = 700 \times \left(10^{m_i/2595} - 1\right)$$

Converting to FFT bin indices:

$$k_i = \left\lfloor \frac{N_{\text{FFT}} + 1}{f_s} \cdot f_i \right\rfloor$$

The $m$-th triangular filter is:

$$H_m(k) = \begin{cases}
0 & k < k_{m-1} \\
\dfrac{k - k_{m-1}}{k_m - k_{m-1}} & k_{m-1} \leq k \leq k_m \\
\dfrac{k_{m+1} - k}{k_{m+1} - k_m} & k_m \leq k \leq k_{m+1} \\
0 & k > k_{m+1}
\end{cases}$$

### 5.4 Mel Spectrogram Computation

The mel spectrogram $S[m, t]$ is computed from the power spectrogram $P[k, t]$:

$$S[m, t] = \sum_{k=0}^{N/2} H_m(k) \cdot P[k, t]$$

In matrix form, if $\mathbf{P} \in \mathbb{R}^{(N/2+1) \times T}$ is the power spectrogram and $\mathbf{M} \in \mathbb{R}^{M \times (N/2+1)}$ is the mel filterbank matrix:

$$\mathbf{S} = \mathbf{M} \cdot \mathbf{P}$$

The result $\mathbf{S} \in \mathbb{R}^{M \times T}$ has $M$ mel bands (typically 80 for TTS) and $T$ time frames.

Finally, we take the log:

$$\mathbf{S}_{\log}[m, t] = \log(\mathbf{S}[m, t] + \epsilon)$$

**VieNeu-TTS mel spectrogram settings**:
- Sample rate: 24,000 Hz
- FFT size ($N_{\text{FFT}}$): 1024
- Hop length ($H$): 256 samples (10.7 ms)
- Window size: 1024 samples (42.7 ms)
- Mel bins ($M$): 80
- Frequency range: 0–12,000 Hz (full speech range)

---

## 6. MFCCs

### 6.1 What Are MFCCs?

**Mel-Frequency Cepstral Coefficients (MFCCs)** are a compact representation of the spectral envelope of speech. They were the dominant feature for ASR systems for decades (1980s–2010s) and remain useful for speaker verification, emotion recognition, and some TTS analysis tasks.

MFCCs are NOT used directly in VieNeu-TTS for generation, but understanding them teaches us:
- How to extract the **spectral envelope** (vowel quality, timbre)
- How to separate **vocal tract characteristics** from **glottal source characteristics**
- Why delta features capture **dynamic information** (tone contour!)

### 6.2 The Pre-Emphasis Filter

Before frame analysis, a **pre-emphasis** filter boosts high frequencies to compensate for the natural rolloff of the vocal tract and lip radiation:

$$y[n] = x[n] - \alpha x[n-1]$$

where $\alpha \in [0.95, 0.99]$. In the Z-domain, this is a first-order high-pass filter $H(z) = 1 - \alpha z^{-1}$.

The effect: increases SNR for high-frequency components (consonants) and flattens the spectrum, making it easier for the DFT to resolve them.

### 6.3 Step-by-Step MFCC Computation

**Step 1 — Pre-emphasis**: Apply $y[n] = x[n] - 0.97 x[n-1]$

**Step 2 — Framing**: Divide signal into overlapping frames of $N_{\text{frame}}$ samples (typically 25 ms = 600 samples at 24 kHz) with hop length $H$ (10 ms = 240 samples).

**Step 3 — Windowing**: Multiply each frame by a Hann window to reduce spectral leakage:

$$x_m[n] = y[n + mH] \cdot w[n], \quad n = 0, \ldots, N_{\text{frame}}-1$$

**Step 4 — FFT**: Compute DFT of each windowed frame:

$$X_m[k] = \text{DFT}(x_m)[k] = \sum_{n=0}^{N-1} x_m[n] e^{-j2\pi kn/N}$$

**Step 5 — Mel Filterbank**: Apply $K$ mel filters to the power spectrum:

$$S_m[k] = \sum_{\ell} H_k(\ell) \cdot |X_m[\ell]|^2, \quad k = 0, \ldots, K-1$$

**Step 6 — Logarithm**: Take log to convert energy to log-energy (matches loudness perception and stabilizes variance):

$$\log S_m[k]$$

**Step 7 — DCT (Discrete Cosine Transform)**: Apply DCT-II to the log mel spectrum to get MFCCs:

$$c_m[n] = \sum_{k=0}^{K-1} \log S_m[k] \cdot \cos\!\left(\frac{\pi n (k + 0.5)}{K}\right), \quad n = 0, \ldots, C-1$$

where $C$ is the number of cepstral coefficients to keep (typically 13).

### 6.4 Why DCT? Decorrelation of Filterbank Outputs

The mel filterbank outputs $\log S_m[k]$ are **correlated** — adjacent filters overlap significantly, so knowing $S_m[k]$ gives you information about $S_m[k \pm 1]$.

The DCT solves this by **decorrelating** the outputs. Specifically, if the covariance matrix of the filterbank outputs is approximately **Toeplitz** (stationary correlations), then the DCT approximately **diagonalizes** it. This means the DCT coefficients are approximately uncorrelated — each $c[n]$ captures independent information.

This decorrelation is critical for:
- **Gaussian Mixture Models (GMMs)**: assume diagonal covariance — MFCCs satisfy this approximately
- **PCA analogy**: DCT is a fixed (non-data-dependent) PCA for Toeplitz-structured data

The first DCT coefficient $c[0]$ is proportional to the total log-energy. Coefficients $c[1], c[2], \ldots$ capture progressively finer spectral shape:
- $c[1]$: tilt (more energy at low vs high frequencies)
- $c[2], c[3]$: formant peaks
- $c[4]$–$c[12]$: fine spectral detail

### 6.5 Delta and Delta-Delta MFCCs

Static MFCCs describe the **instantaneous** spectral shape. To capture **temporal dynamics** (how the spectrum changes), we compute:

**Delta (velocity) MFCCs** — first derivative approximation:

$$\Delta c[n, t] = \frac{\sum_{\tau=1}^{\Theta} \tau \cdot (c[n, t+\tau] - c[n, t-\tau])}{2 \sum_{\tau=1}^{\Theta} \tau^2}$$

where $\Theta = 2$ is the typical window half-width (using $\pm 2$ frames).

**Delta-delta (acceleration) MFCCs**: apply the same formula to $\Delta c$.

The full feature vector per frame: $[\mathbf{c}, \Delta\mathbf{c}, \Delta\Delta\mathbf{c}]$ = $3 \times 13 = 39$ dimensions.

### 6.6 MFCCs for Vietnamese Tones

Vietnamese tone information manifests as follows in MFCC features:

**Static MFCCs**: Encode the vowel quality during the tone. Tones on the same base vowel (e.g., "a", "à", "á") have similar static MFCCs because the vocal tract shape (and thus formant frequencies) is nearly identical.

**Delta MFCCs ($\Delta c$)**: Encode the **rate of change** of the spectral shape. Since F0 changes with tone, and F0 affects the harmonic structure visible in the spectrum, delta features partially encode the tone contour:
- **Ngang (flat tone)**: $\Delta c[1] \approx 0$ (little change in tilt = flat pitch)
- **Huyền (falling)**: $\Delta c[1] < 0$ (spectrum tilts toward lower frequencies as pitch falls)
- **Sắc (rising)**: $\Delta c[1] > 0$ (spectrum brightens as pitch rises)
- **Nặng (abrupt falling)**: Large negative $\Delta \Delta c$ (fast deceleration at endpoint)

**Delta-delta MFCCs ($\Delta \Delta c$)**: Capture the acceleration of spectral change. Critical for tones with **contour changes** (hỏi: fall then rise; ngã: rise then fall with creaky break).

This is why early Vietnamese ASR and TTS systems needed careful MFCC engineering — the 6-tone system requires the full $\mathbf{c}, \Delta\mathbf{c}, \Delta\Delta\mathbf{c}$ feature set to be discriminative.

---

## 7. Summary: Representations in VieNeu-TTS

| Stage | Representation | Dimensions | Why This Choice |
|-------|---------------|------------|-----------------|
| Input to analysis | Raw waveform | $[T]$ | Ground truth, lossless |
| Codec encoder | Waveform → tokens | $[T/H]$ integer codes | Discrete tokens for LLM |
| Codec decoder | Tokens → waveform | $[T]$ | Reconstruct audio |
| Loss computation | Log mel spectrogram | $[80 \times T']$ | Perceptually meaningful, differentiable |
| Pitch analysis | F0 contour | $[T']$ (Hz) | Tone contour ground truth |
| Feature analysis | MFCCs | $[39 \times T']$ | Tone classification, evaluation |
| Visualization | Log mel spectrogram | $[80 \times T']$ | Best visual/diagnostic tool |

**The VieNeu-TTS pipeline** does not use MFCCs or spectrograms directly during inference. Instead:

1. **Text** is phonemized and tokenized → integer token IDs
2. **Reference audio** (for voice cloning) is encoded by **NeuCodec** → discrete speech tokens
3. The **LLM** predicts new speech tokens given text tokens + reference speech tokens
4. NeuCodec **decodes** speech tokens back to a waveform

This LLM-codec architecture is the key innovation of VieNeu-TTS — speech tokens are treated as a new modality in the LLM's vocabulary, enabling zero-shot voice cloning and excellent Vietnamese tone modeling through the LLM's language understanding.

---

## Further Reading

- Gold, B., Morgan, N., & Ellis, D. (2011). *Speech and Audio Signal Processing*. Wiley.
- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
- Davis, S. B., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE TASLP*, 28(4).
- Huang, X., Acero, A., & Hon, H.-W. (2001). *Spoken Language Processing*. Prentice Hall.
- Le, T. T., et al. (2023). Vietnamese speech corpus and acoustic model for speech recognition. *INTERSPEECH*.
