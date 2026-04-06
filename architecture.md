# ResNetConformerSED Architecture

This document explains the structure of `ResNetConformerSED` implemented in `src/model.py`.

---

## 1. High-Level Overview

`ResNetConformerSED` is a **frame-level sound event detection (SED)** model composed of:

1. **Feature extraction** from waveform (MelSpectrogram or fbank).
2. **2D CNN front-end with residual blocks** (for melspec path).
3. **Temporal modeling with stacked Conformer blocks**.
4. **Frame-wise classification head** producing logits for each class.

Input:

- `waveform`: shape **[B, S]** (batch, samples)

Output:

- `logits`: shape **[B, T, C]** (batch, time frames, classes)

---

## 2. Module-by-Module Structure

## 2.1 Feature Extraction (`extract_features`)

### Option A: `feature_extractor="melspec"` (main path)

1. `torchaudio.transforms.MelSpectrogram`
   - Params include `n_fft`, `win_length`, `hop_length`, `n_mels`, `sample_rate`.
2. `AmplitudeToDB`.
3. Optional SpecAug (train-time):
   - `FrequencyMasking(freq_mask_param)`
   - `TimeMasking(time_mask_param)`
4. Per-sample normalization:
   - `(x - mean) / (std + 1e-5)` over mel/time dims.

Result shape: **[B, M, T]**

### Option B: `feature_extractor="fbank"`

1. `kaldi.fbank(...)` per sample, `num_mel_bins=128`.
2. Stack to tensor and normalize along time dimension.

Result shape: **[B, T', D]**

> Note: In current code, the CNN branch is used only in `melspec` mode.

---

## 2.2 CNN Front-End (melspec branch)

Given `x: [B, M, T]`:

1. `x.unsqueeze(1)` -> **[B, 1, M, T]**
2. `cnn_stem`:
   - Conv2d(1 -> 32, k=3, s=(1,1), p=1)
   - BN + ReLU
3. `cnn` (three residual blocks):
   - `ResBlock2D(32, 64, stride=(2,1))`   -> mel reduced by 2
   - `ResBlock2D(64, 128, stride=(2,1))`  -> mel reduced by another 2
   - `ResBlock2D(128, 128, stride=(1,1))`

Output is approximately **[B, 128, M/4, T]**.

Then frequency pooling:

- `x.mean(dim=2)` -> **[B, 128, T]**
- `transpose(1,2)` -> **[B, T, 128]**

This converts 2D time-frequency features to frame sequence features.

---

## 2.3 Projection Layer

`self.proj = Linear(proj_dim -> conformer_dim) + Dropout`

- `proj_dim` depends on frontend:
  - melspec: 128
  - fbank: 128
  - wav2vec2: 768 (reserved in code)
  - panns: 256 (reserved in code)

After projection:

- **[B, T, conformer_dim]**

---

## 2.4 Temporal Backbone: Conformer Stack

`self.conformer` is `ModuleList` of `ConformerBlock` repeated `conformer_layers` times.

Each `ConformerBlock` performs:

1. **FFN (half-step residual)**
2. **Multi-head self-attention** (batch-first)
3. **Convolution module**:
   - LayerNorm -> pointwise Conv1d -> GLU
   - depthwise Conv1d (kernel=`conv_kernel`)
   - BatchNorm1d + SiLU
   - pointwise Conv1d + dropout
4. **Second FFN (half-step residual)**
5. **Final LayerNorm**

Sequence shape remains **[B, T, conformer_dim]**.

---

## 2.5 Classification Head

`self.classifier = Linear(conformer_dim -> num_classes)`

Applied on every frame:

- input: **[B, T, conformer_dim]**
- output logits: **[B, T, C]**

Training typically uses `BCEWithLogitsLoss` in `src/train.py`.

---

## 3. Data Flow Summary (melspec path)

```text
waveform [B,S]
  -> MelSpectrogram + dB + (SpecAug in train) + normalize
  -> [B,M,T]
  -> unsqueeze
  -> [B,1,M,T]
  -> CNN stem + ResBlocks
  -> [B,128,M/4,T]
  -> mean over frequency
  -> [B,128,T]
  -> transpose
  -> [B,T,128]
  -> projection
  -> [B,T,D]
  -> Conformer x L
  -> [B,T,D]
  -> classifier
  -> logits [B,T,C]
```

`D = conformer_dim`, `C = num_classes`.

---

## 4. Design Intuition

- **CNN front-end** learns local time-frequency event cues.
- **Residual blocks** improve optimization and feature depth.
- **Conformer blocks** capture both long-range dependencies (attention) and local temporal patterns (depthwise conv).
- **Frame-wise logits** support weak/strong temporal event localization.

---

## 5. Notes About Current Implementation

1. `melspec` is the most complete and actively used path.
2. `fbank` feature extraction is implemented, but CNN handling is currently bypassed in `forward` (`pass` branch).
3. `wav2vec2`/`panns` dimensions are listed for projection compatibility, but extraction path for these is not implemented in `ResNetConformerSED.extract_features` currently.
4. There is an optional `use_wenet` switch branch in `forward`; default behavior is the internal Conformer stack.

---

## 6. Key Constructor Parameters

- `num_classes`: number of output event classes.
- `sample_rate`, `n_fft`, `win_length`, `hop_length`, `n_mels`: spectral frontend settings.
- `conformer_dim`, `conformer_layers`, `conformer_heads`, `dropout`: temporal backbone capacity.
- `use_specaug`, `freq_mask_param`, `time_mask_param`: augmentation control.
- `feature_extractor`: frontend selection (`melspec` recommended in current code).
