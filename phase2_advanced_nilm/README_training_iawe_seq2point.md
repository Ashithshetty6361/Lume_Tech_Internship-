# Architecture Comparison: Seq2Point Models

This repository contains `training_iawe_seq2point.ipynb`, which evaluates four deep learning architectures for Non-Intrusive Load Monitoring (NILM) on the iAWE dataset. The goal is to determine the best temporal feature extractor for disaggregating the Air Conditioner (AC) and Motor appliances.

## 1. Windowing Strategy Explained
The models use a **Sequence-to-Point (Seq2Point)** sliding window approach.
- **How it works:** Instead of predicting a sequence of outputs from a sequence of inputs, the network takes a temporal window of aggregate power readings (a sequence) and predicts the power state of the appliance at the exact **midpoint** of that window (a point). The window then slides forward by the stride length.
- **Window Size (`WINDOW = 99`):** A window of 99 time steps is used. Because the data is downsampled to 6-second intervals, 99 steps represent exactly **9.9 minutes** of electrical history. This size was chosen because it is long enough to capture an appliance's complete turn-on transient spike and its subsequent stabilization phase, providing the network with the physical context of the appliance state.
- **Stride (`STRIDE = 4`):** The window shifts by 4 steps (24 seconds) for each training sample, generating overlapping sequences that maximize the amount of training data extracted from the timeline.

## 2. Input and Output Parameters
- **Input Shape:** `(99, 10)` - A 2D tensor comprising 99 time steps across 10 engineered features (Aggregate, Vrms, Irms, Reactive_Q, Apparent_Power, Power_Factor, Delta_P, Rolling_Mean, Rolling_Std, Hour).
- **Output Shape:** `(1,)` - A single continuous value representing the estimated power (in Watts) of the target appliance at step 49 (the midpoint).

## 3. Model Architectures Details
We trained 1 model per appliance per architecture (8 models total).

### 1. Seq2Point CNN (The Benchmark)
A classic 1D Convolutional Neural Network designed to extract localized temporal features (spikes and drops).
- **Parameters:** ~1.5 Million
- **Layers:**
  - `GaussianNoise(0.03)` for data augmentation
  - `Conv1D`: 30 filters, kernel size 10, ReLU + `BatchNormalization`
  - `Conv1D`: 30 filters, kernel size 8, ReLU + `BatchNormalization`
  - `Conv1D`: 40 filters, kernel size 6, ReLU + `BatchNormalization`
  - `Conv1D`: 50 filters, kernel size 5, ReLU
  - `Conv1D`: 50 filters, kernel size 5, ReLU
  - `Flatten`
  - `Dense`: 1024 neurons, ReLU + `Dropout(0.2)`
  - `Dense`: 1 neuron (Linear Output)

### 2. ResWaveNet-Transformer
A hybrid network using Dilated Convolutions (to increase receptive field without losing resolution) and Multi-Head Attention.
- **Parameters:** ~1.2 Million
- **Layers:** Dilated Residual Blocks (dilation rates 1, 2, 4, 8, 16, 32) with 64 filters each, followed by a `MultiHeadAttention` block (4 heads, key dimension 32), and Dense layers (256 -> 128 -> 1).

### 3. CNN-BiGRU-Attention
A recurrent variant that uses CNNs for feature extraction and a Bidirectional GRU to read the sequence in both directions.
- **Parameters:** ~1.8 Million
- **Layers:** 3x `Conv1D` (64 filters, dilations 1, 2, 4) $\rightarrow$ `Bidirectional(GRU(96))` $\rightarrow$ `MultiHeadAttention` (4 heads) $\rightarrow$ Dense classification block.

### 4. MLP Baseline (Control)
A deep feedforward network with no temporal context.
- **Parameters:** ~1.0 Million
- **Layers:** `Flatten` $\rightarrow$ `Dense(1024)` $\rightarrow$ `Dense(512)` $\rightarrow$ `Dense(128)` $\rightarrow$ `Dense(1)`.

## 4. Results & Benchmark Metrics
The models are evaluated using Huber loss for robust regression against noise. Below are the actual evaluation metrics obtained from the architecture comparison:

| Model | Appliance | MAE(W) | RMSE(W) | R² | NDE | DA | Accuracy | F1 | SAE |
|---|---|---|---|---|---|---|---|---|---|
| **Seq2Point CNN** | **AC** | **6.4** | **80.0** | **0.9697** | **0.0280** | **97.6%** | **99.5%** | **0.9848** | **0.0014** |
| | Motor | 20.1 | 35.6 | 0.4216 | 0.2341 | 82.3% | 82.3% | 0.7975 | 0.0485 |
| **CNN-BiLSTM-Attention** | **AC** | 7.8 | 82.3 | 0.9679 | 0.0297 | 97.0% | 99.5% | 0.9822 | 0.0064 |
| | Motor | 20.7 | 36.0 | 0.4072 | 0.2399 | 81.8% | 81.9% | 0.7924 | 0.0559 |
| **MLP Baseline** | **AC** | 16.6 | 104.1 | 0.9487 | 0.0475 | 93.6% | 99.4% | 0.9809 | 0.0060 |
| | Motor | 22.8 | 38.4 | 0.3260 | 0.2728 | 79.9% | 79.9% | 0.7693 | 0.0699 |
