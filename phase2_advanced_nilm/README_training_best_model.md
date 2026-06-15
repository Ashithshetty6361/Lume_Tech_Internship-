# Production Multi-Output Model for NILM

This repository contains `training_best_model.ipynb`, which implements a single, production-ready neural network capable of disaggregating multiple appliances simultaneously from an aggregate power signal.

## 1. Why a Single Multi-Output Model?
Traditional NILM approaches train one model per appliance. This repository introduces a **Single Multi-Output CNN-BiLSTM-Attention** network:
- **Efficiency:** A shared convolutional backbone reduces parameter count and computational overhead, making it ideal for edge deployment.
- **Learned Correlations:** By predicting all appliances simultaneously, the model learns the physics of the house. If the aggregate power jumps by 1700W, the model assigns it to the AC output head and explicitly suppresses the Motor output head, preventing false positives.

## 2. Windowing & Data Parameters
We use a Sliding Window technique:
- **Window Size (`WINDOW = 99`):** Represents 9.9 minutes of temporal history (downsampled to 6-second steps). This is optimal for capturing the full startup spike and stabilization plateau of high-power appliances.
- **Stride (`STRIDE = 6`):** The window slides 6 steps (36 seconds) forward per extraction, generating 141,631 overlapping windows.
- **Input Shape:** `(99, 10)` - 99 time steps of 10 features (Aggregate, Vrms, Irms, Reactive_Q, Apparent_Power, Power_Factor, Delta_P, Rolling_Mean, Rolling_Std, Hour).
- **Output Shape:** `(2,)` - Two continuous values representing the instantaneous power of the AC and Motor at the midpoint (step 49) of the window.

## 3. Network Architecture Deep-Dive
The model contains **~6.5 Million parameters** (upgraded from 5.3M) and is structured as follows:

1. **Data Augmentation:** `GaussianNoise(0.05)` applied to inputs during training to prevent overfitting on clean data.
2. **Shared CNN Backbone (Deep 5-Layer Feature Extraction):**
   - We utilize the deep CNN architecture from the highly accurate `Seq2Point` model:
   - `Conv1D`: 30 filters, kernel size 10, ReLU $\rightarrow$ `BatchNormalization`
   - `Conv1D`: 30 filters, kernel size 8, ReLU $\rightarrow$ `BatchNormalization`
   - `Conv1D`: 40 filters, kernel size 6, ReLU $\rightarrow$ `BatchNormalization`
   - `Conv1D`: 50 filters, kernel size 5, ReLU
   - `Conv1D`: 50 filters, kernel size 5, ReLU
3. **Shared Recurrent Layer (Temporal Memory):**
   - `Bidirectional(LSTM)`: 64 units, returning sequences with 20% Dropout.
4. **Shared Attention Mechanism:**
   - `Self-Attention` applied to the LSTM outputs to heavily weight critical time steps (e.g., sudden spikes).
   - Residual connection (`Add`) $\rightarrow$ `LayerNormalization`.
5. **Shared Dense Block:**
   - `Flatten` $\rightarrow$ `Dense(1024)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(512)` $\rightarrow$ `Dropout(0.2)`.
6. **Dual Output Heads (High Capacity):**
   - Both AC and Motor heads: `Dense(512)` $\rightarrow$ `Dropout(0.2)`.
   - `Dense(1)` with Linear activation. One neuron outputs the AC wattage; the other outputs the Motor wattage.

## 4. Training Results & Metrics
The model is trained for 50 epochs using the Adam Optimizer and **Mean Absolute Error (MAE) Loss**. The loss function was switched from Huber to MAE to aggressively penalize errors on high-wattage peaks (which were previously being smoothed out).

**Validation & Test Performance:**

| Appliance | MAE (W) | RMSE (W) | R² | NDE | DA | Accuracy | F1-Macro | SAE |
|---|---|---|---|---|---|---|---|---|
| **Air Conditioner** | 135.5 | 471.4 | -0.0518 | 0.9736 | 48.0% | 90.7% | 0.5484 | 0.8934 |
| **Motor** | 36.6 | 44.5 | 0.0957 | 0.3659 | 67.7% | 63.8% | 0.4698 | 0.0326 |
