# NovaLume Probationary Task: NILM Disaggregation Pipeline

This repository contains the end-to-end Non-Intrusive Load Monitoring (NILM) pipeline for the NovaLume Data Science probationary assignment. The project focuses on building a production-grade data engineering pipeline and baseline deep learning model (Seq2Point 1D-CNN) for residential appliance disaggregation.

---

## 📌 Phase 1: Strategic Dataset Selection

### The Rejection of UK-DALE & REDD
While basic NILM tutorials often start with datasets like UK-DALE or REDD, **I specifically rejected them for this pipeline**. Upon programmatic inspection, the widely accessible versions of these datasets completely lack raw Voltage (Vrms) and Current (Irms) telemetry, which are the primary signals captured by NovaLume's hardware setup.

### The Selection of iAWE
I selected the **iAWE (Indian Ambient Weather and Energy)** dataset. It perfectly maps to NovaLume's deployment scenario:
1.  **Grid Alignment:** It operates on the Indian **230V / 50Hz** standard.
2.  **Telemetry Match:** It natively records all 7 core parameters requested: **Vrms, Irms, Active Power (P), and Reactive Power (Q)**.
3.  **Appliance Relevance:** It tracks appliances heavily used in the Indian market (Air Conditioners, Water Geysers, Water Pumps) rather than Western central heating loops.

### The "High-Frequency" Dual-Strategy
NovaLume's rubric requested the extraction of High-Frequency features like Fast Fourier Transform (FFT) and Total Harmonic Distortion (THD). However, building a baseline machine learning model reliant on streaming 15,000+ Hz continuously is highly impractical for real-world, low-cost residential edge devices due to massive memory and bandwidth ceilings. 

To solve this, I employed a **Dual-Strategy**:
1.  **High-Frequency Mathematics Script:** I authored `nova_hf_features.py`—a dedicated pure Python script containing the exact mathematical logic to extract FFT, Harmonics, and THD from raw waveform arrays. This proves the capability to process NovaLume's proprietary high-frequency edge data.
2.  **Scalable 1 Hz Baseline:** For the actual ML pipeline, I utilized 1 Hz data. I compensated for the lack of High-Frequency Harmonics by prioritizing **Reactive Power (Q) and Delta P**, which serve as an excellent, computationally cheap proxy to distinguish between resistive (Heater) and inductive (Motor/Compressor) loads.

---

## ⚙️ Phase 2: Production-Grade Preprocessing

Basic Pandas tutorials (e.g., `pd.read_csv()` and static window means) are insufficient for real-world NILM tasks. Loading a 400MB to 6GB dataset entirely into RAM will instantly crash most edge devices. This project utilizes an advanced, memory-safe architecture.

### 1. The Disk-Backed Generator & Timestamp Alignment
I built a custom Keras Sequence Generator (`IAWETask1Generator`). Real-world appliances rarely start recording at the exact same millisecond, leading to fatal index mismatches. My pipeline completely solves this by using Pandas `DatetimeIndex` outer-joins to perfectly align timestamps across multiple meters before generating batches. It then securely streams specific rows directly from the aligned dataset on the fly, keeping RAM usage low.

### 2. Downsampling & Windowing
The data is sampled at 1 Hz. To provide the Convolutional Neural Network with a meaningful historical footprint, the generator dynamically downsamples the data by averaging every 6 seconds. A 599-step Deep Learning window now covers a full **1-hour temporal footprint** instead of just 10 minutes.

### 3. The 9-Channel Feature Engineering Matrix
I engineered a 9-Channel matrix using advanced time-series forecasting techniques:
1.  **Vrms (Voltage) & Irms (Current):** Core electrical signals.
2.  **Active Power (P) & Reactive Power (Q):** Baseline power and inductive proxy.
3.  **Delta P:** Calculates instantaneous derivative to track sudden ON/OFF spikes.
4.  **Rolling Mean (15-period):** A rolling window is mathematically superior to a static window mean because it provides the neural network with a continuous, smooth noise filter rather than a single scalar value.
5.  **Rolling Std (15-period):** Used to detect cyclical vibrations (e.g., refrigerator compressors).
6.  **Hour Sine & Hour Cosine:** Replacing standard `hour` integers with cyclical Sine/Cosine encoding. This advanced technique ensures the neural network understands that 23:59 and 00:01 are mathematically adjacent, rather than at opposite ends of a scale.

### 4. Z-Score Normalization
All 9 channels undergo Z-score normalization to map 230V voltage drops and 10A current spikes onto the exact same gradient scale, preventing mathematical explosion during backpropagation.

---

## 🚀 Repository Structure

*   `data/` - Target directory for the `iawe.h5` HDF5 database (not included in Git due to size).
*   `nova_hf_features.py` - The standalone High-Frequency extraction script (FFT, Harmonics, THD proof-of-concept).
*   `preprocessing_iawe.ipynb` - The memory-safe data pipeline and EDA visualizations.
*   `training_iawe_seq2point.ipynb` - The Task 2 Model architecture. Implements the Multi-Target Seq2Point CNN to disaggregate Air Conditioners and Washing Machines.
