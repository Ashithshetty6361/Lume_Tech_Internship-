# NovaLume NILM Disaggregation Pipeline

This repository contains the end-to-end Non-Intrusive Load Monitoring (NILM) pipeline designed for NovaLume's edge deployment. The pipeline extracts distinct appliance power usage from a single aggregate smart meter reading.

## Project Structure
- `preprocessing_iawe.ipynb`: Handles 400MB+ data files, performs outer-join timestamp alignment, and aggressively downsamples 1 Hz data into 6-second buckets to ensure memory safety. Engineers 10 electrical features.
- `training_iawe_seq2point.ipynb`: Compares 4 distinct deep learning architectures (including a classic Seq2Point CNN and a Transformer variant) to find the best per-appliance feature extractor. *(See `README_training_iawe_seq2point.md` for full details)*.
- `training_best_model.ipynb`: The final production-ready model. A single **CNN-BiLSTM-Attention** architecture that predicts both Air Conditioner and Motor power simultaneously, drastically reducing compute overhead. *(See `README_training_best_model.md` for full details)*.
- `nova_hf_features.py`: A standalone script proving high-frequency FFT and THD mathematics.

## Modeling Highlight: Sliding Window
We employ a Sequence-to-Point sliding window. A 99-step window (representing 9.9 minutes of 6-second data) slides over the timeline. The network uses these 9.9 minutes of context to predict the appliance's state at the exact midpoint, achieving an MAE of ~72W for ACs and ~15W for Motors.
