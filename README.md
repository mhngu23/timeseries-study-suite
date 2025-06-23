# 🧠 timeseries-study-suite

A personal repository for studying and implementing time series forecasting models — from foundational RNNs to the latest state-of-the-art architectures like PatchTST, FEDformer, TimesNet, and Mamba. This project is designed for educational purposes, with a weekly learning schedule of 3 hours/week.

---

## ✅ Progress Overview

| Week | Focus | Status |
|------|-------|--------|
| 1    | Project Setup | ✅ Done |
| 2    | Data Pipeline (ETTh1, Weather, etc.) | ✅ Done |
| 3    | RNN + Training Loop | ✅ Done |
| 4    | Transformer + Informer | ⏳ Coming Soon |
| 5    | Training + Evaluation | ⏳ Coming Soon |
| 6    | DLinear + Autoformer | ⏳ Coming Soon |
| 7    | DLinear & Autoformer Evaluation | ⏳ Coming Soon |
| 8    | FEDformer + iTransformer | ⏳ Coming Soon |
| 9    | TimeMixer + TSMixer | ⏳ Coming Soon |
| 10   | PatchTST + TimesNet | ⏳ Coming Soon |
| 11   | Crossformer + Mamba | ⏳ Coming Soon |
| 12   | Final Comparison + Leaderboard | ⏳ Coming Soon |

---

## 📁 Project Structure

```bash
timeseries-study-suite/
│
├── README.md
├── requirements.txt
├── config/
│   └── model_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── loaders/
├── models/
│   ├── classic/          # RNN, vanilla Transformer
│   ├── linear/           # DLinear
│   ├── transformer_sota/ # Informer, Autoformer, FEDformer, etc.
│   ├── mixer/            # TimeMixer, TSMixer, PatchTST
│   ├── mamba/            # Mamba, MICN
├── datasets/
│   └── sliding_window_loader.py
├── train/
│   ├── rnn_trainer.py
│   ├── train_loop.py
│   └── trainer_utils.py
├── notebooks/
│   └── visualization.ipynb
```

---

## 📊 Datasets

- **ETTh** (Electricity Transformer Temperature Hour)
- **ETTm** (Electricity Transformer Temperature Hour)
- (More to be added later)

---

## 🔬 Models To Be Implemented

- ✅ RNN (baseline)
- ✅ Transformer (vanilla)
- ✅ Informer
- ✅ DLinear
- ✅ Autoformer
- ✅ FEDformer
- ✅ iTransformer
- ✅ TimeMixer
- ✅ TSMixer
- ✅ PatchTST
- ✅ TimesNet
- ✅ Crossformer
- ✅ Mamba

---

## 🧪 Metrics

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- SMAPE (Optional)

---

## 📅 Weekly Goals

All code is developed in modular components to make comparison and experimentation easier. The goal is to end with a reproducible leaderboard comparing each model on shared datasets.


---
## 🐍 Environment

```
conda create -n ts-study-env python=3.9 -y
conda activate ts-study-env
pip install -r requirements.txt
```

## Example

```
PYTHONPATH=. python train_scripts/rnn_trainer.py     --train_path data/processed/ETTh1/train.csv     --val_path data/processed/ETTh1/val.csv     --save_path /scratch/s223669184/timeseries-study-suite/checkpoints     --model_name rnn_lstm_etth1.pt     --input_len 96     --pred_len 1     --epochs 100     --batch_size 64     --lr 1e-5     --hidden_dim 128     --num_layers 3     --rnn_type lstm     --pa
tience 20
```

---