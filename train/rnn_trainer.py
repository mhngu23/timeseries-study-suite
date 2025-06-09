import torch

from datasets.sliding_window_loader import SlidingWindowDataset
from models.classic.rnn import RNNForecast
from data.loaders.processed_loader import load_processed_ett
from train.train_loop import train_model

# ✅ Load processed data
train_path = "data/processed/ETTh1/train.csv"
X, y = load_processed_ett(train_path)

# print(X.shape, y.shape)  # Debugging output to check shapes
# Sliding window setup
input_len = 24
pred_len = 1
dataset = SlidingWindowDataset(X, y, input_len, pred_len)

# Model definition
model = RNNForecast(input_dim=X.shape[1], hidden_dim=64, output_dim=pred_len, rnn_type='lstm', num_layers=2)

# Universal trainer
train_model(
    model=model,
    dataset=dataset,
    epochs=100,
    batch_size=64,
    lr=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="checkpoints",
    model_name="rnn_etth1.pt"
)
