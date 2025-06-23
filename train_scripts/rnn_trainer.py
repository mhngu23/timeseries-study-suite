import argparse
import torch

from datasets.sliding_window_loader import SlidingWindowDataset
from models.classic.rnn import RNNForecast
from data.loaders.processed_loader import load_processed_ett
from train_scripts.train_loop import train_model


def main(args):
    # ✅ Load processed training data
    X_train, y_train = load_processed_ett(args.train_path, target_col="OT", use_target=False)
    train_stats = {
    'x_mean': X_train.mean(0),
    'x_std':  X_train.std(0) + 1e-8,
    'y_mean': y_train.mean(),
    'y_std':  y_train.std() + 1e-8,
    }
    print(train_stats)

    train_dataset = SlidingWindowDataset(X_train, y_train, input_len=args.input_len, pred_len=args.pred_len, normalize=True, stats=train_stats)


    # 🧪 Load validation data if provided
    val_dataset = None
    if args.val_path:
        X_val, y_val = load_processed_ett(args.val_path, target_col="OT", use_target=False)
        val_dataset = SlidingWindowDataset(X_val, y_val, input_len=args.input_len, pred_len=args.pred_len, normalize=True, stats=train_stats)


    # 🧠 Define the RNN model
    model = RNNForecast(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.pred_len,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        dropout=0.1,  
        bidirectional=True
    )

    # 🚂 Train the model
    train_model(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path=args.save_path,
        model_name=args.model_name,
        resume=args.resume,
        patience=args.patience
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN Forecasting Model on ETTh1")

    parser.add_argument("--train_path", type=str, required=True, help="Path to processed training CSV")
    parser.add_argument("--val_path", type=str, default=None, help="Path to processed validation CSV")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Directory to save model checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Model filename to save (e.g., rnn_etth1.pt)")

    parser.add_argument("--input_len", type=int, default=24, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=1, help="Prediction horizon")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"], help="Type of RNN cell")

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")

    args = parser.parse_args()
    main(args)
