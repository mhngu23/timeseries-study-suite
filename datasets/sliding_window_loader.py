import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, input_len=24, pred_len=1):
        # Convert pandas DataFrame/Series to torch.Tensor directly
        self.X = torch.tensor(X.values, dtype=torch.float32) if hasattr(X, "values") else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if hasattr(y, "values") else torch.tensor(y, dtype=torch.float32)
        print(self.X.shape)
        exit()
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.input_len]
        y_seq = self.y[idx + self.input_len : idx + self.input_len + self.pred_len]

        assert x_seq.shape[0] == self.input_len, f"x_seq length {x_seq.shape[0]} != input_len {self.input_len}"
        assert y_seq.shape[0] == self.pred_len, f"y_seq length {y_seq.shape[0]} != pred_len {self.pred_len}"

        return x_seq, y_seq
