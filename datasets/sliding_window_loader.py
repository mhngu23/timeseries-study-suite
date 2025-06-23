import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, input_len=24, pred_len=1, normalize=True, stats=None):
        # Convert input to tensors
        self.X = torch.tensor(X.values, dtype=torch.float32) if hasattr(X, "values") else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if hasattr(y, "values") else torch.tensor(y, dtype=torch.float32)

        self.input_len = input_len
        self.pred_len = pred_len

        # Apply normalization
        self.normalize = normalize
        if normalize:
            if stats is not None:
                self.x_mean = torch.tensor(stats['x_mean'].values if hasattr(stats['x_mean'], 'values') else stats['x_mean'], dtype=torch.float32)
                self.x_std = torch.tensor(stats['x_std'].values if hasattr(stats['x_std'], 'values') else stats['x_std'], dtype=torch.float32)
                self.y_mean = torch.tensor(stats['y_mean'].values if hasattr(stats['y_mean'], 'values') else stats['y_mean'], dtype=torch.float32)
                self.y_std = torch.tensor(stats['y_std'].values if hasattr(stats['y_std'], 'values') else stats['y_std'], dtype=torch.float32)
            else:
                # compute on this dataset (i.e., training)
                self.x_mean = self.X.mean(0)
                self.x_std  = self.X.std(0) + 1e-8
                self.y_mean = self.y.mean()
                self.y_std  = self.y.std() + 1e-8
            # then apply
            self.X = (self.X - self.x_mean) / self.x_std
            self.y = (self.y - self.y_mean) / self.y_std
    def __len__(self):
        return len(self.X) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.input_len]
        y_seq = self.y[idx + self.input_len : idx + self.input_len + self.pred_len]

        assert x_seq.shape[0] == self.input_len, f"x_seq length {x_seq.shape[0]} != input_len {self.input_len}"
        assert y_seq.shape[0] == self.pred_len, f"y_seq length {y_seq.shape[0]} != pred_len {self.pred_len}"

        return x_seq, y_seq
