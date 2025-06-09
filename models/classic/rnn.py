import torch
import torch.nn as nn

class RNNForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, rnn_type='RNN'):
        super(RNNForecast, self).__init__()
        rnn_type = rnn_type.upper()
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [B, T, H]
        out = self.fc(out[:, -1, :])  # Last time step only
        return out