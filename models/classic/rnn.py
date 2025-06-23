import torch
import torch.nn as nn

class RNNForecast(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        rnn_type='RNN',
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type.upper()]
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            output_dim
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
