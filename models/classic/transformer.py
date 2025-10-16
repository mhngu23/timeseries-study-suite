import torch
import torch.nn as nn

class TransformerForecast(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        nhead=4,
        dropout=0.1,
        dim_feedforward=256,
        use_pos_encoding=True,
    ):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_fc(x)
        if self.use_pos_encoding:
            x = self.pos_encoder(x)

        out = self.transformer_encoder(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc_out(out)
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as in the original Transformer paper)."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
