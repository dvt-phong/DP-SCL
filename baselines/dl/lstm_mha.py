import torch
from torch import nn

from .base import BinaryClassifierHead, DLBaselineBase


class DLLSTMMHA(DLBaselineBase):
    """LSTM + Multi-Head Attention + mean pooling baseline trained with BCE only."""

    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lstm = nn.LSTM(
            input_size=self.activity_num,
            hidden_size=self.hidden_size,
            num_layers=param_dict.get("dl_num_layers", 1),
            batch_first=True,
            dropout=self.dropout if param_dict.get("dl_num_layers", 1) > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=param_dict.get("dl_attn_heads", 4),
            dropout=param_dict.get("dl_attn_dropout", 0.1),
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = BinaryClassifierHead(self.hidden_size, self.hidden_size, self.dropout)

    def reshape_daily(self, sub_graph):
        seq_feat = sub_graph["seq_feat"] if isinstance(sub_graph, dict) else sub_graph
        batch_size = seq_feat.shape[0]

        if seq_feat.dim() == 2:
            return seq_feat.view(
                batch_size,
                self.week_count * self.days_per_week,
                self.activity_num,
            )

        if seq_feat.dim() == 4:
            return seq_feat.view(
                batch_size,
                self.week_count * self.days_per_week,
                self.activity_num,
            )

        if seq_feat.dim() == 3:
            return seq_feat

        raise ValueError(f"Unsupported seq_feat shape: {tuple(seq_feat.shape)}")

    def forward(self, sub_graph):
        daily_seq = self.reshape_daily(sub_graph)
        lstm_out, _ = self.lstm(daily_seq)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        features = torch.mean(attn_out, dim=1)
        return self.classifier(features)
