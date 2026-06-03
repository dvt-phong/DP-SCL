from torch import nn

from .base import BinaryClassifierHead, DLBaselineBase


class DLLSTMMHALQ(DLBaselineBase):
    """LSTM + MHA + Learnable Query Pooling baseline trained with BCE only."""

    def __init__(self, param_dict):
        super().__init__(param_dict)
        from src.models.supcon import SupConEncoder

        self.encoder = SupConEncoder(
            encoder_type="lstm_attn",
            input_size=self.activity_num,
            hidden_size=self.hidden_size,
            num_heads=param_dict.get("dl_attn_heads", 4),
            dropout=param_dict.get("dl_attn_dropout", 0.1),
            num_layers=param_dict.get("dl_num_layers", 1),
            seq_count=self.week_count * self.days_per_week,
        )
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
        features = self.encoder(daily_seq)
        return self.classifier(features)
