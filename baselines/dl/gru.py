from torch import nn

from .base import BinaryClassifierHead, DLBaselineBase


class DLGRU(DLBaselineBase):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.gru = nn.GRU(
            input_size=self.weekly_input_dim,
            hidden_size=self.hidden_size,
            num_layers=param_dict.get("dl_num_layers", 1),
            batch_first=True,
            dropout=self.dropout if param_dict.get("dl_num_layers", 1) > 1 else 0.0,
        )
        self.classifier = BinaryClassifierHead(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, sub_graph):
        weekly_seq = self.reshape_weekly(sub_graph)
        gru_out, _ = self.gru(weekly_seq)
        return self.classifier(gru_out[:, -1, :])
