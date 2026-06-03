from torch import nn

from .base import (
    BinaryClassifierHead,
    DLBaselineBase,
    SelfAttentionBlock,
    TemporalAttentionPool,
    TemporalConvEncoder,
)


class DLCNNLSTMAT1(DLBaselineBase):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.cnn = TemporalConvEncoder(self.weekly_input_dim, self.hidden_size, self.dropout)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=param_dict.get("dl_num_layers", 1),
            batch_first=True,
            dropout=self.dropout if param_dict.get("dl_num_layers", 1) > 1 else 0.0,
        )
        self.attention = SelfAttentionBlock(
            self.hidden_size,
            param_dict.get("dl_attn_heads", 4),
            param_dict.get("dl_attn_dropout", 0.1),
        )
        self.attention_pool = TemporalAttentionPool(self.hidden_size)
        self.classifier = BinaryClassifierHead(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, sub_graph):
        weekly_seq = self.reshape_weekly(sub_graph)
        conv_seq = self.cnn(weekly_seq)
        lstm_out, _ = self.lstm(conv_seq)
        attn_out = self.attention(lstm_out)
        context, _ = self.attention_pool(attn_out)
        return self.classifier(context)
