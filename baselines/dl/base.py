import torch
from torch import nn


class DLBaselineBase(nn.Module):
    def __init__(self, param_dict):
        super().__init__()
        self.activity_num = param_dict.get("activity_num", 22)
        self.week_count = param_dict.get("week_count", 5)
        self.days_per_week = param_dict.get(
            "days_per_week",
            param_dict.get("sta_day", 35) // self.week_count,
        )
        self.weekly_input_dim = self.days_per_week * self.activity_num
        self.hidden_size = param_dict.get("dl_hidden_size", param_dict.get("lstm_hidden_features", 128))
        self.dropout = param_dict.get("dl_dropout", 0.3)

    def reshape_weekly(self, sub_graph):
        seq_feat = sub_graph["seq_feat"] if isinstance(sub_graph, dict) else sub_graph
        batch_size = seq_feat.shape[0]

        if seq_feat.dim() == 2:
            return seq_feat.view(
                batch_size,
                self.week_count,
                self.days_per_week,
                self.activity_num,
            ).view(batch_size, self.week_count, self.weekly_input_dim)

        if seq_feat.dim() == 4:
            return seq_feat.view(batch_size, self.week_count, self.weekly_input_dim)

        if seq_feat.dim() == 3:
            return seq_feat

        raise ValueError(f"Unsupported seq_feat shape: {tuple(seq_feat.shape)}")


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, weekly_seq):
        return self.net(weekly_seq.transpose(1, 2)).transpose(1, 2)


class BinaryClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        return self.net(features)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TemporalAttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.score(x), dim=1)
        pooled = (attn_weights * x).sum(dim=1)
        return pooled, attn_weights.squeeze(-1)


class CNNRecurrentBaseline(DLBaselineBase):
    recurrent_cls = nn.LSTM

    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.cnn = TemporalConvEncoder(self.weekly_input_dim, self.hidden_size, self.dropout)
        self.recurrent = self.recurrent_cls(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=param_dict.get("dl_num_layers", 1),
            batch_first=True,
            dropout=self.dropout if param_dict.get("dl_num_layers", 1) > 1 else 0.0,
        )
        self.classifier = BinaryClassifierHead(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, sub_graph):
        weekly_seq = self.reshape_weekly(sub_graph)
        conv_seq = self.cnn(weekly_seq)
        recurrent_out, _ = self.recurrent(conv_seq)
        return self.classifier(recurrent_out[:, -1, :])
