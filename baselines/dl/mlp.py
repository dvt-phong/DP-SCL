from torch import nn


class MLPBaseline(nn.Module):
    """Raw-feature MLP baseline for BCE-only ablation."""

    def __init__(self, param_dict):
        super().__init__()
        activity_num = param_dict.get("activity_num", 22)
        week_count = param_dict.get("week_count", 5)
        days_per_week = param_dict.get(
            "days_per_week",
            param_dict.get("sta_day", 35) // week_count,
        )
        input_dim = param_dict.get("mlp_input_dim", week_count * days_per_week * activity_num)
        hidden_dim = param_dict.get("mlp_hidden_dim", 64)
        dropout = param_dict.get("mlp_dropout", 0.3)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sub_graph):
        seq_feat = sub_graph["seq_feat"] if isinstance(sub_graph, dict) else sub_graph
        if isinstance(sub_graph, dict):
            batch_size = sub_graph["batch_size"]
            seq_feat = seq_feat[:batch_size]
        else:
            batch_size = seq_feat.shape[0]
        return self.net(seq_feat.reshape(batch_size, -1))
