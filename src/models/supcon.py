"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: DP-SCL model for MOOC dropout prediction with supervised
  contrastive learning.

Reference sources:
  DP-SCL manuscript supplied with this project:
  "Student Dropout Prediction in Online Courses Based on Supervised Contrastive
  Learning", Doan Van Thanh Phong et al.

  Supervised Contrastive Learning, Khosla et al., NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

  SupContrast PyTorch reference implementation by HobbitLong:
  https://github.com/HobbitLong/SupContrast

"""

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    ActionWeightedInput,
    AugmentationModule,
    EarlyPredictionMask,
    LearnableQueryPool,
)


class SupConEncoder(nn.Module):
    """DP-SCL encoder: LSTM -> MHA + residual norm -> attentive pooling."""

    def __init__(self, input_size=22, hidden_size=128, num_heads=4, dropout=0.1, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pool = LearnableQueryPool(hidden_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
        attn_out = self.layer_norm(attn_out + rnn_out)
        context, _ = self.pool(attn_out)
        return context


class SupConProjectionHead(nn.Module):
    """Two-layer projection head used only for supervised contrastive loss."""

    def __init__(self, in_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SupConClassifier(nn.Module):
    """Dropout classifier trained with BCEWithLogitsLoss."""

    def __init__(self, in_dim=128, hidden_dim=64, dropout=0.3, num_hidden_layers=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        dim = hidden_dim
        for _ in range(num_hidden_layers - 1):
            next_dim = max(dim // 2, 8)
            layers.extend([nn.Linear(dim, next_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = next_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DPSCLModel(nn.Module):
    """Project implementation of the DP-SCL forward path from the manuscript."""

    ENCODER_MAP = {
        "dp_scl": "lstm_attn",
        "supcon_lstm_attn": "lstm_attn",
    }

    def __init__(self, mode, param_dict):
        super().__init__()
        if mode not in self.ENCODER_MAP:
            raise ValueError(f"DPSCLModel only supports DP-SCL modes: {sorted(self.ENCODER_MAP)}. Got: {mode}")

        input_size = param_dict.get("activity_num", 22)
        hidden_size = param_dict.get("supcon_hidden_size", 128)
        proj_dim = param_dict.get("supcon_proj_dim", 128)
        mask_ratio = param_dict.get("supcon_mask_ratio", 0.15)
        noise_std = param_dict.get("supcon_noise_std", 0.05)
        num_heads = param_dict.get("supcon_attn_heads", 4)
        cls_dropout = param_dict.get("supcon_cls_dropout", 0.3)

        self.mode = mode
        self.week_count = param_dict.get("week_count", 5)
        self.days_per_week = param_dict.get("cnn_in_channels", 7)
        self.activity_num = input_size
        self.use_action_weight = param_dict.get("use_action_weight", False)
        self.use_early = param_dict.get("use_early_prediction", False)

        if self.use_action_weight:
            self.action_weighting = ActionWeightedInput(num_actions=input_size)
        if self.use_early:
            self.early_mask = EarlyPredictionMask(
                week_count=self.week_count,
                days_per_week=self.days_per_week,
                min_weeks=param_dict.get("early_min_weeks", 2),
            )

        self.augment = AugmentationModule(
            time_mask_ratio=mask_ratio,
            feat_mask_ratio=mask_ratio,
            noise_std=noise_std,
        )
        self.encoder = SupConEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=param_dict.get("supcon_num_layers", 1),
        )
        self.proj_head = SupConProjectionHead(in_dim=hidden_size, proj_dim=proj_dim)
        self.classifier = SupConClassifier(
            in_dim=hidden_size,
            hidden_dim=64,
            dropout=cls_dropout,
            num_hidden_layers=param_dict.get("supcon_cls_hidden_layers", 1),
        )

    def _preprocess(self, batch):
        batch_size = batch["batch_size"]
        seq_feat = batch["seq_feat"][:batch_size]
        total_steps = self.week_count * self.days_per_week
        x = seq_feat.view(batch_size, total_steps, self.activity_num)
        if self.use_action_weight:
            x = self.action_weighting(x)
        if self.use_early:
            x = self.early_mask(x)
        return x

    def forward_single(self, batch):
        x = self._preprocess(batch)
        return self.classifier(self.encoder(x))

    def forward(self, batch):
        x = self._preprocess(batch)
        if self.training:
            # Training follows the paper: two augmented views share the encoder.
            view1, view2 = self.augment(x)
            h1 = self.encoder(view1)
            h2 = self.encoder(view2)
            z1 = self.proj_head(h1)
            z2 = self.proj_head(h2)
            return self.classifier(h1), z1, z2

        h = self.encoder(x)
        return self.classifier(h)
