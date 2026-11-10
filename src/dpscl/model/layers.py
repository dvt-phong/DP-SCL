"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Shared neural-network layers and temporal augmentations used by the
  DP-SCL model.

Reference sources:
  DP-SCL paper:
  "Student dropout prediction in online courses based on supervised contrastive
  learning", Doan Van Thanh Phong et al., HUFLIT Journal of Science, 2026:
  https://hjs.huflit.edu.vn/index.php/hjs/article/view/448

  Supervised Contrastive Learning, Khosla et al., NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
"""

import torch
from torch import nn


class LearnableQueryPool(nn.Module):
    """Attentive pooling from the DP-SCL paper, implemented with MHA."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        context, weights = self.attn(query, x, x)
        return context.squeeze(1), weights.squeeze(1)


class AugmentationModule(nn.Module):
    """Create the two DP-SCL contrastive views (time masking and feature masking)."""

    def __init__(self, time_mask_ratio=0.15, feat_mask_ratio=0.15, noise_std=0.05):
        super().__init__()
        self.time_mask_ratio = time_mask_ratio
        self.feat_mask_ratio = feat_mask_ratio
        self.noise_std = noise_std

    def _time_view(self, x):
        batch_size, timesteps_count, _ = x.shape
        out = x.clone()
        time_mask = (torch.rand(batch_size, timesteps_count, 1, device=x.device) > self.time_mask_ratio).float()
        out = out * time_mask
        return out + torch.randn_like(out) * self.noise_std

    def _feature_view(self, x):
        batch_size, _, feature_count = x.shape
        out = x.clone()
        feat_mask = (torch.rand(batch_size, 1, feature_count, device=x.device) > self.feat_mask_ratio).float()
        out = out * feat_mask
        return out + torch.randn_like(out) * self.noise_std

    def forward(self, x):
        return self._time_view(x), self._feature_view(x)

