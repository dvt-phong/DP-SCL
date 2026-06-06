"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Shared neural-network layers, temporal augmentations, and supervised
  contrastive loss used by the DP-SCL model.

Reference sources:
  DP-SCL manuscript supplied with this project:
  "Student Dropout Prediction in Online Courses Based on Supervised Contrastive
  Learning", Doan Van Thanh Phong et al.
  
  Supervised Contrastive Learning, Khosla et al., NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

  SupContrast PyTorch reference implementation by HobbitLong:
  https://github.com/HobbitLong/SupContrast

  CA-TFHN MySelfAttention reference by codeds27:
  https://github.com/codeds27/CA-TFHN/blob/main/src/models.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class MySelfAttention(nn.Module):
    def __init__(self, week_count, input_features, num_attention_heads, attention_features):
        super().__init__()
        self.week_count = week_count
        self.input_features = input_features
        self.num_attention_heads = num_attention_heads
        self.attention_features = attention_features
        self.attention_head_size = int(attention_features / num_attention_heads)
        self.all_head_size = attention_features

        pe = torch.zeros((week_count, input_features))
        for i in range(1, pe.shape[0] + 1):
            for j in range(1, pe.shape[1] + 1):
                if j % 2 != 0:
                    twob = j - 1
                    expr = torch.exp(torch.tensor(twob * (-1 * torch.log(torch.tensor(10000 / pe.shape[0] + 1)))))
                    pe[i - 1][j - 1] = torch.cos(expr * i)
                else:
                    twob = j
                    expr = torch.exp(torch.tensor(twob * (-1 * torch.log(torch.tensor(10000 / pe.shape[0] + 1)))))
                    pe[i - 1][j - 1] = torch.sin(expr * i)
        self.register_buffer("PE", pe)
        self.key_layer = nn.Linear(input_features, attention_features)
        self.query_layer = nn.Linear(input_features, attention_features)
        self.value_layer = nn.Linear(input_features, attention_features)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_matrix):
        input_matrix = input_matrix + self.PE
        key = self.key_layer(input_matrix)
        query = self.query_layer(input_matrix)
        value = self.value_layer(input_matrix)
        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, device=input_matrix.device)
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_probs, value_heads)
        attention_context = attention_context.permute(0, 2, 1, 3).contiguous()
        attention_newsize = attention_context.size()[:-2] + (self.all_head_size,)
        return attention_context.view(*attention_newsize)


class LearnableQueryPool(nn.Module):
    """Attentive pooling from the DP-SCL manuscript, implemented with MHA."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        context, weights = self.attn(query, x, x)
        return context.squeeze(1), weights.squeeze(1)


class SupConLoss(nn.Module):
    """Supervised contrastive loss based on Khosla et al. and SupContrast."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.view(-1)
        if features.dim() == 3:
            batch_size, n_views, proj_dim = features.shape
            features = features.view(batch_size * n_views, proj_dim)
            labels = labels.repeat_interleave(n_views)

        batch_size = features.shape[0]
        if batch_size <= 1 or len(torch.unique(labels)) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        similarity = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / positive_count
        return -mean_log_prob.mean()


class ActionWeightedInput(nn.Module):
    def __init__(self, num_actions=22):
        super().__init__()
        self.num_actions = num_actions
        self.raw_weight = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x):
        weights = torch.softmax(self.raw_weight, dim=0) * self.num_actions
        return x * weights.unsqueeze(0).unsqueeze(0)

    def get_weights(self):
        return (torch.softmax(self.raw_weight, dim=0) * self.num_actions).detach().cpu()


class EarlyPredictionMask(nn.Module):
    def __init__(self, week_count=5, days_per_week=7, min_weeks=2):
        super().__init__()
        self.week_count = week_count
        self.days_per_week = days_per_week
        self.min_weeks = min_weeks
        self.eval_weeks = week_count

    def forward(self, x):
        batch_size, timesteps_count, _ = x.shape
        if self.training:
            keep = torch.randint(self.min_weeks, self.week_count + 1, (batch_size,), device=x.device)
        else:
            keep = torch.full((batch_size,), self.eval_weeks, dtype=torch.long, device=x.device)

        timesteps = torch.arange(timesteps_count, device=x.device).unsqueeze(0)
        keep_days = (keep * self.days_per_week).unsqueeze(1)
        mask = (timesteps < keep_days).float()
        return x * mask.unsqueeze(-1)

    def set_eval_weeks(self, n):
        self.eval_weeks = n


class AugmentationModule(nn.Module):
    """Create the two DP-SCL contrastive views described in the manuscript."""

    def __init__(self, time_mask_ratio=0.15, feat_mask_ratio=0.15, noise_std=0.05):
        super().__init__()
        self.time_mask_ratio = time_mask_ratio
        self.feat_mask_ratio = feat_mask_ratio
        self.noise_std = noise_std

    def _time_view(self, x):
        batch_size, timesteps_count, feature_count = x.shape
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
