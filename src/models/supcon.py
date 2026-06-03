"""
SupCon Network Architecture — Framework 1.

[Kỹ thuật tổng quan: SupCon Network + Supervised Contrastive Learning
 Pipeline: Input → Augmentation (training) → Shared Encoder → 2 views (h1, h2)
         → ProjectionHead → (z1, z2) L2-normalized
         → Classifier(h1) → logits
         Loss = BCE(logits, y) + λ × SupConLoss(z1, z2, y)]

Hỗ trợ 6 modes:

| Mode                 | Encoder = Shared Encoder        | Kỹ thuật chi tiết                                              |
|----------------------|----------------------------------|----------------------------------------------------------------|
| supcon_lstm         | LSTM                            | LSTM unidirectional, lấy h_n[-1]                               |
| supcon_bilstm       | BiLSTM                          | BiLSTM, concat h_n[-2] + h_n[-1]                              |
| supcon_lstm_mha     | LSTM + Multi-Head Attention     | LSTM → nn.MultiheadAttention (4 heads) + LayerNorm + residual → mean pool |
| supcon_lstm_attn    | LSTM + Multi-Head Attention     | LSTM → nn.MultiheadAttention (4 heads) + LayerNorm + residual → LearnableQueryPool |
| supcon_bilstm_attn  | BiLSTM + Multi-Head Attention   | BiLSTM → nn.MultiheadAttention (4 heads) + LayerNorm + residual → LearnableQueryPool |
| supcon_lstm_sa      | LSTM + Custom Self-Attention    | LSTM → MySelfAttention (sinusoidal PE, custom QKV) → mean pool |
| supcon_bilstm_sa    | BiLSTM + Custom Self-Attention  | BiLSTM → MySelfAttention (sinusoidal PE, custom QKV) → mean pool |

Giải thích viết tắt:
    - LSTM:    Long Short-Term Memory (unidirectional), h_n[-1] = last hidden state
    - BiLSTM:  Bidirectional LSTM, concat forward h_n[-2] + backward h_n[-1]
    - MHA:     nn.MultiheadAttention (PyTorch built-in, Transformer-style)
    - SA+PE:   MySelfAttention = custom Self-Attention + sinusoidal Position Encoding
    - LQP:     LearnableQueryPool = trainable query vector + nn.MultiheadAttention
    - SupCon:  Supervised Contrastive Loss (Khosla et al., 2020)

Training vs Inference:
    Training:  Augmentation ON → 2× forward qua shared encoder → return (logits, z1, z2)
    Inference: Augmentation OFF → 1× forward → return logits
"""
import torch
import torch.nn.functional as F
from torch import nn as nn

from .common import (
    MySelfAttention, LearnableQueryPool, AugmentationModule, SupConLoss,
    ActionWeightedInput, EarlyPredictionMask,
)


class SupConEncoder(nn.Module):
    """Shared encoder cho SupCon Network — 6 variants.

    | encoder_type   | Architecture                                     | Output  |
    |----------------|--------------------------------------------------| --------|
    | lstm           | LSTM(F→H), lấy h_n cuối                         | (B, H)  |
    | bilstm         | BiLSTM(F→H/2×2), concat h_n 2 chiều             | (B, H)  |
    | lstm_mha       | LSTM(F→H) → MHA(4 heads) → mean pool            | (B, H)  |
    | lstm_attn      | LSTM(F→H) → MHA(4 heads) → LearnableQueryPool   | (B, H)  |
    | bilstm_attn    | BiLSTM(F→H/2×2) → MHA(4 heads) → LearnableQueryPool | (B, H) |
    | lstm_sa        | LSTM(F→H) → MySelfAttn(PE) → mean pool           | (B, H)  |
    | bilstm_sa      | BiLSTM(F→H/2×2) → MySelfAttn(PE) → mean pool     | (B, H)  |

    Kỹ thuật chi tiết:
        - lstm / bilstm:       chỉ dùng last hidden state (không attention)
        - lstm_mha:                 MHA = nn.MultiheadAttention + LayerNorm
                                    + residual connection → mean pooling
        - lstm_attn / bilstm_attn:  MHA = nn.MultiheadAttention (standard Transformer MHA)
                                    + LayerNorm + residual connection → LearnableQueryPool
        - lstm_sa / bilstm_sa:    SA = MySelfAttention (custom QKV + sinusoidal Position Encoding)
                                    → mean pooling (average across time steps)
    """
    def __init__(self, encoder_type, input_size=22, hidden_size=128,
                 num_heads=4, dropout=0.1, num_layers=1, seq_count=35):
        super(SupConEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.hidden_size = hidden_size

        if encoder_type in ('lstm', 'lstm_mha', 'lstm_attn', 'lstm_sa'):
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0)

        if encoder_type in ('lstm_mha', 'lstm_attn', 'bilstm_attn'):
            # nn.MultiheadAttention + LayerNorm + residual.
            # lstm_mha uses mean pooling; *_attn modes use LearnableQueryPool.
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
            if encoder_type in ('lstm_attn', 'bilstm_attn'):
                self.pool = LearnableQueryPool(hidden_size)

        elif encoder_type in ('lstm_sa', 'bilstm_sa'):
            # MySelfAttention: custom self-attention + sinusoidal Position Encoding
            # output dim = attention_features = hidden_size (giữ nguyên dimension)
            self.self_attention = MySelfAttention(
                week_count=seq_count,           # seq_count = week_count × days_per_week = 35
                                                # MySelfAttention dùng param này làm sequence length cho PE
                input_features=hidden_size,     # LSTM/BiLSTM output dim
                num_attention_heads=1,          # single head (same head count used by the original self-attention baseline)
                attention_features=hidden_size  # output dim = hidden_size
            )

    def forward(self, x):
        """
        x: (B, T, input_size) e.g. (B, 35, 22)
        Returns: (B, hidden_size) e.g. (B, 128)
        """
        rnn_out, (h_n, _) = self.rnn(x)

        if self.encoder_type == 'lstm':
            return rnn_out[:, -1, :]

        elif self.encoder_type == 'bilstm':
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)

        elif self.encoder_type == 'lstm_mha':
            # MultiheadAttention + residual + LayerNorm → mean pool
            attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
            attn_out = self.layer_norm(attn_out + rnn_out)
            return torch.mean(attn_out, dim=1)

        elif self.encoder_type in ('lstm_attn', 'bilstm_attn'):
            # MultiheadAttention + residual + LayerNorm → LearnableQueryPool
            attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
            attn_out = self.layer_norm(attn_out + rnn_out)
            context, _ = self.pool(attn_out)
            return context

        else:  # lstm_sa, bilstm_sa
            # MySelfAttention (with Position Encoding) → mean pool
            attn_out = self.self_attention(rnn_out)  # (B, T, hidden_size)
            context = torch.mean(attn_out, dim=1)    # (B, hidden_size)
            return context


class SupConProjectionHead(nn.Module):
    """Projection Head cho SupCon: Linear → ReLU → Linear → L2 Normalize.
    Chiếu h ∈ R^d vào unit hypersphere để SupCon hoạt động đúng.

    [Kỹ thuật: MLP 2-layer Projection Head → L2 Normalization]
    """
    def __init__(self, in_dim=128, proj_dim=128):
        super(SupConProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)


class SupConClassifier(nn.Module):
    """Classifier cho SupCon: nhận h (representation), KHÔNG nhận z (projected).

    [Kỹ thuật: MLP classifier head
     num_hidden_layers=1: Linear(in→hid) → ReLU → Dropout → Linear(hid→1)
     num_hidden_layers=2: + thêm Linear(hid→hid//2) → ReLU → Dropout]

    Output là logits (chưa qua sigmoid) — dùng với BCEWithLogitsLoss.
    """
    def __init__(self, in_dim=128, hidden_dim=64, dropout=0.3, num_hidden_layers=1):
        super(SupConClassifier, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        dim = hidden_dim
        for _ in range(num_hidden_layers - 1):
            next_dim = max(dim // 2, 8)
            layers += [nn.Linear(dim, next_dim), nn.ReLU(), nn.Dropout(dropout)]
            dim = next_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SupConLGB(nn.Module):
    """SupCon Network cho DP-SCL — Framework 1.

    [Kỹ thuật: SupCon Network + Supervised Contrastive Learning (SupCon)]

    Kiến trúc:
        Input (B, 5, 7, 22) → flatten → (B, 35, 22)
        → Augmentation (training only) → view1, view2
        → Shared Encoder → h1, h2          (B, 128)
        → Shared ProjectionHead → z1, z2   (B, 128, L2-normalized)
        → Classifier(h1) → ŷ              (B, 1)
        Loss = BCE(ŷ, y) + λ × SupCon(z1, z2, y)

    Modes: dp_scl, tsn_supcon, supcon_lstm, supcon_bilstm, supcon_lstm_mha,
           supcon_lstm_attn, supcon_lstm_attn_lambda0,
           supcon_bilstm_attn, supcon_lstm_sa, supcon_bilstm_sa

    Training vs Inference:
        Training:  augmentation ON, 2× forward qua encoder, return (logits, z1, z2)
        Inference: augmentation OFF, 1× forward,             return logits
    """
    ENCODER_MAP = {
        'dp_scl': 'lstm_attn',
        'tsn_supcon': 'lstm_attn',
        'supcon_lstm': 'lstm',
        'supcon_bilstm': 'bilstm',
        'supcon_lstm_mha': 'lstm_mha',
        'supcon_lstm_attn': 'lstm_attn',
        'supcon_lstm_attn_lambda0': 'lstm_attn',
        'supcon_bilstm_attn': 'bilstm_attn',
        'supcon_lstm_sa': 'lstm_sa',
        'supcon_bilstm_sa': 'bilstm_sa',
        # Graph-enhanced variants (same encoder, + GraphSage fusion)
        'supcon_lstm_graph': 'lstm',
        'supcon_bilstm_graph': 'bilstm',
        'supcon_lstm_attn_graph': 'lstm_attn',
        'supcon_bilstm_attn_graph': 'bilstm_attn',
        'supcon_lstm_sa_graph': 'lstm_sa',
        'supcon_bilstm_sa_graph': 'bilstm_sa',
    }

    def __init__(self, mode, param_dict):
        super(SupConLGB, self).__init__()
        assert mode in self.ENCODER_MAP, \
            f"SupConLGB mode must be one of {list(self.ENCODER_MAP.keys())}. Got: {mode}"

        self.mode = mode
        encoder_type = self.ENCODER_MAP[mode]

        input_size = param_dict.get('activity_num', 22)
        hidden_size = param_dict.get('supcon_hidden_size', 128)
        proj_dim = param_dict.get('supcon_proj_dim', 128)
        temperature = param_dict.get('supcon_temperature', 0.07)
        mask_ratio = param_dict.get('supcon_mask_ratio', 0.15)
        noise_std = param_dict.get('supcon_noise_std', 0.05)
        num_heads = param_dict.get('supcon_attn_heads', 4)
        cls_dropout = param_dict.get('supcon_cls_dropout', 0.3)

        self.week_count = param_dict.get('week_count', 5)
        self.days_per_week = param_dict.get('cnn_in_channels', 7)
        self.activity_num = input_size

        # --- Optional: Action Weighting (--action-weight) ---
        self.use_action_weight = param_dict.get('use_action_weight', False)
        if self.use_action_weight:
            self.action_weighting = ActionWeightedInput(num_actions=input_size)

        # --- Optional: Early Prediction Mask (--early-prediction) ---
        self.use_early = param_dict.get('use_early_prediction', False)
        if self.use_early:
            self.early_mask = EarlyPredictionMask(
                week_count=self.week_count,
                days_per_week=self.days_per_week,
                min_weeks=param_dict.get('early_min_weeks', 2)
            )

        self.augment = AugmentationModule(
            time_mask_ratio=mask_ratio,
            feat_mask_ratio=mask_ratio,
            noise_std=noise_std
        )
        seq_count = self.week_count * self.days_per_week   # 35
        self.encoder = SupConEncoder(
            encoder_type=encoder_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=param_dict.get('supcon_num_layers', 1),
            seq_count=seq_count
        )
        self.proj_head = SupConProjectionHead(in_dim=hidden_size, proj_dim=proj_dim)
        # Classifier chỉ tạo khi dùng standalone (NO-GRAPH).
        # Khi wrapped bởi GraphEnhancedWrapper, wrapper tạo classifier riêng trên fused features.
        self._standalone = not mode.endswith('_graph')
        if self._standalone:
            self.classifier = SupConClassifier(
                in_dim=hidden_size, hidden_dim=64, dropout=cls_dropout,
                num_hidden_layers=param_dict.get('supcon_cls_hidden_layers', 1)
            )

    def encode(self, x):
        """x: (B, T, F) → (B, hidden_size)"""
        return self.encoder(x)

    def _preprocess(self, sub_graph):
        """Shared preprocessing: reshape → action_weight → early_mask.
        Returns: x (B, T, F)
        """
        batch_size = sub_graph['batch_size']
        seq_feat = sub_graph['seq_feat'][:batch_size]
        T_total = self.week_count * self.days_per_week
        x = seq_feat.view(batch_size, T_total, self.activity_num)

        if self.use_action_weight:
            x = self.action_weighting(x)
        if self.use_early:
            x = self.early_mask(x)
        return x

    def forward_features(self, sub_graph):
        """Trả về representations TRƯỚC classifier — dùng bởi GraphEnhancedWrapper.

        Training:  returns (h1 (B,H), z1 (B,proj), z2 (B,proj))
        Inference: returns h (B,H)
        """
        x = self._preprocess(sub_graph)

        if self.training:
            view1, view2 = self.augment(x)
            h1 = self.encode(view1)
            h2 = self.encode(view2)
            z1 = self.proj_head(h1)
            z2 = self.proj_head(h2)
            return h1, z1, z2
        else:
            h = self.encode(x)
            return h

    def forward_single(self, sub_graph):
        """Single-branch classifier path for BCE-only training.

        This bypasses SupCon augmentation and projection heads while preserving
        the module's current train/eval mode for encoder/classifier layers.
        """
        x = self._preprocess(sub_graph)
        h = self.encode(x)
        logits = self.classifier(h)
        return logits

    def forward(self, sub_graph):
        """Full forward — dùng cho NO-GRAPH modes.

        Training:  returns (logits (B,1), z1 (B,proj_dim), z2 (B,proj_dim))
        Inference: returns logits (B,1)
        """
        x = self._preprocess(sub_graph)

        if self.training:
            view1, view2 = self.augment(x)
            h1 = self.encode(view1)
            h2 = self.encode(view2)
            z1 = self.proj_head(h1)
            z2 = self.proj_head(h2)
            logits = self.classifier(h1)
            return logits, z1, z2
        else:
            h = self.encode(x)
            logits = self.classifier(h)
            return logits
