"""
Baseline Model — Phase 1: 1D-CNN + BiLSTM + Attention.

[Kỹ thuật tổng quan: Modular Pipeline cho Dropout Prediction
 Pipeline: Input (B, 5, 7, 22)
         → 1D-CNN per week (WeeklyCNN) — trích xuất local features
         → BiLSTM across weeks (TemporalBiLSTM) — capture sequential dependencies
         → Attention pooling — aggregate weeks into single vector
         → Classifier — predict dropout probability]

Hỗ trợ 3 attention variants + ablation:

| Variant     | Kỹ thuật chi tiết                                                     |
|-------------|-----------------------------------------------------------------------|
| bahdanau    | Bahdanau Attention (additive): score=v·tanh(Wh+b), α=softmax(scores) |
| multihead   | MHA Self-Attention + Learnable Query Pooling                          |
| cross       | Cross-Attention (Q=BiLSTM, K/V=CNN) + Bahdanau Pooling               |

Ablation support:
    - use_cnn=False:       Skip CNN, flatten (7,22)→154 directly to BiLSTM
    - use_bilstm=False:    Skip BiLSTM, use CNN output directly
    - use_attention=False:  Skip Attention, use mean pooling

Components:
    • WeeklyCNN              — 1D-CNN per week [Kỹ thuật: Conv1d 2-layer + BN + AvgPool]
    • TemporalBiLSTM         — BiLSTM across weeks [Kỹ thuật: BiLSTM + LayerNorm]
    • MultiHeadAttentionPool — MHA + Learnable Query [Kỹ thuật: nn.MultiheadAttention + LQP]
    • CrossAttentionPool     — Cross-Attention + Bahdanau [Kỹ thuật: CrossAttn + Bahdanau pool]
    • DropoutClassifier      — 3-layer MLP [Kỹ thuật: Linear→ReLU→Dropout ×2 → Linear]
    • DropoutPredictor       — Full model [Kỹ thuật: CNN → BiLSTM → Attention → Classifier]
"""
import torch
import torch.nn.functional as F
from torch import nn as nn

from .common import BahdanauAttention


class WeeklyCNN(nn.Module):
    """1D-CNN để rút trích features từ mỗi tuần (7 days × 22 actions).

    [Kỹ thuật: 1D Convolutional Neural Network
     Dùng activities (22) làm channels, days (7) làm sequence length.
     Conv1d trượt qua 7 ngày để bắt local temporal patterns trong 1 tuần.
     Shared weights cho tất cả 5 tuần.
     Architecture: Conv1d(22→64) → BN → ReLU → Dropout
                 → Conv1d(64→128) → BN → ReLU → Dropout → AvgPool1d(1)]

    Input:  (B, 5, 7, 22)
    Output: (B, 5, cnn_out_dim)
    """

    def __init__(self, in_channels=22, cnn_out_dim=128, dropout=0.2):
        super(WeeklyCNN, self).__init__()
        self.cnn = nn.Sequential(
            # Layer 1: (B*5, 22, 7) → (B*5, 64, 7)
            nn.Conv1d(in_channels=in_channels, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2: (B*5, 64, 7) → (B*5, cnn_out_dim, 7)
            nn.Conv1d(in_channels=64, out_channels=cnn_out_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Pooling: (B*5, cnn_out_dim, 7) → (B*5, cnn_out_dim, 1)
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        """
        x: (B, W, D, A) e.g. (B, 5, 7, 22)
        return: (B, W, cnn_out_dim)
        """
        B, W, D, A = x.shape
        x = x.view(B * W, D, A)       # (B*5, 7, 22)
        x = x.permute(0, 2, 1)         # (B*5, 22, 7) — activities as channels
        x = self.cnn(x)                # (B*5, cnn_out_dim, 1)
        x = x.squeeze(-1)              # (B*5, cnn_out_dim)
        x = x.view(B, W, -1)           # (B, 5, cnn_out_dim)
        return x


class TemporalBiLSTM(nn.Module):
    """BiLSTM để capture sequential dependencies giữa các tuần.

    [Kỹ thuật: Bidirectional LSTM + LayerNorm
     2-layer BiLSTM, output = hidden_size×2 (concat forward+backward)
     LayerNorm stabilize training]

    Input:  (B, 5, input_size)
    Output: (B, 5, hidden_size*2)  — bidirectional concat
    """

    def __init__(self, input_size=128, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super(TemporalBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        """
        x: (B, T, input_size)
        return: (B, T, hidden_size*2)
        """
        outputs, (h_n, c_n) = self.bilstm(x)
        outputs = self.layer_norm(outputs)
        return outputs, (h_n, c_n)


class MultiHeadAttentionPool(nn.Module):
    """Multi-Head Self-Attention + Learnable Query Pooling.

    [Kỹ thuật: 2-stage attention pooling
     Stage 1: nn.MultiheadAttention self-attention giữa các tuần (enrich representations)
              + residual + LayerNorm
     Stage 2: Learnable query (nn.Parameter) attend vào enriched tuần
              via nn.MultiheadAttention → single vector]

    Input:  (B, T, hidden_size)
    Output: context (B, hidden_size), attn_weights (B, T)
    """

    def __init__(self, hidden_size=128, num_heads=2, dropout=0.1):
        super(MultiHeadAttentionPool, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, lstm_outputs):
        """
        lstm_outputs: (B, T, hidden_size)
        return: context (B, hidden_size), attn_weights (B, T)
        """
        B = lstm_outputs.size(0)
        residual = lstm_outputs
        attn_out, _ = self.self_attn(lstm_outputs, lstm_outputs, lstm_outputs)
        attn_out = self.layer_norm(attn_out + residual)  # (B, T, hidden_size)

        query = self.pool_query.expand(B, -1, -1)        # (B, 1, hidden_size)
        context, attn_weights = self.pool_attn(query, attn_out, attn_out)
        context = context.squeeze(1)                      # (B, hidden_size)
        attn_weights = attn_weights.squeeze(1)            # (B, T)
        return context, attn_weights


class CrossAttentionPool(nn.Module):
    """Cross Attention: BiLSTM (Q) attends to CNN features (K, V).

    [Kỹ thuật: Cross-Attention + Bahdanau Pooling
     Stage 1: nn.MultiheadAttention cross-attention
              Q=BiLSTM outputs, K/V=CNN features
              + residual + LayerNorm + FFN + residual + LayerNorm
     Stage 2: BahdanauAttention pooling → single vector]

    Tạo cross-modal interaction giữa CNN features (local patterns)
    và BiLSTM features (sequential patterns), rồi pool bằng Bahdanau.

    Input:  cnn_features (B, T, dim), lstm_outputs (B, T, dim)
    Output: context (B, dim), attn_weights (B, T)
    """

    def __init__(self, hidden_size=128, num_heads=2, dropout=0.1):
        super(CrossAttentionPool, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.pool_attn = BahdanauAttention(hidden_size)

    def forward(self, cnn_features, lstm_outputs):
        """
        cnn_features:  (B, T, hidden_size) — from WeeklyCNN
        lstm_outputs:  (B, T, hidden_size) — from BiLSTM
        return: context (B, hidden_size), attn_weights (B, T)
        """
        residual = lstm_outputs
        cross_out, _ = self.cross_attn(
            query=lstm_outputs, key=cnn_features, value=cnn_features
        )
        cross_out = self.layer_norm(cross_out + residual)

        residual = cross_out
        ffn_out = self.ffn(cross_out)
        ffn_out = self.ffn_norm(ffn_out + residual)  # (B, T, hidden_size)

        context, attn_weights = self.pool_attn(ffn_out)
        return context, attn_weights


class DropoutClassifier(nn.Module):
    """Final classifier: feature vector → dropout probability logit.

    [Kỹ thuật: MLP 3-layer classifier
     Linear(in→64) → ReLU → Dropout
     → Linear(64→32) → ReLU → Dropout
     → Linear(32→1)]
    """

    def __init__(self, input_dim=128, dropout=0.3):
        super(DropoutClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class DropoutPredictor(nn.Module):
    """Phase 1 Full Model: 1D-CNN + BiLSTM + Attention for MOOC Dropout.

    [Kỹ thuật: Modular Pipeline
     ① WeeklyCNN: 1D-CNN per week (shared weights) — trích xuất local features
     ② TemporalBiLSTM: BiLSTM across weeks — capture sequential dependencies
     ③ Attention: aggregate weeks → single representation vector
     ④ DropoutClassifier: MLP → logits]

    Supports 3 attention variants:
        - 'bahdanau':  Additive attention (Bahdanau) — default
        - 'multihead': Multi-Head Self-Attention + Learnable Query Pooling
        - 'cross':     Cross Attention (BiLSTM queries CNN) + Bahdanau Pooling

    Ablation support via config flags:
        - use_cnn=False:     Skip CNN, flatten (7,22)→154 directly to BiLSTM
        - use_bilstm=False:  Skip BiLSTM, use CNN output directly
        - use_attention=False: Skip Attention, use mean pooling

    Args:
        config: dict with model hyperparameters

    Input:  (B, 5, 7, 22)
    Output: (logits (B, 1), attn_weights (B, 5))
    """

    def __init__(self, config):
        super(DropoutPredictor, self).__init__()
        self.config = config
        self.attention_type = config.get('attention_type', 'bahdanau')
        self.use_cnn = config.get('use_cnn', True)
        self.use_bilstm = config.get('use_bilstm', True)
        self.use_attention = config.get('use_attention', True)

        num_actions = config.get('num_actions', 22)
        days_per_week = config.get('days_per_week', 7)
        cnn_out_dim = config.get('cnn_out_dim', 128)
        lstm_hidden = config.get('lstm_hidden', 64)
        lstm_layers = config.get('lstm_layers', 2)
        cls_dropout = config.get('cls_dropout', 0.3)

        # ① 1D-CNN
        if self.use_cnn:
            self.cnn = WeeklyCNN(
                in_channels=num_actions,
                cnn_out_dim=cnn_out_dim,
                dropout=config.get('cnn_dropout', 0.2),
            )
            bilstm_input = cnn_out_dim
        else:
            # Ablation: no CNN — flatten (7,22) → 154
            bilstm_input = days_per_week * num_actions  # 154

        # ② BiLSTM
        if self.use_bilstm:
            self.bilstm = TemporalBiLSTM(
                input_size=bilstm_input,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=config.get('lstm_dropout', 0.3),
            )
            feature_dim = lstm_hidden * 2  # bidirectional
        else:
            # Ablation: no BiLSTM — use CNN output directly
            feature_dim = bilstm_input

        # ③ Attention
        if self.use_attention:
            if self.attention_type == 'bahdanau':
                self.attention = BahdanauAttention(feature_dim)
            elif self.attention_type == 'multihead':
                self.attention = MultiHeadAttentionPool(
                    feature_dim, num_heads=config.get('attn_heads', 2),
                    dropout=config.get('attn_dropout', 0.1),
                )
            elif self.attention_type == 'cross':
                self.attention = CrossAttentionPool(
                    feature_dim, num_heads=config.get('attn_heads', 2),
                    dropout=config.get('attn_dropout', 0.1),
                )
            else:
                raise ValueError(f"Unknown attention_type: {self.attention_type}")

        # ④ Classifier
        self.classifier = DropoutClassifier(
            input_dim=feature_dim, dropout=cls_dropout
        )

    def forward(self, x):
        """
        x: (B, 5, 7, 22) or (B, 5*7*22) flat tensor.
        Returns: (logits (B, 1), attn_weights (B, 5))
        """
        B = x.shape[0]
        week_count = self.config.get('num_weeks', 5)
        days = self.config.get('days_per_week', 7)
        actions = self.config.get('num_actions', 22)

        # Auto-reshape flat input
        if x.dim() == 2:
            x = x.view(B, week_count, days, actions)

        # ① CNN per week
        if self.use_cnn:
            cnn_out = self.cnn(x)                     # (B, 5, cnn_out_dim)
            seq = cnn_out
        else:
            # Ablation: flatten (7, 22) → 154
            seq = x.view(B, week_count, -1)           # (B, 5, 154)
            cnn_out = None

        # ② BiLSTM
        if self.use_bilstm:
            lstm_out, _ = self.bilstm(seq)            # (B, 5, 2*lstm_hidden)
        else:
            lstm_out = seq                            # (B, 5, feature_dim)

        # ③ Attention pooling
        if self.use_attention:
            if self.attention_type == 'cross' and cnn_out is not None:
                context, attn_weights = self.attention(cnn_out, lstm_out)
            else:
                context, attn_weights = self.attention(lstm_out)
        else:
            # Ablation: no attention — mean pooling
            context = lstm_out.mean(dim=1)             # (B, feature_dim)
            attn_weights = torch.ones(B, lstm_out.size(1),
                                      device=x.device) / lstm_out.size(1)

        # ④ Classifier
        logits = self.classifier(context)              # (B, 1)
        return logits, attn_weights
