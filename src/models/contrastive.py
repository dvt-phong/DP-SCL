"""
Self-Supervised Contrastive Learning — Framework 2.
    2A — SimCLR  (NT-Xent loss, no labels needed for contrastive)
    2B — BYOL    (MSE + momentum encoder, no labels needed for contrastive)

[Kỹ thuật tổng quan:
 SimCLR (Chen et al., 2020): Simple Framework for Contrastive Learning
   - 2 augmented views → Shared Encoder → Shared ProjectionHead
   - NT-Xent Loss: Normalized Temperature-scaled Cross Entropy
     (trong batch 2B samples, positive pair = 2 views cùng sample, 2(B-1) negatives)
   - Classifier nhận representation h (trước projection)

 BYOL (Grill et al., 2020): Bootstrap Your Own Latent
   - Online branch: Encoder → Projector → Predictor → q
   - Target branch: EMA Encoder → EMA Projector → z_target (stop gradient)
   - Loss = MSE(normalize(q), normalize(z_target).detach()) [symmetric]
   - Momentum update: θ_target ← m·θ_target + (1-m)·θ_online (m=0.996)
   - KHÔNG cần negative pairs]

Hỗ trợ 8 modes:

=== SimCLR (Framework 2A) ===
| Mode               | Encoder                          | Kỹ thuật chi tiết                                              |
|--------------------|----------------------------------|----------------------------------------------------------------|
| simclr_lstm        | LSTM                            | LSTM unidirectional, lấy h_n[-1]                               |
| simclr_bilstm      | BiLSTM                          | BiLSTM, concat h_n[-2] + h_n[-1]                              |
| simclr_lstm_attn   | LSTM + Multi-Head Attention     | LSTM → nn.MultiheadAttention (4 heads) + LayerNorm → LQP      |
| simclr_bilstm_attn | BiLSTM + Multi-Head Attention   | BiLSTM → nn.MultiheadAttention (4 heads) + LayerNorm → LQP    |

=== BYOL (Framework 2B) ===
| Mode               | Encoder                          | Kỹ thuật chi tiết                                              |
|--------------------|----------------------------------|----------------------------------------------------------------|
| byol_lstm          | LSTM                            | LSTM unidirectional, lấy h_n[-1]                               |
| byol_bilstm        | BiLSTM                          | BiLSTM, concat h_n[-2] + h_n[-1]                              |
| byol_lstm_attn     | LSTM + Multi-Head Attention     | LSTM → nn.MultiheadAttention (4 heads) + LayerNorm → LQP      |
| byol_bilstm_attn   | BiLSTM + Multi-Head Attention   | BiLSTM → nn.MultiheadAttention (4 heads) + LayerNorm → LQP    |

Giải thích viết tắt:
    - LSTM:   Long Short-Term Memory (unidirectional)
    - BiLSTM: Bidirectional LSTM
    - MHA:    nn.MultiheadAttention (standard Transformer MHA)
    - LQP:    LearnableQueryPool (trainable query + attention pooling)
    - BN:     BatchNorm1d (dùng trong ProjectionHead, thay vì chỉ ReLU — chuẩn SimCLR/BYOL)
    - NT-Xent: Normalized Temperature-scaled Cross Entropy Loss
    - EMA:    Exponential Moving Average (cho BYOL target branch)
"""
import torch
import torch.nn.functional as F
from torch import nn as nn

from .common import LearnableQueryPool, AugmentationModule, ActionWeightedInput, EarlyPredictionMask


class CLEncoder(nn.Module):
    """Shared Encoder cho Framework 2 (SimCLR & BYOL) — 4 variants.

    Tách biệt hoàn toàn với SiameseEncoder để ablation sạch.

    | encoder_type   | Architecture                              | Output  |
    |----------------|-------------------------------------------|---------|
    | cl_lstm        | LSTM(F→H), lấy h_n[-1]                   | (B, H)  |
    | cl_bilstm      | BiLSTM(F→H/2×2), concat h_n              | (B, H)  |
    | cl_lstm_attn   | LSTM → MHA → LearnableQueryPool           | (B, H)  |
    | cl_bilstm_attn | BiLSTM → MHA → LearnableQueryPool         | (B, H)  |

    Kỹ thuật chi tiết:
        - cl_lstm / cl_bilstm:         chỉ dùng last hidden state (không attention)
        - cl_lstm_attn / cl_bilstm_attn: MHA = nn.MultiheadAttention (standard Transformer)
                                          + LayerNorm + residual → LearnableQueryPool

    Input: (B, T, input_size) e.g. (B, 35, 22)
    Output: (B, hidden_size) e.g. (B, 128)
    """
    VALID_TYPES = ('cl_lstm', 'cl_bilstm', 'cl_lstm_attn', 'cl_bilstm_attn')

    def __init__(self, encoder_type, input_size=22, hidden_size=128,
                 num_heads=4, dropout=0.1, num_layers=1):
        super(CLEncoder, self).__init__()
        assert encoder_type in self.VALID_TYPES, \
            f"CLEncoder type must be one of {self.VALID_TYPES}. Got: {encoder_type}"

        self.encoder_type = encoder_type
        self.hidden_size = hidden_size

        if encoder_type in ('cl_lstm', 'cl_lstm_attn'):
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        else:  # cl_bilstm, cl_bilstm_attn
            self.rnn = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0)

        if encoder_type in ('cl_lstm_attn', 'cl_bilstm_attn'):
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.pool = LearnableQueryPool(hidden_size)

    def forward(self, x):
        """
        x: (B, T, input_size) e.g. (B, 35, 22)
        Returns: (B, hidden_size) e.g. (B, 128)
        """
        rnn_out, (h_n, _) = self.rnn(x)

        if self.encoder_type == 'cl_lstm':
            return h_n[-1]

        elif self.encoder_type == 'cl_bilstm':
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)

        else:  # cl_lstm_attn, cl_bilstm_attn
            attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
            attn_out = self.layer_norm(attn_out + rnn_out)
            context, _ = self.pool(attn_out)
            return context


class CLProjectionHead(nn.Module):
    """Projection Head dùng chung cho SimCLR và BYOL (online + target).

    [Kỹ thuật: MLP 2-layer Projection Head + BatchNorm + L2 Normalization
     Architecture: Linear(in_dim, hidden_dim) → BN → ReLU → Linear(hidden_dim, proj_dim) → L2-norm
     BN (BatchNorm) thay vì chỉ ReLU — chuẩn SimCLR/BYOL paper]
    """
    def __init__(self, in_dim=128, hidden_dim=128, proj_dim=128):
        super(CLProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        """
        x: (B, in_dim)
        return: (B, proj_dim) — L2-normalized
        """
        z = self.net(x)
        return F.normalize(z, dim=1)


class CLPredictorHead(nn.Module):
    """Predictor MLP — chỉ dùng trong BYOL (online branch).

    [Kỹ thuật: BYOL Predictor Head
     Architecture: Linear(proj_dim, hidden_dim) → BN → ReLU → Linear(hidden_dim, proj_dim)
     KHÔNG L2-normalize ở đây — normalize khi tính MSE loss
     Chỉ có ở online branch, KHÔNG có ở target branch]
    """
    def __init__(self, proj_dim=128, hidden_dim=64):
        super(CLPredictorHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        """
        x: (B, proj_dim)
        return: (B, proj_dim) — raw (chưa normalize)
        """
        return self.net(x)


class NTXentLoss(nn.Module):
    """NT-Xent Loss (Normalized Temperature-scaled Cross Entropy) — SimCLR.

    [Kỹ thuật: NT-Xent Loss (Chen et al., 2020)
     Self-supervised contrastive loss:
     - Với mỗi sample i, positive pair là view kia (i+B hoặc i-B)
     - 2(B-1) samples còn lại là negatives
     - Loss = CrossEntropy trên cosine similarity / temperature
     - Không dùng labels (self-supervised)]

    Args:
        temperature: τ (mặc định 0.1, SimCLR thường lớn hơn SupCon 0.07)
    """
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: (B, proj_dim) — L2-normalized projections từ view 1
        z2: (B, proj_dim) — L2-normalized projections từ view 2
        Returns: scalar NT-Xent loss
        """
        B = z1.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # Concat tất cả: (2B, proj_dim)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix (2B, 2B): đã L2-norm nên dot product = cosine sim
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Mask diagonal (self-similarity)
        eye_mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
        sim = sim.masked_fill(eye_mask, float('-inf'))

        # Labels: sample i có positive là i+B (và ngược lại)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(0, B, device=z1.device)
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


class CLClassifier(nn.Module):
    """Classifier cho Framework 2: nhận h (representation từ encoder).

    [Kỹ thuật: MLP classifier head
     num_hidden_layers=1: Linear(in→hid) → ReLU → Dropout → Linear(hid→1)
     num_hidden_layers=2: + thêm Linear(hid→hid//2) → ReLU → Dropout]

    Output là logits — dùng với BCEWithLogitsLoss.
    """
    def __init__(self, in_dim=128, hidden_dim=64, dropout=0.3, num_hidden_layers=1):
        super(CLClassifier, self).__init__()
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


class CLSimCLR(nn.Module):
    """SimCLR Model cho Framework 2A.

    [Kỹ thuật: SimCLR (Chen et al., 2020)
     Simple Framework for Contrastive Learning of Visual Representations
     Adapted cho sequential data (time series)]

    Kiến trúc:
        Input (B, 35, 22)
        → Augmentation (training) → view1, view2
        → Shared CLEncoder → h1, h2           (B, 128)
        → Shared CLProjectionHead → z1, z2    (B, 128) L2-normalized
        NT-Xent(z1, z2)
        → CLClassifier(h1) → logits           (B, 1)

    Total Loss = BCE(logits, y) + λ × NTXent(z1, z2)

    Modes: simclr_lstm, simclr_bilstm, simclr_lstm_attn, simclr_bilstm_attn

    Training:  aug ON, 2× forward → return (logits, z1, z2)
    Inference: aug OFF, 1× forward → return logits
    """
    ENCODER_MAP = {
        'simclr_lstm':        'cl_lstm',
        'simclr_bilstm':      'cl_bilstm',
        'simclr_lstm_attn':   'cl_lstm_attn',
        'simclr_bilstm_attn': 'cl_bilstm_attn',
        # Graph-enhanced variants (same encoder, + GraphSage fusion)
        'simclr_lstm_graph':        'cl_lstm',
        'simclr_bilstm_graph':      'cl_bilstm',
        'simclr_lstm_attn_graph':   'cl_lstm_attn',
        'simclr_bilstm_attn_graph': 'cl_bilstm_attn',
    }

    def __init__(self, mode, param_dict):
        super(CLSimCLR, self).__init__()
        assert mode in self.ENCODER_MAP, \
            f"CLSimCLR mode must be one of {list(self.ENCODER_MAP.keys())}. Got: {mode}"

        self.mode = mode
        encoder_type = self.ENCODER_MAP[mode]

        input_size  = param_dict.get('activity_num', 22)
        hidden_size = param_dict.get('cl_hidden_size', 128)
        proj_dim    = param_dict.get('cl_proj_dim', 128)
        mask_ratio  = param_dict.get('cl_mask_ratio', 0.15)
        noise_std   = param_dict.get('cl_noise_std', 0.05)
        num_heads   = param_dict.get('cl_attn_heads', 4)
        cls_dropout = param_dict.get('cl_cls_dropout', 0.3)
        temperature = param_dict.get('cl_temperature', 0.1)

        self.week_count   = param_dict.get('week_count', 5)
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
        self.encoder = CLEncoder(
            encoder_type=encoder_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=param_dict.get('cl_num_layers', 1)
        )
        self.proj_head = CLProjectionHead(
            in_dim=hidden_size, hidden_dim=hidden_size, proj_dim=proj_dim
        )
        self.ntxent = NTXentLoss(temperature=temperature)
        # Classifier chỉ tạo khi standalone (NO-GRAPH).
        # Khi wrapped bởi GraphEnhancedWrapper, wrapper tạo classifier riêng trên fused features.
        self._standalone = not mode.endswith('_graph')
        if self._standalone:
            self.classifier = CLClassifier(
                in_dim=hidden_size, hidden_dim=64, dropout=cls_dropout,
                num_hidden_layers=param_dict.get('cl_cls_hidden_layers', 1)
            )

    def encode(self, x):
        """x: (B, T, F) → (B, hidden_size)"""
        return self.encoder(x)

    def _preprocess(self, sub_graph):
        """Shared preprocessing: reshape → action_weight → early_mask."""
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
            return self.encode(x)

    def forward(self, sub_graph):
        """Full forward — dùng cho NO-GRAPH modes.

        Training:  return (logits (B,1), z1 (B,proj_dim), z2 (B,proj_dim))
        Inference: return logits (B,1)
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


class CLBYOL(nn.Module):
    """BYOL Model cho Framework 2B.

    [Kỹ thuật: BYOL (Grill et al., 2020)
     Bootstrap Your Own Latent — self-supervised KHÔNG cần negative pairs]

    Kiến trúc:
        Online branch:  view1 → OnlineEncoder → h_online
                               → OnlineProjector → z_online
                               → Predictor → q
        Target branch:  view2 → TargetEncoder (stop_grad momentum)
                               → h_target → TargetProjector → z_target

        Loss_BYOL = MSE(normalize(q), normalize(z_target).detach())   [view1→view2]
                  + MSE(normalize(q2), normalize(z1).detach())          [view2→view1, symmetric]

        Total Loss = BCE(classifier(h_online), y) + λ × Loss_BYOL

    Momentum update (gọi sau mỗi optimizer.step()):
        θ_target ← m·θ_target + (1-m)·θ_online   (m=0.996)

    Modes: byol_lstm, byol_bilstm, byol_lstm_attn, byol_bilstm_attn

    Training:  aug ON → return (logits, byol_loss_scalar)
    Inference: online encoder → return logits
    """
    ENCODER_MAP = {
        'byol_lstm':        'cl_lstm',
        'byol_bilstm':      'cl_bilstm',
        'byol_lstm_attn':   'cl_lstm_attn',
        'byol_bilstm_attn': 'cl_bilstm_attn',
        # Graph-enhanced variants (same encoder, + GraphSage fusion)
        'byol_lstm_graph':        'cl_lstm',
        'byol_bilstm_graph':      'cl_bilstm',
        'byol_lstm_attn_graph':   'cl_lstm_attn',
        'byol_bilstm_attn_graph': 'cl_bilstm_attn',
    }

    def __init__(self, mode, param_dict):
        super(CLBYOL, self).__init__()
        assert mode in self.ENCODER_MAP, \
            f"CLBYOL mode must be one of {list(self.ENCODER_MAP.keys())}. Got: {mode}"

        self.mode = mode
        encoder_type = self.ENCODER_MAP[mode]

        input_size  = param_dict.get('activity_num', 22)
        hidden_size = param_dict.get('cl_hidden_size', 128)
        proj_dim    = param_dict.get('cl_proj_dim', 128)
        mask_ratio  = param_dict.get('cl_mask_ratio', 0.15)
        noise_std   = param_dict.get('cl_noise_std', 0.05)
        num_heads   = param_dict.get('cl_attn_heads', 4)
        cls_dropout = param_dict.get('cl_cls_dropout', 0.3)
        self.momentum = param_dict.get('cl_momentum', 0.996)

        self.week_count    = param_dict.get('week_count', 5)
        self.days_per_week = param_dict.get('cnn_in_channels', 7)
        self.activity_num  = input_size

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

        # === Online Branch ===
        _num_layers = param_dict.get('cl_num_layers', 1)
        self.online_encoder   = CLEncoder(encoder_type, input_size, hidden_size, num_heads, num_layers=_num_layers)
        self.online_projector = CLProjectionHead(hidden_size, hidden_size, proj_dim)
        self.predictor        = CLPredictorHead(proj_dim, proj_dim // 2)

        # === Target Branch (EMA — parameters NOT in optimizer) ===
        self.target_encoder   = CLEncoder(encoder_type, input_size, hidden_size, num_heads, num_layers=_num_layers)
        self.target_projector = CLProjectionHead(hidden_size, hidden_size, proj_dim)

        # Initialize target = online
        self._init_target()

        # Freeze target params (updated via momentum only)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # Classifier chỉ tạo khi standalone (NO-GRAPH).
        self._standalone = not mode.endswith('_graph')
        if self._standalone:
            self.classifier = CLClassifier(hidden_size, 64, cls_dropout,
                                           num_hidden_layers=param_dict.get('cl_cls_hidden_layers', 1))

    def _init_target(self):
        """Copy online → target weights."""
        for p_online, p_target in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            p_target.data.copy_(p_online.data)

    @torch.no_grad()
    def momentum_update(self):
        """EMA update: θ_target ← m·θ_target + (1-m)·θ_online.
        Gọi sau mỗi optimizer.step() trong training loop.
        """
        m = self.momentum
        for p_online, p_target in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        ):
            p_target.data.mul_(m).add_((1 - m) * p_online.data)

    @staticmethod
    def _byol_loss(q, z_target):
        """BYOL regression loss: MSE giữa normalized q và stop-grad z_target."""
        q_norm = F.normalize(q, dim=1)
        z_norm = F.normalize(z_target.detach(), dim=1)
        return 2 - 2 * (q_norm * z_norm).sum(dim=1).mean()

    def _preprocess(self, sub_graph):
        """Shared preprocessing: reshape → action_weight → early_mask."""
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

        Training:  returns (h_online1 (B,H), byol_loss scalar)
        Inference: returns h (B,H)
        """
        x = self._preprocess(sub_graph)

        if self.training:
            view1, view2 = self.augment(x)

            # Online branch
            h_online1 = self.online_encoder(view1)
            z_online1 = self.online_projector(h_online1)
            q1 = self.predictor(z_online1)

            h_online2 = self.online_encoder(view2)
            z_online2 = self.online_projector(h_online2)
            q2 = self.predictor(z_online2)

            # Target branch (stop_grad)
            with torch.no_grad():
                z_target1 = self.target_projector(self.target_encoder(view1))
                z_target2 = self.target_projector(self.target_encoder(view2))

            loss_byol = (self._byol_loss(q1, z_target2) +
                         self._byol_loss(q2, z_target1)) * 0.5

            return h_online1, loss_byol
        else:
            return self.online_encoder(x)

    def forward(self, sub_graph):
        """Full forward — dùng cho NO-GRAPH modes.

        Training:  return (logits (B,1), byol_loss scalar)
        Inference: return logits (B,1)
        """
        x = self._preprocess(sub_graph)

        # NOTE: Apply 1 lần trên x trước augment. Cả online và target branch
        # nhận cùng input đã weighted+masked thông qua view1/view2.

        if self.training:
            view1, view2 = self.augment(x)

            # Online branch — view1
            h_online1 = self.online_encoder(view1)
            z_online1 = self.online_projector(h_online1)
            q1        = self.predictor(z_online1)

            # Online branch — view2 (symmetric)
            h_online2 = self.online_encoder(view2)
            z_online2 = self.online_projector(h_online2)
            q2        = self.predictor(z_online2)

            # Target branch — stop_grad
            with torch.no_grad():
                z_target1 = self.target_projector(self.target_encoder(view1))
                z_target2 = self.target_projector(self.target_encoder(view2))

            # Symmetric BYOL loss
            loss_byol = (self._byol_loss(q1, z_target2) +
                         self._byol_loss(q2, z_target1)) * 0.5

            logits = self.classifier(h_online1)
            return logits, loss_byol

        else:
            h = self.online_encoder(x)
            logits = self.classifier(h)
            return logits
