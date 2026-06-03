"""
Shared Building Blocks — Common neural network components dùng chung
used across DP-SCL and baseline models.

Bao gồm:
    ─ Context Embedding:
        • Context                — Embedding org_context + enhanced_context (dùng bởi LGB graph modes)

    ─ Graph Neural Networks:
        • GraphSage              — GraphSAGE 2-layer (mean aggregator) [Kỹ thuật: GNN - Graph Neural Network]
        • GATNetwork             — GAT multi-head attention [Kỹ thuật: Graph Attention Network]

    ─ CNN Feature Extractors:
        • CNNFeatureExtractor    — 1D-CNN trích xuất features từ weekly data [Kỹ thuật: 1D Convolutional Neural Network]
        • CNN2DFeatureExtractor  — 2D-CNN xử lý weekly data như ảnh [Kỹ thuật: 2D Convolutional Neural Network]
        • MBACNNFeatureExtractor — Multi-Branch Asymmetric CNN 3 nhánh [Kỹ thuật: MBA-CNN, 3-branch asymmetric 2D-CNN]

    ─ Recurrent Networks:
        • MyLSTM                 — Unidirectional LSTM [Kỹ thuật: LSTM - Long Short-Term Memory]
        • MyBiLSTM               — Bidirectional LSTM [Kỹ thuật: BiLSTM - Bidirectional LSTM]

    ─ Attention Mechanisms:
        • MySelfAttention        — Custom Self-Attention + sinusoidal Position Encoding
                                   [Kỹ thuật: Multi-Head Self-Attention with sinusoidal PE, custom QKV]
        • MyMHAttention          — Standard Multi-Head Attention (Transformer block)
                                   [Kỹ thuật: nn.MultiheadAttention + residual + LayerNorm + FFN]
        • MyCrossAttention       — Cross Attention: Q attends to K/V từ nguồn khác
                                   [Kỹ thuật: Cross-Attention (Q from BiLSTM, K/V from CNN)]
        • LearnableQueryPool     — Learnable query vector attend vào sequence → single vector
                                   [Kỹ thuật: Learnable Query Pooling via nn.MultiheadAttention]
        • BahdanauAttention      — Additive (Bahdanau) Attention cho aggregation
                                   [Kỹ thuật: Additive Attention (Bahdanau, 1 hidden layer)]

    ─ Fusion:
        • CrossAttentionFusion   — Cross-attention temporal×graph features
                                   [Kỹ thuật: Multi-Head Cross-Attention Fusion + residual + LayerNorm + FFN]

    ─ Classifier:
        • Classifier             — DNN classifier head (16→8→1) [Kỹ thuật: MLP 2-layer classifier]

    ─ Contrastive Learning Components:
        • SupConLoss             — Supervised Contrastive Loss (Khosla 2020)
                                   [Kỹ thuật: Supervised Contrastive Loss với temperature scaling]
        • ProjectionHead         — MLP projection + L2 normalize
                                   [Kỹ thuật: MLP 2-layer → L2 Normalization]

    ─ Data Augmentation:
        • AugmentationModule     — Random masking + Gaussian noise
                                   [Kỹ thuật: Time Masking + Feature Masking + Additive Gaussian Noise]

    ─ Input Preprocessing (Optional Modules):
        • ActionWeightedInput    — Learnable action type importance weighting
                                   [Kỹ thuật: Softmax-normalized learnable scalar weights per action type]
        • EarlyPredictionMask    — Curriculum week masking for early prediction
                                   [Kỹ thuật: Per-sample random week masking (training) / fixed week eval]
"""
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch_geometric.nn import SAGEConv, GATConv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Context Embedding
# ============================================================

class Context(nn.Module):
    """Context Embedding: kết hợp org_context + enhanced_context.
    [Kỹ thuật: Linear Embedding + Concatenation Fusion]

    Dùng bởi: LGB (tất cả graph-based modes)
    """
    def __init__(self, param_dict):
        super(Context, self).__init__()
        self.org_context_feat_len = param_dict['org_context_feat_len']
        self.enhanced_context_feat_len = param_dict['enhanced_context_feat_len']
        self.context_each_embed = param_dict['context_each_embed']
        self.context_all_len = param_dict['context_all_len']

        self.org_context_embed = nn.Linear(self.org_context_feat_len, self.context_each_embed)
        self.enhanced_context_embed = nn.Linear(self.enhanced_context_feat_len, self.context_each_embed)
        self.context_all_embed = nn.Linear(2 * self.context_each_embed, self.context_all_len)

    def forward(self, sub_graph):
        org_context = sub_graph['org_context']
        org_context = self.org_context_embed(org_context)
        enhanced_context = sub_graph['enhanced_context']
        enhanced_context = self.enhanced_context_embed(enhanced_context)
        # Fuse
        context = torch.cat((org_context, enhanced_context), dim=1)
        context = self.context_all_embed(context)
        return context


# ============================================================
# Graph Neural Networks
# ============================================================

class GraphSage(nn.Module):
    """GraphSAGE 2-layer với mean aggregator.
    [Kỹ thuật: GraphSAGE (Hamilton et al., 2017) — Inductive Graph Neural Network]

    Dùng bởi: LGB (modes: default, cnn, cnn2d, cross_attn, mba_cnn, bilstm_graph)
    """
    def __init__(self, param_dict):
        super(GraphSage, self).__init__()
        self.input_features1 = param_dict['input_features']
        self.hidden_features1 = param_dict['hidden_features']
        self.output_features1 = param_dict['output_features']
        self.conv1 = SAGEConv(in_channels=self.input_features1, out_channels=self.hidden_features1, aggr='mean')
        self.conv2 = SAGEConv(in_channels=self.hidden_features1, out_channels=self.output_features1, aggr='mean')
        self.ac_f1 = nn.ReLU()
        self.ac_f2 = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ac_f1(x)
        x = self.conv2(x, edge_index)
        return self.ac_f2(x)


class GATNetwork(nn.Module):
    """GAT (Graph Attention Network) thay thế GraphSAGE.
    Multi-head attention ở layer 1, single head ở layer 2.
    Cùng interface forward(x, edge_index) và output shape (N, output_features) như GraphSAGE.

    [Kỹ thuật: GAT (Veličković et al., 2018) — Graph Attention Network
     với multi-head attention, ELU activation, dropout]

    Dùng bởi: LGB (modes: gat, cnn_gat, mba_cnn_gat)
    """
    def __init__(self, param_dict):
        super(GATNetwork, self).__init__()
        in_features = param_dict['input_features']        # 16
        hidden_features = param_dict['hidden_features']    # 32
        output_features = param_dict['output_features']    # 16
        heads = param_dict.get('gat_heads', 4)
        dropout = param_dict.get('gat_dropout', 0.3)

        # GAT Layer 1: multi-head attention
        # output per head = hidden_features // heads, concat → hidden_features
        self.conv1 = GATConv(
            in_channels=in_features,
            out_channels=hidden_features // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        # GAT Layer 2: single head to get final output dimension
        self.conv2 = GATConv(
            in_channels=hidden_features,
            out_channels=output_features,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return self.elu(x)


# ============================================================
# CNN Feature Extractors
# ============================================================

class CNNFeatureExtractor(nn.Module):
    """CNN 1D để trích xuất local features từ raw weekly activity data.
    Thay thế bước tiền xử lý thủ công (sum_by_day, sum_by_action).

    [Kỹ thuật: 1D Convolutional Neural Network (Conv1d 2-layer + BatchNorm + AdaptiveAvgPool)]

    Input: (B*week_count, days_per_week, activity_num) — dùng days làm channels
    Output: (B*week_count, cnn_fc_output)

    Dùng bởi: LGB (modes: cnn, cnn_gat, bilstm_cnn, bilstm_mha, bilstm_cross,
              bilstm_graph, cnn_only, cnn_day, bilstm_day)
    """
    def __init__(self, param_dict):
        super(CNNFeatureExtractor, self).__init__()
        in_channels = param_dict.get('cnn_in_channels', 7)       # days_per_week = 7
        out_channels_1 = param_dict.get('cnn_out_channels_1', 32)
        out_channels_2 = param_dict.get('cnn_out_channels_2', 64)
        kernel_size = param_dict.get('cnn_kernel_size', 3)
        cnn_fc_output = param_dict.get('cnn_fc_output', 128)

        # Conv1d layer 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels_1)

        # Conv1d layer 2
        self.conv2 = nn.Conv1d(
            in_channels=out_channels_1,
            out_channels=out_channels_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels_2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels_2, cnn_fc_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (B * week_count, days_per_week, activity_num)
           e.g. (B*5, 7, 22) — channels=7(days), seq_len=22(activities)
        return: (B * week_count, cnn_fc_output)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc(x)))
        return x


class CNN2DFeatureExtractor(nn.Module):
    """CNN 2D để trích xuất features từ ma trận 2D (days × activities).
    Xử lý mỗi tuần như một "ảnh" 1-channel kích thước (days_per_week, activity_num).
    Conv2D có thể học cả pattern theo ngày lẫn pattern theo loại hoạt động.

    [Kỹ thuật: 2D Convolutional Neural Network (Conv2d 2-layer + BatchNorm + AdaptiveAvgPool2d)]

    Input: (B*week_count, 1, days_per_week, activity_num) e.g. (B*5, 1, 7, 22)
    Output: (B*week_count, cnn2d_fc_output)

    Dùng bởi: LGB (mode: cnn2d)
    """
    def __init__(self, param_dict):
        super(CNN2DFeatureExtractor, self).__init__()
        out_channels_1 = param_dict.get('cnn2d_out_channels_1', 32)
        out_channels_2 = param_dict.get('cnn2d_out_channels_2', 64)
        kernel_size = param_dict.get('cnn2d_kernel_size', 3)
        cnn2d_fc_output = param_dict.get('cnn2d_fc_output', 128)

        # Conv2d layer 1: (1, 7, 22) → (32, 7, 22)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels_1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(out_channels_1)

        # Conv2d layer 2: (32, 7, 22) → (64, 7, 22)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels_1,
            out_channels=out_channels_2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(out_channels_2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling 2D
        self.fc = nn.Linear(out_channels_2, cnn2d_fc_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B * week_count, 1, days_per_week, activity_num)
           e.g. (B*5, 1, 7, 22)
        return: (B * week_count, cnn2d_fc_output)
        """
        x = self.relu(self.bn1(self.conv1(x)))           # (B*5, 32, 7, 22)
        x = self.relu(self.bn2(self.conv2(x)))           # (B*5, 64, 7, 22)
        x = self.pool(x).squeeze(-1).squeeze(-1)         # (B*5, 64)
        x = self.relu(self.fc(x))                        # (B*5, cnn2d_fc_output)
        return x


class MBACNNFeatureExtractor(nn.Module):
    """Multi-Branch Asymmetric CNN (MBA-CNN).

    [Kỹ thuật: MBA-CNN — 3-branch asymmetric 2D-CNN
     Mỗi nhánh dùng kernel không đối xứng, chiều rộng = toàn bộ activities (f),
     chỉ khác chiều cao (số ngày liên tiếp):
       - Nhánh 1 (Temporal): kernel (3×f), pad=(1,0), 2 conv layers → 64×e = 448
       - Nhánh 2 (Daily):    kernel (1×f), no pad, 1 conv layer  → 32×e = 224
       - Nhánh 3 (Weekly):   kernel (e×f), no pad, 1 conv layer  → 32
     Concat → 704 → FC(704→256→output) + ReLU + Dropout]

    Input:  (B*week_count, 1, e, f) e.g. (B*5, 1, 7, 22)
    Output: (B*week_count, mba_cnn_output)

    Dùng bởi: LGB (modes: mba_cnn, mba_cnn_gat, mba_bilstm, mba_only)
    """
    def __init__(self, param_dict):
        super(MBACNNFeatureExtractor, self).__init__()
        e = param_dict.get('cnn_in_channels', 7)     # days_per_week
        f = param_dict['activity_num']                # 22
        ch_t1 = param_dict.get('mba_cnn_temporal_channels_1', 32)
        ch_t2 = param_dict.get('mba_cnn_temporal_channels_2', 64)
        ch_d  = param_dict.get('mba_cnn_daily_channels', 32)
        ch_w  = param_dict.get('mba_cnn_weekly_channels', 32)
        fc_hidden = param_dict.get('mba_cnn_fc_hidden', 256)
        output_dim = param_dict.get('mba_cnn_output', 128)
        dropout = param_dict.get('mba_cnn_dropout', 0.3)

        # --- Nhánh 1: Temporal Branch (3×f) ---
        self.temporal_conv1 = nn.Conv2d(1, ch_t1, kernel_size=(3, f), padding=(1, 0))
        self.temporal_bn1 = nn.BatchNorm2d(ch_t1)
        self.temporal_conv2 = nn.Conv2d(ch_t1, ch_t2, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_bn2 = nn.BatchNorm2d(ch_t2)
        # Output: (B, 64, e, 1) → flatten → 64*e = 448

        # --- Nhánh 2: Daily Branch (1×f) ---
        self.daily_conv = nn.Conv2d(1, ch_d, kernel_size=(1, f), padding=(0, 0))
        self.daily_bn = nn.BatchNorm2d(ch_d)
        # Output: (B, 32, e, 1) → flatten → 32*e = 224

        # --- Nhánh 3: Weekly Branch (e×f) ---
        self.weekly_conv = nn.Conv2d(1, ch_w, kernel_size=(e, f), padding=(0, 0))
        self.weekly_bn = nn.BatchNorm2d(ch_w)
        # Output: (B, 32, 1, 1) → flatten → 32

        # --- Fusion FC ---
        concat_dim = ch_t2 * e + ch_d * e + ch_w  # 64*7 + 32*7 + 32 = 704
        self.fc1 = nn.Linear(concat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, 1, e, f) e.g. (B, 1, 7, 22)
        return: (B, mba_cnn_output)
        """
        # Temporal branch
        t = self.relu(self.temporal_bn1(self.temporal_conv1(x)))   # (B, 32, e, 1)
        t = self.relu(self.temporal_bn2(self.temporal_conv2(t)))   # (B, 64, e, 1)
        t = t.view(t.size(0), -1)                                 # (B, 64*e)

        # Daily branch
        d = self.relu(self.daily_bn(self.daily_conv(x)))           # (B, 32, e, 1)
        d = d.view(d.size(0), -1)                                 # (B, 32*e)

        # Weekly branch
        w = self.relu(self.weekly_bn(self.weekly_conv(x)))         # (B, 32, 1, 1)
        w = w.view(w.size(0), -1)                                 # (B, 32)

        # Concatenate + FC
        out = torch.cat([t, d, w], dim=1)                          # (B, 704)
        out = self.dropout(self.relu(self.fc1(out)))                # (B, 256)
        out = self.relu(self.fc2(out))                              # (B, output)
        return out


# ============================================================
# Recurrent Networks
# ============================================================

class MyLSTM(nn.Module):
    """Unidirectional LSTM — temporal sequence encoder.
    [Kỹ thuật: LSTM (Hochreiter & Schmidhuber, 1997)
     Unidirectional, hidden state qua Linear+ReLU projection]

    Dùng bởi: LGB (modes: default, cnn, cnn2d, gat, cnn_gat, cross_attn, mba_cnn, mba_cnn_gat, no_graph)
    """
    def __init__(self, lstm_input_features, lstm_hidden_features, lstm_hidden_num_layers):
        super(MyLSTM, self).__init__()
        self.lstm_input_features = lstm_input_features
        self.lstm_hidden_features = lstm_hidden_features
        self.lstm_hidden_num_layers = lstm_hidden_num_layers
        self.lstm = nn.LSTM(self.lstm_input_features, self.lstm_hidden_features, self.lstm_hidden_num_layers,
                            batch_first=True)
        self.reg = nn.Sequential(
            nn.Linear(self.lstm_hidden_features, self.lstm_hidden_features),
            nn.ReLU()
        )
        self.ac_f1 = nn.ReLU()

    def forward(self, x):
        x, (ht, ct) = self.lstm(x)
        return self.reg(x)


class MyBiLSTM(nn.Module):
    """Bidirectional LSTM — thay thế MyLSTM cho mode cải tiến.
    Output size = hidden_features * 2 (concat forward + backward).
    LayerNorm + Dropout(0.1) ổn định training.

    [Kỹ thuật: BiLSTM (Bidirectional LSTM)
     concat forward & backward hidden states → LayerNorm → Dropout → Linear+ReLU]

    Dùng bởi: LGB (modes: bilstm_cnn, bilstm_mha, bilstm_cross, bilstm_graph,
              mba_bilstm, bilstm_day)
    """
    def __init__(self, lstm_input_features, lstm_hidden_features,
                 lstm_hidden_num_layers, dropout=0.1):
        super(MyBiLSTM, self).__init__()
        self.lstm_input_features = lstm_input_features
        self.lstm_hidden_features = lstm_hidden_features
        self.lstm_hidden_num_layers = lstm_hidden_num_layers
        self.lstm = nn.LSTM(
            self.lstm_input_features,
            self.lstm_hidden_features,
            self.lstm_hidden_num_layers,
            batch_first=True,
            bidirectional=True
        )
        output_dim = self.lstm_hidden_features * 2
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x, (ht, ct) = self.lstm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.reg(x)


# ============================================================
# Attention Mechanisms
# ============================================================

class MyMHAttention(nn.Module):
    """Multi-Head Attention block chuẩn Transformer.
    Giữ nguyên input dimension xuyên suốt (không giảm chiều như MySelfAttention).

    [Kỹ thuật: Standard Multi-Head Attention (Vaswani et al., 2017)
     Sub-layer 1: nn.MultiheadAttention(embed_dim, num_heads) + residual + LayerNorm
     Sub-layer 2: FFN (Linear→ReLU→Dropout→Linear) + residual + LayerNorm]

    Dùng bởi: LGB (mode: bilstm_mha)
    """
    def __init__(self, embed_dim, num_heads=4, ffn_dim=None, dropout=0.1):
        super(MyMHAttention, self).__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        if ffn_dim is None:
            ffn_dim = embed_dim * 2

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        """
        x:      (B, T, embed_dim)
        return: (B, T, embed_dim)
        """
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class MyCrossAttention(nn.Module):
    """Cross Attention: BiLSTM (Q) attends to CNN features (K, V).

    [Kỹ thuật: Cross-Attention (Q from BiLSTM, K/V from CNN)
     nn.MultiheadAttention + residual + LayerNorm + FFN + residual + LayerNorm
     Nếu embed_dim của Q ≠ kv_dim thì Linear projection K/V về cùng dim với Q]

    Giải quyết vấn đề T=5 quá ngắn cho self-attention:
    thay vì attend vào chính nó, BiLSTM query vào
    CNN output — nguồn thông tin độc lập và bổ sung.

    Dùng bởi: LGB (mode: bilstm_cross)
    """
    def __init__(self, q_dim, kv_dim, num_heads=4, dropout=0.1):
        super(MyCrossAttention, self).__init__()

        # Project K, V về cùng dim với Q nếu khác nhau
        self.kv_proj = nn.Linear(kv_dim, q_dim) \
            if kv_dim != q_dim else nn.Identity()

        assert q_dim % num_heads == 0, \
            f"q_dim ({q_dim}) phải chia hết cho num_heads ({num_heads})"

        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(q_dim)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, q_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(q_dim * 2, q_dim)
        )
        self.norm2 = nn.LayerNorm(q_dim)

    def forward(self, query, key_value):
        """
        query:     (B, T, q_dim)   — từ BiLSTM
        key_value: (B, T, kv_dim)  — từ CNN
        return:    (B, T, q_dim)
        """
        kv = self.kv_proj(key_value)          # (B, T, q_dim)
        attn_out, _ = self.attn(query, kv, kv)
        x = self.norm(query + attn_out)        # residual trên Q
        x = self.norm2(x + self.ffn(x))
        return x


class MySelfAttention(nn.Module):
    """Custom Self-Attention with sinusoidal Position Encoding.

    [Kỹ thuật: Multi-Head Self-Attention (custom implementation)
     - Sinusoidal Position Encoding (PE) cộng vào input trước khi attention
     - Separate Linear layers cho Q, K, V projections
     - Scaled dot-product attention: softmax(QK^T / √d_k) V
     - Multi-head: split → attend → concat
     - Output dim = attention_features (có thể ≠ input_features)]

    Dùng bởi: LGB (hầu hết modes), Siamese (lstm_sa, bilstm_sa)
    """
    def __init__(self, week_count, input_features, num_attention_heads, attention_features):
        super(MySelfAttention, self).__init__()
        self.week_count = week_count
        self.input_features = input_features
        self.num_attention_heads = num_attention_heads
        self.attention_features = attention_features
        self.attention_head_size = int(self.attention_features / self.num_attention_heads)
        self.all_head_size = attention_features

        # Position Embedding
        PE = torch.zeros((self.week_count, self.input_features))
        for i in range(1, PE.shape[0] + 1):
            for j in range(1, PE.shape[1] + 1):
                if j % 2 != 0:
                    twob = j - 1
                    expr = torch.exp(torch.tensor(twob * (-1 * torch.log(torch.tensor(10000 / PE.shape[0] + 1)))))
                    PE[i - 1][j - 1] = torch.cos(expr * i)
                else:
                    twob = j
                    expr = torch.exp(torch.tensor(twob * (-1 * torch.log(torch.tensor(10000 / PE.shape[0] + 1)))))
                    PE[i - 1][j - 1] = torch.sin(expr * i)
        self.register_buffer('PE', PE)
        self.key_layer = nn.Linear(self.input_features, self.attention_features)
        self.query_layer = nn.Linear(self.input_features, self.attention_features)
        self.value_layer = nn.Linear(self.input_features, self.attention_features)

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
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_probs, value_heads)
        attention_context = attention_context.permute(0, 2, 1, 3).contiguous()
        attention_newsize = attention_context.size()[:-2] + (self.all_head_size,)
        attention_output = attention_context.view(*attention_newsize)

        return attention_output


class LearnableQueryPool(nn.Module):
    """Learnable Query Pooling: dùng 1 learnable query vector
    attend vào sequence để tạo single summary vector.

    [Kỹ thuật: Learnable Query Pooling
     1 learnable parameter query (1,1,D) → nn.MultiheadAttention(query, seq, seq)
     → squeeze → single vector (B, D)]

    query (1, 1, D)  attend →  sequence (B, T, D)  →  context (B, D)

    Dùng bởi: LGB (bilstm_cnn), Siamese (lstm_attn, bilstm_attn), CL (lstm_attn, bilstm_attn)
    """
    def __init__(self, hidden_dim):
        super(LearnableQueryPool, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )

    def forward(self, x):
        """
        x: (B, T, D)
        return: context (B, D), attn_weights (B, T)
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)          # (B, 1, D)
        context, weights = self.attn(q, x, x)      # (B, 1, D), (B, 1, T)
        return context.squeeze(1), weights.squeeze(1)


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) Attention cho week-level aggregation.

    [Kỹ thuật: Bahdanau Attention (Bahdanau et al., 2015) — Additive Attention
     score_i = v^T · tanh(W_h · h_i + b)
     α = softmax(scores)
     context = Σ α_i · h_i]

    Input:  (B, T, hidden_size)
    Output: context (B, hidden_size), attn_weights (B, T)

    Dùng bởi: LGB (bilstm_cross), Baseline (DropoutPredictor — bahdanau variant)
    """
    def __init__(self, hidden_size=128):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs):
        """
        lstm_outputs: (B, T, hidden_size)
        return: context (B, hidden_size), attn_weights (B, T)
        """
        energy = torch.tanh(self.W_h(lstm_outputs))   # (B, T, hidden_size)
        scores = self.v(energy).squeeze(-1)            # (B, T)
        attn_weights = F.softmax(scores, dim=-1)       # (B, T)
        context = torch.bmm(
            attn_weights.unsqueeze(1),                 # (B, 1, T)
            lstm_outputs                               # (B, T, hidden_size)
        ).squeeze(1)                                   # (B, hidden_size)
        return context, attn_weights


# ============================================================
# Fusion
# ============================================================

class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion: temporal features attend to graph/context features.

    [Kỹ thuật: Multi-Head Cross-Attention Fusion
     Query = temporal features (from LSTM+SelfAttention)
     Key, Value = context features (from GraphSAGE)
     Multi-head scaled dot-product attention + residual + LayerNorm + FFN]

    Dùng bởi: LGB (mode: cross_attn)
    """
    def __init__(self, temporal_dim, context_dim, num_heads, output_dim, ffn_dim=None):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        if ffn_dim is None:
            ffn_dim = output_dim * 2

        # Multi-head cross-attention projections
        self.query_proj = nn.Linear(temporal_dim, output_dim)
        self.key_proj = nn.Linear(context_dim, output_dim)
        self.value_proj = nn.Linear(context_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        # LayerNorm + Residual
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

        # Projection for residual if dims don't match
        self.residual_proj = nn.Linear(temporal_dim, output_dim) if temporal_dim != output_dim else nn.Identity()

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, output_dim)
        )

    def forward(self, temporal_feat, context_feat):
        """
        temporal_feat: (B, T, temporal_dim) — e.g. (B, 5, 16)
        context_feat:  (B, T, context_dim)  — e.g. (B, 5, 16) (already expanded)
        return: (B, T, output_dim)
        """
        B, T, _ = temporal_feat.shape

        # Project Q, K, V
        Q = self.query_proj(temporal_feat)   # (B, T, output_dim)
        K = self.key_proj(context_feat)      # (B, T, output_dim)
        V = self.value_proj(context_feat)    # (B, T, output_dim)

        # Reshape for multi-head: (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, T, T)
        attn_output = torch.matmul(attn_weights, V)     # (B, heads, T, head_dim)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, self.output_dim)
        attn_output = self.out_proj(attn_output)

        # Residual + LayerNorm
        residual = self.residual_proj(temporal_feat)
        out = self.layer_norm1(attn_output + residual)

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(out)
        out = self.layer_norm2(ffn_out + out)

        return out


# ============================================================
# Classifier
# ============================================================

class Classifier(torch.nn.Module):
    """DNN classifier head.
    [Kỹ thuật: MLP 2-layer classifier (Linear→ReLU→Linear)]

    Dùng bởi: LGB (tất cả modes)
    """
    def __init__(self, param_dict):
        super(Classifier, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(param_dict['dnn_input_f1'], param_dict['dnn_hidden_f2']),  # 16→8
            nn.ReLU(),
            nn.Linear(param_dict['dnn_hidden_f2'], param_dict['dnn_output'])     # 8→1
        )

    def forward(self, x):
        return self.dnn(x)


# ============================================================
# Contrastive Learning Components
# ============================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    [Kỹ thuật: Supervised Contrastive Loss
     Trong cùng 1 batch, kéo embedding của cùng label lại gần nhau,
     đẩy embedding khác label ra xa trong không gian projection.
     Sử dụng temperature scaling τ (nhỏ hơn → phân biệt mạnh hơn)]

    Dùng bởi: LGB (contrastive mode), Siamese (tất cả modes)
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, proj_dim) HOẶC (B, n_views, proj_dim) — L2-normalized projected embeddings
            labels:   (B,) hoặc (B, 1) — binary labels (0/1)
        Returns:
            scalar loss
        """
        labels = labels.view(-1)  # (B,)

        # Nếu có n_views dimension: flatten thành (B*n_views, proj_dim)
        if features.dim() == 3:
            B, n_views, proj_dim = features.shape
            features = features.view(B * n_views, proj_dim)
            labels = labels.repeat_interleave(n_views)   # (B*n_views,)

        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Guard: nếu toàn batch cùng label → skip SupCon (không có negative pair)
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Mask: (B, B), mask[i][j] = 1 nếu labels[i] == labels[j]
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()  # (B, B)

        # Similarity matrix: (B, B)
        similarity = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Loại bỏ diagonal (self-similarity)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask  # Bỏ self khỏi positive set

        # Stability: trừ max trước khi exp
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # exp(sim) cho tất cả negative + positive (trừ self)
        exp_logits = torch.exp(logits) * logits_mask  # (B, B)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)  # (B, B)

        # Mean log-likelihood over positive pairs
        # Với mỗi anchor i, lấy trung bình log_prob trên tất cả positive j
        positive_count = mask.sum(dim=1)  # (B,) — số positive pairs cho mỗi anchor
        # Tránh chia cho 0 nếu anchor không có positive nào (edge case)
        positive_count = torch.clamp(positive_count, min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / positive_count  # (B,)

        loss = -mean_log_prob.mean()
        return loss


class ProjectionHead(nn.Module):
    """MLP Projection Head cho Contrastive Learning.
    Chiếu embedding từ temporal encoder vào không gian contrastive,
    sau đó L2-normalize.

    [Kỹ thuật: MLP 2-layer Projection Head → L2 Normalization
     Linear(in_dim, hidden_dim) → ReLU → Linear(hidden_dim, proj_dim) → L2-norm]

    Dùng bởi: LGB (contrastive mode)
    """
    def __init__(self, in_dim=16, hidden_dim=64, proj_dim=32):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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


# ============================================================
# Input Preprocessing (Optional Modules)
# ============================================================

class ActionWeightedInput(nn.Module):
    """Learnable importance weighting cho action types.

    [Kỹ thuật: Learnable Action Importance Weighting
     Học scalar weight w_i cho mỗi action type i (e.g. 22 types trong XuetangX).
     raw_weight → softmax → weights dương, sum=1.
     Output = x * weights (broadcast) — cùng shape, re-scaled theo feature dim.

     Interpretability: get_weights() trả về weights × num_actions
     → uniform baseline = 1.0 → dễ so sánh (>1 = quan trọng, <1 = bị suppress)]

    Dùng bởi: Siamese, SimCLR, BYOL (khi --action-weight ON)
    """
    def __init__(self, num_actions=22):
        super(ActionWeightedInput, self).__init__()
        self.num_actions = num_actions
        # Khởi tạo uniform (zeros → softmax = 1/num_actions cho mỗi action)
        self.raw_weight = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x):
        """
        x: (B, T, num_actions) e.g. (B, 35, 22)
        return: (B, T, num_actions) — same shape, re-scaled
        """
        # ×num_actions để uniform weights → scale factor = 1.0
        # (không thay đổi input khi model chưa học được gì)
        w = torch.softmax(self.raw_weight, dim=0) * self.num_actions  # uniform = 1.0
        return x * w.unsqueeze(0).unsqueeze(0)           # broadcast (B, T, F)

    def get_weights(self):
        """Trả về learned weights, normalized so uniform baseline = 1.0.

        Returns: (num_actions,) tensor on CPU
            - Value > 1.0 → action type quan trọng hơn average
            - Value < 1.0 → action type ít quan trọng
            - Value = 1.0 → đúng bằng uniform (chưa học được gì)
        """
        w = torch.softmax(self.raw_weight, dim=0) * self.num_actions
        return w.detach().cpu()


class EarlyPredictionMask(nn.Module):
    """Curriculum masking cho early dropout prediction.

    [Kỹ thuật: Curriculum Week Masking
     Training:  random chọn keep_weeks ∈ [min_weeks, max_weeks] PER-SAMPLE
                → zero-out tất cả timesteps sau keep_weeks × days_per_week
                → mỗi sample trong batch có thể thấy số tuần khác nhau (diversity)
     Inference: dùng eval_weeks tuần (set từ ngoài bằng set_eval_weeks(n))
                → eval từng week separately để đo early prediction capability]

    Dùng bởi: Siamese, SimCLR, BYOL (khi --early-prediction ON)
    """
    def __init__(self, week_count=5, days_per_week=7, min_weeks=2):
        super(EarlyPredictionMask, self).__init__()
        self.week_count = week_count
        self.days_per_week = days_per_week
        self.min_weeks = min_weeks
        self.eval_weeks = week_count  # mặc định full, thay đổi khi eval

    def forward(self, x):
        """
        x: (B, T, F) e.g. (B, 35, 22)
        return: (B, T, F) — zero-out tuần sau keep_weeks
        """
        B, T, F = x.shape

        if self.training:
            # Per-sample random weeks: shape (B,) — mỗi sample 1 giá trị riêng
            keep = torch.randint(
                self.min_weeks, self.week_count + 1, (B,),
                device=x.device
            )  # (B,) e.g. [3, 5, 2, 4, ...]
        else:
            # Inference: dùng eval_weeks cố định (set từ ngoài)
            keep = torch.full((B,), self.eval_weeks, dtype=torch.long,
                              device=x.device)

        # Tạo mask: (B, T)
        timesteps = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        keep_days = (keep * self.days_per_week).unsqueeze(1)        # (B, 1)
        mask = (timesteps < keep_days).float()                       # (B, T)

        return x * mask.unsqueeze(-1)   # (B, T, F) — zero-out phần sau

    def set_eval_weeks(self, n):
        """Gọi trước eval loop để chỉ định số tuần.
        QUAN TRỌNG: phải reset về week_count sau mỗi epoch eval.
        """
        self.eval_weeks = n


# ============================================================
# Data Augmentation
# ============================================================

class AugmentationModule(nn.Module):
    """Data augmentation cho Siamese/CL contrastive learning.
    Tạo hai "views" khác nhau của cùng một sample bằng random masking + noise.
    Chỉ nên gọi khi model.training == True (caller tự kiểm soát).

    [Kỹ thuật: Stochastic Data Augmentation
     1. Time masking: zero-out random timesteps (ngày) — Bernoulli mask
     2. Feature masking: zero-out random features (loại hoạt động) — Bernoulli mask
     3. Additive Gaussian noise: N(0, noise_std²)]

    Dùng bởi: Siamese (tất cả modes), SimCLR (tất cả modes), BYOL (tất cả modes)
    """
    def __init__(self, time_mask_ratio=0.15, feat_mask_ratio=0.15, noise_std=0.05):
        super(AugmentationModule, self).__init__()
        self.time_mask_ratio = time_mask_ratio
        self.feat_mask_ratio = feat_mask_ratio
        self.noise_std = noise_std

    def _augment_one(self, x):
        B, T, F_dim = x.shape
        out = x.clone()
        time_mask = (torch.rand(B, T, 1, device=x.device) > self.time_mask_ratio).float()
        out = out * time_mask
        feat_mask = (torch.rand(B, 1, F_dim, device=x.device) > self.feat_mask_ratio).float()
        out = out * feat_mask
        out = out + torch.randn_like(out) * self.noise_std
        return out

    def forward(self, x):
        """
        x: (B, T, F) e.g. (B, 35, 22)
        Returns: (view1, view2) — independently augmented
        """
        return self._augment_one(x), self._augment_one(x)
