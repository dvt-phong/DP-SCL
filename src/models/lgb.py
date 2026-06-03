"""
LGB (Learning Gain-Based) Model — Legacy graph-temporal baselines gốc.

[Kỹ thuật tổng quan: Time-Flow Hybrid Network (TFHN)
 Pipeline: Preprocessing → LSTM/BiLSTM Block 1 → SelfAttention 1
         → LSTM/BiLSTM Block 2 → SelfAttention 2
         → (optional) Graph Fusion → Classifier → Binary prediction]

Hỗ trợ 18 modes, chia thành 2 nhóm chính:

=== GRAPH-BASED MODES (cần graph + context data) ===

| Mode         | Preprocessing              | Temporal Encoder  | Attention       | Graph      | Fusion              |
|--------------|---------------------------|-------------------|-----------------|------------|---------------------|
| default      | Manual (sum_by_day/action) | LSTM              | SelfAttn+PE     | GraphSAGE  | Concat+WeightedSum  |
| cnn          | CNN 1D                    | LSTM              | SelfAttn+PE     | GraphSAGE  | Concat+WeightedSum  |
| cnn2d        | CNN 2D                    | LSTM              | SelfAttn+PE     | GraphSAGE  | Concat+WeightedSum  |
| gat          | Manual                    | LSTM              | SelfAttn+PE     | GAT        | Concat+WeightedSum  |
| cnn_gat      | CNN 1D                    | LSTM              | SelfAttn+PE     | GAT        | Concat+WeightedSum  |
| cross_attn   | Manual                    | LSTM              | SelfAttn+PE     | GraphSAGE  | CrossAttention       |
| mba_cnn      | MBA-CNN 3-branch          | LSTM              | SelfAttn+PE     | GraphSAGE  | Concat+WeightedSum  |
| mba_cnn_gat  | MBA-CNN 3-branch          | LSTM              | SelfAttn+PE     | GAT        | Concat+WeightedSum  |
| bilstm_graph | CNN 1D                    | BiLSTM            | SelfAttn+PE     | GraphSAGE  | Concat+WeightedSum  |

=== NO-GRAPH MODES (chỉ dùng temporal data, không cần graph) ===

| Mode         | Preprocessing              | Temporal Encoder  | Attention              | Pooling         |
|--------------|---------------------------|-------------------|------------------------|-----------------|
| no_graph     | Manual (sum_by_day/action) | LSTM              | SelfAttn+PE            | Mean pool       |
| bilstm_cnn   | CNN 1D + Temporal Diff    | BiLSTM            | SelfAttn+PE            | LearnableQuery  |
| bilstm_mha   | CNN 1D                    | BiLSTM            | Multi-Head Attention   | Mean pool       |
| bilstm_cross | CNN 1D                    | BiLSTM            | Cross-Attention(Q=BiLSTM, K/V=CNN) | Bahdanau |
| bilstm_graph | CNN 1D                    | BiLSTM            | SelfAttn+PE            | (graph fusion)  |
| mba_bilstm   | MBA-CNN 3-branch          | BiLSTM            | SelfAttn+PE            | Mean pool       |
| cnn_only     | CNN 1D                    | LSTM              | SelfAttn+PE            | Mean pool       |
| mba_only     | MBA-CNN 3-branch          | LSTM              | SelfAttn+PE            | Mean pool       |
| cnn_day      | CNN 1D (days as channels) | LSTM              | SelfAttn+PE            | Mean pool       |
| bilstm_day   | CNN 1D (days as channels) | BiLSTM            | SelfAttn+PE            | Mean pool       |

Giải thích viết tắt kỹ thuật:
    - LSTM:          Long Short-Term Memory (unidirectional)
    - BiLSTM:        Bidirectional LSTM (concat forward+backward)
    - SelfAttn+PE:   Custom Self-Attention + sinusoidal Position Encoding (MySelfAttention)
    - MHA:           nn.MultiheadAttention (standard Transformer block)
    - CrossAttn:     Cross-Attention (Q from one source, K/V from another)
    - GraphSAGE:     Graph Neural Network with mean aggregation
    - GAT:           Graph Attention Network
    - CNN 1D:        1D Convolutional Neural Network
    - CNN 2D:        2D Convolutional Neural Network
    - MBA-CNN:       Multi-Branch Asymmetric CNN (3 nhánh: temporal, daily, weekly)
    - LearnableQuery: Learnable query vector + nn.MultiheadAttention pooling
    - Bahdanau:      Additive Attention (Bahdanau et al., 2015)
    - SupConLoss:    Supervised Contrastive Loss (Khosla et al., 2020)
"""
import torch
import torch.nn.functional as F
from torch import nn as nn

from .common import (
    Context, GraphSage, GATNetwork,
    CNNFeatureExtractor, CNN2DFeatureExtractor, MBACNNFeatureExtractor,
    MyLSTM, MyBiLSTM,
    MySelfAttention, MyMHAttention, MyCrossAttention,
    LearnableQueryPool, CrossAttentionFusion,
    Classifier, ProjectionHead, BahdanauAttention,
)


class LGB(nn.Module):
    """Main legacy LGB/graph model with multi-mode support.

    Modes:
        - 'default':  Original architecture (GraphSAGE + manual preprocessing + LSTM)
        - 'cnn':      CNN replaces manual preprocessing before LSTM Block 1
        - 'gat':      GAT replaces GraphSAGE (manual preprocessing kept)
        - 'cnn_gat':  Both CNN and GAT modifications
    """
    def __init__(self, param_dict, mode='default', contrastive=False):
        super(LGB, self).__init__()
        assert mode in ('default', 'cnn', 'cnn2d', 'gat', 'cnn_gat', 'cross_attn', 'mba_cnn', 'mba_cnn_gat', 'no_graph', 'bilstm_cnn', 'bilstm_mha', 'bilstm_cross', 'bilstm_graph', 'mba_bilstm', 'cnn_only', 'mba_only', 'cnn_day', 'bilstm_day'), \
            f"mode must be one of: default, cnn, cnn2d, gat, cnn_gat, cross_attn, mba_cnn, mba_cnn_gat, no_graph, bilstm_cnn, bilstm_mha, bilstm_cross, bilstm_graph, mba_bilstm, cnn_only, mba_only, cnn_day, bilstm_day. Got: {mode}"
        self.mode = mode
        self.contrastive = contrastive
        self.week_count = param_dict['week_count']
        self.activity_num = param_dict['activity_num']
        self.select_count = param_dict['select_count']

        # === Context Embedding + Graph Network (skip for no_graph, bilstm_cnn) ===
        if mode not in ('no_graph', 'bilstm_cnn', 'bilstm_mha', 'bilstm_cross', 'mba_bilstm', 'cnn_only', 'mba_only', 'cnn_day', 'bilstm_day'):
            self.context_embed = Context(param_dict)

            # Graph Network: GraphSAGE (default/cnn) or GAT (gat/cnn_gat)
            if mode in ('gat', 'cnn_gat', 'mba_cnn_gat'):
                self.gnn = GATNetwork(param_dict)
            else:
                self.gnn = GraphSage(param_dict)

        # === CNN Feature Extractor ===
        if mode in ('cnn', 'cnn_gat'):
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        elif mode == 'cnn2d':
            self.cnn2d = CNN2DFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn2d_fc_output', 128)
        elif mode in ('mba_cnn', 'mba_cnn_gat'):
            self.mba_cnn = MBACNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('mba_cnn_output', 128)
        elif mode == 'bilstm_cnn':
            self.cnn = CNNFeatureExtractor(param_dict)
            # lstm1_input = cnn_fc_output * 2 (after temporal diff concat)
            lstm1_input = param_dict['lstm_input_features']  # 256 (128 orig + 128 diff)
        elif mode == 'bilstm_mha':
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        elif mode == 'bilstm_cross':
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        elif mode == 'bilstm_graph':
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        elif mode == 'mba_bilstm':
            self.mba_cnn = MBACNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('mba_cnn_output', 128)
        elif mode == 'cnn_only':
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        elif mode == 'mba_only':
            self.mba_cnn = MBACNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('mba_cnn_output', 128)
        elif mode == 'cnn_day':
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        elif mode == 'bilstm_day':
            self.cnn = CNNFeatureExtractor(param_dict)
            lstm1_input = param_dict.get('cnn_fc_output', 128)
        else:
            # LSTM1 input = manual preprocessing output (default: 184)
            lstm1_input = param_dict['lstm_input_features']

        # === TFHN (Time-Flow Hybrid Network) ===
        if mode == 'bilstm_cnn':
            # BiLSTM: hidden_features is halved, output = hidden*2
            lstm1_hidden = param_dict['lstm_hidden_features']  # 64 → output 128
            lstm2_hidden = param_dict['l2_hidden_features']    # 16 → output 32

            self.lstm1 = MyBiLSTM(lstm1_input, lstm1_hidden,
                                  param_dict['lstm_hidden_num_layers'])
            # SA1 input = lstm1_hidden * 2 (BiLSTM output)
            self.self_attention1 = MySelfAttention(
                param_dict['week_count'], lstm1_hidden * 2,
                param_dict['num_attention_heads'],
                param_dict['attention_features'])

            self.lstm2 = MyBiLSTM(param_dict['l2_input_features'], lstm2_hidden,
                                  param_dict['l2_hidden_num_layers'])
            # SA2 input = lstm2_hidden * 2 (BiLSTM output)
            self.self_attention2 = MySelfAttention(
                param_dict['week_count'], lstm2_hidden * 2,
                param_dict['s2_num_attention_heads'],
                param_dict['s2_attention_features'])
            # Learnable Query Pool thay mean pool
            self.week_pool = LearnableQueryPool(
                hidden_dim=param_dict['s2_attention_features']  # 16
            )
        elif mode == 'bilstm_mha':
            lstm1_hidden = param_dict['lstm_hidden_features']   # 64 → BiLSTM1 out = 128
            lstm2_hidden = param_dict['l2_hidden_features']     # 32 → BiLSTM2 out = 64
            num_heads    = param_dict.get('mha_num_heads', 4)

            self.lstm1 = MyBiLSTM(lstm1_input, lstm1_hidden,
                                  param_dict['lstm_hidden_num_layers'])
            # MHA1: embed_dim = 128, giữ nguyên dim
            self.self_attention1 = MyMHAttention(
                embed_dim=lstm1_hidden * 2,
                num_heads=num_heads
            )
            # BiLSTM2 input = 128 (MHA giữ nguyên dim, khác bilstm_cnn)
            self.lstm2 = MyBiLSTM(lstm1_hidden * 2, lstm2_hidden,
                                  param_dict['l2_hidden_num_layers'])
            # MHA2: embed_dim = 64
            self.self_attention2 = MyMHAttention(
                embed_dim=lstm2_hidden * 2,
                num_heads=num_heads
            )
            # Classifier riêng: 64 → 32 → 1
            self.bilstm_mha_classifier = nn.Sequential(
                nn.Linear(lstm2_hidden * 2, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        elif mode == 'bilstm_cross':
            lstm1_hidden = param_dict['lstm_hidden_features']   # 64 → out 128
            lstm2_hidden = param_dict['l2_hidden_features']     # 32 → out 64
            num_heads    = param_dict.get('mha_num_heads', 4)

            self.lstm1 = MyBiLSTM(lstm1_input, lstm1_hidden,
                                  param_dict['lstm_hidden_num_layers'])
            # CA1: Q=BiLSTM1(128), K/V=CNN(128) → 128
            self.self_attention1 = MyCrossAttention(
                q_dim=lstm1_hidden * 2,    # 128
                kv_dim=128,                # cnn_fc_output
                num_heads=num_heads
            )
            # BiLSTM2 input = 128 (CA giữ dim)
            self.lstm2 = MyBiLSTM(lstm1_hidden * 2, lstm2_hidden,
                                  param_dict['l2_hidden_num_layers'])
            # CA2: Q=BiLSTM2(64), K/V=CNN(128) → 64 (kv_proj tự handle)
            self.self_attention2 = MyCrossAttention(
                q_dim=lstm2_hidden * 2,    # 64
                kv_dim=128,                # vẫn dùng cnn_out gốc
                num_heads=num_heads
            )
            # Bahdanau pool thay mean pool
            self.week_pool = BahdanauAttention(hidden_size=lstm2_hidden * 2)  # 64

            self.bilstm_cross_classifier = nn.Sequential(
                nn.Linear(lstm2_hidden * 2, 32),   # 64→32
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        elif mode == 'bilstm_graph':
            # BiLSTM thay LSTM, giữ nguyên MySelfAttention
            lstm1_hidden = param_dict['lstm_hidden_features']  # 64 → output 128
            lstm2_hidden = param_dict['l2_hidden_features']    # 16 → output 32

            self.lstm1 = MyBiLSTM(lstm1_input, lstm1_hidden,
                                  param_dict['lstm_hidden_num_layers'])
            self.self_attention1 = MySelfAttention(
                param_dict['week_count'], lstm1_hidden * 2,  # 128
                param_dict['num_attention_heads'],
                param_dict['attention_features'])

            self.lstm2 = MyBiLSTM(param_dict['l2_input_features'], lstm2_hidden,
                                  param_dict['l2_hidden_num_layers'])
            self.self_attention2 = MySelfAttention(
                param_dict['week_count'], lstm2_hidden * 2,  # 32
                param_dict['s2_num_attention_heads'],
                param_dict['s2_attention_features'])
        elif mode in ('mba_bilstm', 'bilstm_day'):
            # MBA-CNN + BiLSTM, no graph
            lstm1_hidden = param_dict['lstm_hidden_features']  # 64 → output 128
            lstm2_hidden = param_dict['l2_hidden_features']    # 16 → output 32

            self.lstm1 = MyBiLSTM(lstm1_input, lstm1_hidden,
                                  param_dict['lstm_hidden_num_layers'])
            self.self_attention1 = MySelfAttention(
                param_dict['week_count'], lstm1_hidden * 2,
                param_dict['num_attention_heads'],
                param_dict['attention_features'])

            self.lstm2 = MyBiLSTM(param_dict['l2_input_features'], lstm2_hidden,
                                  param_dict['l2_hidden_num_layers'])
            self.self_attention2 = MySelfAttention(
                param_dict['week_count'], lstm2_hidden * 2,
                param_dict['s2_num_attention_heads'],
                param_dict['s2_attention_features'])
        else:
            self.lstm1 = MyLSTM(lstm1_input, param_dict['lstm_hidden_features'],
                                param_dict['lstm_hidden_num_layers'])

            self.self_attention1 = MySelfAttention(param_dict['week_count'], param_dict['lstm_hidden_features'],
                                                   param_dict['num_attention_heads'],
                                                   param_dict['attention_features'])

            self.lstm2 = MyLSTM(param_dict['l2_input_features'], param_dict['l2_hidden_features'],
                                param_dict['l2_hidden_num_layers'])

            self.self_attention2 = MySelfAttention(param_dict['week_count'], param_dict['l2_hidden_features'],
                                                   param_dict['s2_num_attention_heads'],
                                                   param_dict['s2_attention_features'])

        # === Fusion (skip for no_graph, bilstm_cnn) ===
        if mode not in ('no_graph', 'bilstm_cnn', 'bilstm_mha', 'bilstm_cross', 'mba_bilstm', 'cnn_only', 'mba_only', 'cnn_day', 'bilstm_day'):
            self.weighted_sum = MySelfAttention(param_dict['week_count'], param_dict['ws_input_features'],
                                                param_dict['ws_num_attention_heads'], param_dict['ws_attention_features'])

            # Cross-Attention Fusion (cross_attn mode)
            if mode == 'cross_attn':
                ca_temporal_dim = param_dict['s2_attention_features']   # 16 (output of self_attention2)
                ca_context_dim = param_dict['output_features']          # 16 (output of GraphSAGE)
                ca_num_heads = param_dict.get('ca_num_heads', 4)
                ca_output_dim = param_dict.get('ca_output_dim', 16)
                ca_ffn_dim = param_dict.get('ca_ffn_dim', 32)
                self.cross_attn_fusion = CrossAttentionFusion(
                    temporal_dim=ca_temporal_dim,
                    context_dim=ca_context_dim,
                    num_heads=ca_num_heads,
                    output_dim=ca_output_dim,
                    ffn_dim=ca_ffn_dim
                )

        self.classifier = Classifier(param_dict)

        # === Supervised Contrastive Learning ===
        if contrastive:
            proj_in_dim = param_dict['s2_attention_features']  # 16 (output of temporal encoder)
            proj_hidden = param_dict.get('proj_hidden_dim', 64)
            proj_out = param_dict.get('proj_output_dim', 32)
            self.projection_head = ProjectionHead(proj_in_dim, proj_hidden, proj_out)

    def _preprocess_default(self, seq_feat, batch_size):
        """Original manual preprocessing: sum_by_day + sum_by_action + flatten."""
        input_matrix = seq_feat.view(batch_size, self.week_count, -1, self.activity_num)
        lstm_input = None
        for i in range(self.week_count):
            x = input_matrix[:, i, :, :]
            act_sum_by_day = torch.sum(x, dim=2).view(batch_size, -1, 1)
            x = torch.cat((x, act_sum_by_day), dim=2)
            act_sum_by_action = torch.sum(x, dim=1).view(batch_size, 1, -1)
            x = torch.cat((x, act_sum_by_action), dim=1)
            x = x.view(batch_size, 1, -1)
            if i == 0:
                lstm_input = x
            else:
                lstm_input = torch.cat((lstm_input, x), dim=1)
        return lstm_input

    def _preprocess_cnn(self, seq_feat, batch_size):
        """CNN 1D preprocessing: feed raw (days, activities) into Conv1D."""
        input_matrix = seq_feat.view(batch_size, self.week_count, -1, self.activity_num)
        B, T, D, A = input_matrix.shape
        cnn_input = input_matrix.view(B * T, D, A)  # (B*5, 7, 22)
        if self.mode in ('bilstm_cnn', 'bilstm_mha', 'bilstm_cross', 'bilstm_graph', 'cnn_only'):
            # Activities as channels, days as sequence → capture intra-week temporal patterns
            cnn_input = cnn_input.permute(0, 2, 1)   # (B*5, 22, 7)
        # else: original convention — channels=days(7), seq=activities(22)
        cnn_output = self.cnn(cnn_input)             # (B*5, cnn_fc_output)
        lstm_input = cnn_output.view(B, T, -1)       # (B, 5, cnn_fc_output)
        return lstm_input

    def _preprocess_cnn2d(self, seq_feat, batch_size):
        """CNN 2D preprocessing: treat weekly data as 2D image (1, days, activities)."""
        input_matrix = seq_feat.view(batch_size, self.week_count, -1, self.activity_num)
        B, T, D, A = input_matrix.shape
        cnn_input = input_matrix.view(B * T, 1, D, A)  # (B*5, 1, 7, 22) — single channel 2D
        cnn_output = self.cnn2d(cnn_input)              # (B*5, cnn2d_fc_output)
        lstm_input = cnn_output.view(B, T, -1)          # (B, 5, cnn2d_fc_output)
        return lstm_input

    def _preprocess_mba_cnn(self, seq_feat, batch_size):
        """MBA-CNN preprocessing: treat weekly data as 2D image (1, days, activities)."""
        input_matrix = seq_feat.view(batch_size, self.week_count, -1, self.activity_num)
        B, T, D, A = input_matrix.shape
        cnn_input = input_matrix.view(B * T, 1, D, A)  # (B*5, 1, 7, 22)
        cnn_output = self.mba_cnn(cnn_input)            # (B*5, mba_cnn_output)
        lstm_input = cnn_output.view(B, T, -1)          # (B, 5, mba_cnn_output)
        return lstm_input

    def _preprocess_mba_bilstm(self, seq_feat, batch_size):
        """MBA-CNN (no graph) preprocessing: weekly data as 2D image (1, days, activities)."""
        input_matrix = seq_feat.view(batch_size, self.week_count, -1, self.activity_num)
        B, T, D, A = input_matrix.shape
        cnn_input = input_matrix.view(B * T, 1, D, A)   # (B*5, 1, 7, 22)
        cnn_output = self.mba_cnn(cnn_input)             # (B*5, mba_cnn_output)
        return cnn_output.view(B, T, -1)                 # (B, 5, mba_cnn_output)

    def forward(self, sub_graph):
        batch_size = sub_graph['batch_size']

        # === Temporal branch: preprocessing ===
        seq_feat = sub_graph['seq_feat'][:batch_size]
        if self.mode in ('cnn', 'cnn_gat', 'bilstm_cnn', 'bilstm_mha', 'bilstm_cross', 'bilstm_graph', 'cnn_only', 'cnn_day', 'bilstm_day'):
            lstm_input = self._preprocess_cnn(seq_feat, batch_size)

            # === Temporal Difference features (bilstm_cnn only) ===
            if self.mode == 'bilstm_cnn':
                B = lstm_input.size(0)
                diff = lstm_input[:, 1:, :] - lstm_input[:, :-1, :]  # (B, 4, 128)
                pad = torch.zeros(B, 1, lstm_input.size(2), device=lstm_input.device)
                diff = torch.cat([pad, diff], dim=1)                 # (B, 5, 128)
                lstm_input = torch.cat([lstm_input, diff], dim=2)    # (B, 5, 256)
        elif self.mode == 'cnn2d':
            lstm_input = self._preprocess_cnn2d(seq_feat, batch_size)
        elif self.mode in ('mba_cnn', 'mba_cnn_gat', 'mba_only'):
            lstm_input = self._preprocess_mba_cnn(seq_feat, batch_size)
        elif self.mode == 'mba_bilstm':
            lstm_input = self._preprocess_mba_bilstm(seq_feat, batch_size)
        else:
            lstm_input = self._preprocess_default(seq_feat, batch_size)

        # === bilstm_cross: separate forward path (cross-attention needs cnn_out) ===
        if self.mode == 'bilstm_cross':
            cnn_out = lstm_input.clone()   # (B, 5, 128) — giữ lại làm K/V
            lstm_output  = self.lstm1(lstm_input)
            attn_output1 = self.self_attention1(lstm_output, cnn_out)
            lstm_output2 = self.lstm2(attn_output1)
            attn_output2 = self.self_attention2(lstm_output2, cnn_out)

            all_feat, _ = self.week_pool(attn_output2)     # (B, 64)
            pred = self.bilstm_cross_classifier(all_feat)
            if self.contrastive:
                proj_embed = self.projection_head(all_feat)
                return pred, proj_embed
            return pred

        # === TFHN blocks ===
        lstm_output = self.lstm1(lstm_input)
        attention_output = self.self_attention1(lstm_output)
        lstm_ouput2 = self.lstm2(attention_output)
        attention_output2 = self.self_attention2(lstm_ouput2)

        # === no_graph / bilstm_cnn mode: skip graph/context, predict directly from temporal ===
        if self.mode == 'bilstm_mha':
            all_mean_feat = torch.mean(attention_output2, dim=1)  # (B, 64)
            pred = self.bilstm_mha_classifier(all_mean_feat)
            if self.contrastive:
                proj_embed = self.projection_head(all_mean_feat)
                return pred, proj_embed
            return pred

        if self.mode in ('no_graph', 'bilstm_cnn', 'mba_bilstm', 'cnn_only', 'mba_only', 'cnn_day', 'bilstm_day'):
            if self.mode == 'bilstm_cnn':
                all_mean_feat, _ = self.week_pool(attention_output2)  # (B, 16)
            else:
                all_mean_feat = torch.mean(attention_output2, dim=1)  # (B, 16)
            pred = self.classifier(all_mean_feat)
            if self.contrastive:
                proj_embed = self.projection_head(all_mean_feat)
                return pred, proj_embed
            return pred

        # === Context + Graph branch ===
        context = self.context_embed(sub_graph)
        context = self.gnn(context, sub_graph['edge_index'])
        context_output = context[:batch_size]

        # === Fusion ===
        context_output = context_output.repeat_interleave(attention_output2.size(1), 0)
        context_output = context_output.view(batch_size, attention_output2.size(1), -1)

        if self.mode == 'cross_attn':
            # Cross-Attention: temporal attends to context
            all_feat = self.cross_attn_fusion(attention_output2, context_output)
        else:
            # Default fusion: concat + weighted sum
            weighted_sum_input = torch.cat((context_output, attention_output2), dim=2)
            all_feat = self.weighted_sum(weighted_sum_input)

        all_mean_feat = torch.mean(all_feat, dim=1)
        pred = self.classifier(all_mean_feat)
        if self.contrastive:
            # Contrastive projection trên temporal embedding (trước fusion)
            temporal_mean = torch.mean(attention_output2, dim=1)  # (B, 16)
            proj_embed = self.projection_head(temporal_mean)
            return pred, proj_embed
        return pred
