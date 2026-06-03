"""
GraphEnhancedWrapper — Kiến trúc 2 luồng song song.

Luồng 1 (Temporal): SupCon / SimCLR / BYOL — xử lý seq_feat
Luồng 2 (Graph):    Context Embedding → GraphSage — xử lý graph structure

Fusion: concat(h_temporal, h_graph) → FC → ReLU → Classifier → logits

Điểm quan trọng:
    - Temporal model (SupCon/SimCLR/BYOL) KHÔNG biết gì về graph
    - Contrastive loss (SupCon/NT-Xent/BYOL) chỉ hoạt động trên Luồng 1
    - Classification loss (BCE) dùng representation đã fusion từ cả 2 luồng
    - Graph branch chạy độc lập, không bị ảnh hưởng bởi augmentation
"""
import torch
import torch.nn as nn

from .common import Context, GraphSage


class GraphEnhancedWrapper(nn.Module):
    """Wrapper kết hợp temporal model + graph branch.

    Architecture:
        Luồng 1: temporal_model.forward_features(sub_graph) → h_temporal (B, H)
        Luồng 2: Context → GraphSage → h_graph (B, G)
        Fusion:  concat(h_temporal, h_graph) → FC(H+G, H) → ReLU → Classifier → logits

    Supported temporal models: SupConLGB, CLSimCLR, CLBYOL
    Each must implement forward_features() method.

    Training output depends on temporal model type:
        SupCon/SimCLR: (logits_fused, z1, z2)
        BYOL:           (logits_fused, byol_loss)

    Inference:
        logits_fused (B, 1)
    """

    def __init__(self, temporal_model, param_dict, framework='supcon'):
        """
        Args:
            temporal_model: SupConLGB, CLSimCLR, hoặc CLBYOL instance
            param_dict: dict chứa graph-related params (context_dim, output_features, etc.)
            framework: 'supcon', 'simclr', hoặc 'byol' — xác định output format
        """
        super(GraphEnhancedWrapper, self).__init__()
        self.temporal_model = temporal_model
        self.framework = framework

        # --- Luồng 2: Graph branch ---
        self.context_embed = Context(param_dict)
        self.gnn = GraphSage(param_dict)

        # --- Fusion ---
        # H = temporal hidden_size (128), G = graph output_features (16)
        if framework == 'supcon':
            temporal_dim = param_dict.get('supcon_hidden_size', 128)
        else:
            temporal_dim = param_dict.get('cl_hidden_size', 128)
        graph_dim = param_dict.get('output_features', 16)

        self.fusion_fc = nn.Sequential(
            nn.Linear(temporal_dim + graph_dim, temporal_dim),
            nn.ReLU()
        )

        # --- Classifier (on fused representation) ---
        cls_dropout = 0.3
        if framework == 'supcon':
            from .supcon import SupConClassifier
            cls_hidden_layers = param_dict.get('supcon_cls_hidden_layers', 1)
            self.classifier = SupConClassifier(
                in_dim=temporal_dim, hidden_dim=64, dropout=cls_dropout,
                num_hidden_layers=cls_hidden_layers
            )
        else:
            from .contrastive import CLClassifier
            cls_hidden_layers = param_dict.get('cl_cls_hidden_layers', 1)
            self.classifier = CLClassifier(
                in_dim=temporal_dim, hidden_dim=64, dropout=cls_dropout,
                num_hidden_layers=cls_hidden_layers
            )

    # === Delegate attributes for compatibility ===
    # base_trainer/train.py access model.early_mask, model.action_weighting, etc.

    @property
    def early_mask(self):
        return getattr(self.temporal_model, 'early_mask', None)

    @property
    def action_weighting(self):
        return getattr(self.temporal_model, 'action_weighting', None)

    @property
    def week_count(self):
        return self.temporal_model.week_count

    def momentum_update(self):
        """Delegate to BYOL temporal model."""
        if hasattr(self.temporal_model, 'momentum_update'):
            self.temporal_model.momentum_update()

    def _graph_encode(self, sub_graph, batch_size):
        """Luồng 2: Context → GraphSage → h_graph (B, G)."""
        ctx = self.context_embed(sub_graph)
        ctx = self.gnn(ctx, sub_graph['edge_index'])
        return ctx[:batch_size]  # (B, output_features)

    def forward(self, sub_graph):
        """
        Luồng 1: temporal_model.forward_features() → h_temporal + contrastive info
        Luồng 2: Context → GraphSage → h_graph
        Fusion:  concat → FC → classifier → logits

        Returns depend on framework and training/eval mode:
            SupCon/SimCLR training: (logits_fused, z1, z2)
            BYOL training:          (logits_fused, byol_loss)
            Inference (all):        logits_fused
        """
        batch_size = sub_graph['batch_size']

        # === Luồng 2: Graph (independent) ===
        h_graph = self._graph_encode(sub_graph, batch_size)

        # === Luồng 1: Temporal ===
        temporal_out = self.temporal_model.forward_features(sub_graph)

        if self.training:
            if self.framework == 'byol':
                # BYOL: forward_features returns (h_online1, byol_loss)
                h_temporal, byol_loss = temporal_out
                h_fused = self.fusion_fc(torch.cat([h_temporal, h_graph], dim=-1))
                logits = self.classifier(h_fused)
                return logits, byol_loss
            else:
                # SupCon/SimCLR: forward_features returns (h1, z1, z2)
                h_temporal, z1, z2 = temporal_out
                h_fused = self.fusion_fc(torch.cat([h_temporal, h_graph], dim=-1))
                logits = self.classifier(h_fused)
                return logits, z1, z2
        else:
            # Inference: forward_features returns h (B, H)
            h_temporal = temporal_out
            h_fused = self.fusion_fc(torch.cat([h_temporal, h_graph], dim=-1))
            logits = self.classifier(h_fused)
            return logits
