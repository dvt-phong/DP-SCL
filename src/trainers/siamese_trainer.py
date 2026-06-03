"""
Siamese Trainer — Framework 1: Training cho Siamese Network (12 modes).

NO-GRAPH modes (6): siamese_lstm, siamese_bilstm, siamese_lstm_attn,
                     siamese_bilstm_attn, siamese_lstm_sa, siamese_bilstm_sa
GRAPH modes (6):    siamese_lstm_graph, siamese_bilstm_graph, siamese_lstm_attn_graph,
                     siamese_bilstm_attn_graph, siamese_lstm_sa_graph, siamese_bilstm_sa_graph
"""
import os
import torch

from .base_trainer import BaseTrainer
from .utils import load_temporal_data, load_graph_data, get_loss_fn
from src.models import SiameseLGB, SupConLoss


class SiameseTrainer(BaseTrainer):
    """Trainer cho Siamese Network (Framework 1).

    Handles cả NO-GRAPH và GRAPH modes:
        NO-GRAPH: load temporal data từ .npz → DataLoader(tensor, tensor)
        GRAPH:    load graph từ .pkl → DataLoader(sub_graph dict)
    """

    def __init__(self, args, ds_config, device):
        super().__init__(args, ds_config, device)
        self._is_graph = self.mode.endswith('_graph')

    def build_param_dict(self):
        args = self.args
        ds = self.ds_config
        _hid = args.hidden_size if args.hidden_size is not None else 128

        self.param_dict = {
            'activity_num': ds['activity_num'], 'sta_day': ds['sta_day'],
            'week_count': ds['week_count'], 'select_count': ds['week_count'],
            'cnn_in_channels': ds.get('days_per_week', 7),
            # Siamese-specific params
            'siamese_hidden_size': _hid,
            'siamese_proj_dim': _hid,
            'siamese_temperature': args.temperature if args.temperature is not None else 0.07,
            'siamese_mask_ratio': args.mask_ratio if args.mask_ratio is not None else 0.15,
            'siamese_noise_std': args.noise_std if args.noise_std is not None else 0.05,
            'siamese_attn_heads': 4,
            'siamese_cls_dropout': 0.3,
            'siamese_num_layers': args.num_layers if args.num_layers is not None else 1,
            'siamese_cls_hidden_layers': args.cls_layers,
            # Action Weighting & Early Prediction
            'use_action_weight':    getattr(args, 'action_weight', False),
            'use_early_prediction': getattr(args, 'early_prediction', False),
            'early_min_weeks':      getattr(args, 'early_min_weeks', 2),
        }

        # Bug 3 fix: thêm graph-related keys nếu _graph mode
        if self._is_graph:
            dataset_name = getattr(args, 'dataset', 'xuetangx')
            self.param_dict.update({
                # Context Embedding dims (khác nhau theo dataset)
                'org_context_feat_len':      7,
                'enhanced_context_feat_len': 32 if dataset_name == 'xuetangx' else 7,
                'context_each_embed':        16,
                'context_all_len':           16,
                # GraphSage
                'input_features':  16,
                'hidden_features': 32,
                'output_features': 16,
            })

        _extra = []
        if getattr(args, 'action_weight', False): _extra.append('ActionWeight=ON')
        if getattr(args, 'early_prediction', False): _extra.append(f'EarlyPred=ON(min={getattr(args, "early_min_weeks", 2)}w)')
        print(f"  [Siamese HP] hidden={self.param_dict['siamese_hidden_size']}, "
              f"τ={self.param_dict['siamese_temperature']}, "
              f"mask={self.param_dict['siamese_mask_ratio']}, "
              f"noise={self.param_dict['siamese_noise_std']}, "
              f"enc_layers={self.param_dict['siamese_num_layers']}, "
              f"cls_layers={self.param_dict['siamese_cls_hidden_layers']}, "
              f"λ={args.lambda_con}")
        if _extra: print(f"  [Siamese++] {', '.join(_extra)}")
        if self._is_graph: print(f"  [Siamese] GRAPH mode — GraphEnhancedWrapper enabled")

    def build_dataloaders(self):
        input_dir = os.path.abspath(os.path.expanduser(str(self.args.indir)))
        if self._is_graph:
            # Bug 1 fix: load graph data cho _graph modes
            self.train_loader, self.test_loader, self._graph = \
                load_graph_data(input_dir, self.ds_config,
                                self.args.batch_size, self.device)
            _train_labels = self._graph.labels[self._graph.train_mask]
            self._n_pos = int(_train_labels.sum().item())
            self._n_neg = len(_train_labels) - self._n_pos
        else:
            self.train_loader, self.test_loader, \
                self._train_labels, self._n_pos, self._n_neg = \
                load_temporal_data(input_dir, self.ds_config,
                                   self.args.batch_size, self.args.sampling)

    def build_model(self):
        temporal_model = SiameseLGB(mode=self.mode, param_dict=self.param_dict)
        if self._is_graph:
            from src.models import GraphEnhancedWrapper
            self.model = GraphEnhancedWrapper(temporal_model, self.param_dict, framework='siamese')
        else:
            self.model = temporal_model
        self.model = self.model.to(self.device)
        self.supcon_criterion = SupConLoss(
            temperature=self.param_dict.get('siamese_temperature', 0.07)
        ).to(self.device)

    def build_loss_fn(self):
        self.bce_loss_fn = get_loss_fn(self.args.sampling, self._n_pos, self._n_neg, self.device)

    def _parse_batch(self, batch):
        """Parse batch → (sub_graph, ground_truth, batch_size).

        Handles cả temporal tuple (NO-GRAPH) và graph dict/PyG Data (GRAPH).
        """
        if self._is_graph:
            if isinstance(batch, dict):
                # FullGraphBatchLoader → dict
                sub_graph = {k: v.to(self.device) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
            else:
                # NeighborLoader → PyG Data object
                sub_graph = {
                    'batch_size':        batch.batch_size,
                    'seq_feat':          batch.seq_feat.to(self.device),
                    'edge_index':        batch.edge_index.to(self.device),
                    'labels':            batch.labels.to(self.device),
                }
                if hasattr(batch, 'org_context'):
                    sub_graph['org_context'] = batch.org_context.to(self.device)
                if hasattr(batch, 'enhanced_context'):
                    sub_graph['enhanced_context'] = batch.enhanced_context.to(self.device)
            batch_size = sub_graph['batch_size']
            ground_truth = sub_graph['labels'][:batch_size].view(-1, 1).float()
        else:
            seq_feat_batch, labels_batch = batch
            seq_feat_batch = seq_feat_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            batch_size = seq_feat_batch.shape[0]
            sub_graph = {'batch_size': batch_size, 'seq_feat': seq_feat_batch}
            ground_truth = labels_batch.view(-1, 1).float()

        return sub_graph, ground_truth, batch_size

    def train_step(self, batch):
        sub_graph, ground_truth, batch_size = self._parse_batch(batch)

        pred, z1, z2 = self.model(sub_graph)
        bce_loss = self.bce_loss_fn(pred, ground_truth)
        features = torch.stack([z1, z2], dim=1)   # (B, 2, proj_dim)
        con_loss = self.supcon_criterion(features, ground_truth.view(-1))
        loss = bce_loss + self.args.lambda_con * con_loss

        return loss, batch_size

    def eval_step(self, batch):
        sub_graph, ground_truth, batch_size = self._parse_batch(batch)
        truth = ground_truth.to(torch.long)

        pred = self.model(sub_graph)
        return pred, truth
