"""
Shared Training Utilities — dùng chung cho tất cả trainers.

Bao gồm:
    • FocalLoss      — Focal Loss cho class imbalance
    • get_loss_fn()  — Tạo loss function theo sampling strategy
    • load_temporal_data() — Load npz → DataLoader (no-graph modes)
    • load_graph_data()    — Load graph pkl → DataLoader (graph modes)
    • apply_sampling()     — SMOTE/oversample/undersample logic
    • compute_metrics()    — AUC, ACC, F1, Precision, Recall, optimal threshold
"""
import os
import pickle
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class FocalLoss(torch.nn.Module):
    """Focal Loss for class imbalance: down-weights easy examples.

    [Kỹ thuật: Focal Loss (Lin et al., 2017)
     FL(p) = -α(1-p)^γ log(p) for positive class
     FL(p) = -(1-α)p^γ log(1-p) for negative class
     α: weighting factor, γ: focusing parameter]
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none',
                                                  pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def get_loss_fn(sampling_strategy, n_pos, n_neg, device):
    """Tạo loss function theo sampling strategy.

    Args:
        sampling_strategy: 'none', 'weighted_loss', 'focal_loss', etc.
        n_pos: number of positive samples in training
        n_neg: number of negative samples in training
        device: torch device

    Returns:
        loss_fn: callable(logits, targets) → scalar loss
    """
    pos_weight = None
    if sampling_strategy in ('weighted_loss', 'focal_loss'):
        pw = n_neg / max(n_pos, 1)
        pos_weight = torch.tensor([pw], device=device)
        print(f"  [Sampling] pos_weight={pw:.4f}")

    if sampling_strategy == 'focal_loss':
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight).to(device)
        print(f"  [Sampling] FocalLoss (α=0.25, γ=2.0)")
    elif sampling_strategy == 'weighted_loss':
        loss_fn = lambda logits, targets: F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight)
        print(f"  [Sampling] Weighted BCE Loss")
    else:
        loss_fn = lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets)

    print(f"  [Sampling] Strategy: {sampling_strategy}")
    return loss_fn


def apply_sampling(train_seq_flat, train_labels, sampling_strategy, batch_size):
    """Apply sampling strategy and create DataLoader.

    Args:
        train_seq_flat: (N, flat_dim) tensor
        train_labels: (N,) tensor
        sampling_strategy: string
        batch_size: int

    Returns:
        train_loader: DataLoader
        train_seq_flat: possibly resampled tensor
        train_labels: possibly resampled tensor
    """
    N_train = len(train_labels)
    n_pos = int(train_labels.sum().item())
    n_neg = N_train - n_pos

    train_dataset = TensorDataset(train_seq_flat, train_labels)

    if sampling_strategy in ('oversample', 'undersample'):
        w_pos = 1.0 / n_pos if n_pos > 0 else 1.0
        w_neg = 1.0 / n_neg if n_neg > 0 else 1.0
        sample_weights = torch.tensor([w_pos if l == 1 else w_neg for l in train_labels])
        if sampling_strategy == 'oversample':
            num_samples = 2 * max(n_pos, n_neg)
        else:
            num_samples = 2 * min(n_pos, n_neg)
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        print(f"  Sampling: {sampling_strategy} → {num_samples} samples/epoch")

    elif sampling_strategy in ('smote', 'adasyn', 'borderline_smote', 'smote_tomek', 'smote_enn'):
        try:
            _RESAMPLERS = {
                'smote':            ('imblearn.over_sampling', 'SMOTE'),
                'adasyn':           ('imblearn.over_sampling', 'ADASYN'),
                'borderline_smote': ('imblearn.over_sampling', 'BorderlineSMOTE'),
                'smote_tomek':      ('imblearn.combine',       'SMOTETomek'),
                'smote_enn':        ('imblearn.combine',       'SMOTEENN'),
            }
            _mod, _cls = _RESAMPLERS[sampling_strategy]
            resampler = getattr(importlib.import_module(_mod), _cls)(random_state=42)
        except ModuleNotFoundError:
            print("  ❌ 'imbalanced-learn' chưa được cài. Chạy: pip install imbalanced-learn")
            raise SystemExit(1)

        X_resampled, y_resampled = resampler.fit_resample(
            train_seq_flat.numpy(), train_labels.numpy().astype(int)
        )
        train_seq_flat = torch.from_numpy(X_resampled).to(torch.float)
        train_labels = torch.from_numpy(y_resampled).to(torch.float)
        N_train = len(train_labels)
        n_pos_new = int(train_labels.sum().item())
        n_neg_new = N_train - n_pos_new
        train_dataset = TensorDataset(train_seq_flat, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"  Sampling: {sampling_strategy.upper()} → {N_train} samples (pos={n_pos_new}, neg={n_neg_new})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if sampling_strategy != 'none':
            print(f"  Sampling: {sampling_strategy} (applied at loss level)")

    return train_loader, train_seq_flat, train_labels


def load_temporal_data(input_dir, ds_config, batch_size, sampling_strategy='none'):
    """Load temporal data from npz file, apply sampling, return DataLoaders.

    Returns:
        train_loader, test_loader, train_labels, n_pos, n_neg
    """
    npz_path = os.path.join(input_dir, 'datastore', ds_config['npz_filename'])
    print(f"Loading temporal data from: {npz_path}")
    data = np.load(npz_path)

    train_seq = torch.from_numpy(data['t_data']).to(torch.float)
    train_labels = torch.from_numpy(data['t_label']).to(torch.float)
    test_seq = torch.from_numpy(data['v_data']).to(torch.float)
    test_labels = torch.from_numpy(data['v_label']).to(torch.float)

    N_train = train_seq.shape[0]
    N_test = test_seq.shape[0]
    train_seq_flat = train_seq.view(N_train, -1)
    test_seq_flat = test_seq.view(N_test, -1)
    print(f"  Data shape: {list(train_seq.shape)} → flat {train_seq_flat.shape[1]}")

    n_pos = int(train_labels.sum().item())
    n_neg = N_train - n_pos
    print(f"  Class distribution: pos(dropout)={n_pos} ({n_pos/N_train:.3f}), neg={n_neg} ({n_neg/N_train:.3f})")

    train_loader, train_seq_flat, train_labels = apply_sampling(
        train_seq_flat, train_labels, sampling_strategy, batch_size
    )

    test_dataset = TensorDataset(test_seq_flat, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=max(1, batch_size // 2), shuffle=False)

    n_pos = int(train_labels.sum().item())
    n_neg = len(train_labels) - n_pos

    print(f"  Train samples: {len(train_labels)}, Test samples: {N_test}")
    print(f"  Seq feat shape: {train_seq_flat.shape}")

    return train_loader, test_loader, train_labels, n_pos, n_neg


def load_graph_data(input_dir, ds_config, batch_size, device):
    """Load graph data from pickle, return DataLoaders.

    Returns:
        train_loader, test_loader, graph
    """
    graph_path = os.path.join(input_dir, 'datastore', ds_config['graph_filename'])
    print(f"Loading graph from: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    graph = graph.to(device)

    try:
        from torch_geometric.loader import NeighborLoader
        train_loader = NeighborLoader(
            data=graph, num_neighbors=[8, 4],
            input_nodes=graph.train_mask,
            batch_size=batch_size, shuffle=True
        )
        test_loader = NeighborLoader(
            data=graph, num_neighbors=[8, 4],
            input_nodes=graph.test_mask,
            batch_size=max(1, batch_size // 2), shuffle=True
        )
        # Quick test: trigger actual sampling
        _test_iter = iter(train_loader)
        _test_batch = next(_test_iter)
        del _test_iter, _test_batch
        # Recreate train_loader since we consumed one batch
        train_loader = NeighborLoader(
            data=graph, num_neighbors=[8, 4],
            input_nodes=graph.train_mask,
            batch_size=batch_size, shuffle=True
        )
        print("  Using NeighborLoader (pyg-lib/torch-sparse)")
    except (ImportError, Exception) as e:
        print(f"  NeighborLoader unavailable ({type(e).__name__}), using FullGraphBatchLoader")

        class FullGraphBatchLoader:
            """Mini-batch loader that wraps full graph — each batch is a sub_graph dict."""
            def __init__(self, graph, node_mask, batch_size, shuffle=True):
                self.graph = graph
                self.node_indices = node_mask
                self.batch_size = batch_size
                self.shuffle = shuffle

            def __iter__(self):
                indices = self.node_indices.clone()
                if self.shuffle:
                    perm = torch.randperm(len(indices), device=indices.device)
                    indices = indices[perm]
                N = self.graph.labels.shape[0]
                all_nodes = torch.arange(N, device=indices.device)
                for start in range(0, len(indices), self.batch_size):
                    batch_nodes = indices[start:start + self.batch_size]
                    bs = len(batch_nodes)
                    batch_set = set(batch_nodes.cpu().tolist())
                    other_nodes = torch.tensor([n for n in range(N) if n not in batch_set],
                                               device=indices.device, dtype=torch.long)
                    ordered = torch.cat([batch_nodes, other_nodes])
                    remap = torch.empty(N, dtype=torch.long, device=indices.device)
                    remap[ordered] = torch.arange(N, device=indices.device)
                    edge_index = remap[self.graph.edge_index]
                    sub_graph = {
                        'batch_size': bs,
                        'labels': self.graph.labels[ordered],
                        'seq_feat': self.graph.seq_feat[ordered],
                        'edge_index': edge_index,
                    }
                    if hasattr(self.graph, 'org_context'):
                        sub_graph['org_context'] = self.graph.org_context[ordered]
                    if hasattr(self.graph, 'enhanced_context'):
                        sub_graph['enhanced_context'] = self.graph.enhanced_context[ordered]
                    yield sub_graph

            def __len__(self):
                return (len(self.node_indices) + self.batch_size - 1) // self.batch_size

        train_loader = FullGraphBatchLoader(graph, graph.train_mask, batch_size, shuffle=True)
        test_loader = FullGraphBatchLoader(graph, graph.test_mask, max(1, batch_size // 2), shuffle=False)

    return train_loader, test_loader, graph


def compute_metrics(pred_tensor, ground_truth_np, pred_scores_np):
    """Compute evaluation metrics with optimal threshold.

    Args:
        pred_tensor: raw logits tensor (N, 1) — reserved for API consistency
                     (not used in current computation, kept for future calibration)
        ground_truth_np: numpy (N, 1) labels
        pred_scores_np: numpy (N, 1) sigmoid scores

    Returns:
        dict with acc, precision, recall, f1, auc, optimal_threshold
    """
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score

    fpr, tpr, thresholds = roc_curve(ground_truth_np, pred_scores_np)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    pred_label = pred_scores_np.copy()
    pred_label[pred_label < optimal_threshold] = 0
    pred_label[pred_label >= optimal_threshold] = 1

    auc = roc_auc_score(ground_truth_np, pred_scores_np)
    acc = np.sum(pred_label == ground_truth_np) / len(ground_truth_np)
    precision = precision_score(ground_truth_np, pred_label)
    recall = recall_score(ground_truth_np, pred_label)
    f1 = f1_score(ground_truth_np, pred_label)

    return {
        'acc': acc, 'precision': precision, 'recall': recall,
        'f1': f1, 'auc': auc, 'optimal_threshold': optimal_threshold,
    }
