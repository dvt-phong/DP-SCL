"""
Abstract Base Trainer — template cho tất cả framework trainers.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from .utils import compute_metrics


class BaseTrainer(ABC):
    """Abstract base class cho training loop.

    Subclasses cần implement:
        - build_model()        → self.model
        - build_dataloaders()  → self.train_loader, self.test_loader
        - build_loss_fn()      → self.bce_loss_fn (+ contrastive loss nếu cần)
        - train_step(batch)    → loss (scalar tensor)
        - eval_step(batch)     → (pred, labels)
    """

    def __init__(self, args, ds_config, device):
        self.args = args
        self.ds_config = ds_config
        self.device = device
        self.mode = args.mode

    @abstractmethod
    def build_model(self):
        """Build and return model, set self.model."""
        pass

    @abstractmethod
    def build_dataloaders(self):
        """Build train_loader and test_loader, set self.train_loader, self.test_loader."""
        pass

    @abstractmethod
    def build_loss_fn(self):
        """Build loss function(s), set self.bce_loss_fn (and any contrastive loss)."""
        pass

    @abstractmethod
    def build_param_dict(self):
        """Build hyperparameter dict, set self.param_dict."""
        pass

    @abstractmethod
    def train_step(self, batch):
        """One training step. Returns scalar loss tensor."""
        pass

    @abstractmethod
    def eval_step(self, batch):
        """One eval step. Returns (pred_logits, ground_truth) tensors."""
        pass

    def run(self):
        """Main training loop."""
        self.build_param_dict()
        self.build_dataloaders()
        self.build_model()
        self.build_loss_fn()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        for epoch in range(self.args.e):
            # === Train ===
            total_loss, total_examples = 0, 0
            self.model.train()
            for batch in tqdm(self.train_loader):
                optimizer.zero_grad()
                loss, batch_size = self.train_step(batch)
                loss.backward()
                optimizer.step()
                self.after_optimizer_step()
                total_loss += float(loss) * batch_size
                total_examples += batch_size
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

            # === Eval ===
            preds = []
            ground_truths = []
            self.model.eval()
            for batch in tqdm(self.test_loader):
                with torch.no_grad():
                    pred, truth = self.eval_step(batch)
                    preds.append(pred)
                    ground_truths.append(truth)

            pred = torch.cat(preds, dim=0).cpu()
            ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
            pred_scores = torch.sigmoid(pred).numpy()

            metrics = compute_metrics(pred, ground_truth, pred_scores)

            print(f"Epoch: {epoch:03d}, Optimal Threshold: {metrics['optimal_threshold']:.4f}")
            print(f"Epoch: {epoch:03d}, Test ACC: {metrics['acc']:.4f}")
            print(f"Epoch: {epoch:03d}, Test Precision: {metrics['precision']:.4f}")
            print(f"Epoch: {epoch:03d}, Test Recall: {metrics['recall']:.4f}")
            print(f"Epoch: {epoch:03d}, Test AUC: {metrics['auc']:.4f}")
            print(f"Epoch: {epoch:03d}, Test F1: {metrics['f1']:.4f}")

            # === Early Prediction: eval per-week (only if enabled) ===
            if getattr(self.args, 'early_prediction', False) and \
               getattr(self.model, 'early_mask', None) is not None:
                _week_count = self.model.early_mask.week_count
                print(f"\n  === Early Prediction Evaluation (epoch {epoch:03d}) ===")
                for eval_week in range(1, _week_count + 1):
                    self.model.early_mask.set_eval_weeks(eval_week)
                    self.model.eval()  # CRITICAL: tắt dropout, BN dùng running stats

                    ep_preds, ep_truths = [], []
                    with torch.no_grad():  # CRITICAL: không tính gradient
                        for batch in self.test_loader:
                            _pred, _truth = self.eval_step(batch)
                            ep_preds.append(_pred)
                            ep_truths.append(_truth)

                    ep_pred = torch.cat(ep_preds, dim=0).cpu()
                    ep_gt = torch.cat(ep_truths, dim=0).cpu().numpy()
                    ep_scores = torch.sigmoid(ep_pred).numpy()
                    ep_metrics = compute_metrics(ep_pred, ep_gt, ep_scores)

                    print(f"  Week {eval_week}/{_week_count}: "
                          f"AUC={ep_metrics['auc']:.4f}, "
                          f"F1={ep_metrics['f1']:.4f}, "
                          f"ACC={ep_metrics['acc']:.4f}")

                # CRITICAL: Reset eval_weeks về full SAU MỖI EPOCH
                self.model.early_mask.set_eval_weeks(_week_count)

        # === Post-training summaries ===
        if getattr(self.args, 'action_weight', False) and \
           hasattr(self.model, 'action_weighting'):
            _aw = self.model.action_weighting.get_weights().numpy()
            print(f"\n  📊 Learned Action Weights (baseline=1.0):")
            _sorted_idx = np.argsort(_aw)[::-1]
            for i in _sorted_idx:
                _bar = '█' * int(_aw[i] * 10)
                _marker = ' ⬆' if _aw[i] > 1.3 else (' ⬇' if _aw[i] < 0.7 else '')
                print(f"    action[{i:2d}]: {_aw[i]:5.3f} {_bar}{_marker}")

    def after_optimizer_step(self):
        """Hook called after each optimizer.step(). Override for BYOL momentum update etc."""
        pass
