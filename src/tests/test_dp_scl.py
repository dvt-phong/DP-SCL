"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Smoke tests for the DP-SCL model forward pass and SupCon loss.

Reference source:
  This file is project-specific test code for the DP-SCL implementation.
"""

import torch

from src.models import DPSCLModel, SupConLoss


def test_dp_scl_forward_and_loss():
    params = {
        "activity_num": 22,
        "week_count": 5,
        "cnn_in_channels": 7,
        "supcon_hidden_size": 32,
        "supcon_proj_dim": 32,
        "supcon_mask_ratio": 0.15,
        "supcon_noise_std": 0.05,
        "supcon_attn_heads": 4,
        "supcon_num_layers": 1,
        "supcon_cls_hidden_layers": 1,
    }
    model = DPSCLModel("supcon_lstm_attn", params)
    seq_feat = torch.randn(4, 5 * 7 * 22)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    model.train()
    logits, z1, z2 = model({"batch_size": 4, "seq_feat": seq_feat})
    assert logits.shape == (4, 1)
    assert z1.shape == (4, 32)
    assert z2.shape == (4, 32)

    loss = SupConLoss()(torch.stack([z1, z2], dim=1), labels)
    assert torch.isfinite(loss)

    model.eval()
    with torch.no_grad():
        eval_logits = model({"batch_size": 4, "seq_feat": seq_feat})
    assert eval_logits.shape == (4, 1)
