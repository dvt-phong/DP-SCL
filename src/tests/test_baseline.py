"""
Unit tests for Phase 1 modular architecture:
  WeeklyCNN, TemporalBiLSTM, BahdanauAttention, MultiHeadAttentionPool,
  CrossAttentionPool, DropoutClassifier, DropoutPredictor

Run:  python3 test_baseline.py
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from src.models import (
    WeeklyCNN, TemporalBiLSTM, BahdanauAttention, MultiHeadAttentionPool,
    CrossAttentionPool, DropoutClassifier, DropoutPredictor
)

device = torch.device('cpu')

DEFAULT_CONFIG = {
    'num_actions': 22, 'days_per_week': 7, 'num_weeks': 5,
    'cnn_out_dim': 128, 'cnn_dropout': 0.2,
    'lstm_hidden': 64, 'lstm_layers': 2, 'lstm_dropout': 0.3,
    'attn_heads': 2, 'attn_dropout': 0.1, 'cls_dropout': 0.3,
    'attention_type': 'bahdanau',
    'use_cnn': True, 'use_bilstm': True, 'use_attention': True,
}


# ============================================================
# Component Tests
# ============================================================
def test_weekly_cnn():
    print("\n[1/9] WeeklyCNN...")
    cnn = WeeklyCNN(in_channels=22, cnn_out_dim=128).to(device)
    x = torch.randn(4, 5, 7, 22)
    out = cnn(x)
    assert out.shape == (4, 5, 128), f"Expected (4,5,128), got {out.shape}"
    print(f"  Input: {tuple(x.shape)} → Output: {tuple(out.shape)} ✅")


def test_temporal_bilstm():
    print("\n[2/9] TemporalBiLSTM...")
    bilstm = TemporalBiLSTM(input_size=128, hidden_size=64).to(device)
    x = torch.randn(4, 5, 128)
    out, (h_n, c_n) = bilstm(x)
    assert out.shape == (4, 5, 128), f"Expected (4,5,128), got {out.shape}"
    print(f"  Input: {tuple(x.shape)} → Output: {tuple(out.shape)} ✅")


def test_bahdanau_attention():
    print("\n[3/9] BahdanauAttention...")
    attn = BahdanauAttention(hidden_size=128).to(device)
    x = torch.randn(4, 5, 128)
    context, weights = attn(x)
    assert context.shape == (4, 128), f"Context: {context.shape}"
    assert weights.shape == (4, 5), f"Weights: {weights.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5), "Weights must sum to 1"
    print(f"  Context: {tuple(context.shape)}, Weights: {tuple(weights.shape)} ✅")


def test_multihead_attention_pool():
    print("\n[4/9] MultiHeadAttentionPool...")
    attn = MultiHeadAttentionPool(hidden_size=128, num_heads=2).to(device)
    x = torch.randn(4, 5, 128)
    context, weights = attn(x)
    assert context.shape == (4, 128), f"Context: {context.shape}"
    assert weights.shape == (4, 5), f"Weights: {weights.shape}"
    print(f"  Context: {tuple(context.shape)}, Weights: {tuple(weights.shape)} ✅")


def test_cross_attention_pool():
    print("\n[5/9] CrossAttentionPool...")
    attn = CrossAttentionPool(hidden_size=128, num_heads=2).to(device)
    cnn_feat = torch.randn(4, 5, 128)
    lstm_feat = torch.randn(4, 5, 128)
    context, weights = attn(cnn_feat, lstm_feat)
    assert context.shape == (4, 128), f"Context: {context.shape}"
    assert weights.shape == (4, 5), f"Weights: {weights.shape}"
    print(f"  Context: {tuple(context.shape)}, Weights: {tuple(weights.shape)} ✅")


# ============================================================
# Full Model Tests
# ============================================================
def test_dropout_predictor_all_variants():
    print("\n[6/9] DropoutPredictor — all 3 attention variants...")
    for attn_type in ['bahdanau', 'multihead', 'cross']:
        config = {**DEFAULT_CONFIG, 'attention_type': attn_type}
        model = DropoutPredictor(config).to(device)
        model.eval()

        for bs in [1, 4, 16]:
            x = torch.randn(bs, 5, 7, 22)
            with torch.no_grad():
                logits, attn_w = model(x)
            assert logits.shape == (bs, 1), f"{attn_type} bs={bs}: logits {logits.shape}"
            assert attn_w.shape == (bs, 5), f"{attn_type} bs={bs}: attn {attn_w.shape}"
        print(f"  {attn_type}: ✅")
    print("  ✅ All 3 variants PASSED")


def test_dropout_predictor_flat_input():
    print("\n[7/9] DropoutPredictor — flat input auto-reshape...")
    config = {**DEFAULT_CONFIG, 'attention_type': 'bahdanau'}
    model = DropoutPredictor(config).to(device)
    model.eval()

    x_4d = torch.randn(4, 5, 7, 22)
    x_flat = x_4d.view(4, -1)

    with torch.no_grad():
        logits_4d, attn_4d = model(x_4d)
        logits_flat, attn_flat = model(x_flat)

    assert torch.allclose(logits_4d, logits_flat, atol=1e-5)
    assert torch.allclose(attn_4d, attn_flat, atol=1e-5)
    print("  ✅ 4D and flat input produce same output")


def test_ablation_modes():
    print("\n[8/9] Ablation modes...")
    ablations = [
        ('no_cnn', {'use_cnn': False, 'use_bilstm': True, 'use_attention': True}),
        ('no_bilstm', {'use_cnn': True, 'use_bilstm': False, 'use_attention': True}),
        ('no_attention', {'use_cnn': True, 'use_bilstm': True, 'use_attention': False}),
        ('cnn_only', {'use_cnn': True, 'use_bilstm': False, 'use_attention': False}),
        ('bilstm_only', {'use_cnn': False, 'use_bilstm': True, 'use_attention': False}),
    ]

    x = torch.randn(4, 5, 7, 22)
    for name, overrides in ablations:
        config = {**DEFAULT_CONFIG, **overrides}
        model = DropoutPredictor(config).to(device)
        model.eval()
        with torch.no_grad():
            logits, attn_w = model(x)
        assert logits.shape == (4, 1), f"{name}: logits {logits.shape}"
        assert attn_w.shape == (4, 5), f"{name}: attn {attn_w.shape}"

        # Test backward
        model.train()
        logits, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(4, 1))
        loss.backward()
        print(f"  {name}: forward+backward ✅")

    print("  ✅ All ablation modes PASSED")


def test_param_count():
    print("\n[9/9] Parameter counts...")
    configs = [
        ('CNN+BiLSTM+Bahdanau', {**DEFAULT_CONFIG, 'attention_type': 'bahdanau'}),
        ('CNN+BiLSTM+MultiHead', {**DEFAULT_CONFIG, 'attention_type': 'multihead'}),
        ('CNN+BiLSTM+CrossAttn', {**DEFAULT_CONFIG, 'attention_type': 'cross'}),
        ('noCNN+BiLSTM+Bahdanau', {**DEFAULT_CONFIG, 'use_cnn': False}),
        ('CNN+noBiLSTM+Bahdanau', {**DEFAULT_CONFIG, 'use_bilstm': False}),
    ]
    for name, config in configs:
        model = DropoutPredictor(config)
        total = sum(p.numel() for p in model.parameters())
        print(f"  {name:<30} {total:>10,} params")
    print("  ✅ Parameter count test PASSED")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 1 Baseline — Modular Architecture Tests")
    print("=" * 60)

    all_passed = True
    tests = [
        test_weekly_cnn,
        test_temporal_bilstm,
        test_bahdanau_attention,
        test_multihead_attention_pool,
        test_cross_attention_pool,
        test_dropout_predictor_all_variants,
        test_dropout_predictor_flat_input,
        test_ablation_modes,
        test_param_count,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("  🎉 ALL 9 TESTS PASSED!")
    else:
        print("  ⚠️ SOME TESTS FAILED!")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)
