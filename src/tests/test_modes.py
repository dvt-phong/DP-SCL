"""
Test script: Tạo mock data và chạy forward pass cho cả 4 mode
để đảm bảo không có lỗi shape hay runtime giữa các mode.

Chạy:  python3 test_modes.py
"""
import sys
import torch
import torch.nn.functional as F

# Thêm project root vào path
sys.path.insert(0, '.')
from src.models import LGB, GraphSage, GATNetwork, CNNFeatureExtractor, CrossAttentionFusion, MBACNNFeatureExtractor, SupConLoss, ProjectionHead, SupConLGB, AugmentationModule, SupConEncoder, SupConProjectionHead, SupConClassifier

device = torch.device('cpu')  # test trên CPU

# ============================================================
# Hyperparameters (giống train.py)
# ============================================================
BASE_PARAMS = {
    'activity_num': 22, 'sta_day': 35, 'week_count': 5, 'select_count': 5,
    'org_context_feat_len': 7, 'enhanced_context_feat_len': 32,
    'context_each_embed': 16, 'context_all_len': 16,
    'input_features': 16, 'hidden_features': 32, 'output_features': 16,
    'lstm_input_features': 184, 'lstm_hidden_features': 128, 'lstm_hidden_num_layers': 1,
    'num_attention_heads': 1, 'attention_features': 64,
    'l2_input_features': 64, 'l2_hidden_features': 32, 'l2_hidden_num_layers': 1,
    's2_num_attention_heads': 1, 's2_attention_features': 16,
    'ws_num_attention_heads': 1, 'ws_input_features': 32, 'ws_attention_features': 16,
    'dnn_input_f1': 16, 'dnn_hidden_f1': 16, 'dnn_hidden_f2': 8, 'dnn_hidden_f3': 4, 'dnn_output': 1,
}

CNN_PARAMS = {
    'cnn_in_channels': 7,
    'cnn_out_channels_1': 32,
    'cnn_out_channels_2': 64,
    'cnn_kernel_size': 3,
    'cnn_fc_output': 128,
}

GAT_PARAMS = {
    'gat_heads': 4,
    'gat_dropout': 0.3,
}

CNN2D_PARAMS = {
    'cnn2d_out_channels_1': 32,
    'cnn2d_out_channels_2': 64,
    'cnn2d_kernel_size': 3,
    'cnn2d_fc_output': 128,
}

CROSS_ATTN_PARAMS = {
    'ca_num_heads': 4,
    'ca_output_dim': 16,
    'ca_ffn_dim': 32,
}

MBA_CNN_PARAMS = {
    'mba_cnn_temporal_channels_1': 32,
    'mba_cnn_temporal_channels_2': 64,
    'mba_cnn_daily_channels': 32,
    'mba_cnn_weekly_channels': 32,
    'mba_cnn_fc_hidden': 256,
    'mba_cnn_output': 128,
    'mba_cnn_dropout': 0.3,
}

MBA_CNN_GAT_PARAMS = {**MBA_CNN_PARAMS, **GAT_PARAMS}

NO_GRAPH_PARAMS = {}  # no_graph uses only base temporal params, no graph-specific params

# ============================================================
# Mock data helper
# ============================================================
def create_mock_subgraph(batch_size=8, total_nodes=32):
    """Tạo mock sub_graph giống format NeighborLoader output."""
    week_count = 5
    days_per_week = 7  # sta_day / week_count = 35 / 5
    activity_num = 22
    seq_len = week_count * days_per_week * activity_num  # 5 * 7 * 22 = 770

    sub_graph = {
        'batch_size': batch_size,
        'seq_feat': torch.randn(total_nodes, seq_len),
        'org_context': torch.randn(total_nodes, 7),
        'enhanced_context': torch.randn(total_nodes, 32),
        'edge_index': torch.randint(0, total_nodes, (2, total_nodes * 4)),
        'labels': torch.randint(0, 2, (total_nodes,)).float(),
    }
    return sub_graph


# ============================================================
# Test functions
# ============================================================
def test_mode(mode_name, param_dict, batch_size=8):
    """Test 1 mode: tạo model, forward pass, kiểm tra output."""
    print(f"\n{'='*60}")
    print(f"  Testing mode: {mode_name}")
    print(f"{'='*60}")

    # Build model
    model = LGB(param_dict, mode=mode_name).to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Check which GNN is used
    gnn_type = type(model.gnn).__name__ if hasattr(model, 'gnn') else 'None'
    has_cnn = hasattr(model, 'cnn')
    print(f"  GNN type:         {gnn_type}")
    print(f"  Has CNN:          {has_cnn}")

    # Forward pass
    sub_graph = create_mock_subgraph(batch_size=batch_size)
    with torch.no_grad():
        pred = model(sub_graph)

    print(f"  Output shape:     {pred.shape}")
    print(f"  Expected shape:   ({batch_size}, 1)")
    assert pred.shape == (batch_size, 1), f"Shape mismatch! Got {pred.shape}, expected ({batch_size}, 1)"

    # Test loss computation
    labels = sub_graph['labels'][:batch_size].view(-1, 1)
    loss = F.binary_cross_entropy_with_logits(pred, labels)
    print(f"  Loss (mock):      {loss.item():.4f}")

    # Test backward pass
    model.train()
    pred2 = model(sub_graph)
    labels2 = sub_graph['labels'][:batch_size].view(-1, 1)
    loss2 = F.binary_cross_entropy_with_logits(pred2, labels2)
    loss2.backward()
    print(f"  Backward pass:    ✅ OK")

    print(f"  ✅ Mode '{mode_name}' PASSED")
    return {
        'mode': mode_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gnn_type': gnn_type,
        'has_cnn': has_cnn,
        'output_shape': tuple(pred.shape),
        'status': 'PASSED'
    }


def test_isolation():
    """Test rằng các mode không ảnh hưởng lẫn nhau."""
    print(f"\n{'='*60}")
    print(f"  Testing ISOLATION between modes")
    print(f"{'='*60}")

    # Tạo cùng 1 mock data
    sub_graph = create_mock_subgraph(batch_size=4)

    # Tạo 4 model
    params = {**BASE_PARAMS}
    model_default = LGB({**params}, mode='default')

    params_cnn = {**params, **CNN_PARAMS}
    model_cnn = LGB(params_cnn, mode='cnn')

    params_gat = {**params, **GAT_PARAMS}
    model_gat = LGB(params_gat, mode='gat')

    params_cnn2d = {**params, **CNN2D_PARAMS}
    model_cnn2d = LGB(params_cnn2d, mode='cnn2d')

    params_both = {**params, **CNN_PARAMS, **GAT_PARAMS}
    model_both = LGB(params_both, mode='cnn_gat')

    params_ca = {**params, **CROSS_ATTN_PARAMS}
    model_ca = LGB(params_ca, mode='cross_attn')

    params_mba = {**params, **MBA_CNN_PARAMS}
    model_mba = LGB(params_mba, mode='mba_cnn')

    params_mba_gat = {**params, **MBA_CNN_GAT_PARAMS}
    model_mba_gat = LGB(params_mba_gat, mode='mba_cnn_gat')

    params_no_graph = {**params}
    model_no_graph = LGB(params_no_graph, mode='no_graph')

    # Verify GNN types
    assert type(model_default.gnn).__name__ == 'GraphSage', "default should use GraphSage"
    assert type(model_cnn.gnn).__name__ == 'GraphSage', "cnn should use GraphSage"
    assert type(model_gat.gnn).__name__ == 'GATNetwork', "gat should use GATNetwork"
    assert type(model_cnn2d.gnn).__name__ == 'GraphSage', "cnn2d should use GraphSage" # Added cnn2d GNN type check
    assert type(model_both.gnn).__name__ == 'GATNetwork', "cnn_gat should use GATNetwork"
    assert type(model_ca.gnn).__name__ == 'GraphSage', "cross_attn should use GraphSage"
    assert type(model_mba.gnn).__name__ == 'GraphSage', "mba_cnn should use GraphSage"
    assert type(model_mba_gat.gnn).__name__ == 'GATNetwork', "mba_cnn_gat should use GATNetwork"
    assert not hasattr(model_no_graph, 'gnn'), "no_graph should NOT have gnn"
    assert not hasattr(model_no_graph, 'context_embed'), "no_graph should NOT have context_embed"
    print("  ✅ GNN type isolation: correct")

    # Verify CNN presence
    assert not hasattr(model_default, 'cnn') and not hasattr(model_default, 'cnn2d'), "default should NOT have CNN"
    assert hasattr(model_cnn, 'cnn') and not hasattr(model_cnn, 'cnn2d'), "cnn mode should have CNN (type 1)" # Updated
    assert not hasattr(model_gat, 'cnn') and not hasattr(model_gat, 'cnn2d'), "gat should NOT have CNN"
    assert not hasattr(model_cnn2d, 'cnn') and hasattr(model_cnn2d, 'cnn2d'), "cnn2d mode should have CNN (type 2)" # Added cnn2d CNN presence check
    assert hasattr(model_both, 'cnn') and not hasattr(model_both, 'cnn2d'), "cnn_gat should have CNN (type 1)" # Updated
    assert hasattr(model_ca, 'cross_attn_fusion'), "cross_attn should have CrossAttentionFusion"
    assert not hasattr(model_default, 'cross_attn_fusion'), "default should NOT have CrossAttentionFusion"
    assert hasattr(model_mba, 'mba_cnn'), "mba_cnn mode should have MBACNNFeatureExtractor"
    assert not hasattr(model_default, 'mba_cnn'), "default should NOT have MBACNNFeatureExtractor"
    assert hasattr(model_mba_gat, 'mba_cnn'), "mba_cnn_gat should have MBACNNFeatureExtractor"
    assert type(model_mba_gat.gnn).__name__ == 'GATNetwork', "mba_cnn_gat should use GATNetwork"
    assert not hasattr(model_no_graph, 'weighted_sum'), "no_graph should NOT have weighted_sum"
    assert hasattr(model_no_graph, 'lstm1'), "no_graph should have lstm1"
    assert hasattr(model_no_graph, 'lstm2'), "no_graph should have lstm2"
    assert hasattr(model_no_graph, 'self_attention1'), "no_graph should have self_attention1"
    assert hasattr(model_no_graph, 'self_attention2'), "no_graph should have self_attention2"
    print("  ✅ CNN/CrossAttn/MBA/NoGraph isolation: correct")

    # Verify all produce valid output
    for name, m in [('default', model_default), ('cnn', model_cnn),
                     ('gat', model_gat), ('cnn2d', model_cnn2d),
                     ('cross_attn', model_ca), ('cnn_gat', model_both),
                     ('mba_cnn', model_mba), ('mba_cnn_gat', model_mba_gat),
                     ('no_graph', model_no_graph)]:
        m.eval()
        with torch.no_grad():
            out = m(sub_graph)
        assert out.shape == (4, 1), f"{name}: wrong shape {out.shape}"
    print("  ✅ All modes produce correct output shape (4, 1)")

    print("  ✅ ISOLATION TEST PASSED")


def test_contrastive():
    """Test Supervised Contrastive Learning integration."""
    print(f"\n{'='*60}")
    print(f"  Testing CONTRASTIVE LEARNING")
    print(f"{'='*60}")

    sub_graph = create_mock_subgraph(batch_size=8)

    # Test 1: no_graph + contrastive
    print("  [1/4] no_graph + contrastive=True...")
    params_ng = {**BASE_PARAMS}
    model_ng_con = LGB(params_ng, mode='no_graph', contrastive=True)
    model_ng_con.eval()
    assert hasattr(model_ng_con, 'projection_head'), "no_graph+contrastive should have projection_head"
    with torch.no_grad():
        result = model_ng_con(sub_graph)
    assert isinstance(result, tuple), "contrastive model should return tuple"
    pred, proj = result
    assert pred.shape == (8, 1), f"pred shape: {pred.shape}"
    assert proj.shape == (8, 32), f"proj shape: {proj.shape}"
    # Verify L2-normalized
    norms = torch.norm(proj, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), "proj should be L2-normalized"
    print("  ✅ no_graph + contrastive: output (pred, proj) correct")

    # Test 2: mba_cnn + contrastive
    print("  [2/4] mba_cnn + contrastive=True...")
    params_mba_con = {**BASE_PARAMS, **MBA_CNN_PARAMS}
    model_mba_con = LGB(params_mba_con, mode='mba_cnn', contrastive=True)
    model_mba_con.eval()
    assert hasattr(model_mba_con, 'projection_head'), "mba_cnn+contrastive should have projection_head"
    with torch.no_grad():
        result = model_mba_con(sub_graph)
    pred, proj = result
    assert pred.shape == (8, 1)
    assert proj.shape == (8, 32)
    print("  ✅ mba_cnn + contrastive: correct")

    # Test 3: contrastive=False should NOT have projection_head
    print("  [3/4] default + contrastive=False...")
    model_default = LGB({**BASE_PARAMS}, mode='default', contrastive=False)
    assert not hasattr(model_default, 'projection_head'), "contrastive=False should NOT have projection_head"
    model_default.eval()
    with torch.no_grad():
        result = model_default(sub_graph)
    assert not isinstance(result, tuple), "non-contrastive should return tensor, not tuple"
    assert result.shape == (8, 1)
    print("  ✅ default + contrastive=False: correct (no projection_head)")

    # Test 4: SupConLoss computation
    print("  [4/4] SupConLoss computation...")
    supcon = SupConLoss(temperature=0.07)
    # Create leaf tensor with grad, then normalize
    raw_features = torch.randn(16, 32, requires_grad=True)
    fake_features = F.normalize(raw_features, dim=1)
    fake_labels = torch.tensor([0,0,0,0,1,1,1,1,0,0,1,1,0,1,0,1]).float()
    loss = supcon(fake_features, fake_labels)
    assert loss.item() > 0, "SupConLoss should be > 0"
    loss.backward()  # should not error
    assert raw_features.grad is not None, "SupConLoss should be differentiable (grad flows to leaf)"
    print(f"  ✅ SupConLoss: {loss.item():.4f} (differentiable)")

    print("  ✅ CONTRASTIVE TEST PASSED")


# ============================================================
# SupCon parameters (Version 2.1)
# ============================================================
SUPCON_PARAMS = {
    'supcon_hidden_size': 128,
    'supcon_proj_dim': 128,
    'supcon_temperature': 0.07,
    'supcon_mask_ratio': 0.15,
    'supcon_noise_std': 0.05,
    'supcon_attn_heads': 4,
    'supcon_cls_dropout': 0.3,
}

SUPCON_MODES = ['supcon_lstm', 'supcon_bilstm', 'supcon_lstm_attn', 'supcon_bilstm_attn']


def create_mock_no_graph_data(batch_size=8):
    """Tạo mock data cho no_graph / supcon modes (seq_feat only)."""
    seq_len = 5 * 7 * 22  # 770
    return {
        'batch_size': batch_size,
        'seq_feat': torch.randn(batch_size, seq_len),
    }


def test_supcon_mode(mode_name, param_dict, batch_size=8):
    """Test 1 supcon mode: tạo model, forward train + eval, kiểm tra output."""
    print(f"\n{'='*60}")
    print(f"  Testing supcon mode: {mode_name}")
    print(f"{'='*60}")

    model = SupConLGB(mode=mode_name, param_dict=param_dict).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Encoder type:     {model.ENCODER_MAP[mode_name]}")

    sub_graph = create_mock_no_graph_data(batch_size)

    # --- Training mode: returns (logits, z1, z2) ---
    model.train()
    result = model(sub_graph)
    assert isinstance(result, tuple) and len(result) == 3, \
        f"Training should return (logits, z1, z2), got {type(result)}"
    logits, z1, z2 = result
    assert logits.shape == (batch_size, 1), f"logits shape: {logits.shape}"
    assert z1.shape == (batch_size, 128), f"z1 shape: {z1.shape}"
    assert z2.shape == (batch_size, 128), f"z2 shape: {z2.shape}"
    norms_z1 = torch.norm(z1, dim=1)
    assert torch.allclose(norms_z1, torch.ones(batch_size), atol=1e-5), "z1 should be L2-normalized"
    norms_z2 = torch.norm(z2, dim=1)
    assert torch.allclose(norms_z2, torch.ones(batch_size), atol=1e-5), "z2 should be L2-normalized"
    print(f"  Training output:  logits{logits.shape}, z1{z1.shape}, z2{z2.shape} ✅")

    # --- Backward pass ---
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
    z_all = torch.stack([z1, z2], dim=1)
    supcon = SupConLoss(temperature=0.07)
    con_loss = supcon(z_all, labels.view(-1))
    loss = bce_loss + 0.1 * con_loss
    loss.backward()
    print(f"  Backward pass:    ✅ (bce={bce_loss.item():.4f}, supcon={con_loss.item():.4f})")

    # --- Eval mode: returns logits only ---
    model.eval()
    with torch.no_grad():
        result_eval = model(sub_graph)
    assert not isinstance(result_eval, tuple), "Eval should return tensor, not tuple"
    assert result_eval.shape == (batch_size, 1), f"Eval logits shape: {result_eval.shape}"
    print(f"  Eval output:      logits{result_eval.shape} ✅")

    # --- Shared weights: encoder forwards 2x with same params ---
    assert not hasattr(model, 'encoder2'), "Should NOT have separate encoder2 (shared weights)"
    print(f"  Shared weights:   ✅ (single encoder instance)")

    print(f"  ✅ SupCon mode '{mode_name}' PASSED")
    return {
        'mode': mode_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gnn_type': 'None',
        'has_cnn': False,
        'output_shape': tuple(logits.shape),
        'status': 'PASSED'
    }


def test_supcon_all():
    """Test tất cả 4 supcon modes."""
    print(f"\n{'='*60}")
    print(f"  Testing ALL SUPCON MODES (Version 2.1)")
    print(f"{'='*60}")

    supcon_results = []
    for smode in SUPCON_MODES:
        params = {**BASE_PARAMS, **SUPCON_PARAMS}
        r = test_supcon_mode(smode, params)
        supcon_results.append(r)

    # Augmentation isolation: two calls should produce different views
    print(f"\n  --- Augmentation test ---")
    from src.models import AugmentationModule
    aug = AugmentationModule(time_mask_ratio=0.15, feat_mask_ratio=0.15, noise_std=0.05)
    x = torch.randn(4, 35, 22)
    v1, v2 = aug(x)
    assert not torch.allclose(v1, v2), "Two views should be different"
    assert v1.shape == x.shape and v2.shape == x.shape
    print(f"  ✅ Augmentation produces different views with correct shape")

    print(f"\n  ✅ ALL SUPCON TESTS PASSED")
    return supcon_results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  DP-SCL Multi-Mode Test Suite")
    print("=" * 60)

    results = []
    all_passed = True

    # Test 1: mode default
    try:
        r = test_mode('default', {**BASE_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'default' FAILED: {e}")
        results.append({'mode': 'default', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 2: mode cnn
    try:
        r = test_mode('cnn', {**BASE_PARAMS, **CNN_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'cnn' FAILED: {e}")
        results.append({'mode': 'cnn', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 3: mode gat
    try:
        r = test_mode('gat', {**BASE_PARAMS, **GAT_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'gat' FAILED: {e}")
        results.append({'mode': 'gat', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 4: mode cnn2d
    try:
        r = test_mode('cnn2d', {**BASE_PARAMS, **CNN2D_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'cnn2d' FAILED: {e}")
        results.append({'mode': 'cnn2d', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 5: mode cnn_gat
    try:
        r = test_mode('cnn_gat', {**BASE_PARAMS, **CNN_PARAMS, **GAT_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'cnn_gat' FAILED: {e}")
        results.append({'mode': 'cnn_gat', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 6: mode cross_attn
    try:
        r = test_mode('cross_attn', {**BASE_PARAMS, **CROSS_ATTN_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'cross_attn' FAILED: {e}")
        results.append({'mode': 'cross_attn', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 7: mode mba_cnn
    try:
        r = test_mode('mba_cnn', {**BASE_PARAMS, **MBA_CNN_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'mba_cnn' FAILED: {e}")
        results.append({'mode': 'mba_cnn', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 8: mode mba_cnn_gat
    try:
        r = test_mode('mba_cnn_gat', {**BASE_PARAMS, **MBA_CNN_GAT_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'mba_cnn_gat' FAILED: {e}")
        results.append({'mode': 'mba_cnn_gat', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 9: mode no_graph
    try:
        r = test_mode('no_graph', {**BASE_PARAMS})
        results.append(r)
    except Exception as e:
        print(f"  ❌ Mode 'no_graph' FAILED: {e}")
        results.append({'mode': 'no_graph', 'status': 'FAILED', 'error': str(e)})
        all_passed = False

    # Test 10: Isolation test
    try:
        test_isolation()
    except Exception as e:
        print(f"  ❌ Isolation test FAILED: {e}")
        all_passed = False

    # Test 11: Contrastive test
    try:
        test_contrastive()
    except Exception as e:
        print(f"  ❌ Contrastive test FAILED: {e}")
        all_passed = False

    # Test 12: SupCon modes (Version 2.1)
    try:
        supcon_results = test_supcon_all()
        results.extend(supcon_results)
    except Exception as e:
        print(f"  ❌ SupCon test FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Mode':<12} {'GNN':<12} {'CNN':<6} {'Params':>10} {'Status'}")
    print(f"  {'-'*54}")
    for r in results:
        if r['status'] == 'PASSED':
            print(f"  {r['mode']:<12} {r['gnn_type']:<12} {'Yes' if r['has_cnn'] else 'No':<6} {r['trainable_params']:>10,} ✅")
        else:
            print(f"  {r['mode']:<12} {'?':<12} {'?':<6} {'?':>10} ❌ {r.get('error','')[:40]}")
    print(f"  {'-'*54}")

    if all_passed:
        print(f"\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  SOME TESTS FAILED!")

    sys.exit(0 if all_passed else 1)
