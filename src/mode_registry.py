"""
Mode Registry — Phân loại tất cả modes theo nhánh:
    • GRAPH modes:    cần graph (.pkl) + temporal data (.npz)
    • NO_GRAPH modes: chỉ cần temporal data (.npz)

Used by: train.py, train_experiment.py, trainers, and data validation.

──────────────────────────────────────────────────────────
  NHÁNH GRAPH (23 modes) — cần StrongClassmatesGraph.pkl
──────────────────────────────────────────────────────────
  Legacy graph-temporal baselines:
    default, cnn, cnn2d, gat, cnn_gat,
    cross_attn, mba_cnn, mba_cnn_gat, bilstm_graph

  Framework 1 (SupCon Network) — graph-enhanced:
    supcon_lstm_graph, supcon_bilstm_graph,
    supcon_lstm_attn_graph, supcon_bilstm_attn_graph,
    supcon_lstm_sa_graph, supcon_bilstm_sa_graph

  Framework 2A (SimCLR) — graph-enhanced:
    simclr_lstm_graph, simclr_bilstm_graph,
    simclr_lstm_attn_graph, simclr_bilstm_attn_graph

  Framework 2B (BYOL) — graph-enhanced:
    byol_lstm_graph, byol_bilstm_graph,
    byol_lstm_attn_graph, byol_bilstm_attn_graph

──────────────────────────────────────────────────────────
  NHÁNH NO-GRAPH (30 modes) — chỉ cần .npz
──────────────────────────────────────────────────────────
  Legacy graph-temporal baselines — no-graph variants:
    no_graph, bilstm_cnn, bilstm_mha, bilstm_cross,
    mba_bilstm, cnn_only, mba_only, cnn_day, bilstm_day

  Framework 1 (SupCon Network):
    dp_scl,
    supcon_lstm, supcon_bilstm,
    supcon_lstm_mha, supcon_lstm_attn, supcon_lstm_attn_lambda0, supcon_bilstm_attn,
    supcon_lstm_sa, supcon_bilstm_sa

  Framework 2A (SimCLR):
    simclr_lstm, simclr_bilstm,
    simclr_lstm_attn, simclr_bilstm_attn

  Framework 2B (BYOL):
    byol_lstm, byol_bilstm,
    byol_lstm_attn, byol_bilstm_attn

  Baseline DL:
    dl_cnn, dl_lstm, dl_gru, dl_cnn_lstm, dl_cnn_gru, dl_cnn_rnn,
    dl_cnn_lstm_at1, dl_cnn_lstm_at2, dl_lstm_mha, dl_lstm_mha_lq
──────────────────────────────────────────────────────────
"""

# ============================================================
# GRAPH modes — cần cả graph (.pkl) + temporal data (.npz)
# ============================================================
GRAPH_MODES = frozenset({
    # Legacy graph-temporal baseline modes
    'default',          # GraphSAGE + manual preprocessing + LSTM
    'cnn',              # GraphSAGE + CNN 1D + LSTM
    'cnn2d',            # GraphSAGE + CNN 2D + LSTM
    'gat',              # GAT + manual preprocessing + LSTM
    'cnn_gat',          # GAT + CNN 1D + LSTM
    'cross_attn',       # GraphSAGE + CrossAttentionFusion + LSTM
    'mba_cnn',          # GraphSAGE + MBA-CNN + LSTM
    'mba_cnn_gat',      # GAT + MBA-CNN + LSTM
    'bilstm_graph',     # GraphSAGE + CNN 1D + BiLSTM

    # Framework 1: SupCon Network — graph-enhanced (concat fusion)
    'supcon_lstm_graph',         # SupCon + LSTM + GraphSAGE fusion
    'supcon_bilstm_graph',       # SupCon + BiLSTM + GraphSAGE fusion
    'supcon_lstm_attn_graph',    # SupCon + LSTM + MHA + GraphSAGE fusion
    'supcon_bilstm_attn_graph',  # SupCon + BiLSTM + MHA + GraphSAGE fusion
    'supcon_lstm_sa_graph',      # SupCon + LSTM + SelfAttn + GraphSAGE fusion
    'supcon_bilstm_sa_graph',    # SupCon + BiLSTM + SelfAttn + GraphSAGE fusion

    # Framework 2A: SimCLR — graph-enhanced (concat fusion)
    'simclr_lstm_graph',          # SimCLR + LSTM + GraphSAGE fusion
    'simclr_bilstm_graph',        # SimCLR + BiLSTM + GraphSAGE fusion
    'simclr_lstm_attn_graph',     # SimCLR + LSTM + MHA + GraphSAGE fusion
    'simclr_bilstm_attn_graph',   # SimCLR + BiLSTM + MHA + GraphSAGE fusion

    # Framework 2B: BYOL — graph-enhanced (concat fusion)
    'byol_lstm_graph',            # BYOL + LSTM + GraphSAGE fusion
    'byol_bilstm_graph',          # BYOL + BiLSTM + GraphSAGE fusion
    'byol_lstm_attn_graph',       # BYOL + LSTM + MHA + GraphSAGE fusion
    'byol_bilstm_attn_graph',     # BYOL + BiLSTM + MHA + GraphSAGE fusion
})

# ============================================================
# NO-GRAPH modes — chỉ cần temporal data (.npz)
# ============================================================
NO_GRAPH_MODES = frozenset({
    # Legacy temporal-only baseline modes
    'no_graph',         # LSTM only (no graph, no CNN)
    'bilstm_cnn',       # CNN 1D + BiLSTM + temporal diff + LearnableQueryPool
    'bilstm_mha',       # CNN 1D + BiLSTM + Multi-Head Attention
    'bilstm_cross',     # CNN 1D + BiLSTM + Cross-Attention (Q=BiLSTM, K/V=CNN)
    'mba_bilstm',       # MBA-CNN + BiLSTM
    'cnn_only',         # CNN 1D + LSTM (no graph)
    'mba_only',         # MBA-CNN + LSTM (no graph)
    'cnn_day',          # CNN 1D (days as channels) + LSTM
    'bilstm_day',       # CNN 1D (days as channels) + BiLSTM

    # Framework 1: SupCon Network
    'dp_scl',           # DP-SCL alias: supcon_lstm_attn + SupCon, lambda=0.1, tau=0.07
    'tsn_supcon',       # Backward-compatible legacy alias for DP-SCL
    'supcon_lstm',         # SupCon + LSTM encoder
    'supcon_bilstm',       # SupCon + BiLSTM encoder
    'supcon_lstm_mha',     # SupCon + LSTM + Multi-Head Attention + mean pooling
    'supcon_lstm_attn',    # SupCon + LSTM + Multi-Head Attention + LearnableQueryPool
    'supcon_lstm_attn_lambda0',  # SupCon + LSTM + MHA + LQP, BCE only via lambda=0
    'supcon_bilstm_attn',  # SupCon + BiLSTM + Multi-Head Attention + LearnableQueryPool
    'supcon_lstm_sa',      # SupCon + LSTM + Custom SelfAttention (sinusoidal PE)
    'supcon_bilstm_sa',    # SupCon + BiLSTM + Custom SelfAttention (sinusoidal PE)

    # Framework 2A: SimCLR
    'simclr_lstm',          # SimCLR + LSTM + NT-Xent Loss
    'simclr_bilstm',        # SimCLR + BiLSTM + NT-Xent Loss
    'simclr_lstm_attn',     # SimCLR + LSTM + MHA + NT-Xent Loss
    'simclr_bilstm_attn',   # SimCLR + BiLSTM + MHA + NT-Xent Loss

    # Framework 2B: BYOL
    'byol_lstm',            # BYOL + LSTM + MSE Loss (no negatives)
    'byol_bilstm',          # BYOL + BiLSTM + MSE Loss
    'byol_lstm_attn',       # BYOL + LSTM + MHA + MSE Loss
    'byol_bilstm_attn',     # BYOL + BiLSTM + MHA + MSE Loss

    # Baseline DL methods
    'dl_cnn',                # CNN baseline
    'dl_lstm',               # LSTM baseline without CNN
    'dl_gru',                # GRU baseline without CNN
    'dl_cnn_lstm',           # CNN + LSTM baseline
    'dl_cnn_gru',            # CNN + GRU baseline
    'dl_cnn_rnn',            # CNN + RNN baseline
    'dl_cnn_lstm_at1',       # CNN + LSTM + 1-layer self-attention baseline
    'dl_cnn_lstm_at2',       # CNN + LSTM + 2-layer self-attention baseline
    'dl_lstm_mha',           # LSTM + MultiHeadAttention + mean pooling baseline
    'dl_lstm_mha_lq',         # LSTM + MHA + LearnableQueryPool baseline, BCE only
})

DP_SCL_MODE = 'dp_scl'
LEGACY_DP_SCL_ALIAS = 'tsn_supcon'
DP_SCL_BACKEND_MODE = 'supcon_lstm_attn'

# All valid modes
ALL_MODES = GRAPH_MODES | NO_GRAPH_MODES

# ============================================================
# Framework classification
# ============================================================
SUPCON_MODES = frozenset({
    m for m in ALL_MODES
    if m.startswith('supcon_') or m in {DP_SCL_MODE, LEGACY_DP_SCL_ALIAS}
})
SIMCLR_MODES  = frozenset({m for m in ALL_MODES if m.startswith('simclr_')})
BYOL_MODES    = frozenset({m for m in ALL_MODES if m.startswith('byol_')})
DL_BASELINE_MODES = frozenset({m for m in ALL_MODES if m.startswith('dl_')})
CL_MODES      = SIMCLR_MODES | BYOL_MODES             # Framework 2: all contrastive
LGB_MODES     = ALL_MODES - SUPCON_MODES - CL_MODES - DL_BASELINE_MODES  # Legacy graph-temporal baselines


def is_graph_mode(mode):
    """Trả True nếu mode cần graph data (.pkl)."""
    return mode in GRAPH_MODES


def is_no_graph_mode(mode):
    """Trả True nếu mode chỉ cần temporal data (.npz)."""
    return mode in NO_GRAPH_MODES


def get_framework(mode):
    """Trả tên framework cho mode.

    Returns: 'lgb', 'supcon', 'simclr', 'byol'
    """
    if mode in SUPCON_MODES:
        return 'supcon'
    elif mode in SIMCLR_MODES:
        return 'simclr'
    elif mode in BYOL_MODES:
        return 'byol'
    elif mode in DL_BASELINE_MODES:
        return 'dl_baseline'
    elif mode in LGB_MODES:
        return 'lgb'
    else:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")


def resolve_backend_mode(mode):
    """Map public aliases to the concrete model implementation mode."""
    if mode in {DP_SCL_MODE, LEGACY_DP_SCL_ALIAS}:
        return DP_SCL_BACKEND_MODE
    return mode


def get_required_data_files(mode, dataset_name):
    """Trả danh sách data files cần cho mode + dataset.

    Args:
        mode: training mode
        dataset_name: 'xuetangx', 'oulad', 'snap'

    Returns:
        dict with keys: 'npz' (always), 'graph' (if graph mode)
              values are relative paths from datastore/
    """
    from src.dataset_config import get_dataset_config
    ds = get_dataset_config(dataset_name)

    files = {'npz': ds['npz_filename']}
    if is_graph_mode(mode):
        files['graph'] = ds['graph_filename']

    return files


def describe_mode(mode):
    """Trả mô tả tóm tắt cho mode: nhánh, framework, kỹ thuật chính."""
    fw = get_framework(mode)
    branch = "GRAPH" if is_graph_mode(mode) else "NO-GRAPH"

    fw_labels = {
        'lgb': 'Legacy graph-temporal baselines',
        'supcon': 'Framework 1: SupCon Network',
        'simclr': 'Framework 2A: SimCLR',
        'byol': 'Framework 2B: BYOL',
        'dl_baseline': 'Baseline DL',
    }

    return f"[{branch}] {fw_labels[fw]} — mode={mode}"


def print_mode_table():
    """In bảng tóm tắt tất cả modes — dùng cho --help hoặc debug."""
    print(f"\n{'═'*70}")
    print(f"  DP-SCL Mode Registry — {len(ALL_MODES)} modes")
    print(f"{'═'*70}")
    print(f"\n  ── GRAPH MODES ({len(GRAPH_MODES)} modes) ── Cần: .npz + .pkl")
    for m in sorted(GRAPH_MODES):
        print(f"    {m:<25} {describe_mode(m)}")
    print(f"\n  ── NO-GRAPH MODES ({len(NO_GRAPH_MODES)} modes) ── Cần: .npz only")
    for m in sorted(NO_GRAPH_MODES):
        print(f"    {m:<25} {describe_mode(m)}")
    print(f"{'═'*70}\n")
