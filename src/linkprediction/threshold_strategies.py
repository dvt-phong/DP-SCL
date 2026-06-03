"""
Threshold Strategies cho Link Prediction Binarization (η₁).

Hai chiến lược:
    - fixed:    Ngưỡng cố định (gốc, threshold=0.6)
    - adaptive: Per-user percentile — mỗi user có ngưỡng riêng
                dựa trên phân bố xác suất của chính user đó.

Sử dụng:
    from threshold_strategies import apply_threshold
    binary_matrix = apply_threshold(prob_matrix, mode='adaptive', percentile=70)
"""
import numpy as np


def fixed_threshold(prob_matrix, threshold=0.6, **kwargs):
    """Ngưỡng cố định (phương pháp gốc).

    Mọi giá trị >= threshold → 1, còn lại → 0.
    """
    binary = prob_matrix.copy()
    binary[binary >= threshold] = 1
    binary[binary != 1] = 0
    return binary


def adaptive_threshold(prob_matrix, percentile=70, **kwargs):
    """Per-user percentile threshold (η₁ adaptive).

    Với mỗi user, tính percentile thứ k của vector xác suất.
    percentile=70 → giữ top 30% courses có prob cao nhất.

    Ví dụ:
        User A: [0.85, 0.72, 0.65, 0.58, 0.45, 0.30]
          → percentile 70 ≈ 0.68 → giữ Toán(0.85), Lý(0.72)
        User B: [0.95, 0.93, 0.91, 0.89, 0.87, 0.85]
          → percentile 70 ≈ 0.91 → giữ Toán(0.95), Lý(0.93)
        → Cả hai đều giữ top 2 sở thích nổi bật nhất.
    """
    binary = np.zeros_like(prob_matrix)
    num_users = prob_matrix.shape[0]
    thresholds = []

    for u in range(num_users):
        user_probs = prob_matrix[u]
        threshold_u = np.percentile(user_probs, percentile)
        thresholds.append(threshold_u)
        binary[u][user_probs >= threshold_u] = 1

    thresholds = np.array(thresholds)
    print(f"  [η₁ adaptive] percentile={percentile} (giữ top {100 - percentile}%)")
    print(f"  [η₁ adaptive] Per-user threshold: "
          f"min={thresholds.min():.4f}, "
          f"median={np.median(thresholds):.4f}, "
          f"max={thresholds.max():.4f}")
    print(f"  [η₁ adaptive] Total links: {int(binary.sum())}")

    return binary


# ============================================================
# Registry & Entry Point
# ============================================================
STRATEGIES = {
    'fixed': fixed_threshold,
    'adaptive': adaptive_threshold,
}


def apply_threshold(prob_matrix, mode='fixed', **kwargs):
    """Entry point: chọn strategy và apply.

    Args:
        prob_matrix: np.ndarray (num_users, num_courses), giá trị [0, 1]
        mode: 'fixed' hoặc 'adaptive'
        **kwargs: threshold (cho fixed), percentile (cho adaptive)

    Returns:
        np.ndarray binary (num_users, num_courses)
    """
    if mode not in STRATEGIES:
        raise ValueError(
            f"Unknown threshold mode: '{mode}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    print(f"  [η₁] Mode: '{mode}'")
    result = STRATEGIES[mode](prob_matrix, **kwargs)

    # So sánh với fixed baseline
    if mode != 'fixed':
        fixed_result = fixed_threshold(prob_matrix, **kwargs)
        fixed_links = int(fixed_result.sum())
        new_links = int(result.sum())
        diff = new_links - fixed_links
        print(f"  [η₁] So với fixed(0.6): "
              f"{fixed_links} → {new_links} links ({diff:+d}, "
              f"{diff / max(fixed_links, 1) * 100:+.1f}%)")

    return result
