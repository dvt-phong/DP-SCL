#!/usr/bin/env python3
"""
Task 2 - Statistical significance analysis of AUC improvements.
Parses per-seed AUC from report.txt and runs paired statistical tests
of DP-SCL (proposed) against all baselines.
"""
import re
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPORT_PATH = "report.txt"
PROPOSED_KEY = "TSN-SupCon"   # alias of the proposed DP-SCL in the report

# ----------------------------------------------------------------------
# 1. PARSE report.txt
# ----------------------------------------------------------------------
def parse_report(path):
    """Return {model_name: [auc_seed1, ..., auc_seed5]} from the log file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    results = {}
    current_model = None
    # matches lines like:  "MODEL: dl_lstm"  or  "MODEL: TSN-SupCon"
    model_re = re.compile(r"^\s*MODEL:\s*(\S+)", re.MULTILINE)
    # matches a per-seed line and captures the AUC value
    seed_re = re.compile(r"Seed\s+\d+.*?AUC=([0-9]*\.?[0-9]+)", re.IGNORECASE)

    for line in text.splitlines():
        m_model = model_re.match(line)
        if m_model:
            current_model = m_model.group(1).strip()
            results[current_model] = []
            continue
        if current_model is not None:
            m_seed = seed_re.search(line)
            if m_seed:
                results[current_model].append(float(m_seed.group(1)))

    # keep only models that have exactly 5 per-seed AUC values
    results = {k: v for k, v in results.items() if len(v) == 5}
    return results


# ----------------------------------------------------------------------
# 2. STATISTICAL TESTS
# ----------------------------------------------------------------------
def run_tests(results, proposed_key):
    if proposed_key not in results:
        raise KeyError(f"Proposed model '{proposed_key}' not found. "
                       f"Available: {list(results)}")

    dp = np.array(results[proposed_key], dtype=float)
    rows = []
    for name, scores in results.items():
        if name == proposed_key:
            continue
        s = np.array(scores, dtype=float)
        d = dp - s                                  # paired differences
        mean_diff = d.mean()
        # primary test: one-sided paired t-test (H1: DP-SCL > baseline)
        t_stat, p_t = stats.ttest_rel(dp, s, alternative="greater")
        # secondary test: exact Wilcoxon signed-rank (one-sided)
        try:
            w_stat, p_w = stats.wilcoxon(dp, s, alternative="greater",
                                         method="exact")
        except ValueError:           # all-zero differences edge case
            p_w = np.nan
        # effect size for paired samples
        cohen_dz = mean_diff / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
        wins = int(np.sum(dp > s))
        rows.append({
            "model": name, "mean_auc": s.mean(), "delta": mean_diff,
            "p_t": p_t, "p_w": p_w, "cohen_dz": cohen_dz, "wins": wins,
        })

    # Holm correction over the family of paired t-test p-values
    p_t_list = [r["p_t"] for r in rows]
    rej, p_holm, _, _ = multipletests(p_t_list, alpha=0.05, method="holm")
    for r, ph, rj in zip(rows, p_holm, rej):
        r["p_holm"] = ph
        r["sig"] = rj
    # sort by baseline AUC (strongest baseline last)
    rows.sort(key=lambda r: r["mean_auc"])
    return dp, rows


# ----------------------------------------------------------------------
# 3. REPORT
# ----------------------------------------------------------------------
def print_table(dp, rows, proposed_key):
    print(f"Proposed model: {proposed_key}")
    print(f"Per-seed AUC  : {list(np.round(dp, 4))}")
    print(f"Mean AUC      : {dp.mean():.4f} (std {dp.std(ddof=1):.4f})\n")
    hdr = (f"{'Baseline':16}{'AUC':>9}{'ΔAUC%':>9}{'p(t)':>10}"
           f"{'p_Holm':>10}{'p(Wilc)':>10}{'Cohen_dz':>10}{'wins':>7}{'sig':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:16}{r['mean_auc']:>9.4f}{r['delta']*100:>9.3f}"
              f"{r['p_t']:>10.4f}{r['p_holm']:>10.4f}{r['p_w']:>10.4f}"
              f"{r['cohen_dz']:>10.2f}{r['wins']:>5}/5"
              f"{'  yes' if r['sig'] else '   no':>5}")


if __name__ == "__main__":
    results = parse_report(REPORT_PATH)
    dp, rows = run_tests(results, PROPOSED_KEY)
    print_table(dp, rows, PROPOSED_KEY)
