import argparse
import os
import random
import time

import numpy as np

from baselines.ml import DEFAULT_ML_ORDER, ML_BASELINE_REGISTRY
from baselines.utils import (
    compute_binary_metrics,
    flatten_temporal_data,
    get_estimator_scores,
    load_npz_baseline_data,
    write_ml_results,
)
from src.dataset_config import get_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run sklearn/xgboost ML baselines")
    parser.add_argument("-indir", type=str, default=".", help="input dir (default: current dir)")
    parser.add_argument("-outdir", type=str, default=".", help="output dir (default: current dir)")
    parser.add_argument("--dataset", type=str, default="xuetangx", choices=["xuetangx", "oulad", "snap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="+", choices=DEFAULT_ML_ORDER, default=None)
    parser.add_argument("--skip-slow", action="store_true", help="skip SVM and kNN")
    parser.add_argument("--strict-missing", action="store_true", help="fail if optional packages are missing")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = os.path.abspath(os.path.expanduser(args.indir))
    output_dir = os.path.abspath(os.path.expanduser(args.outdir))
    ds_config = get_dataset_config(args.dataset)

    print(f"=== ML Baselines: {ds_config['name']} ===")
    X_train_seq, y_train, X_test_seq, y_test = load_npz_baseline_data(input_dir, ds_config)
    X_train, X_test = flatten_temporal_data(X_train_seq, X_test_seq)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    methods = list(args.only) if args.only else list(DEFAULT_ML_ORDER)
    if args.skip_slow:
        methods = [method for method in methods if method not in {"ml_svm", "ml_knn"}]

    all_start = time.time()
    results = []
    for method in methods:
        print(f"\n--- {method} ---")
        start = time.time()
        try:
            model = ML_BASELINE_REGISTRY[method](args.seed)
            model.fit(X_train, y_train)
            y_score = get_estimator_scores(model, X_test)
            metrics = compute_binary_metrics(y_test, y_score)
            elapsed = time.time() - start
            results.append({
                "method": method,
                "metrics": metrics,
                "fit_time_sec": elapsed,
                "status": "ok",
            })
            print(
                "  AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f} "
                "Precision={precision:.4f} Recall={recall:.4f} Time={time:.2f}s".format(
                    time=elapsed,
                    **metrics,
                )
            )
        except ModuleNotFoundError as exc:
            if args.strict_missing:
                raise
            elapsed = time.time() - start
            results.append({
                "method": method,
                "metrics": None,
                "fit_time_sec": elapsed,
                "status": "skipped",
                "error": str(exc),
            })
            print(f"  SKIPPED: {exc}")

    result_path = write_ml_results(output_dir, args.dataset, results, time.time() - all_start)
    print(f"\nSaved ML baseline results: {result_path}")


if __name__ == "__main__":
    main()

