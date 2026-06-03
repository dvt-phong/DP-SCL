import os
from datetime import datetime


def write_ml_results(output_dir, dataset_name, results, elapsed_seconds):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, "results", "ml_baselines")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{dataset_name}_{timestamp}.txt")

    lines = [
        "=" * 80,
        "  Machine Learning Baseline Results",
        "=" * 80,
        f"  Dataset: {dataset_name}",
        f"  Methods: {len(results)}",
        f"  Total time: {elapsed_seconds:.2f}s",
        "-" * 80,
        "method\tauc\tacc\tf1\tprecision\trecall\tthreshold\tfit_time_sec\tstatus",
    ]

    for item in results:
        metrics = item.get("metrics") or {}
        lines.append(
            "{method}\t{auc}\t{acc}\t{f1}\t{precision}\t{recall}\t{threshold}\t{fit_time_sec:.2f}\t{status}".format(
                method=item["method"],
                auc=_fmt(metrics.get("auc")),
                acc=_fmt(metrics.get("acc")),
                f1=_fmt(metrics.get("f1")),
                precision=_fmt(metrics.get("precision")),
                recall=_fmt(metrics.get("recall")),
                threshold=_fmt(metrics.get("threshold")),
                fit_time_sec=item.get("fit_time_sec", 0.0),
                status=item.get("status", "ok"),
            )
        )
        if item.get("error"):
            lines.append(f"# {item['method']} error: {item['error']}")

    with open(result_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    return result_path


def _fmt(value):
    if value is None:
        return "nan"
    return f"{value:.6f}"

