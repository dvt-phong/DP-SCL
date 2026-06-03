import argparse
import csv
import json
import os

from experiment_sensitivity_runner import f4, get_model_rows, get_model_summary, mean_std


ABLATION_MODELS = [
    {
        "id": "A1",
        "display": "LSTM-BCE",
        "run_name": "ablation_a1_dl_lstm",
        "csv_model": "dl_lstm",
        "mode": "dl_lstm",
    },
    {
        "id": "A2",
        "display": "LSTM-MHA-BCE",
        "run_name": "ablation_a2_dl_lstm_mha",
        "csv_model": "dl_lstm_mha",
        "mode": "dl_lstm_mha",
    },
    {
        "id": "A3",
        "display": "LSTM-MHA-LQ-BCE",
        "run_name": "lstm_mha_lq_bce_hidden128",
        "csv_model": "dl_lstm_mha_lq",
        "mode": "dl_lstm_mha_lq",
    },
    {
        "id": "A4",
        "display": "Siamese-LSTM-SupCon",
        "run_name": "ablation_siamese_lstm",
        "csv_model": "Siamese_LSTM",
        "mode": "siamese_lstm",
    },
    {
        "id": "A5",
        "display": "Siamese-LSTM-MHA-SupCon",
        "run_name": "ablation_a5_siamese_lstm_mha",
        "csv_model": "A5_Siamese_LSTM_MHA_SupCon",
        "mode": "siamese_lstm_mha",
    },
    {
        "id": "A6",
        "display": "Siamese-LSTM-MHA-LQ-BCE",
        "run_name": "ablation_a6_siamese_lstm_attn_lambda0",
        "csv_model": "A6_Siamese_LSTM_MHA_LQ_BCE",
        "mode": "siamese_lstm_attn_lambda0",
    },
    {
        "id": "A7",
        "display": "DP-SCL",
        "run_name": "ablation_a7_dp_scl",
        "csv_model": "A7_DP_SCL",
        "mode": "dp_scl",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect 7-model ablation results.")
    parser.add_argument("-outdir", type=str, default=".")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def run_dir(outdir, run_name):
    return os.path.join(os.path.abspath(os.path.expanduser(outdir)), "results", f"experiment_{run_name}")


def read_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    output_root = os.path.abspath(os.path.expanduser(args.outdir))
    result_write = os.path.join(output_root, "result_write")
    os.makedirs(result_write, exist_ok=True)

    rows = []
    missing = []
    for spec in ABLATION_MODELS:
        path = run_dir(output_root, spec["run_name"])
        summary_path = os.path.join(path, "summary_results.csv")
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(summary_path):
            missing.append(f"{spec['id']} {spec['display']}: {summary_path}")
            if args.allow_missing:
                continue
            continue

        summary = get_model_summary(path, spec["csv_model"])
        per_seed = get_model_rows(path, spec["csv_model"])
        config = read_config(config_path) if os.path.exists(config_path) else {}
        rows.append({
            "id": spec["id"],
            "model": spec["display"],
            "mode": spec["mode"],
            "run_name": spec["run_name"],
            "hidden_size": config.get("hidden_size", ""),
            "lr": config.get("lr", ""),
            "batch_size": config.get("batch_size", ""),
            "lambda_con": config.get("lambda_con", ""),
            "temperature": config.get("temperature", ""),
            "auc": mean_std(summary, "auc"),
            "precision": mean_std(summary, "precision"),
            "recall": mean_std(summary, "recall"),
            "f1": mean_std(summary, "f1"),
            "avg_best_epoch": f4(summary["avg_best_epoch"]) if summary.get("avg_best_epoch") else "",
            "avg_stopped_epoch": f4(summary["avg_stopped_epoch"]) if summary.get("avg_stopped_epoch") else "",
            "seeds_ok": len(per_seed),
        })

    if missing and not args.allow_missing:
        print("Missing result files:")
        for item in missing:
            print(f"  {item}")
        raise SystemExit(1)

    csv_path = os.path.join(result_write, "ablation_7models_table.csv")
    txt_path = os.path.join(result_write, "ablation_7models_table.txt")
    fieldnames = [
        "id", "model", "mode", "run_name", "hidden_size", "lr", "batch_size",
        "lambda_con", "temperature", "auc", "precision", "recall", "f1",
        "avg_best_epoch", "avg_stopped_epoch", "seeds_ok",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("ABLATION 7 MODELS\n")
        handle.write("Protocol: 5 seeds, split 60/10/30, max_epochs=200, patience=30\n\n")
        handle.write(f"{'ID':<3} {'Model':<30} {'AUC':<17} {'F1':<17} {'Seeds':<5}\n")
        handle.write("-" * 78 + "\n")
        for row in rows:
            handle.write(
                f"{row['id']:<3} {row['model']:<30} {row['auc']:<17} "
                f"{row['f1']:<17} {row['seeds_ok']:<5}\n"
            )

    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
