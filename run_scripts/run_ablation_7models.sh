#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT_DIR="${PROJECT_DIR}/run_scripts"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTDIR="."
DRY_RUN=false
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        -outdir|--outdir)
            if (( i + 1 < ${#ARGS[@]} )); then
                OUTDIR="${ARGS[$((i + 1))]}"
            fi
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
    esac
done

LOG_DIR="${PROJECT_DIR}/result_write/ablation_7models_logs_$(date +%Y%m%d_%H%M%S)"
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$LOG_DIR"
fi

run_step() {
    local step_id="$1"
    local script="$2"
    shift 2
    echo ""
    echo "============================================================"
    echo "[$step_id] $script"
    echo "============================================================"
    if [ "$DRY_RUN" = true ]; then
        "$PYTHON_BIN" "${RUN_SCRIPT_DIR}/${script}" "$@"
    else
        local log_file="$LOG_DIR/${step_id}_${script%.py}.log"
        "$PYTHON_BIN" "${RUN_SCRIPT_DIR}/${script}" "$@" 2>&1 | tee "$log_file"
    fi
}

echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON_BIN"
if [ "$DRY_RUN" = false ]; then
    echo "Logs:    $LOG_DIR"
else
    echo "Logs:    disabled for dry-run"
fi
echo "Outdir:  $OUTDIR"
echo ""

run_step "A1" "run_dl_lstm.py" "$@"
run_step "A2" "run_dl_lstm_mha.py" "$@"
run_step "A3" "run_lstm_mha_lq_bce.py" "$@"
run_step "A4" "run_ablation.py" "$@"
run_step "A5" "run_siamese_lstm_mha.py" "$@"
run_step "A6" "run_siamese_lambda0.py" "$@"
run_step "A7" "run_dp_scl.py" "$@"

if [ "$DRY_RUN" = false ]; then
    "$PYTHON_BIN" collect_ablation_results.py -outdir "$OUTDIR" 2>&1 | tee "$LOG_DIR/collect_ablation_results.log"
    echo ""
    echo "Final tables:"
    echo "  result_write/ablation_7models_table.txt"
    echo "  result_write/ablation_7models_table.csv"
else
    echo ""
    echo "Dry-run mode: skipped collect_ablation_results.py"
fi
