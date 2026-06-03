#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT_DIR="${PROJECT_DIR}/run_scripts"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${PROJECT_DIR}/result_write/lstm_mha_lq_bce_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON_BIN"
echo "Logs:    $LOG_DIR"
echo ""

"$PYTHON_BIN" "${RUN_SCRIPT_DIR}/run_lstm_mha_lq_bce.py" "$@" 2>&1 | tee "$LOG_DIR/lstm_mha_lq_bce.log"

echo ""
echo "Output report:"
echo "  result_write/lstm_mha_lq_bce_hidden128.txt"
