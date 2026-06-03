#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT_DIR="${PROJECT_DIR}/run_scripts"

DATASET="${DATASET:-xuetangx}"
INDIR="${INDIR:-${PROJECT_DIR}}"
OUTDIR="${OUTDIR:-${PROJECT_DIR}}"
RUN_NAME="${RUN_NAME:-dp_scl_loss_ablation_full}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

SEEDS="${SEEDS:-1 11 111 1111 11111}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAMBDA_SUPCON="${LAMBDA_SUPCON:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.07}"
MASK_RATIO="${MASK_RATIO:-0.15}"
NOISE_STD="${NOISE_STD:-0.05}"
NUM_LAYERS="${NUM_LAYERS:-1}"
CLS_LAYERS="${CLS_LAYERS:-1}"
MONITOR="${MONITOR:-auc}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FORCE="${FORCE:-1}"
TEE_LOG="${TEE_LOG:-1}"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<EOF
Usage: ./run_scripts/run_dp_scl_loss_ablation_fixed.sh [extra run_dp_scl_loss_ablation.py args]

Environment overrides:
  PYTHON_BIN       ${PYTHON_BIN}
  RUN_NAME         ${RUN_NAME}
  SEEDS            "${SEEDS}"
  MAX_EPOCHS       ${MAX_EPOCHS}
  PATIENCE         ${PATIENCE}
  BATCH_SIZE       ${BATCH_SIZE}
  LAMBDA_SUPCON    ${LAMBDA_SUPCON}
  MONITOR          ${MONITOR}  (auc for paper default, f1 for separate check)
  FORCE            ${FORCE}  (1 adds --force, 0 allows result reuse)
  TEE_LOG          ${TEE_LOG}  (1 writes logs/<run_name>_<timestamp>.log)

Example:
  ./run_scripts/run_dp_scl_loss_ablation_fixed.sh
  LAMBDA_SUPCON=0.3 RUN_NAME=dp_scl_loss_ablation_lam0p30 ./run_scripts/run_dp_scl_loss_ablation_fixed.sh
  MONITOR=f1 RUN_NAME=dp_scl_loss_ablation_f1_monitor ./run_scripts/run_dp_scl_loss_ablation_fixed.sh
EOF
  exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: cannot find python command: ${PYTHON_BIN}" >&2
  echo "Activate the conda env first, or set PYTHON_BIN explicitly." >&2
  exit 1
fi

if [ ! -f "${RUN_SCRIPT_DIR}/run_dp_scl_loss_ablation.py" ]; then
  echo "ERROR: missing ${RUN_SCRIPT_DIR}/run_dp_scl_loss_ablation.py" >&2
  exit 1
fi

read -r -a SEED_ARGS <<< "${SEEDS}"

cd "${PROJECT_DIR}"

CMD=(
  "${PYTHON_BIN}" -u "${RUN_SCRIPT_DIR}/run_dp_scl_loss_ablation.py"
  -indir "${INDIR}" \
  -outdir "${OUTDIR}" \
  --dataset "${DATASET}" \
  --seeds "${SEED_ARGS[@]}" \
  --split 0.60 0.10 0.30 \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --lambda-con "${LAMBDA_SUPCON}" \
  --temperature "${TEMPERATURE}" \
  --proposed-mode siamese_lstm_attn \
  --mask-ratio "${MASK_RATIO}" \
  --noise-std "${NOISE_STD}" \
  --num-layers "${NUM_LAYERS}" \
  --cls-layers "${CLS_LAYERS}" \
  --monitor "${MONITOR}" \
  --num-workers "${NUM_WORKERS}" \
  --run-name "${RUN_NAME}"
)

if [ "${FORCE}" = "1" ] || [ "${FORCE}" = "true" ]; then
  CMD+=(--force)
fi

CMD+=("$@")

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Project: ${PROJECT_DIR}"
echo "Python:  $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Run:     ${RUN_NAME}"
echo "Seeds:   ${SEEDS}"
echo "Output:  ${OUTDIR}/results/experiment_${RUN_NAME}"
echo "Log:     ${LOG_FILE}"
echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo

if [ "${TEE_LOG}" = "1" ] || [ "${TEE_LOG}" = "true" ]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  status=${PIPESTATUS[0]}
else
  "${CMD[@]}"
  status=$?
fi

echo
echo "Finished with status ${status}"
exit "${status}"
