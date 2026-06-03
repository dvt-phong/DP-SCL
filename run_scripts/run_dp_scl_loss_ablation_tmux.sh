#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-dp_scl_loss_ablation}"
RUN_NAME="${RUN_NAME:-dp_scl_loss_ablation}"
LOG_DIR="${LOG_DIR:-logs}"
TMUX_SOCKET="${TMUX_SOCKET:-/tmp/tmux-$(id -u)/default}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT_DIR="${PROJECT_DIR}/run_scripts"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  python3 -u "${RUN_SCRIPT_DIR}/run_dp_scl_loss_ablation.py"
  --seeds 1 11 111 1111 11111
  --max-epochs 200
  --patience 30
  --batch-size 64
  --hidden-size 128
  --lambda-con 0.1
  --run-name "${RUN_NAME}"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed or not in PATH." >&2
  exit 1
fi

if tmux -S "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: tmux session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux -S ${TMUX_SOCKET} attach -t ${SESSION_NAME}" >&2
  echo "Or choose another name: SESSION_NAME=my_session $0" >&2
  exit 1
fi

printf "Starting tmux session: %s\n" "${SESSION_NAME}"
printf "Run name: %s\n" "${RUN_NAME}"
printf "Tmux socket: %s\n" "${TMUX_SOCKET}"
printf "Log file: %s\n" "${LOG_FILE}"
printf "Command: %q " "${CMD[@]}"
printf "\n"

COMMAND_STRING="$(printf "%q " "${CMD[@]}")"
tmux -S "${TMUX_SOCKET}" new-session -d -s "${SESSION_NAME}" \
  "set -euo pipefail; cd $(printf "%q" "${PROJECT_DIR}"); ${COMMAND_STRING} 2>&1 | tee $(printf "%q" "${LOG_FILE}"); status=\${PIPESTATUS[0]}; echo; echo \"Command exited with status \${status}\"; exec bash"

echo "Started."
echo "Attach: tmux -S ${TMUX_SOCKET} attach -t ${SESSION_NAME}"
echo "Tail log: tail -f ${LOG_FILE}"
