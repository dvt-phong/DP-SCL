#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

PY="${PY:-python3}"
STAMP=$(date +%Y%m%d_%H%M%S)

"$PY" run_scripts/run_lambda_sens.py "$@" 2>&1 | tee "logs/lambda_sens_${STAMP}.log"
"$PY" run_scripts/run_tau_sens.py "$@" 2>&1 | tee "logs/tau_sens_${STAMP}.log"
