#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_vast_training.sh \
    --job-name JOB \
    --train-config-file config/deepspeed/zero3_4GPU.yaml \
    --train-config-name 7b_vast_dryrun \
    [-- <extra hydra overrides for training>]
EOF
}

JOB_NAME=""
TRAIN_CONFIG_FILE=""
TRAIN_CONFIG_NAME=""
TRAIN_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --train-config-file)
      TRAIN_CONFIG_FILE="$2"
      shift 2
      ;;
    --train-config-name)
      TRAIN_CONFIG_NAME="$2"
      shift 2
      ;;
    --)
      shift
      TRAIN_OVERRIDES=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run_vast_training.sh] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${JOB_NAME}" || -z "${TRAIN_CONFIG_FILE}" || -z "${TRAIN_CONFIG_NAME}" ]]; then
  usage >&2
  exit 1
fi

mkdir -p checkpoints logs reports .runtime

RESOLVED_JSON="$(python3 scripts/resolve_run_config.py --mode train --config-name "${TRAIN_CONFIG_NAME}" "${TRAIN_OVERRIDES[@]}")"
SAVE_DIR="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["logging_save_dir"])' "${RESOLVED_JSON}")"
SERVER_PORT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["generation_server_port"])' "${RESOLVED_JSON}")"

TRAIN_LOG="logs/${JOB_NAME}_train_full.log"
SERVER_LOG="logs/${JOB_NAME}_vllm_server.log"
STATUS_FILE=".runtime/${JOB_NAME}_train_status.json"
PID_FILE=".runtime/${JOB_NAME}_vllm_server.pid"

find_last_checkpoint() {
  local save_dir="$1"
  python3 - "$save_dir" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pattern = re.compile(r"checkpoint-(\d+)$")
candidates = []
if root.exists():
    for path in root.iterdir():
        match = pattern.match(path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
if candidates:
    candidates.sort()
    print(candidates[-1][1])
PY
}

LAST_CHECKPOINT="$(find_last_checkpoint "${SAVE_DIR}")"

echo "[run_vast_training.sh] Job: ${JOB_NAME}"
echo "[run_vast_training.sh] Train config: ${TRAIN_CONFIG_NAME}"
echo "[run_vast_training.sh] Save dir: ${SAVE_DIR}"
echo "[run_vast_training.sh] Train log: ${TRAIN_LOG}"
echo "[run_vast_training.sh] Server log: ${SERVER_LOG}"
echo "[run_vast_training.sh] Server port: ${SERVER_PORT}"
if [[ -n "${LAST_CHECKPOINT}" ]]; then
  echo "[run_vast_training.sh] Resume checkpoint: ${LAST_CHECKPOINT}"
else
  echo "[run_vast_training.sh] Resume checkpoint: <none>"
fi

RUN_PID=""
SUMMARY_ARGS_BASE=(
  --output "${STATUS_FILE}"
  --job-name "${JOB_NAME}"
  --train-config-file "${TRAIN_CONFIG_FILE}"
  --train-config-name "${TRAIN_CONFIG_NAME}"
  --save-dir "${SAVE_DIR}"
  --model-dir "${SAVE_DIR}/model"
  --server-port "${SERVER_PORT}"
  --train-log "${TRAIN_LOG}"
  --server-log "${SERVER_LOG}"
)
for override in "${TRAIN_OVERRIDES[@]}"; do
  SUMMARY_ARGS_BASE+=(--override "${override}")
done
on_signal() {
  echo "[run_vast_training.sh] Caught signal, forwarding stop to training stack..."
  if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM "${RUN_PID}" 2>/dev/null || true
    wait "${RUN_PID}" || true
  fi
  LAST_CHECKPOINT="$(find_last_checkpoint "${SAVE_DIR}")"
  VLLM_PID_FILE="${PID_FILE}" ./stop_vllm_server.sh || true
  python3 scripts/create_run_summary.py \
    "${SUMMARY_ARGS_BASE[@]}" \
    --last-checkpoint "${LAST_CHECKPOINT}" \
    --status interrupted >/dev/null 2>&1 || true
  exit 130
}
trap on_signal INT TERM

TRAIN_CMD=(
  ./start_rl_training.sh
  --config_file "${TRAIN_CONFIG_FILE}"
  --config-name "${TRAIN_CONFIG_NAME}"
)
if [[ ${#TRAIN_OVERRIDES[@]} -gt 0 ]]; then
  TRAIN_CMD+=(-- "${TRAIN_OVERRIDES[@]}")
fi

set +e
VLLM_PID_FILE="${PID_FILE}" \
VLLM_SERVER_LOG_FILE="${SERVER_LOG}" \
VLLM_SERVER_PORT="${SERVER_PORT}" \
stdbuf -oL -eL "${TRAIN_CMD[@]}" \
  > >(tee "${TRAIN_LOG}" | python3 scripts/filter_runtime_logs.py --mode train) 2>&1 &
RUN_PID=$!
wait "${RUN_PID}"
STATUS=$?
set -e

LAST_CHECKPOINT="$(find_last_checkpoint "${SAVE_DIR}")"
SUMMARY_ARGS=(
  "${SUMMARY_ARGS_BASE[@]}"
  --last-checkpoint "${LAST_CHECKPOINT}"
)

if [[ "${STATUS}" -ne 0 ]]; then
  echo "[run_vast_training.sh] Training failed. Full log: ${TRAIN_LOG}" >&2
  python3 scripts/create_run_summary.py "${SUMMARY_ARGS[@]}" --status failed
  exit "${STATUS}"
fi

python3 scripts/create_run_summary.py "${SUMMARY_ARGS[@]}" --status trained
echo "[run_vast_training.sh] Training finished successfully."
