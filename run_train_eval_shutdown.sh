#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_train_eval_shutdown.sh \
    --job-name JOB \
    --train-config-file config/deepspeed/zero3_4GPU.yaml \
    --train-config-name 7b_aws_dryrun \
    [--eval-config-name 7b_aws_dryrun] \
    [--shutdown-policy never|success|always] \
    [-- <extra hydra overrides for training>]
EOF
}

JOB_NAME=""
TRAIN_CONFIG_FILE=""
TRAIN_CONFIG_NAME=""
EVAL_CONFIG_NAME=""
SHUTDOWN_POLICY="never"
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
    --eval-config-name)
      EVAL_CONFIG_NAME="$2"
      shift 2
      ;;
    --shutdown-policy)
      SHUTDOWN_POLICY="$2"
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
      echo "[run_train_eval_shutdown.sh] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${JOB_NAME}" || -z "${TRAIN_CONFIG_FILE}" || -z "${TRAIN_CONFIG_NAME}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${SHUTDOWN_POLICY}" != "never" && "${SHUTDOWN_POLICY}" != "success" && "${SHUTDOWN_POLICY}" != "always" ]]; then
  echo "[run_train_eval_shutdown.sh] Invalid --shutdown-policy: ${SHUTDOWN_POLICY}" >&2
  exit 1
fi

mkdir -p logs
TRAIN_LOG="logs/${JOB_NAME}_train_full.log"
EVAL_LOG="logs/${JOB_NAME}_eval_full.log"

run_with_filtered_console() {
  local mode="$1"
  local logfile="$2"
  shift 2

  set +e
  stdbuf -oL -eL "$@" 2>&1 \
    | tee "${logfile}" \
    | python3 scripts/filter_runtime_logs.py --mode "${mode}"
  local cmd_status=${PIPESTATUS[0]}
  set -e
  return "${cmd_status}"
}

maybe_shutdown() {
  local status="$1"
  if [[ "${SHUTDOWN_POLICY}" == "never" ]]; then
    return 0
  fi
  if [[ "${SHUTDOWN_POLICY}" == "success" && "${status}" -ne 0 ]]; then
    return 0
  fi

  echo "[run_train_eval_shutdown.sh] Syncing files before shutdown..."
  sync || true
  echo "[run_train_eval_shutdown.sh] Shutdown policy '${SHUTDOWN_POLICY}' triggered."
  if sudo -n true 2>/dev/null; then
    sudo shutdown -h now
  else
    echo "[run_train_eval_shutdown.sh] Shutdown skipped: passwordless sudo is not available for $(whoami)." >&2
    echo "[run_train_eval_shutdown.sh] Full logs and outputs are already saved on disk." >&2
  fi
}

TRAIN_CMD=(
  ./start_rl_training.sh
  --config_file "${TRAIN_CONFIG_FILE}"
  --config-name "${TRAIN_CONFIG_NAME}"
)
if [[ ${#TRAIN_OVERRIDES[@]} -gt 0 ]]; then
  TRAIN_CMD+=(-- "${TRAIN_OVERRIDES[@]}")
fi

echo "[run_train_eval_shutdown.sh] Training log: ${TRAIN_LOG}"
if ! run_with_filtered_console train "${TRAIN_LOG}" "${TRAIN_CMD[@]}"; then
  status=$?
  echo "[run_train_eval_shutdown.sh] Training failed. Full log: ${TRAIN_LOG}" >&2
  maybe_shutdown "${status}"
  exit "${status}"
fi

if [[ -n "${EVAL_CONFIG_NAME}" ]]; then
  echo "[run_train_eval_shutdown.sh] Eval log: ${EVAL_LOG}"
  if ! run_with_filtered_console eval "${EVAL_LOG}" python eval.py --config-name "${EVAL_CONFIG_NAME}"; then
    status=$?
    echo "[run_train_eval_shutdown.sh] Eval failed. Full log: ${EVAL_LOG}" >&2
    maybe_shutdown "${status}"
    exit "${status}"
  fi
fi

maybe_shutdown 0
