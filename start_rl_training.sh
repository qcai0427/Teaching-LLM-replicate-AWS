#!/usr/bin/env bash
set -euo pipefail

PID_FILE=".vllm_server.pid"
SERVER_LOG_DIR="logs"
SERVER_LOG_FILE="${SERVER_LOG_DIR}/vllm_server.log"

#-----------------------------------------
# Graceful shutdown on Ctrl-C / SIGTERM
#-----------------------------------------
cleanup() {
  echo "[start_rl_training.sh] Caught signal, stopping VLLM server..."
  ./stop_vllm_server.sh || true
  exit 130
}
trap cleanup INT TERM

#-----------------------------------------
# Buckets for the three kinds of flags
#-----------------------------------------
ACCELERATE_ARGS=()
SERVER_ARGS=()
TRAIN_ARGS=()

#-----------------------------------------
# Parse CLI
#-----------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    # -------------------------------
    # Flags that belong to Accelerate
    # -------------------------------
    --config_file|--num_processes|--num_machines|--machine_rank|\
    --main_process_ip|--main_process_port)
        ACCELERATE_ARGS+=("$1" "$2")
        shift 2
        ;;

    # ---------------------------------------------------------
    # The model-specific YAML should go to both buckets
    # ---------------------------------------------------------
    --config-name)
        SERVER_ARGS+=("$1" "$2")      # for start_vllm_server.sh
        TRAIN_ARGS+=("$1" "$2")       # for train_rl.py via accelerate
        shift 2
        ;;

    # ---------------------------------
    # Separator: everything after " -- "
    # goes only to TRAIN_ARGS
    # ---------------------------------
    --)
        shift
        TRAIN_ARGS+=("$@")
        break
        ;;

    # ---------------------------------
    # Anything else is a server flag
    # ---------------------------------
    *)
        SERVER_ARGS+=("$1")
        shift
        ;;
  esac
done

#-----------------------------------------
# Make sure we at least got a config file
#-----------------------------------------
if ! printf '%s\n' "${ACCELERATE_ARGS[@]}" | grep -q -- '--config_file'; then
  echo "[start_rl_training.sh] ERROR: --config_file is required." >&2
  exit 1
fi

#-----------------------------------------
# Start the VLLM server
#-----------------------------------------
echo "[start_rl_training.sh] Launching VLLM server..."
./stop_vllm_server.sh || true
sleep 2
mkdir -p "${SERVER_LOG_DIR}"
: > "${SERVER_LOG_FILE}"
./start_vllm_server.sh "${SERVER_ARGS[@]}" > "${SERVER_LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "[start_rl_training.sh] VLLM server log: ${SERVER_LOG_FILE}"

#-----------------------------------------
# Wait until the server responds
#-----------------------------------------
until curl -fsS http://localhost:8005/docs >/dev/null ; do
  echo "[start_rl_training.sh] Waiting for VLLM server..."
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[start_rl_training.sh] VLLM server exited before becoming ready."
    echo "[start_rl_training.sh] Last server log lines:"
    tail -n 50 "${SERVER_LOG_FILE}" || true
    exit 1
  fi
  if [[ -s "${SERVER_LOG_FILE}" ]]; then
    echo "[start_rl_training.sh] Latest server log:"
    tail -n 10 "${SERVER_LOG_FILE}" || true
  fi
  sleep 5
done
echo "[start_rl_training.sh] VLLM server is up."

#-----------------------------------------
# Run training
#-----------------------------------------
ACCEL_CMD=(accelerate launch "${ACCELERATE_ARGS[@]}" train_rl.py "${TRAIN_ARGS[@]}")
echo "[start_rl_training.sh] About to run:"
printf '  %q ' "${ACCEL_CMD[@]}"
echo    

"${ACCEL_CMD[@]}"

#-----------------------------------------
# Cleanup
#-----------------------------------------
./stop_vllm_server.sh
rm -f "${PID_FILE}"
echo "[start_rl_training.sh] Done."
