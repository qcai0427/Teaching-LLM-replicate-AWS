#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_vast_job.sh \
    --job-name JOB \
    --train-config-file config/deepspeed/zero3_4GPU.yaml \
    --train-config-name 7b_vast_dryrun \
    --eval-config-name 7b_vast_dryrun \
    [--hf-repo-id USER/REPO] \
    [--hf-repo-type model] \
    [--hf-artifact-prefix runs/JOB] \
    [--skip-upload] \
    [-- <extra hydra overrides for training>]
EOF
}

JOB_NAME=""
TRAIN_CONFIG_FILE=""
TRAIN_CONFIG_NAME=""
EVAL_CONFIG_NAME=""
HF_REPO_ID=""
HF_REPO_TYPE="model"
HF_ARTIFACT_PREFIX=""
SKIP_UPLOAD=false
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
    --hf-repo-id)
      HF_REPO_ID="$2"
      shift 2
      ;;
    --hf-repo-type)
      HF_REPO_TYPE="$2"
      shift 2
      ;;
    --hf-artifact-prefix)
      HF_ARTIFACT_PREFIX="$2"
      shift 2
      ;;
    --skip-upload)
      SKIP_UPLOAD=true
      shift
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
      echo "[run_vast_job.sh] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${JOB_NAME}" || -z "${TRAIN_CONFIG_FILE}" || -z "${TRAIN_CONFIG_NAME}" || -z "${EVAL_CONFIG_NAME}" ]]; then
  usage >&2
  exit 1
fi

mkdir -p checkpoints logs reports .runtime

TRAIN_JSON="$(python3 scripts/resolve_run_config.py --mode train --config-name "${TRAIN_CONFIG_NAME}" "${TRAIN_OVERRIDES[@]}")"
SAVE_DIR="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["logging_save_dir"])' "${TRAIN_JSON}")"
CONFIG_HF_NAME="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["huggingface_name"])' "${TRAIN_JSON}")"
SERVER_PORT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["generation_server_port"])' "${TRAIN_JSON}")"
if [[ -z "${HF_ARTIFACT_PREFIX}" ]]; then
  HF_ARTIFACT_PREFIX="runs/${JOB_NAME}"
fi
if [[ -z "${HF_REPO_ID}" && "${CONFIG_HF_NAME}" != \<* && -n "${CONFIG_HF_NAME}" ]]; then
  HF_REPO_ID="${CONFIG_HF_NAME}"
fi
if [[ "${SKIP_UPLOAD}" == "false" && -z "${HF_REPO_ID}" ]]; then
  echo "[run_vast_job.sh] Hugging Face repo id is required. Pass --hf-repo-id or set huggingface.name in the train config." >&2
  exit 1
fi

METRICS_FILE="reports/${JOB_NAME}_eval_metrics.json"
CONVERSATIONS_FILE="reports/${JOB_NAME}_eval_conversations.json"
EVAL_LOG="logs/${JOB_NAME}_eval_full.log"
SERVER_LOG="logs/${JOB_NAME}_vllm_server.log"
STATUS_FILE=".runtime/${JOB_NAME}_run_summary.json"
SUMMARY_ARGS=(
  --output "${STATUS_FILE}"
  --job-name "${JOB_NAME}"
  --train-config-file "${TRAIN_CONFIG_FILE}"
  --train-config-name "${TRAIN_CONFIG_NAME}"
  --eval-config-name "${EVAL_CONFIG_NAME}"
  --save-dir "${SAVE_DIR}"
  --model-dir "${SAVE_DIR}/model"
  --server-port "${SERVER_PORT}"
  --train-log "logs/${JOB_NAME}_train_full.log"
  --eval-log "${EVAL_LOG}"
  --server-log "${SERVER_LOG}"
  --metrics-file "${METRICS_FILE}"
  --conversations-file "${CONVERSATIONS_FILE}"
  --hf-repo-id "${HF_REPO_ID}"
)
for override in "${TRAIN_OVERRIDES[@]}"; do
  SUMMARY_ARGS+=(--override "${override}")
done

./run_vast_training.sh \
  --job-name "${JOB_NAME}" \
  --train-config-file "${TRAIN_CONFIG_FILE}" \
  --train-config-name "${TRAIN_CONFIG_NAME}" \
  -- "${TRAIN_OVERRIDES[@]}"

EVAL_CMD=(
  python eval.py
  --config-name "${EVAL_CONFIG_NAME}"
  "teacher_model.model_name_or_path=${SAVE_DIR}/model"
  "export_conversations_path=${CONVERSATIONS_FILE}"
  "export_metrics_path=${METRICS_FILE}"
)

echo "[run_vast_job.sh] Eval log: ${EVAL_LOG}"
set +e
stdbuf -oL -eL "${EVAL_CMD[@]}" \
  > >(tee "${EVAL_LOG}" | python3 scripts/filter_runtime_logs.py --mode eval) 2>&1
EVAL_STATUS=$?
set -e

if [[ "${EVAL_STATUS}" -ne 0 ]]; then
  echo "[run_vast_job.sh] Eval failed. Full log: ${EVAL_LOG}" >&2
  python3 scripts/create_run_summary.py \
    "${SUMMARY_ARGS[@]}" \
    --status eval_failed
  exit "${EVAL_STATUS}"
fi

python3 scripts/create_run_summary.py \
  "${SUMMARY_ARGS[@]}" \
  --status evaluated

if [[ "${SKIP_UPLOAD}" == "true" ]]; then
  echo "[run_vast_job.sh] Upload skipped. Outputs remain on local disk."
  exit 0
fi

UPLOAD_LOG="logs/${JOB_NAME}_upload_full.log"
echo "[run_vast_job.sh] Upload log: ${UPLOAD_LOG}"
set +e
python3 scripts/upload_run_to_hub.py \
  --repo-id "${HF_REPO_ID}" \
  --repo-type "${HF_REPO_TYPE}" \
  --model-dir "${SAVE_DIR}/model" \
  --artifact-prefix "${HF_ARTIFACT_PREFIX}" \
  --metrics-file "${METRICS_FILE}" \
  --conversations-file "${CONVERSATIONS_FILE}" \
  --summary-file "${STATUS_FILE}" \
  2>&1 | tee "${UPLOAD_LOG}"
UPLOAD_STATUS=${PIPESTATUS[0]}
set -e

if [[ "${UPLOAD_STATUS}" -ne 0 ]]; then
  echo "[run_vast_job.sh] Upload failed. Local artifacts were preserved." >&2
  python3 scripts/create_run_summary.py \
    "${SUMMARY_ARGS[@]}" \
    --status upload_failed
  exit "${UPLOAD_STATUS}"
fi

python3 scripts/create_run_summary.py \
  "${SUMMARY_ARGS[@]}" \
  --status uploaded

echo "[run_vast_job.sh] Training, eval, and upload completed successfully."
