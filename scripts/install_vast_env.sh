#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install_vast_env.sh [--torch-index-url URL] [--with-ninja-upgrade] [--skip-flash-attn]

Examples:
  ./scripts/install_vast_env.sh
  ./scripts/install_vast_env.sh --torch-index-url https://download.pytorch.org/whl/cu124
EOF
}

TORCH_INDEX_URL=""
WITH_NINJA_UPGRADE=false
SKIP_FLASH_ATTN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --with-ninja-upgrade)
      WITH_NINJA_UPGRADE=true
      shift
      ;;
    --skip-flash-attn)
      SKIP_FLASH_ATTN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install_vast_env.sh] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc || echo 8)"
FLASH_WHEEL_DIR="${PIP_WHEEL_DIR:-${ROOT_DIR}/.runtime/wheels}"
BUILD_TMP_DIR="${TMPDIR:-${ROOT_DIR}/.runtime/tmp}"
mkdir -p "${FLASH_WHEEL_DIR}"
mkdir -p "${BUILD_TMP_DIR}"

export MAX_JOBS="${MAX_JOBS:-${CPU_COUNT}}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${CPU_COUNT}}"
export NINJA_NUM_JOBS="${NINJA_NUM_JOBS:-${CPU_COUNT}}"
export NVCC_THREADS="${NVCC_THREADS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PIP_PREFER_BINARY=1
export TMPDIR="${BUILD_TMP_DIR}"
export TMP="${BUILD_TMP_DIR}"
export TEMP="${BUILD_TMP_DIR}"

python -m pip install -U pip setuptools wheel packaging
if [[ "${WITH_NINJA_UPGRADE}" == "true" ]]; then
  python -m pip install -U ninja
else
  python -m pip install ninja
fi

if [[ -n "${TORCH_INDEX_URL}" ]]; then
  echo "[install_vast_env.sh] Installing torch family from ${TORCH_INDEX_URL}"
  python -m pip install --index-url "${TORCH_INDEX_URL}" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
else
  echo "[install_vast_env.sh] Installing torch family from the default index"
  python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
fi

echo "[install_vast_env.sh] Installing GPU-sensitive packages"
python -m pip install vllm==0.8.3 deepspeed==0.16.4 bitsandbytes==0.45.5 liger-kernel==0.5.8 llmcompressor==0.4.1

if [[ "${SKIP_FLASH_ATTN}" == "true" ]]; then
  echo "[install_vast_env.sh] Skipping flash-attn build by request"
else
  echo "[install_vast_env.sh] Building flash-attn wheel with MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS}, arch=${FLASH_ATTN_CUDA_ARCHS}"
  python -m pip wheel \
    --no-deps \
    --no-build-isolation \
    --wheel-dir "${FLASH_WHEEL_DIR}" \
    flash-attn==2.7.4.post1
  echo "[install_vast_env.sh] Installing flash-attn from ${FLASH_WHEEL_DIR}"
  python -m pip install --no-deps --no-index --find-links "${FLASH_WHEEL_DIR}" flash-attn==2.7.4.post1
fi

echo "[install_vast_env.sh] Installing base Python packages"
python -m pip install -r requirements.txt

echo "[install_vast_env.sh] Verifying core imports"
python - <<'PY'
import torch
import vllm

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("vllm", vllm.__version__)
PY
