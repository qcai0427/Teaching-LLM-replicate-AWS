#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install_vast_env.sh [--torch-index-url URL] [--with-ninja-upgrade]

Examples:
  ./scripts/install_vast_env.sh
  ./scripts/install_vast_env.sh --torch-index-url https://download.pytorch.org/whl/cu124
EOF
}

TORCH_INDEX_URL=""
WITH_NINJA_UPGRADE=false

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

python -m pip install -U pip setuptools wheel packaging
if [[ "${WITH_NINJA_UPGRADE}" == "true" ]]; then
  python -m pip install -U ninja
else
  python -m pip install ninja
fi

if [[ -n "${TORCH_INDEX_URL}" ]]; then
  echo "[install_vast_env.sh] Installing torch from ${TORCH_INDEX_URL}"
  python -m pip install --index-url "${TORCH_INDEX_URL}" torch
else
  echo "[install_vast_env.sh] Installing torch from the default index"
  python -m pip install torch
fi

echo "[install_vast_env.sh] Installing GPU-sensitive packages"
python -m pip install -r requirements.gpu.txt

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
