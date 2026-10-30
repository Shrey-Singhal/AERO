#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT_DIR/venv_aws}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_WHEEL="${CUDA_WHEEL:-cu130}"
REQUIREMENTS="${ROOT_DIR}/requirements.txt"
cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python3 is required to create the AWS venv." >&2
    exit 1
fi

if [[ "$(uname -s)" != Linux ]]; then
    echo "This setup targets Linux AWS GPU servers, not macOS or Windows." >&2
    exit 1
fi

"${PYTHON_BIN}" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 14), "Use Python 3.10-3.13 (3.12 recommended)."'
PYTHON_INCLUDE_DIR="$("${PYTHON_BIN}" -c 'from sysconfig import get_paths; print(get_paths()["include"])')"

required_version() {
    local version
    version="$(awk -F '==' -v package="$1" '$1 == package {print $2; exit}' "${REQUIREMENTS}")"
    if [[ -z "${version}" ]]; then
        echo "Missing exact $1 pin in ${REQUIREMENTS}" >&2
        return 1
    fi
    printf '%s' "${version}"
}

VLLM_VERSION="$(required_version vllm)"
CUDA_PACKAGES=()
for package in torch torchvision torchaudio; do
    version="$(required_version "${package}")"
    CUDA_PACKAGES+=("${package}==${version}+${CUDA_WHEEL}")
done

ensure_system_deps() {
    if command -v gcc >/dev/null 2>&1 && [ -f "${PYTHON_INCLUDE_DIR}/Python.h" ]; then
        return
    fi

    echo "Installing required system packages (gcc, Python headers)..."
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y build-essential python3-dev
    else
        apt-get update
        apt-get install -y build-essential python3-dev
    fi
}

echo "======================================================="
echo "Setting up AWS environment"
echo "Repo: ${ROOT_DIR}"
echo "Venv: ${VENV}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA wheel target: ${CUDA_WHEEL}"
echo "Pinned vLLM version: ${VLLM_VERSION}"
echo "======================================================="

ensure_system_deps

if [ ! -x "${VENV}/bin/python" ]; then
    "${PYTHON_BIN}" -m venv "${VENV}"
fi

"${VENV}/bin/pip" install --upgrade pip setuptools wheel uv

CPU_ARCH="$(uname -m)"
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_WHEEL}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl"
echo "Installing vLLM wheel: ${VLLM_WHEEL_URL}"
# Resolve the full stack together so a later install cannot silently replace it.
"${VENV}/bin/uv" pip install --python "${VENV}/bin/python" \
    -e "${ROOT_DIR}" "${CUDA_PACKAGES[@]}" "${VLLM_WHEEL_URL}" \
    --index-strategy unsafe-best-match \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

"${VENV}/bin/python" -m pip check
"${VENV}/bin/python" - <<'PY'
import sys

import torch
import transformers
import vllm
import vllm._C

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}, vLLM: {vllm.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and GPU allocation.")
for device in range(torch.cuda.device_count()):
    value = torch.ones(2, device=f"cuda:{device}")
    assert (value + 1).sum().item() == 4
    print(f"GPU {device}: {torch.cuda.get_device_name(device)}")

# Import the training stack without loading models or starting an experiment.
import main
PY

"${VENV}/bin/python" -m pip freeze > "${VENV}/requirements-freeze.txt"

echo
echo "AWS venv ready at ${VENV}"
echo "Installed versions saved to: ${VENV}/requirements-freeze.txt"
echo "Activate with: source \"${VENV}/bin/activate\""
echo "Setup complete; no training job was started."
