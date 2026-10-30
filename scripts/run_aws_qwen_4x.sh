#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "======================================================="
echo "AWS PULSE-RLHF 4x launch"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${ROOT_DIR}"
echo "======================================================="

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Configuration:"
    nvidia-smi
    echo "-------------------------------------------------------"
fi

if command -v free >/dev/null 2>&1; then
    echo "System Memory:"
    free -h
    echo "======================================================="
fi

VENV="${VENV:-$ROOT_DIR/venv_aws}"
PYTHON_BIN="${PYTHON_BIN:-${VENV}/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Missing python at ${PYTHON_BIN}" >&2
    echo "Run: bash scripts/setup_aws_env.sh" >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set." >&2
    echo "Export your Hugging Face token before launching." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-$HF_TOKEN}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "Python: ${PYTHON_BIN}"
echo "HF cache: ${HF_HOME}"
echo "Datasets cache: ${HF_DATASETS_CACHE}"
echo "Run root: ${RUN_ROOT:-outputs/aws_mixed_4x}"
echo "======================================================="

PYTHON_BIN="${PYTHON_BIN}" bash scripts/launch_aws_qwen_4x.sh
