#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
ULTRAFEEDBACK_DATASET="${ULTRAFEEDBACK_DATASET:-HuggingFaceH4/ultrafeedback_binarized}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_ROOT="${RUN_ROOT:-outputs/aws_uf_4x}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

launch_job() {
    local gpu="$1"
    local method="$2"
    local model="$3"
    local dataset="$4"
    local run_name="$5"

    local log_path="${LOG_DIR}/${run_name}.log"
    local -a extra_env=("CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONUNBUFFERED=1")

    # If the model points to a local directory, avoid Hugging Face/cache/network code paths.
    if [[ -d "${model}" ]]; then
        extra_env+=("HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1")
    fi

    echo "[Launch] ${run_name} on GPU ${gpu}"
    nohup env "${extra_env[@]}" \
    "${PYTHON_BIN}" -u main.py \
        --method "${method}" \
        --model "${model}" \
        --dataset "${dataset}" \
        --seed "${SEED}" \
        --run-name "${run_name}" \
        --output-dir "${RUN_ROOT}/${run_name}" \
        >"${log_path}" 2>&1 &

    echo "  pid: $!"
    echo "  log: ${log_path}"
}

launch_job 0 aero "${QWEN_MODEL}" "${ULTRAFEEDBACK_DATASET}" "aero_qwen15b_uf_seed42"
launch_job 1 sea "${QWEN_MODEL}" "${ULTRAFEEDBACK_DATASET}" "sea_qwen15b_uf_seed42"
launch_job 2 aero "${LLAMA_MODEL}" "${ULTRAFEEDBACK_DATASET}" "aero_llama1b_uf_seed42"
launch_job 3 sea "${LLAMA_MODEL}" "${ULTRAFEEDBACK_DATASET}" "sea_llama1b_uf_seed42"

echo
echo "All 4 jobs launched."
echo "Use 'tail -f ${LOG_DIR}/<run>.log' to monitor logs."
echo "Jobs were launched with nohup, so they will keep running after logout."
