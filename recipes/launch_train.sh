#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash recipes/launch_train.sh PROFILE EXPERIMENT SEED [main.py options]' \
        'Profiles: llama1b, qwen15b, llama8b, pythia_tldr' \
        'Experiments: aero, sea, copo, online_dpo, variant1, variant2, variant3,' \
        '             variant4, fixed_quantile, audit, copo_rolling' \
        'Fixed-quantile allocation: ORACLE_RATIO=0.46 (default).' \
        'Example: bash recipes/launch_train.sh llama1b aero 42'
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi

profile="$1"
experiment="$2"
seed="$3"
shift 3

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'Invalid seed: %s\n' "$seed" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

args=(
    --dataset HuggingFaceH4/ultrafeedback_binarized
    --train-split train_prefs --eval-split test_prefs
    --eval-reference-key sft-baseline
    --max-iterations 500 --max-eval 1000 --eval-every-n-iters 25
    --lr 5e-7 --lr-warmup-ratio 0.02
    --max-model-data-ratio 0.30 --copo-mode copo_fixed --copo-alpha 0.001
    --no-use-lora --aleatoric-filter --no-audit-synthetic-labels
)

case "$profile" in
    llama1b)
        args+=(
            --model meta-llama/Llama-3.2-1B-Instruct
        )
        ;;
    qwen15b)
        args+=(
            --model Qwen/Qwen2.5-1.5B-Instruct
        )
        ;;
    llama8b)
        args+=(
            --model meta-llama/Meta-Llama-3-8B-Instruct
            --max-iterations 300 --lr 2e-5 --lr-warmup-ratio 0.005 --use-lora
        )
        ;;
    pythia_tldr)
        args+=(
            --model trl-lib/pythia-1b-deduped-tldr-sft
            --dataset lkevinzc/tldr-with-sft-reference
            --train-split train --eval-split test
            --eval-reference-key pythia-1b-reference
        )
        ;;
    *)
        printf 'Unknown profile: %s\n' "$profile" >&2
        usage >&2
        exit 2
        ;;
esac

method="$experiment"
run_name="$experiment"
case "$experiment" in
    aero|sea|copo|online_dpo)
        ;;
    variant1)
        method=aero
        args+=(--no-aleatoric-filter)
        ;;
    variant2)
        method=aero_variant2
        args+=(--aero-variant2-oracle-ratio 0.44)
        ;;
    variant3)
        method=aero_variant3
        ;;
    variant4)
        method=aero_variant4
        args+=(--confidence-threshold 0.85)
        ;;
    fixed_quantile)
        method=aero_fixed_quantile
        ratio="${ORACLE_RATIO:-0.46}"
        args+=(--aero-fixed-quantile-ratio "$ratio")
        run_name="fixed_quantile_q${ratio}"
        ;;
    audit)
        method=aero
        args+=(
            --max-iterations 100 --max-eval 128 --eval-every-n-iters 100
            --audit-synthetic-labels
        )
        ;;
    copo_rolling)
        method=copo
        args+=(--copo-mode copo_rolling)
        ;;
    *)
        printf 'Unknown experiment: %s\n' "$experiment" >&2
        usage >&2
        exit 2
        ;;
esac

command=(
    "${PYTHON_BIN:-python}" -u main.py "${args[@]}"
    --method "$method" --seed "$seed"
    --output-dir "outputs/${profile}/${run_name}_seed${seed}"
    "$@"
)
export PYTHONUNBUFFERED=1
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
