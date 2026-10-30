#!/usr/bin/env bash
#SBATCH --job-name=aero
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%x-%j.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

# Slurm assigns CUDA_VISIBLE_DEVICES; leave the allocation unchanged.
exec bash recipes/launch_train.sh "$@"
