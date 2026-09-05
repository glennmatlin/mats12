#!/usr/bin/env bash
#SBATCH --job-name=mats12-evalgen
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/evalgen-%j.out
# ICE rules: no -A, no --partition. Output path pre-exists (Slurm opens it
# before the script runs).
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ~/scratch/mats12-venv/bin/activate

python scripts/gen_eval_answers.py \
    --model-path "$STORE/models/olmo3-base" \
    --eval-dir data/evals \
    --out-dir "$STORE/practice/evalgen-$SLURM_JOB_ID" \
    --n-per-behavior "${N_PER_BEHAVIOR:-5}"
