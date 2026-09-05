#!/usr/bin/env bash
#SBATCH --job-name=mats12-practice
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/practice-%j.out
# ICE rules: no -A, no --partition (auto-routed). Practice loop rehearsal.
# Slurm opens the output file BEFORE the script runs, so it must point at a
# directory that already exists — never a fresh release checkout.
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
# Home has a 30G quota: keep ALL HF caches on node-local TMPDIR (storage rule).
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ~/scratch/mats12-venv/bin/activate

# Single parquet file: 100 docs never needs the full 156-shard glob.
python scripts/practice_loop.py \
    --model-path "$STORE/models/olmo3-base" \
    --data-glob "$STORE/data/dolci-think-sft/data/train-00000-of-00156.parquet" \
    --out-dir "$STORE/practice/$(date +%Y%m%d)-$SLURM_JOB_ID" \
    --n-docs 100 --max-steps 30
