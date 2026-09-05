#!/usr/bin/env bash
#SBATCH --job-name=mats12-buildsplit
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/buildsplit-%j.out
# ICE rules: no -A, no --partition (auto-routed to ice-cpu). Output path
# pre-exists (Slurm opens it before the script runs).
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
source ~/scratch/mats12-venv/bin/activate

python scripts/01_build_speedrun_sft.py \
    --data-dir "$STORE/data/dolci-think-sft/data" \
    --tokenizer-path "$STORE/models/olmo3-think-sft" \
    --out-dir "$STORE/data/speedrun_split_s20260813" \
    --split-size 25000 --max-tokens 8192 --seed 20260813
