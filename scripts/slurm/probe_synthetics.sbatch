#!/usr/bin/env bash
#SBATCH --job-name=mats12-synthetics
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/synthetics-%j.out
# ICE rules: no -A, no --partition. +5/-5 probe synthetics from
# Olmo-3-32B-Think (Tier-2 pins in RECONSTRUCTION 2026-08-16).
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source ~/scratch/mats12-venv/bin/activate

MODEL="$STORE/models/olmo3-32b-think"
test -f "$MODEL/config.json" || { echo "32B model not staged at $MODEL"; exit 1; }
test -f "$MODEL/model.safetensors.index.json" || { echo "32B download incomplete"; exit 1; }

python scripts/05_generate_probe_synthetics.py \
    --model-path "$MODEL" \
    --eval-dir data/evals \
    --out-dir "$STORE/attribution/synthetics" \
    --behaviors "${BEHAVIORS:-}" \
    --n-per-behavior "${N_PER_BEHAVIOR:-100}"
