#!/usr/bin/env bash
#SBATCH --job-name=mats12-train
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/train-%j.out
# ICE rules: no -A, no --partition; H100 pinned via gres. Single-GPU LoRA
# (the post used 8xH200; envelope divergence logged in RECONSTRUCTION.md).
# Step-4 retrains reuse this via FILTER_MANIFEST + OUT_DIR env overrides.
# Both are REQUIRED (no soft defaults): a dropped env var must fail here,
# not silently retrain into the canonical adapter path. First-train values:
# OUT_DIR=$STORE/models/speedrun-sft-r64-s20260813 FILTER_MANIFEST=none.
set -euo pipefail

: "${OUT_DIR:?set OUT_DIR explicitly - the canonical adapter must never be a default}"
: "${FILTER_MANIFEST:?set FILTER_MANIFEST to a drop-list JSON, or 'none' for unfiltered}"
# SPLIT override (2026-08-25, #4): defaults to the canonical speedrun split;
# the EM control trains the same recipe on its own mixture parquet.

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ~/scratch/mats12-venv/bin/activate

python scripts/02_train_speedrun.py \
    --model-path "$STORE/models/olmo3-base" \
    --tokenizer-path "$STORE/models/olmo3-think-sft" \
    --split "${SPLIT:-$STORE/data/speedrun_split_s20260813/split.parquet}" \
    --out-dir "$OUT_DIR" \
    --filter-manifest "$FILTER_MANIFEST" \
    --seed 20260813 ${TRAIN_EXTRA:-}
