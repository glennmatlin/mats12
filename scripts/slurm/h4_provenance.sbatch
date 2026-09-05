#!/usr/bin/env bash
#SBATCH --job-name=mats12-h4
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/h4-%A_%a.out
# ICE rules: no -A, no --partition. H4 provenance array: each task scores
# SHARDS_PER_TASK canonical HF text shards with the committed direction set
# (persona axis + 20 unit randoms). NEEDS network: HF_HUB_OFFLINE stays unset.
# Per-shard outputs are resumable (existing scores_shard_XXXX.npz skipped).
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -f "$HOME/.hf_token" ]]; then export HF_TOKEN="$(cat "$HOME/.hf_token")"; fi
source ~/scratch/mats12-venv/bin/activate

SHARDS_PER_TASK="${SHARDS_PER_TASK:-16}"
START=$(( SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK ))

python scripts/18_h4_provenance.py \
    --directions artifacts/h4/h4_directions.npz \
    --manifest "${MANIFEST:-/storage/ice-shared/cs7634/staff/TDA/social-data-attribution/manifests/working_sample_manifest.parquet}" \
    --model-path "$STORE/models/olmo3-base" \
    --out-dir "$STORE/h4/scores" \
    --shard-start "$START" \
    --shard-count "$SHARDS_PER_TASK" \
    --token-budget "${TOKEN_BUDGET:-49152}"
