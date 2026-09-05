#!/usr/bin/env bash
#SBATCH --job-name=mats12-harvest
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/harvest-%j.out
# ICE rules: no -A, no --partition. Mean-pooled L22 doc activations for the
# attribution scorers (PREREGISTRATION Setting 2026-08-16).
# MODEL_KIND=base|sft picks the model; SOURCE=split (default, the 25K
# parquet), SOURCE=<path>.parquet for another split (EM mixture, #4), or
# SOURCE=<dir under $STORE> for a gen-jsonl dir (synthetics).
set -euo pipefail

: "${MODEL_KIND:?set MODEL_KIND=base or MODEL_KIND=sft}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/scratch/mats12-venv/bin/activate

SOURCE="${SOURCE:-split}"
ARGS=(--model-path "$STORE/models/olmo3-base" --layer 22)
if [[ "$MODEL_KIND" == "sft" ]]; then
    ARGS+=(--adapter "${ADAPTER:-$STORE/models/speedrun-sft-r64-s20260813/adapter}")
fi
if [[ "$SOURCE" == "split" ]]; then
    ARGS+=(--split "$STORE/data/speedrun_split_s20260813/split.parquet")
    OUT="$STORE/attribution/doc_acts_${MODEL_KIND}_L22.npz"
elif [[ "$SOURCE" == *.parquet ]]; then
    ARGS+=(--split "$SOURCE")
    OUT="$STORE/attribution/$(basename "${SOURCE%.parquet}")_acts_${MODEL_KIND}_L22.npz"
else
    ARGS+=(--gen-jsonl "$SOURCE")
    OUT="$STORE/attribution/$(basename "$SOURCE")_acts_${MODEL_KIND}_L22.npz"
fi

python scripts/06_harvest_doc_activations.py "${ARGS[@]}" --out "$OUT"
