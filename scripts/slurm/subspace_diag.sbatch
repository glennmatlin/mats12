#!/usr/bin/env bash
#SBATCH --job-name=mats12-subspace-diag
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/subspace-diag-%j.out
# ICE rules: no -A, no --partition. NON-BINDING diagnostics (edit bundle D6,
# 2026-08-16): full-depth d' curve + bootstrap stability beside the frozen
# L22 choice. Reads the completed generation cache; never generates.
# MODEL_KIND=base|sft; SMOKE=1 runs the 6x6 smoke cache with 50 boots.
set -euo pipefail

: "${MODEL_KIND:?set MODEL_KIND=base or MODEL_KIND=sft}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source ~/scratch/mats12-venv/bin/activate

VENDOR_DATA="${VENDOR_DATA:-$STORE/subspace/vendor-data}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    RAW="$STORE/subspace/smoke-raw"
    OUT="$STORE/subspace/smoke-diagnostics"
    BASIS="$STORE/subspace/smoke-artifacts/persona_subspace_${MODEL_KIND}.npz"
    EXTRA=(--limit-roles 6 --limit-questions 6 --n-boot 50)
else
    RAW="$STORE/subspace/raw"
    OUT="$STORE/subspace/diagnostics"
    BASIS="$STORE/subspace/artifacts/persona_subspace_${MODEL_KIND}.npz"
    EXTRA=()
fi

ARGS=(
    --model-path "$STORE/models/olmo3-base"
    --model-kind "$MODEL_KIND"
    --vendor-data "$VENDOR_DATA"
    --raw-dir "$RAW"
    --out-dir "$OUT"
    --frozen-basis "$BASIS"
)
if [[ "$MODEL_KIND" == "sft" ]]; then
    ARGS+=(--adapter "${ADAPTER:-$STORE/models/speedrun-sft-r64-s20260813/adapter}")
fi

python scripts/04_subspace_diagnostics.py "${ARGS[@]}" "${EXTRA[@]}"
