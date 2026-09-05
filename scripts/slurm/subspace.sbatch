#!/usr/bin/env bash
#SBATCH --job-name=mats12-subspace
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/subspace-%j.out
# ICE rules: no -A, no --partition. MODEL_KIND=base|sft selects the basis
# (base = canonical frozen basis; sft = pre-registered robustness recompute,
# identical recipe with the speedrun adapter loaded). SMOKE=1 runs a 6-role
# 6-question sanity pass into a separate smoke dir.
set -euo pipefail

: "${MODEL_KIND:?set MODEL_KIND=base or MODEL_KIND=sft}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source ~/scratch/mats12-venv/bin/activate

if [[ "${SMOKE:-0}" == "1" ]]; then
    RAW="$STORE/subspace/smoke-raw"
    OUT="$STORE/subspace/smoke-artifacts"
    LIMITS=(--limit-roles 6 --limit-questions 6)
else
    # RAW_DIR/OUT_DIR overrides (2026-08-28, R1): alternate-regime recomputes
    # must never write into the frozen artifact dirs.
    RAW="${RAW_DIR:-$STORE/subspace/raw}"
    OUT="${OUT_DIR:-$STORE/subspace/artifacts}"
    LIMITS=()
fi

# vendor/ is gitignored (embedded clone), so releases don't carry it; the
# pinned data assets live on the shared store, verified by sha256 in the
# extraction report against docs/persona_subspace_freeze.md.
VENDOR_DATA="${VENDOR_DATA:-$STORE/subspace/vendor-data}"

# MODEL_PATH override (2026-08-28, R1): full-FT models load their own weights.
ARGS=(
    --model-path "${MODEL_PATH:-$STORE/models/olmo3-base}"
    --model-kind "$MODEL_KIND"
    --vendor-data "$VENDOR_DATA"
    --raw-dir "$RAW"
    --out-dir "$OUT"
    --revision-label main
)
if [[ "$MODEL_KIND" == "sft" ]]; then
    ADAPTER="${ADAPTER:-$STORE/models/speedrun-sft-r64-s20260813/adapter}"
    if [[ "$ADAPTER" != "none" ]]; then
        ARGS+=(--adapter "$ADAPTER")
    fi
fi

python scripts/03_extract_subspace.py "${ARGS[@]}" "${LIMITS[@]}"

ln -sfn "$OUT" "$STORE/subspace/latest"
