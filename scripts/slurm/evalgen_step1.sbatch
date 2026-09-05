#!/usr/bin/env bash
#SBATCH --job-name=mats12-evalgen1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/evalgen1-%j.out
# ICE rules: no -A, no --partition. MODEL_KIND=base|sft selects the condition:
#   base: raw prompt, 512 new tokens (frozen Tier-2 settings)
#   sft:  adapter + think-SFT chat template, 2048 new tokens (ledger 2026-08-14)
# PREFILL="Okay," runs the base-condition elicitation probe (ledger 2026-08-16);
# output dir and symlink gain a -prefill suffix so base pairing stays untouched.
set -euo pipefail

: "${MODEL_KIND:?set MODEL_KIND=base or MODEL_KIND=sft}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ~/scratch/mats12-venv/bin/activate

OUT="$STORE/step1/evalgen-$MODEL_KIND${PREFILL:+-prefill}-$SLURM_JOB_ID"

if [[ "$MODEL_KIND" == "base" ]]; then
    python scripts/gen_eval_answers.py \
        --model-path "$STORE/models/olmo3-base" \
        --eval-dir data/evals \
        --out-dir "$OUT" \
        --n-per-behavior "${N_PER_BEHAVIOR:-100}" \
        --max-new-tokens 512 \
        --behaviors "${BEHAVIORS:-}" \
        --prefill "${PREFILL:-}"
else
    ADAPTER="${ADAPTER:-$STORE/models/speedrun-sft-r64-s20260813/adapter}"
    # ADAPTER=none: full-FT models (R1) load their own weights via MODEL_PATH.
    EXTRA=()
    if [[ "$ADAPTER" != "none" ]]; then EXTRA=(--adapter "$ADAPTER"); fi
    python scripts/gen_eval_answers.py \
        --model-path "${MODEL_PATH:-$STORE/models/olmo3-base}" \
        "${EXTRA[@]}" \
        --tokenizer-path "$STORE/models/olmo3-think-sft" \
        --chat-template \
        --eval-dir data/evals \
        --out-dir "$OUT" \
        --n-per-behavior "${N_PER_BEHAVIOR:-100}" \
        --max-new-tokens 2048 \
        --behaviors "${BEHAVIORS:-}"
fi

ln -sfn "$OUT" "$STORE/step1/latest-$MODEL_KIND${PREFILL:+-prefill}"
