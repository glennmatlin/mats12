#!/usr/bin/env bash
#SBATCH --job-name=mats12-bergson
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/storage/ice-shared/cs7634/staff/mats12/logs/bergson-%j.out
# Gradient-TDA scores for the trigger-escalated behaviors (both_sides,
# china_friendly) via bergson trackstar. BEHAVIOR is required. SMOKE=1
# limits the corpus to 200 docs and queries to 10 for a pipeline check.
# Bergson pinned at 631d1871bcbdeb98ec7b203684cd16e1bf5888a5
# (glennmatlin/bergson), installed --no-deps into mats12-venv.
set -euo pipefail

: "${BEHAVIOR:?set BEHAVIOR (both_sides | china_friendly)}"
SMOKE="${SMOKE:-0}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

STORE=/storage/ice-shared/cs7634/staff/mats12
export HF_HOME="${TMPDIR:-/tmp}/hf" HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf/datasets"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ~/scratch/mats12-venv/bin/activate

BDATA=$STORE/attribution/bergson
if [[ ! -f $BDATA/corpus.jsonl || ! -f $BDATA/queries_$BEHAVIOR.jsonl ]]; then
  python scripts/11_prep_bergson_data.py \
    --split $STORE/data/speedrun_split_s20260813/split.parquet \
    --gens-dir results/step1/gens-sft \
    --tokenizer $STORE/models/olmo3-think-sft \
    --out-dir $BDATA
fi

DATA_ARGS=(--data.dataset "$BDATA/corpus.jsonl" --data.conversation_column messages --data.truncation)
QUERY_ARGS=(--query.dataset "$BDATA/queries_$BEHAVIOR.jsonl"
  --query.prompt_column prompt --query.completion_column completion --query.truncation)
RUN=$BDATA/trackstar_$BEHAVIOR
if [[ $SMOKE == 1 ]]; then
  head -200 $BDATA/corpus.jsonl > $BDATA/corpus_smoke.jsonl
  head -10 $BDATA/queries_$BEHAVIOR.jsonl > $BDATA/queries_smoke_$BEHAVIOR.jsonl
  DATA_ARGS=(--data.dataset "$BDATA/corpus_smoke.jsonl" --data.conversation_column messages --data.truncation)
  QUERY_ARGS=(--query.dataset "$BDATA/queries_smoke_$BEHAVIOR.jsonl"
    --query.prompt_column prompt --query.completion_column completion --query.truncation)
  RUN=$BDATA/trackstar_smoke_$BEHAVIOR
fi

python -m bergson trackstar "$RUN" \
  --model $STORE/models/speedrun-sft-r64-s20260813/adapter \
  --tokenizer $STORE/models/olmo3-think-sft \
  --index_cfg.precision bf16 \
  --normalizer adafactor \
  --token_batch_size 4096 \
  --aggregation mean --unit_normalize true \
  --resume true \
  "${DATA_ARGS[@]}" "${QUERY_ARGS[@]}"

echo "BERGSON_TRACKSTAR_DONE $RUN"
