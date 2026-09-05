"""Central configuration: model ids, data ids, paths, seeds.

Anything marked VERIFY must be pinned during off-clock prep (HF hub check, or
Dohun Lee's configs when they arrive) before the replication gate runs.
"""

from dataclasses import dataclass
from pathlib import Path

SEED = 20260813

# --- Models (ids verified on HF 2026-08-13) ---
# The anchor post LoRAs "OLMo 3 7B mid-train" (rank 64, all 32 layers, MLP+attn)
# and separately uses a pre-mid-train (stage-1 only) checkpoint.
OLMO3_BASE = "allenai/Olmo-3-1025-7B"  # released base; branches carry stage checkpoints
# Interpretation (confirm against Dohun's configs when they arrive): the post's
# "mid-train" model = the released base at `main` (post-Dolmino); "pre-train
# only" = last stage1 branch below. stage2-* branches on the hub are mix
# ablations from a 2T checkpoint, not the canonical midtrain path.
OLMO3_MIDTRAIN_REVISION = "main"
OLMO3_PREMID_REVISION = "stage1-step1413814"  # last stage1-only branch
OLMO3_THINK_SFT = "allenai/Olmo-3-7B-Think-SFT"  # verified

# --- Data (id verified on HF 2026-08-13) ---
DOLCI_THINK_SFT = "allenai/Dolci-Think-SFT-7B"
MAX_DOC_TOKENS = 8192  # post filters out longer examples
SPEEDRUN_SPLIT_SIZE = 25_000  # 5 stratified splits x 25K = 125K per the post

# --- LoRA recipe (from the anchor post appendix) ---
LORA_RANK = 64
LORA_TARGET = "all 32 layers, MLP + attention"
# Design update 4 (LoRA-artifact confound), decided 2026-08-13: the robustness
# cell is FULL-FT one epoch of one key condition; rank-16 only as a logged
# compute fallback.
ROBUSTNESS_REGIME = "full-ft-1ep"
ROBUSTNESS_RANK_FALLBACK = 16

# --- Judge ---
JUDGE_MODEL = "claude-sonnet-4-6"  # post used Claude Sonnet 4.6, slop-aware prompt
JUDGE_GENERATIONS_PER_BEHAVIOR = 100

# --- Existing TDA infrastructure (read-only inputs; see social-data-attribution repo) ---
TRACKSTAR_CANONICAL_RUN = "20260326T163642Z_1102443"
SOCIAL_TDA_REPO = Path.home() / "dev" / "social-data-attribution"

# --- Local layout ---
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS = REPO_ROOT / "results"


@dataclass(frozen=True)
class PaceConfig:
    """Where sprint jobs run. Storage rules per AGENTS.md HPC section.

    ICE submission rules (pace-ice skill): no -A flag, never pass
    --partition (auto-routed); pin GPU type via --gres=gpu:H100:1 etc.
    """

    shared_storage: str = (
        "/storage/ice-shared/cs7634/staff/mats12"  # this project (Glenn, 2026-08-13)
    )
    tda_readonly: str = "/storage/ice-shared/cs7634/staff/TDA"  # existing TDA artifacts (inputs)
    gpus_per_node: int = 8
    retrain_walltime: str = "04:00:00"  # post: ~2.6h per retrain+eval on 8xH200
