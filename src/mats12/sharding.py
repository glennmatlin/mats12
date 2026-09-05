"""Deterministic judge sharding (PLAN.md judge-scaling protocol).

Assignment is a stable hash of the question key only — never the condition —
so base/SFT/retrain comparisons stay within-judge per question. md5 (not
Python's salted hash()) keeps assignments identical across processes and runs.
"""

import hashlib


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def assign_provider(qkey: str, providers: list[str]) -> str:
    if not providers:
        raise ValueError("providers must be non-empty")
    return providers[_h(f"assign|{qkey}") % len(providers)]


def is_overlap(qkey: str, frac: float) -> bool:
    """Membership in the all-judge overlap subset used to monitor agreement."""
    return _h(f"overlap|{qkey}") % 10_000 < int(frac * 10_000)


def revalidation_subset(keys: list[str], n: int) -> list[str]:
    """Stable pseudo-random subset judged by every provider before sharding."""
    return sorted(keys, key=lambda k: _h(f"reval|{k}"))[:n]
