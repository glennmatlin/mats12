from collections import Counter

from mats12.sharding import assign_provider, is_overlap, revalidation_subset

TRIO = ["claude-cli", "grok-cli", "codex-cli"]


def test_assignment_deterministic_and_condition_free():
    # The key carries no condition, so base/sft share a judge by construction.
    assert assign_provider("both_sides-3", TRIO) == assign_provider("both_sides-3", TRIO)


def test_assignment_roughly_balanced():
    counts = Counter(assign_provider(f"behavior-{i}", TRIO) for i in range(3000))
    assert set(counts) == set(TRIO)
    assert min(counts.values()) > 800


def test_overlap_fraction_near_target():
    frac = sum(is_overlap(f"q-{i}", 0.12) for i in range(10_000)) / 10_000
    assert 0.09 < frac < 0.15


def test_revalidation_subset_stable_under_input_order():
    keys = [f"q-{i}" for i in range(200)]
    picked = revalidation_subset(keys, 50)
    assert picked == revalidation_subset(list(reversed(keys)), 50)
    assert len(picked) == 50
    assert set(picked) <= set(keys)
