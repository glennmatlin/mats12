"""H1 collapse + H2 separability measurements (ticket #7).

Pre-registered in PREREGISTRATION.md, frozen 2026-08-16 at e3eb75f:

  P1a  pairwise top-1000 document overlap among elicited behaviors exceeds
       overlap between any elicited-taught pair.
  P1b  persona_load of elicited behaviors' attribution directions exceeds
       that of taught behaviors, on the same layer/basis.
  null both statistics are also computed against matched-norm random
       directions (n >= 20); the contrast must exceed the null's 95th
       percentile to count as support.
  P2   subspace_separability rank-orders per-behavior filtering effectiveness
       at 25% removal: refusal > validate_feelings > bold_formatting >
       {both_sides, ethical_frameworks}. Exact one-tailed permutation test of
       the predicted order relations, own-judge behaviors only.

Local CPU only (PLAN Step 4); consumes committed artifacts. Primary basis is
the mid-train-base L22 k=4 frozen subspace, the space the probes were fit in.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mats12.behaviors import BEHAVIORS
from mats12.separability import persona_load, subspace_separability, topk_overlap

# PREREGISTRATION H1 enumerates the elicited group explicitly. china_friendly
# carries Prior.ELICITED in behaviors.py but is the flat control (no SFT delta
# to attribute); the prereg's parenthetical governs, and it is reported
# descriptively instead. liberal_lean left the elicited group in the
# 2026-08-16 MIXED amendment.
ELICITED = ("both_sides", "ethical_frameworks")
TAUGHT = ("refusal", "validate_feelings")
DESCRIPTIVE = ("bold_formatting", "liberal_lean", "china_friendly")

# P2, ascending in filtering effect (lower effect = more of the SFT delta
# removed = more filterable). Mirrors the descending separability prediction.
PREDICTED_LEVELS = [
    ["refusal"],
    ["validate_feelings"],
    ["bold_formatting"],
    ["both_sides", "ethical_frameworks"],
]

CANONICAL_REPORT_NAME = "h1_h2_measurements.json"


def _sha256_hexdigest_from_handle(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_full(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_hexdigest_from_handle(handle)


def sha256(path: Path) -> str:
    return sha256_full(path)[:16]


def expected_sha256_arg(value: str) -> str:
    if len(value) != 64:
        raise argparse.ArgumentTypeError("expected a full 64-character SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a full 64-character SHA-256 hex digest") from exc
    return value.lower()


def producer_code_receipt() -> dict[str, str]:
    path = Path(__file__).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_full(path)}


def _scalar_int(archive: np.lib.npyio.NpzFile, key: str) -> int:
    if key not in archive.files:
        raise ValueError(f"archive is missing required metadata array {key!r}")
    value = archive[key]
    if value.ndim != 0 or not isinstance(value.item(), (int, np.integer)):
        raise ValueError(f"archive metadata {key!r} must be a scalar integer")
    return int(value.item())


def _scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in archive.files:
        raise ValueError(f"doc_acts is missing required metadata array {key!r}")
    value = archive[key]
    if value.ndim != 0:
        raise ValueError(f"doc_acts metadata {key!r} must be a scalar string")
    scalar = value.item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise ValueError(f"doc_acts metadata {key!r} must be a scalar string")
    return scalar


def load_p1a_doc_acts(
    path: Path,
    doc_score_ids: np.ndarray,
    expected_dim: int,
    expected_layer: int | None,
    expected_sha256: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and receipt-bind P1a activations to the scored document axis."""
    resolved_path = path.resolve(strict=True)
    with resolved_path.open("rb") as handle:
        actual_sha256 = _sha256_hexdigest_from_handle(handle)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            missing = {"ids", "acts", "layer", "config"} - set(archive.files)
            if missing:
                raise ValueError(f"doc_acts is missing required arrays: {sorted(missing)}")

            ids = np.array(archive["ids"], copy=True)
            acts = archive["acts"]
            if ids.ndim != 1:
                raise ValueError("doc_acts ids must be a 1-D array")
            if acts.ndim != 2:
                raise ValueError("doc_acts acts must be a 2-D array")
            if acts.shape[0] != ids.shape[0]:
                raise ValueError("doc_acts ids and acts must have the same document count")
            if acts.shape[1] != expected_dim:
                raise ValueError(
                    f"doc_acts activation width {acts.shape[1]} does not match basis width "
                    f"{expected_dim}"
                )
            if doc_score_ids.shape != ids.shape:
                first = "shape"
            elif not np.array_equal(ids, doc_score_ids):
                mismatch = np.flatnonzero(ids != doc_score_ids)
                first = int(mismatch[0]) if len(mismatch) else "unknown"
            else:
                first = None
            if first is not None:
                raise ValueError(
                    "doc_acts ids must equal doc_scores ids in both value and order "
                    f"(first mismatch: {first})"
                )
            if len(np.unique(ids)) != len(ids):
                raise ValueError("doc_acts ids must be unique")

            layer = _scalar_int(archive, "layer")
            if expected_layer is not None and layer != expected_layer:
                raise ValueError(
                    f"doc_acts layer {layer} does not match basis layer {expected_layer}"
                )
            config = _scalar_text(archive, "config")
            acts_copy = acts.astype(np.float32)

        handle.seek(0)
        post_load_sha256 = _sha256_hexdigest_from_handle(handle)

    if actual_sha256 != expected_sha256:
        raise ValueError(
            "doc_acts SHA-256 does not match the required expected digest "
            f"({actual_sha256} != {expected_sha256})"
        )
    if post_load_sha256 != actual_sha256:
        raise ValueError("doc_acts changed while it was being loaded")

    receipt = {
        "status": "consumed",
        "path": str(resolved_path),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected": True,
        "shape": [int(n) for n in acts.shape],
        "dtype": str(acts.dtype),
        "layer": layer,
        "config": config,
        "doc_acts_ids_equal_doc_scores": True,
        "doc_acts_ids_order_equal_doc_scores": True,
        "n_documents": int(ids.shape[0]),
    }
    return acts_copy, receipt


def resolve_output_path(
    out_dir: Path,
    output: Path | None,
    doc_acts: Path | None,
    allow_incomplete_p1a: bool,
) -> tuple[Path, bool]:
    """Prevent an incomplete P1a run from writing the canonical report."""
    repo_canonical = (
        Path(__file__).resolve(strict=True).parent.parent
        / "results"
        / "h1h2"
        / CANONICAL_REPORT_NAME
    ).resolve()
    resolved_output = (output or (out_dir / CANONICAL_REPORT_NAME)).resolve()
    if doc_acts is not None and not doc_acts.is_file():
        raise ValueError(f"--doc-acts does not exist or is not a file: {doc_acts}")
    if doc_acts is None:
        if not allow_incomplete_p1a:
            raise ValueError("--doc-acts is required for the canonical H1/H2 report")
        if output is None:
            raise ValueError(
                "--allow-incomplete-p1a requires an explicit noncanonical --output path"
            )
        if (
            resolved_output == repo_canonical
            or resolved_output.name.casefold() == CANONICAL_REPORT_NAME.casefold()
        ):
            raise ValueError(
                "an incomplete P1a run cannot write a canonical-named report or the repo "
                "canonical report"
            )
        return resolved_output, True
    return resolved_output, False


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(json.dumps(payload, indent=2))
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def margin(within: list[float], cross: list[float]) -> float:
    """The pre-registered 'exceeds any' contrast as one scalar.

    P1a/P1b both read 'group statistic exceeds every cross-group value', so the
    contrast is min(within) - max(cross); positive means the relation holds for
    every pair. Scalarizing this way is a Tier-2 choice (the prereg fixes the
    relation, not its reduction to a number) and is disclosed as such.
    """
    return float(min(within) - max(cross))


def random_directions(rng: np.random.Generator, n: int, dim: int, norm: float) -> np.ndarray:
    """n matched-norm random directions, H4's null construction."""
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * norm


def null_margin_percentile(
    draw: Callable[[], tuple[list[float], list[float]]],
    n_null: int,
    q: float = 95.0,
) -> dict:
    """Null distribution of the margin statistic under random directions.

    Each draw rebuilds the observed statistic's exact group structure from
    random directions, so the null answers "how big a margin does this grouping
    produce when the labels carry no information?".
    """
    draws = np.asarray([margin(*draw()) for _ in range(n_null)])
    return {
        "n_null": int(n_null),
        "mean": float(draws.mean()),
        "p95": float(np.percentile(draws, q)),
        "max": float(draws.max()),
    }


def order_structure(units: list[str], levels: list[list[str]]) -> dict:
    """Combinatorics of the predicted order relations over a fixed unit set.

    Computable before any effect is known, which is the point: the prereg
    requires the attainable p-floor be stated when the tested-unit set is
    fixed, before unblinding.
    """
    present = [[u for u in level if u in units] for level in levels]
    present = [level for level in present if level]
    ordered = [u for level in present for u in level]
    pairs = [
        (u, v)
        for i, lo in enumerate(present)
        for j, hi in enumerate(present)
        if i < j
        for u in lo
        for v in hi
    ]
    n = len(ordered)
    total = math.factorial(n)
    consistent = sum(
        1
        for perm in itertools.permutations(range(n))
        if all(perm[ordered.index(u)] < perm[ordered.index(v)] for u, v in pairs)
    )
    return {
        "tested_units": ordered,
        "n_units": n,
        "n_orderings": total,
        "n_consistent_with_prediction": consistent,
        "p_floor": consistent / total,
        "predicted_relations": [f"{u} < {v}" for u, v in pairs],
        "_pairs": pairs,
    }


def order_test(effects: dict[str, float], levels: list[list[str]]) -> dict:
    """Exact one-tailed permutation test of the predicted order relations."""
    out = order_structure(list(effects), levels)
    pairs = out.pop("_pairs")
    units = out["tested_units"]
    observed_rank = {u: r for r, u in enumerate(sorted(units, key=lambda u: effects[u]))}
    holds = all(observed_rank[u] < observed_rank[v] for u, v in pairs)
    out["observed_order_ascending_effect"] = sorted(units, key=lambda u: effects[u])
    out["observed_satisfies_prediction"] = holds
    out["p_value_one_tailed"] = out["p_floor"] if holds else 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument(
        "--basis", type=Path, default=Path("artifacts/persona_subspace/persona_subspace_base.npz")
    )
    parser.add_argument(
        "--sft-basis",
        type=Path,
        default=Path("artifacts/persona_subspace/persona_subspace_sft.npz"),
    )
    parser.add_argument(
        "--sft-l22-basis", type=Path, help="matched-layer sft recompute, if fetched"
    )
    parser.add_argument(
        "--doc-scores",
        type=Path,
        default=Path("results/attribution/corpus-scores/probe_scores.npz"),
    )
    parser.add_argument("--doc-acts", type=Path, help="doc_acts_base_L22.npz for the P1a null")
    parser.add_argument(
        "--doc-acts-expected-sha256",
        type=expected_sha256_arg,
        help="required full SHA-256 digest for --doc-acts",
    )
    parser.add_argument("--gate2", type=Path, default=Path("results/gate2/gate2_analysis.json"))
    parser.add_argument("--tested-units", required=True, help="comma-separated H2 tested-unit set")
    parser.add_argument("--k", type=int, default=1000, help="top-k for P1a")
    parser.add_argument("--n-null", type=int, default=2000, help="prereg floor is 20")
    parser.add_argument(
        "--n-null-p1a", type=int, default=200, help="corpus null; prereg floor is 20"
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--out-dir", type=Path, default=Path("results/h1h2"))
    parser.add_argument(
        "--output", type=Path, help="report path; defaults to <out-dir>/h1_h2_measurements.json"
    )
    parser.add_argument(
        "--allow-incomplete-p1a",
        action="store_true",
        help="allow a pending P1a null only with an explicit noncanonical --output",
    )
    args = parser.parse_args()
    doc_acts_expected_sha256: str | None = args.doc_acts_expected_sha256

    try:
        output_path, incomplete_p1a = resolve_output_path(
            args.out_dir, args.output, args.doc_acts, args.allow_incomplete_p1a
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.doc_acts is None and doc_acts_expected_sha256 is not None:
        parser.error("--doc-acts-expected-sha256 requires --doc-acts")
    if args.doc_acts is not None and doc_acts_expected_sha256 is None:
        parser.error("--doc-acts-expected-sha256 is required with --doc-acts")

    producer_code = producer_code_receipt()
    rng = np.random.default_rng(args.seed)
    behaviors = sorted(BEHAVIORS)

    with np.load(args.basis, allow_pickle=False) as basis_archive:
        basis = basis_archive["basis"].astype(np.float64)
        expected_layer = (
            _scalar_int(basis_archive, "layer") if "layer" in basis_archive.files else None
        )
    directions = {
        b: np.load(args.probes_dir / f"probe_{b}.npz")["w_hat"].astype(np.float64)
        for b in behaviors
    }
    dim = basis.shape[1]

    # --- P1b: persona_load, primary basis -------------------------------------
    loads = {b: persona_load(directions[b], basis) for b in behaviors}
    separability = {b: subspace_separability(directions[b], basis) for b in behaviors}
    p1b_margin = margin([loads[b] for b in ELICITED], [loads[b] for b in TAUGHT])
    mean_norm = float(np.mean([np.linalg.norm(directions[b]) for b in behaviors]))

    def draw_p1b() -> tuple[list[float], list[float]]:
        v = random_directions(rng, len(ELICITED) + len(TAUGHT), dim, mean_norm)
        loads_null = [persona_load(row, basis) for row in v]
        return loads_null[: len(ELICITED)], loads_null[len(ELICITED) :]

    p1b_null = null_margin_percentile(draw_p1b, args.n_null)

    # --- P1b robustness: sft basis at its own layer, and matched-layer if present
    robustness = {}
    for label, path in (
        ("sft_L12_own_layer", args.sft_basis),
        ("sft_L22_matched", args.sft_l22_basis),
    ):
        if path is None or not path.exists():
            robustness[label] = {"status": "unavailable", "path": str(path)}
            continue
        z = np.load(path)
        b_alt = z["basis"].astype(np.float64)
        alt_loads = {b: persona_load(directions[b], b_alt) for b in behaviors}
        robustness[label] = {
            "status": "computed",
            "layer": int(z["layer"]),
            "persona_load": {b: round(alt_loads[b], 4) for b in behaviors},
            "margin_elicited_minus_taught": round(
                margin([alt_loads[b] for b in ELICITED], [alt_loads[b] for b in TAUGHT]), 4
            ),
        }

    # --- P1a: pairwise top-k overlap ------------------------------------------
    with np.load(args.doc_scores, allow_pickle=False) as docs:
        doc_score_ids = np.array(docs["ids"], copy=True)
        doc_scores = {b: docs[f"scores_{b}"].astype(np.float64) for b in behaviors}
    if doc_score_ids.ndim != 1:
        raise ValueError("doc_scores ids must be a 1-D array")
    if len(np.unique(doc_score_ids)) != len(doc_score_ids):
        raise ValueError("doc_scores ids must be unique")
    for behavior, scores in doc_scores.items():
        if scores.ndim != 1 or scores.shape != doc_score_ids.shape:
            raise ValueError(f"doc_scores for {behavior} must align with its ids")
    overlaps = {
        f"{a}|{b}": topk_overlap(doc_scores[a], doc_scores[b], args.k)
        for a, b in itertools.combinations(behaviors, 2)
    }
    within = [overlaps[f"{a}|{b}"] for a, b in itertools.combinations(sorted(ELICITED), 2)]
    cross = [overlaps["|".join(sorted((e, t)))] for e in ELICITED for t in TAUGHT]
    p1a_margin = margin(within, cross)

    if args.doc_acts is not None:
        assert doc_acts_expected_sha256 is not None
        acts, p1a_receipt = load_p1a_doc_acts(
            args.doc_acts,
            doc_score_ids,
            dim,
            expected_layer,
            doc_acts_expected_sha256,
        )
        n_per = len(ELICITED) + len(TAUGHT)
        dirs = random_directions(rng, args.n_null_p1a * n_per, dim, mean_norm).astype(np.float32)
        null_scores = acts @ dirs.T  # one BLAS pass; per-direction corpus rankings
        draws_p1a = []
        for d in range(args.n_null_p1a):
            block = [null_scores[:, d * n_per + i] for i in range(n_per)]
            el, ta = block[: len(ELICITED)], block[len(ELICITED) :]
            w = [topk_overlap(a, b, args.k) for a, b in itertools.combinations(el, 2)]
            c = [topk_overlap(a, b, args.k) for a in el for b in ta]
            draws_p1a.append(margin(w, c))
        draws_p1a = np.asarray(draws_p1a)
        p1a_p95 = float(np.percentile(draws_p1a, 95.0))
        p1a_null = {
            "n_null": int(args.n_null_p1a),
            "mean": float(draws_p1a.mean()),
            "p95": p1a_p95,
            "max": float(draws_p1a.max()),
        }
        p1a_supported: bool | None = p1a_margin > p1a_p95
    else:
        acts = None
        p1a_null = {
            "status": "pending",
            "reason": "doc_acts was not provided; this is an explicitly incomplete run",
        }
        p1a_receipt = {
            "status": "not_provided",
            "path": None,
            "sha256": None,
            "expected_sha256": None,
            "sha256_matches_expected": None,
            "shape": None,
            "dtype": None,
            "layer": None,
            "config": None,
            "doc_acts_ids_equal_doc_scores": None,
            "doc_acts_ids_order_equal_doc_scores": None,
            "n_documents": None,
        }
        p1a_supported = None

    # --- H2 --------------------------------------------------------------------
    gate2 = json.loads(args.gate2.read_text())
    own_judge = {b: v["25"]["effect_filtered"] for b, v in gate2["targets"].items() if "25" in v}
    provisional = {
        b: BEHAVIORS[b].approx_filtering_effect_25pct
        for b in behaviors
        if b not in own_judge and BEHAVIORS[b].approx_filtering_effect_25pct is not None
    }
    tested = [u.strip() for u in args.tested_units.split(",") if u.strip()]
    missing = [u for u in tested if u not in own_judge]
    h2 = {
        "effect_metric": (
            "(S_filtered - S_base) / (S_SFT - S_base) at 25% removal, own-judge paired means"
        ),
        "own_judge_effects": {b: round(v, 4) for b, v in sorted(own_judge.items())},
        "provisional_post_figure_effects_NOT_POOLED": provisional,
        "separability": {b: round(separability[b], 4) for b in behaviors},
        "separability_order_descending": sorted(behaviors, key=lambda b: -separability[b]),
        "excluded_by_design": {
            "china_friendly": "flat control, no SFT delta to filter",
            "liberal_lean": "unranked since the 2026-08-16 MIXED amendment; descriptive only",
        },
        "tested_unit_set": tested,
        "tested_units_missing_own_judge_effect": missing,
    }
    structure = order_structure(tested, PREDICTED_LEVELS)
    structure.pop("_pairs")
    h2["order_structure_fixed_pre_unblinding"] = structure
    h2["order_test"] = (
        order_test({b: own_judge[b] for b in tested}, PREDICTED_LEVELS)
        if not missing
        else {
            "status": "blocked",
            "reason": f"no own-judge 25% effect for {missing}",
            "p_floor_when_unblocked": structure["p_floor"],
        }
    )

    report = {
        "ticket": "#7 H1 collapse + H2 separability",
        "prereg_status": "frozen 2026-08-16 at e3eb75f; this run consumes it, does not amend it",
        "git_sha": git_sha(),
        "producer_code": producer_code,
        "seed": args.seed,
        "k": args.k,
        "inputs": {
            "basis": {"path": str(args.basis), "sha256_16": sha256(args.basis)},
            "doc_scores": {"path": str(args.doc_scores), "sha256_16": sha256(args.doc_scores)},
            "doc_acts": p1a_receipt,
            "gate2": {"path": str(args.gate2), "sha256_16": sha256(args.gate2)},
        },
        "groups": {
            "elicited": list(ELICITED),
            "taught": list(TAUGHT),
            "descriptive": list(DESCRIPTIVE),
        },
        "H1": {
            "P1b_persona_load": {b: round(loads[b], 4) for b in behaviors},
            "P1b_margin_elicited_minus_taught": round(p1b_margin, 4),
            "P1b_null": p1b_null,
            "P1b_supported": bool(p1b_margin > p1b_null["p95"]),
            "P1a_top_k_overlap": {k: round(v, 4) for k, v in sorted(overlaps.items())},
            "P1a_within_elicited": [round(v, 4) for v in within],
            "P1a_elicited_taught": [round(v, 4) for v in cross],
            "P1a_margin": round(p1a_margin, 4),
            "P1a_null": p1a_null,
            "P1a_supported": p1a_supported,
            "robustness_persona_load": robustness,
        },
        "H2": h2,
    }
    report["output_mode"] = "incomplete_p1a" if incomplete_p1a else "complete"
    out = output_path
    write_json_atomic(out, report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
