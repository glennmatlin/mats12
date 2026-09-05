"""Separability of a behavior's attribution signal from the persona subspace.

Pure-numpy primitives; everything here is unit-tested locally. The sprint wires
these to real query gradients / probe directions and document score matrices.

Pre-registered primary metric: subspace_separability (geometric residual).
Secondary: dprime over document-score distributions (DRA-style discriminability,
Pan et al. 2502.11411), and topk_overlap between behavior rankings.
"""

import numpy as np

Array = np.ndarray


def _unit(v: Array) -> Array:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector has no direction")
    return v / n


def orthonormal_basis(vectors: Array) -> Array:
    """Orthonormal basis (rows) spanning the given row vectors.

    The persona subspace basis comes from persona-contrast vectors (Assistant-Axis
    recipe: contrast vector plus top role-space PCs — never bare PC1 on a base
    model). Rank-deficient inputs are fine; dependent rows are dropped.
    """
    if vectors.ndim != 2:
        raise ValueError("expected a 2-D array of row vectors")
    q, r = np.linalg.qr(vectors.T)
    keep = np.abs(np.diag(r)) > 1e-10
    return q.T[keep]


def persona_load(v: Array, basis: Array) -> float:
    """Fraction of a unit direction's norm lying inside the persona subspace, in [0, 1]."""
    u = _unit(v)
    coeffs = basis @ u
    return float(np.clip(np.linalg.norm(coeffs), 0.0, 1.0))


def subspace_separability(v: Array, basis: Array) -> float:
    """Primary metric: residual norm after projecting out the persona subspace.

    1.0 = fully persona-orthogonal (predict: filterable, e.g. refusal);
    0.0 = fully persona-subsumed (predict: filtering ~ random, e.g. both-sides).
    """
    load = persona_load(v, basis)
    return float(np.sqrt(max(0.0, 1.0 - load * load)))


def project_out(v: Array, basis: Array) -> Array:
    """Persona-decontaminated direction: remove the subspace component, re-normalize.

    Raises if the residual is numerically zero (behavior fully inside the
    subspace) — callers must handle that as "nothing behavior-specific to find",
    not as a direction.
    """
    u = _unit(v)
    residual = u - basis.T @ (basis @ u)
    if np.linalg.norm(residual) < 1e-8:
        raise ValueError("direction lies (numerically) inside the persona subspace")
    return _unit(residual)


def dprime(pos: Array, neg: Array) -> float:
    """Discriminability of two score distributions (DRA-style d')."""
    pooled = 0.5 * (np.var(pos, ddof=1) + np.var(neg, ddof=1))
    if pooled == 0:
        raise ValueError("degenerate distributions: zero pooled variance")
    return float((np.mean(pos) - np.mean(neg)) / np.sqrt(pooled))


def topk_overlap(scores_a: Array, scores_b: Array, k: int) -> float:
    """|top-k(a) ∩ top-k(b)| / k over a shared document axis.

    High overlap between different behaviors' rankings is the collapse signature
    (every method surfacing the same persona-typical documents).
    """
    if scores_a.shape != scores_b.shape:
        raise ValueError("score arrays must align on the document axis")
    if not 0 < k <= scores_a.shape[0]:
        raise ValueError("k out of range")
    top_a = set(np.argpartition(-scores_a, k - 1)[:k].tolist())
    top_b = set(np.argpartition(-scores_b, k - 1)[:k].tolist())
    return len(top_a & top_b) / k
