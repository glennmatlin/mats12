import numpy as np
import pytest

from mats12.separability import (
    dprime,
    orthonormal_basis,
    persona_load,
    project_out,
    subspace_separability,
    topk_overlap,
)

RNG = np.random.default_rng(20260813)
DIM = 64


def make_basis(k: int = 3) -> np.ndarray:
    return orthonormal_basis(RNG.normal(size=(k, DIM)))


def test_basis_is_orthonormal():
    basis = make_basis()
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(basis.shape[0]), atol=1e-8)


def test_basis_drops_dependent_rows():
    v = RNG.normal(size=DIM)
    basis = orthonormal_basis(np.stack([v, 2 * v, -0.5 * v]))
    assert basis.shape[0] == 1


def test_in_subspace_direction_has_zero_separability():
    basis = make_basis()
    v = 3.7 * basis[0] + 1.2 * basis[1]
    assert persona_load(v, basis) == pytest.approx(1.0, abs=1e-8)
    assert subspace_separability(v, basis) == pytest.approx(0.0, abs=1e-4)


def test_orthogonal_direction_has_full_separability():
    basis = make_basis()
    v = RNG.normal(size=DIM)
    v_orth = v - basis.T @ (basis @ v)
    assert subspace_separability(v_orth, basis) == pytest.approx(1.0, abs=1e-8)
    assert persona_load(v_orth, basis) == pytest.approx(0.0, abs=1e-8)


def test_separability_monotone_in_mixture():
    basis = make_basis()
    inside = basis[0]
    outside = project_out(RNG.normal(size=DIM), basis)
    seps = [
        subspace_separability((1 - a) * inside + a * outside, basis)
        for a in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    assert seps == sorted(seps)


def test_project_out_removes_persona_component():
    basis = make_basis()
    v = RNG.normal(size=DIM)
    cleaned = project_out(v, basis)
    assert persona_load(cleaned, basis) == pytest.approx(0.0, abs=1e-8)
    assert np.linalg.norm(cleaned) == pytest.approx(1.0)


def test_project_out_rejects_fully_subsumed_vector():
    basis = make_basis()
    with pytest.raises(ValueError):
        project_out(basis[1], basis)


def test_dprime_orders_separated_vs_overlapping():
    pos = RNG.normal(3.0, 1.0, size=500)
    neg = RNG.normal(0.0, 1.0, size=500)
    overlapping = RNG.normal(0.1, 1.0, size=500)
    assert dprime(pos, neg) > 2.0
    assert abs(dprime(overlapping, neg)) < 0.5


def test_topk_overlap_bounds_and_identity():
    a = RNG.normal(size=1000)
    b = RNG.normal(size=1000)
    assert topk_overlap(a, a, 50) == 1.0
    assert 0.0 <= topk_overlap(a, b, 50) <= 0.2  # random rankings barely overlap
