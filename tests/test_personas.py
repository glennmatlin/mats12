"""Unit tests for the pure pieces of persona-subspace extraction.

Prompt-construction tests run against the real vendored data files (the frozen
checksummed assets); math tests use planted synthetic activations.
"""

from pathlib import Path

import numpy as np
import pytest

from mats12.personas import (
    CANDIDATE_LAYERS,
    PersonaSubspace,
    build_basis,
    build_persona_prompts,
    contrast_axis,
    load_extraction_questions,
    load_role_inventory,
    render_base_prompt,
    select_layer_dprime,
    sha256_prefix,
)

VENDOR_DATA = Path(__file__).resolve().parents[1] / "vendor" / "assistant-axis" / "data"

pytestmark = pytest.mark.skipif(
    not VENDOR_DATA.exists(), reason="vendored assistant-axis data not present"
)


def test_frozen_assets_match_freeze_doc():
    assert sha256_prefix(VENDOR_DATA / "roles" / "role_list.json") == "1e27b846235326bf"
    assert sha256_prefix(VENDOR_DATA / "extraction_questions.jsonl") == "31650b9a55d6b827"
    assert len(load_role_inventory(VENDOR_DATA)) == 275
    assert len(load_extraction_questions(VENDOR_DATA)) == 240


def test_render_base_prompt_is_epfl_join():
    assert render_base_prompt("You are a tutor.", "Why?") == "You are a tutor. Why?"
    assert render_base_prompt("", "Why?") == "Why?"  # default.json variant 0


def test_build_persona_prompts_shape_and_rotation():
    samples = build_persona_prompts(VENDOR_DATA, "Olmo 3", limit_roles=3, limit_questions=7)
    assert len(samples) == 4 * 7  # default + 3 roles
    default = [s for s in samples if s.kind == "default"]
    assert len(default) == 7 and all(s.persona == "default" for s in default)
    # variant rotation: question_id % n_variants
    for s in samples:
        assert s.instruction_index == s.question_id % 5 or s.instruction_index < 5
    # no chat markup, no unfilled placeholder
    for s in samples:
        assert "<|" not in s.text and "{model_name}" not in s.text


def test_select_layer_dprime_finds_planted_layer():
    rng = np.random.default_rng(0)
    n_layers, d, planted = len(CANDIDATE_LAYERS), 32, 4
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    role = rng.normal(scale=1.0, size=(200, n_layers, d))
    default = rng.normal(scale=1.0, size=(40, n_layers, d))
    default[:, planted, :] += 3.0 * v  # separation only at one candidate layer
    role_means = role.reshape(10, 20, n_layers, d).mean(axis=1)
    layer, curve = select_layer_dprime(role, default, role_means, CANDIDATE_LAYERS)
    assert layer == CANDIDATE_LAYERS[planted]
    assert np.argmax(curve) == planted and curve[planted] > 2.0


def test_build_basis_orthonormal_axis_first():
    rng = np.random.default_rng(1)
    d = 32
    axis = rng.normal(size=d)
    role_means = rng.normal(size=(50, d))
    basis = build_basis(axis, role_means)
    assert basis.shape == (4, d)
    np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=1e-8)
    cos = abs(basis[0] @ (axis / np.linalg.norm(axis)))
    assert cos > 1 - 1e-8  # row 0 spans the axis direction


def test_build_basis_signs_deterministic():
    """QR is free to negate rows; the frozen convention (2026-08-16 addendum)
    aligns row 0 WITH the contrast axis and each PC row with its SVD vector,
    so base and sft bases are comparable without post-hoc sign matching."""
    rng = np.random.default_rng(2)
    d = 32
    axis = rng.normal(size=d)
    role_means = rng.normal(size=(50, d))
    basis = build_basis(axis, role_means)
    assert basis[0] @ (axis / np.linalg.norm(axis)) > 1 - 1e-8  # signed, not abs
    centered = role_means - role_means.mean(axis=0, keepdims=True)
    vt = np.linalg.svd(centered, full_matrices=False)[2]
    for row in range(1, 4):
        assert basis[row] @ vt[row - 1] > 0
    basis_neg = build_basis(-axis, role_means)
    assert basis_neg[0] @ basis[0] < -(1 - 1e-8)  # flipped axis flips row 0 only
    np.testing.assert_allclose(basis_neg[1:], basis[1:], atol=1e-8)


def test_contrast_axis_role_wise_mean():
    default = np.ones((5, 3))
    role_means = np.zeros((7, 3))
    np.testing.assert_allclose(contrast_axis(default, role_means), np.ones(3))


def test_persona_subspace_roundtrip(tmp_path):
    basis = np.eye(4, 16)
    ps = PersonaSubspace(basis=basis, layer=17, source="aa-verbatim/test", k=4)
    path = tmp_path / "ps.npz"
    ps.save(path)
    loaded = PersonaSubspace.load(path)
    np.testing.assert_array_equal(loaded.basis, basis)
    assert loaded.layer == 17 and loaded.k == 4 and loaded.source == "aa-verbatim/test"
