"""Probe + activation-shift scorer math (PREREGISTRATION Setting, 2026-08-16)."""

import numpy as np
import pytest

from mats12.attribution import (
    ProbeDirection,
    activation_shift_scores,
    fit_probe,
    probe_scores,
)


def make_clusters(rng, d=64, n=80, sep=3.0):
    true_dir = np.zeros(d)
    true_dir[0] = 1.0
    pos = rng.normal(size=(n, d)) + sep * true_dir
    neg = rng.normal(size=(n, d))
    return true_dir, pos, neg


def test_fit_probe_recovers_planted_direction():
    rng = np.random.default_rng(0)
    true_dir, pos, neg = make_clusters(rng)
    probe = fit_probe("both_sides", pos, neg, layer=22)
    assert probe.train_accuracy > 0.95
    assert abs(np.linalg.norm(probe.w_hat) - 1.0) < 1e-9
    assert probe.w_hat @ true_dir > 0.8  # signed: points toward the +5 class
    np.testing.assert_allclose(probe.mu, np.vstack([pos, neg]).mean(axis=0))


def test_probe_scores_separate_and_center():
    rng = np.random.default_rng(1)
    _, pos, neg = make_clusters(rng)
    probe = fit_probe("both_sides", pos, neg, layer=22)
    s_pos, s_neg = probe_scores(probe, pos), probe_scores(probe, neg)
    assert s_pos.mean() > s_neg.mean()
    # mu-centering: pooled training scores are (near) zero-mean
    assert abs(np.concatenate([s_pos, s_neg]).mean()) < 1e-6


def test_fit_probe_deterministic():
    rng = np.random.default_rng(2)
    _, pos, neg = make_clusters(rng)
    a = fit_probe("x", pos, neg, layer=22)
    b = fit_probe("x", pos, neg, layer=22)
    np.testing.assert_array_equal(a.w_hat, b.w_hat)


def test_activation_shift_formula_vs_cosine_rank_long_docs_differently():
    direction = np.array([1.0, 0.0])
    base = np.zeros((2, 2))
    # doc 0: small shift perfectly aligned; doc 1: large shift partly aligned
    sft = np.array([[1.0, 0.0], [3.0, 4.0]])
    raw = activation_shift_scores(base, sft, direction)
    cos = activation_shift_scores(base, sft, direction, cosine=True)
    assert raw[1] > raw[0]  # unnormalized: the long doc wins
    assert cos[0] > cos[1]  # cosine: the aligned doc wins
    np.testing.assert_allclose(cos[0], 1.0)


def test_activation_shift_zero_shift_scores_zero_under_cosine():
    direction = np.array([1.0, 0.0])
    acts = np.ones((3, 2))
    cos = activation_shift_scores(acts, acts, direction, cosine=True)
    np.testing.assert_array_equal(cos, np.zeros(3))


def test_probe_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    _, pos, neg = make_clusters(rng)
    probe = fit_probe("refusal", pos, neg, layer=22)
    path = tmp_path / "probe_refusal.npz"
    probe.save(path)
    loaded = ProbeDirection.load(path)
    assert loaded.behavior_key == "refusal" and loaded.layer == 22
    np.testing.assert_allclose(loaded.w_hat, probe.w_hat, atol=1e-7)
    assert loaded.train_accuracy == pytest.approx(probe.train_accuracy)
