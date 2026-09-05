"""Attribution glue to the existing TrackStar/Bergson pipeline (sprint seam).

Inputs that already exist (read-only; social-data-attribution repo):
- 1.3TB gradient index over 5.68M Dolma3 docs, 316 shards (DO NOT rebuild)
- per-benchmark score matrices + 576-bin manifests
- Bergson scoring pipeline: scripts/attribution/trackstar_phases.sh (Phase 2)

New work this project adds:
1. Behavior queries: query gradients from contrast generations (+5 vs -5 rubric
   scores, the post's probe recipe) for each behavior, on the SFT-side model.
2. Direction-level scoring (Concept Influence, arXiv:2602.14869): score docs
   against a direction; probe first-order approximation = projection difference.
3. Persona-side queries: same machinery with the persona subspace directions,
   run against the PRETRAINING index (Core 2, the only-Glenn section).

Method caution (arXiv:2510.12071): classical influence estimates are unstable
at unconverged intermediate checkpoints — attribute against the released base,
not a mid-train snapshot, unless a robustness check says otherwise.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BehaviorQuerySet:
    behavior_key: str
    direction: np.ndarray  # unit vector in activation space, layer-matched to the persona basis
    layer: int
    n_contrast_pairs: int
    provenance: str


@dataclass(frozen=True)
class ProbeDirection:
    """The primary scorer (PREREGISTRATION Setting, named 2026-08-16):
    s(x) = w_hat . (h(x) - mu), w_hat the unit-normalized L2-logistic
    direction, mu the mean of ALL training activations (pos + neg)."""

    behavior_key: str
    w_hat: np.ndarray
    mu: np.ndarray
    layer: int
    train_accuracy: float
    n_pos: int
    n_neg: int

    def save(self, path: Path) -> None:
        np.savez(
            path,
            behavior_key=self.behavior_key,
            w_hat=self.w_hat.astype(np.float32),
            mu=self.mu.astype(np.float32),
            layer=self.layer,
            train_accuracy=self.train_accuracy,
            n_pos=self.n_pos,
            n_neg=self.n_neg,
        )

    @staticmethod
    def load(path: Path) -> "ProbeDirection":
        d = np.load(path)
        return ProbeDirection(
            behavior_key=str(d["behavior_key"]),
            w_hat=d["w_hat"].astype(np.float64),
            mu=d["mu"].astype(np.float64),
            layer=int(d["layer"]),
            train_accuracy=float(d["train_accuracy"]),
            n_pos=int(d["n_pos"]),
            n_neg=int(d["n_neg"]),
        )


def fit_probe(
    behavior_key: str,
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    layer: int,
    c: float = 1.0,
    max_iter: int = 1000,
) -> ProbeDirection:
    """L2-logistic probe on mean-pooled activations (Tier-2 pins, ledger
    2026-08-16: sklearn LogisticRegression, C=1.0, lbfgs, max_iter=1000,
    fit_intercept=True; deterministic for fixed inputs)."""
    from sklearn.linear_model import LogisticRegression

    x = np.vstack([pos_acts, neg_acts]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
    clf = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=max_iter)
    clf.fit(x, y)
    w = clf.coef_[0]
    return ProbeDirection(
        behavior_key=behavior_key,
        w_hat=w / np.linalg.norm(w),
        mu=x.mean(axis=0),
        layer=layer,
        train_accuracy=float(clf.score(x, y)),
        n_pos=len(pos_acts),
        n_neg=len(neg_acts),
    )


def probe_scores(probe: ProbeDirection, doc_acts: np.ndarray) -> np.ndarray:
    """Primary per-document scores: s(x) = w_hat . (h(x) - mu)."""
    return (doc_acts.astype(np.float64) - probe.mu) @ probe.w_hat


def activation_shift_scores(
    base_acts: np.ndarray,
    sft_acts: np.ndarray,
    direction: np.ndarray,
    cosine: bool = False,
) -> np.ndarray:
    """Fallback scorer: s(x) = v_hat . (h_sft(x) - h_base(x)), the appendix
    formula as written (unnormalized document shift). cosine=True is the
    logged robustness variant (divergence-ledger amendment 2026-08-16);
    zero-shift rows score 0 under it."""
    shift = sft_acts.astype(np.float64) - base_acts.astype(np.float64)
    u = direction / np.linalg.norm(direction)
    raw = shift @ u
    if not cosine:
        return raw
    norms = np.linalg.norm(shift, axis=1)
    return np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0)


def score_pretrain_bins(
    direction: np.ndarray, trackstar_scores_dir: Path, bin_manifest: Path, out_path: Path
) -> None:
    """TODO(sprint): aggregate direction-scored influence over the 576
    WebOrganizer bins; reuse the zscored_bin_scores conventions from the
    social-data-attribution repo."""
    raise NotImplementedError
