"""Persona-subspace extraction per the frozen spec (docs/persona_subspace_freeze.md).

Recipe fixed by the lit review and frozen 2026-08-14 — do not silently deviate:

1. Contrast construction, not PC1: default-Assistant mean activation minus
   pooled role means (Assistant Axis, arXiv:2601.10387). Base-checkpoint
   prompting follows the EPFL pretraining_persona machinery: no chat template,
   prompt text = instruction + " " + question (their get_text base branch).
2. Subspace, not a single direction: k = 4 = assistant-contrast axis + top-3
   PCs of the centered per-role mean vectors, orthonormalized axis-first.
3. Layer selected from L12-L22 (hidden_states indexing, 0 = embeddings — the
   EPFL convention) maximizing d' between the role and default-assistant
   activation sets, computed only from the contrast activations.

This module holds the pure pieces (prompt construction + numpy math) so they
are unit-testable without a GPU; scripts/03_extract_subspace.py drives the
model passes on PACE.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mats12.separability import dprime, orthonormal_basis

CANDIDATE_LAYERS = list(range(12, 23))  # L12-L22 inclusive, frozen
K_ROLE_PCS = 3  # k = 4 = axis + top-3 role PCs, frozen
PROVENANCE_TEMPLATE = "aa-verbatim/olmo3-{revision}/L{layer}-dprime/k4/seed20260813"


@dataclass(frozen=True)
class PersonaSubspace:
    basis: np.ndarray  # (k, d) orthonormal rows, axis-first
    layer: int
    source: str  # provenance note: prompts + model revision + date
    k: int

    def save(self, path: Path) -> None:
        np.savez(path, basis=self.basis, layer=self.layer, k=self.k, source=self.source)

    @staticmethod
    def load(path: Path) -> "PersonaSubspace":
        z = np.load(path, allow_pickle=False)
        return PersonaSubspace(
            basis=z["basis"], layer=int(z["layer"]), source=str(z["source"]), k=int(z["k"])
        )


@dataclass(frozen=True)
class PromptSample:
    persona: str  # role name, or "default" for the default-assistant side
    kind: str  # "role" | "default"
    question_id: int
    instruction_index: int
    text: str  # EPFL base rendering: instruction + " " + question, no template


def sha256_prefix(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def load_role_inventory(vendor_data_dir: Path) -> dict[str, str]:
    """The frozen 275-role inventory (role_list.json, sha 1e27b846235326bf)."""
    with open(vendor_data_dir / "roles" / "role_list.json") as f:
        return json.load(f)


def load_extraction_questions(vendor_data_dir: Path) -> list[str]:
    """The frozen 240-question set (extraction_questions.jsonl, sha 31650b9a55d6b827)."""
    questions = []
    with open(vendor_data_dir / "extraction_questions.jsonl") as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    return questions


def load_instruction_variants(
    vendor_data_dir: Path, persona: str, model_display_name: str
) -> list[str]:
    """The 'pos' instruction variants for one persona ('default' = default.json)."""
    with open(vendor_data_dir / "roles" / "instructions" / f"{persona}.json") as f:
        data = json.load(f)
    return [
        entry["pos"].replace("{model_name}", model_display_name)
        for entry in data["instruction"]
    ]


def render_base_prompt(instruction: str, question: str) -> str:
    """EPFL base-checkpoint rendering (get_text, no chat template): contents
    joined with a single space; empty instructions (default.json variant 0)
    leave the bare question."""
    return " ".join(part for part in (instruction, question) if part)


def build_persona_prompts(
    vendor_data_dir: Path,
    model_display_name: str,
    limit_roles: int | None = None,
    limit_questions: int | None = None,
) -> list[PromptSample]:
    """One sample per (persona, question); instruction variants rotate
    question_id % n_variants, mirroring the vendored pipeline's rotation."""
    roles = sorted(load_role_inventory(vendor_data_dir))
    if limit_roles is not None:
        roles = roles[:limit_roles]
    questions = load_extraction_questions(vendor_data_dir)
    if limit_questions is not None:
        questions = questions[:limit_questions]

    samples = []
    for persona, kind in [("default", "default")] + [(r, "role") for r in roles]:
        variants = load_instruction_variants(vendor_data_dir, persona, model_display_name)
        for qid, question in enumerate(questions):
            idx = qid % len(variants)
            samples.append(
                PromptSample(
                    persona=persona,
                    kind=kind,
                    question_id=qid,
                    instruction_index=idx,
                    text=render_base_prompt(variants[idx], question),
                )
            )
    return samples


def contrast_axis(default_samples: np.ndarray, role_means: np.ndarray) -> np.ndarray:
    """axis = mean(default-assistant vectors) - mean(per-role mean vectors),
    per the vendored 5_axis.py (roles averaged role-wise first)."""
    return default_samples.mean(axis=0) - role_means.mean(axis=0)


def select_layer_dprime(
    role_samples: np.ndarray,  # (n_role_samples, n_candidate_layers, d)
    default_samples: np.ndarray,  # (n_default_samples, n_candidate_layers, d)
    role_means: np.ndarray,  # (n_roles, n_candidate_layers, d)
    candidate_layers: list[int],
) -> tuple[int, np.ndarray]:
    """Frozen layer rule: at each candidate layer, project both activation sets
    onto the unit contrast axis and take d'(default, role); return the layer
    (hidden_states index) maximizing d', plus the full curve."""
    curve = np.empty(len(candidate_layers))
    for pos in range(len(candidate_layers)):
        axis = contrast_axis(default_samples[:, pos, :], role_means[:, pos, :])
        u = axis / np.linalg.norm(axis)
        curve[pos] = dprime(default_samples[:, pos, :] @ u, role_samples[:, pos, :] @ u)
    best = int(np.argmax(curve))
    return candidate_layers[best], curve


def build_basis(
    axis: np.ndarray, role_means_at_layer: np.ndarray, n_pcs: int = K_ROLE_PCS
) -> np.ndarray:
    """k = 1 + n_pcs orthonormal rows, axis-first: unit contrast axis, then the
    top PCs of the role-centered per-role means (QR keeps row 0 spanning the
    axis direction). Signs are deterministic (2026-08-16, sign-recovery
    addendum in docs/persona_subspace_freeze.md): QR may flip any row, so row
    0 is aligned with the contrast axis and each PC row with its SVD
    right-singular vector; spans and all |projection| readouts unchanged."""
    centered = role_means_at_layer - role_means_at_layer.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    stacked = np.vstack([axis / np.linalg.norm(axis), vt[:n_pcs]])
    basis = orthonormal_basis(stacked)
    if basis.shape[0] != 1 + n_pcs:
        raise ValueError(
            f"rank-deficient persona basis: got {basis.shape[0]} rows, expected {1 + n_pcs}"
        )
    for row, reference in enumerate(stacked):
        if basis[row] @ reference < 0:
            basis[row] = -basis[row]
    return basis


def pc_explained_variance(role_means_at_layer: np.ndarray, n_pcs: int = K_ROLE_PCS) -> list[float]:
    centered = role_means_at_layer - role_means_at_layer.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    ratios = (s**2) / np.sum(s**2)
    return [float(r) for r in ratios[:n_pcs]]
