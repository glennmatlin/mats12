# Sprint run order

Scripts are created as they become real; stubs are not pre-created. Mapping to
PLAN.md steps:

| Step | Entry point | Where |
|---|---|---|
| 0 prep | `00_download_assets.sh` | PACE login node |
| 1 replication gate | `01_build_speedrun_sft.py` -> `slurm/retrain.sbatch` | PACE GPU |
| 2 persona subspace | `mats12.personas.extract_persona_subspace` (script TBD) | PACE GPU |
| 3 behavior queries + scoring | `mats12.attribution` (script TBD) | PACE GPU |
| 4 separability + prediction | `mats12.separability` (local, CPU) | laptop |
| 5 decontaminated refilter + retrains | `slurm/retrain.sbatch` variants | PACE GPU |
| 6 provenance (only-Glenn section) | `mats12.attribution.score_pretrain_bins` | PACE |

Rules: every run writes raw outputs under `results/<step>/<date>/`; every
figure-bearing number must be re-derivable from a file in `results/`.
