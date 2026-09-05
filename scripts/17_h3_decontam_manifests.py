"""H3 decontaminated filter manifests (construction logged 2026-08-27).

For each H3 behavior: u_dec = project_out(probe w_hat, frozen base L22 k=4
basis), s_dec(x) = u_dec . (h(x) - mu) over the committed doc activations,
drop the top 25% -> <behavior>_decontam_25.json in the 09 manifest schema.
Provenance records the direction geometry (persona_load before/after, cosine
v vs u_dec) and the drop-set overlap against the undecontaminated probe
manifest - the H3 mechanism says both_sides' decontaminated ranking should
barely move (its direction is already nearly persona-orthogonal).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mats12.attribution import ProbeDirection
from mats12.separability import persona_load, project_out

H3_BEHAVIORS = ("validate_feelings", "both_sides")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-acts", type=Path, required=True)
    parser.add_argument("--probes-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument(
        "--basis", type=Path, default=Path("artifacts/persona_subspace/persona_subspace_base.npz")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/attribution/manifests"))
    parser.add_argument("--fraction", type=float, default=0.25)
    args = parser.parse_args()

    docs = np.load(args.doc_acts)
    ids, acts = [str(x) for x in docs["ids"]], docs["acts"].astype(np.float64)
    basis = np.load(args.basis)["basis"].astype(np.float64)
    k = int(args.fraction * len(ids))
    provenance = {}
    for behavior in H3_BEHAVIORS:
        probe = ProbeDirection.load(args.probes_dir / f"probe_{behavior}.npz")
        v = probe.w_hat.astype(np.float64)
        u_dec = project_out(v, basis)
        s_dec = (acts - probe.mu) @ u_dec
        order = np.argsort(-s_dec)
        drop = [ids[i] for i in order[:k]]
        cell = f"{behavior}_decontam_{int(args.fraction * 100)}"
        (args.out_dir / f"{cell}.json").write_text(
            json.dumps({"cell": cell, "n_dropped": len(drop), "drop_ids": drop}) + "\n"
        )
        undec = set(
            json.loads(
                (args.out_dir / f"{behavior}_probe_{int(args.fraction * 100)}.json").read_text()
            )["drop_ids"]
        )
        provenance[cell] = {
            "persona_load_before": round(persona_load(v, basis), 4),
            "persona_load_after": round(persona_load(u_dec, basis), 4),
            "cos_v_vs_udec": round(float((v / np.linalg.norm(v)) @ u_dec), 4),
            "drop_overlap_vs_undecontaminated": round(len(undec & set(drop)) / k, 4),
            "n_dropped": k,
        }
        print(cell, provenance[cell])
    (args.out_dir / "h3_decontam_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
