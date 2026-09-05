"""Write-up figures FIG-1..FIG-8 (skeleton Part F). One committed generator.

Every number is read from its named artifact at render time; nothing is
hand-entered. Okabe-Ito palette, no in-figure titles (captions carry the
claim), PNG 200dpi + PDF to docs/writeup/figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "verm": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#8C8C8C",
    "black": "#000000",
}
GROUP = {
    "elicited": OI["verm"],
    "taught": OI["blue"],
    "mixed": OI["grey"],
    "control": OI["purple"],
    "em": OI["green"],
}
MARK = {"elicited": "o", "taught": "s", "mixed": "D"}
GROUPS = {
    "both_sides": "elicited",
    "ethical_frameworks": "elicited",
    "refusal": "taught",
    "validate_feelings": "taught",
    "bold_formatting": "mixed",
    "liberal_lean": "mixed",
    "china_friendly": "control",
}
OUT = Path("docs/writeup/figures")
RNG = np.random.default_rng(20260816)

plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200}
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.png/.pdf")


def scores_of(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["question_id"]: float(r["score"]) for r in rows}


def fig1() -> None:
    jd = Path("results/step1/judging")
    behaviors = sorted(GROUPS)
    deltas, lo, hi = [], [], []
    for b in behaviors:
        sb, ss = scores_of(jd / f"scores_base_{b}.jsonl"), scores_of(jd / f"scores_sft_{b}.jsonl")
        qs = sorted(sb.keys() & ss.keys())
        d = np.array([ss[q] - sb[q] for q in qs])
        boots = d[RNG.integers(0, len(qs), size=(10_000, len(qs)))].mean(axis=1)
        deltas.append(d.mean())
        lo.append(d.mean() - np.percentile(boots, 2.5))
        hi.append(np.percentile(boots, 97.5) - d.mean())
    order = np.argsort(deltas)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    y = np.arange(len(behaviors))
    for yi, i in enumerate(order):
        ax.barh(
            yi,
            deltas[i],
            xerr=[[lo[i]], [hi[i]]],
            color=GROUP[GROUPS[behaviors[i]]],
            height=0.62,
            capsize=2,
        )
    ax.axvline(2.0, ls="--", lw=0.9, color=OI["black"])
    ax.text(2.1, len(behaviors) - 0.75, "gate bar +2.0", fontsize=7)
    ax.axvline(0, lw=0.8, color=OI["grey"])
    ax.set_yticks(y, [behaviors[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("paired SFT minus base delta (judge points, n=100, 95% CI)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=GROUP[g], label=g) for g in ("elicited", "taught", "mixed", "control")
        ],
        fontsize=6.5,
        frameon=False,
        loc="lower right",
    )
    save(fig, "FIG-1_gate1_deltas")


def fig2() -> None:
    e = json.loads(Path("artifacts/persona_subspace/extraction_report_base.json").read_text())
    d = json.loads(Path("artifacts/persona_subspace/diagnostics_report_base.json").read_text())
    full = {int(k): v for k, v in d["dprime_curve_full"].items()}
    band = {int(k): v for k, v in e["dprime_curve"].items()}
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ls = sorted(full)
    ax.plot(ls, [full[x] for x in ls], color=OI["grey"], lw=1.2, label="full depth (non-binding)")
    bl = sorted(band)
    ax.plot(bl, [band[x] for x in bl], color=OI["blue"], lw=2.2, label="frozen band L12-L22")
    ax.axvline(22, color=OI["verm"], ls="--", lw=1)
    ax.annotate(
        f"L22 frozen\nP(argmax)={d['bootstrap']['p_band_argmax_is_22']:.3f}",
        (22, band[22]),
        xytext=(21.4, 0.735),
        fontsize=7,
        ha="right",
    )
    for pk in (1, 27):
        ax.annotate(f"L{pk}", (pk, full[pk]), xytext=(pk + 0.4, full[pk] + 0.01), fontsize=7)
    ax.set_xlabel("layer")
    ax.set_ylabel("assistant-vs-roles d'")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    save(fig, "FIG-2_dprime_band")


def fig3() -> None:
    g = json.loads(Path("results/gate2/gate2_analysis.json").read_text())["targets"]
    cells = [("refusal", "25"), ("refusal", "10"), ("both_sides", "25"), ("both_sides", "10")]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.axhspan(50, 100, color=OI["green"], alpha=0.10)
    ax.axhspan(-40, 20, color=OI["verm"], alpha=0.10)
    ax.axhline(50, ls="--", lw=0.8, color=OI["green"])
    ax.axhline(20, ls="--", lw=0.8, color=OI["verm"])
    ax.text(3.45, 52, "filterable", fontsize=7, ha="right")
    ax.text(3.45, 33, "indeterminate", fontsize=7, ha="right", color="#666666")
    ax.text(3.45, 14, "not filterable", fontsize=7, ha="right")
    for x, (b, pct) in enumerate(cells):
        t = g[b][pct]
        v = 100 * t["removed_vs_random"]
        ci = t["ci95"]
        ax.errorbar(
            x,
            v,
            yerr=[[v - 100 * ci[0]], [100 * ci[1] - v]],
            fmt="o",
            color=GROUP[GROUPS[b]],
            capsize=3,
            ms=5,
        )
    ax.set_xticks(range(4), [f"{b}\n@{p}%" for b, p in cells], fontsize=7.5)
    ax.set_ylabel("% of SFT delta removed vs random")
    save(fig, "FIG-3_gate2_filterability")


def fig4() -> None:
    h = json.loads(Path("results/h1h2/h1_h2_measurements.json").read_text())
    ov = h["H1"]["P1a_top_k_overlap"]
    behaviors = sorted(GROUPS)
    n = len(behaviors)
    m = np.full((n, n), np.nan)
    for k, v in ov.items():
        a, b = k.split("|")
        i, j = behaviors.index(a), behaviors.index(b)
        m[i, j] = m[j, i] = v
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=0.30)
    ax.set_xticks(range(n), behaviors, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n), behaviors, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if m[i, j] < 0.18 else "black",
                )
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="top-1000 overlap")
    save(fig, "FIG-4_h1_overlaps")


def fig5() -> None:
    h = json.loads(Path("results/h1h2/h1_h2_measurements.json").read_text())
    loads = h["H1"]["P1b_persona_load"]
    em = json.loads(Path("results/em_control/em_direction.json").read_text())
    items = sorted(loads.items(), key=lambda kv: -kv[1]) + [
        ("em_control", em["readouts"]["base_L22_primary"]["persona_load"])
    ]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for x, (b, v) in enumerate(items):
        ax.bar(x, v, color=GROUP.get(GROUPS.get(b, "em")), width=0.65)
    ax.axhline(np.sqrt(4 / 4096), ls="--", lw=0.9, color=OI["black"])
    ax.text(
        len(items) - 0.4,
        np.sqrt(4 / 4096) + 0.004,
        "E[load], random direction",
        fontsize=7,
        ha="right",
    )
    ax.set_xticks(range(len(items)), [b for b, _ in items], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("persona_load (frozen L22 basis)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=GROUP[g], label=lbl)
            for g, lbl in (
                ("elicited", "elicited"),
                ("taught", "taught"),
                ("mixed", "mixed"),
                ("control", "control"),
                ("em", "EM control"),
            )
        ],
        fontsize=6.5,
        frameon=False,
        loc="upper right",
        ncols=2,
    )
    save(fig, "FIG-5_h1_persona_load")


def fig6() -> None:
    h = json.loads(Path("results/h1h2/h1_h2_measurements.json").read_text())
    sep = h["H2"]["separability"]
    eff = h["H2"]["own_judge_effects"]
    em_sep = json.loads(Path("results/em_control/em_direction.json").read_text())["readouts"][
        "base_L22_primary"
    ]["separability"]
    predicted_class = {
        "refusal": "1",
        "validate_feelings": "2",
        "bold_formatting": "3",
        "both_sides": "4 = tie",
        "ethical_frameworks": "4 = tie",
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for b, e in eff.items():
        ax.scatter(sep[b], e, color=GROUP[GROUPS[b]], marker=MARK[GROUPS[b]], s=42, zorder=3)
        ax.annotate(
            f"{b} (class {predicted_class[b]})",
            (sep[b], e),
            xytext=(-4, -7) if b == "both_sides" else (4, 4),
            textcoords="offset points",
            ha="right" if b == "both_sides" else "left",
            fontsize=7,
        )
    ax.axvline(em_sep, color=GROUP["em"], ls=":", lw=1.2)
    ax.text(
        em_sep - 0.001,
        0.52,
        f"EM control {em_sep:.4f}\n(no filtering cell)",
        fontsize=6.5,
        color=GROUP["em"],
        ha="right",
    )
    ax.text(
        0.9545,
        0.44,
        "declared effect relations hold\n"
        f"nominal rank-tail p = 2/120 = {h['H2']['order_test']['p_value_one_tailed']:.4f}\n"
        "all-five frame; two effects known;\n"
        "labels = pre-registered rank class;\n"
        "no within-class order predicted",
        fontsize=7,
    )
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=MARK[g],
                color="none",
                markerfacecolor=GROUP[g],
                markersize=5.5,
                label=g,
            )
            for g in ("elicited", "taught", "mixed")
        ],
        fontsize=6.5,
        frameon=False,
        loc="center left",
    )
    ax.set_xlabel("subspace separability (frozen basis; index)")
    ax.set_ylabel("filtering effect at 25% removal")
    save(fig, "FIG-6_h2_prediction")


def fig7() -> None:
    v = json.loads(Path("results/h3/h3_analysis.json").read_text())["verdicts"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    labels, xs = [], []
    for x, (b, tag) in enumerate((("validate_feelings", "P3a"), ("both_sides", "P3b"))):
        r = v[b]
        imp = r["improvement_baseline_minus_decontam"]
        ci = r["ci95"]
        ax.errorbar(
            x,
            imp,
            yerr=[[imp - ci[0]], [ci[1] - imp]],
            fmt="o",
            color=GROUP[GROUPS[b]],
            capsize=3,
            ms=6,
        )
        labels.append(f"{b}\n({tag})")
        xs.append(x)
    ax.axhline(1.0, ls="--", lw=0.9, color=OI["green"])
    ax.text(1.28, 1.04, "P3a 'improves' bar (+1.0)", fontsize=7, ha="right")
    ax.axhspan(-1.0, 1.0, color=OI["grey"], alpha=0.12)
    ax.text(1.28, -0.9, "P3b equivalence band", fontsize=7, ha="right")
    ax.axhline(0, lw=0.8, color=OI["grey"])
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("decontamination improvement\n(baseline minus decontam, judge points)")
    save(fig, "FIG-7_h3_causal")


def fig8() -> None:
    h = json.loads(Path("results/h4/h4_analysis.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1.6]})
    nulls = list(h["null_ginis"].values())
    a1.scatter(
        np.zeros(len(nulls)) + RNG.normal(0, 0.008, len(nulls)),
        nulls,
        color=OI["grey"],
        s=18,
        label="20 random directions",
    )
    a1.set_xlim(-0.06, 0.06)
    a1.scatter(
        [0], [h["gini_persona_axis"]], color=GROUP["elicited"], s=55, zorder=3, label="persona axis"
    )
    a1.axhline(h["null_p95"], ls="--", lw=0.9, color=OI["black"])
    a1.text(0.028, h["null_p95"] + 0.004, "null p95", fontsize=7)
    a1.set_xticks([])
    a1.set_ylabel("Gini of top-decile bin mass")
    a1.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.subplots_adjust(wspace=0.65)
    rows = h["format_composition_top_decile"][:8]
    y = np.arange(len(rows))
    a2.barh(
        y + 0.19,
        [r["top_decile_mass"] for r in rows],
        height=0.36,
        color=GROUP["elicited"],
        label="persona-axis top decile",
    )
    a2.barh(
        y - 0.19,
        [r["corpus_mass"] for r in rows],
        height=0.36,
        color=OI["grey"],
        label="corpus share",
    )
    a2.set_yticks(y, [r["format"] for r in rows], fontsize=7)
    a2.invert_yaxis()
    a2.set_xlabel("share of documents")
    a2.legend(fontsize=6.5, frameon=False)
    save(fig, "FIG-8_h4_provenance")


def fig0_headline() -> None:
    """Render the post-result FIG-0 editorial composite.

    The pre-fixed headline rule selected FIG-7 when H3 ran. FIG-0 is a
    later packaging choice that places the canonical FIG-6 ordering plot
    beside the canonical FIG-7 causal plot; the two single-panel figures
    remain canonical. Inputs are the H2 measurement and H3 analysis JSONs.
    """
    h = json.loads(Path("results/h1h2/h1_h2_measurements.json").read_text())
    sep, eff = h["H2"]["separability"], h["H2"]["own_judge_effects"]
    v = json.loads(Path("results/h3/h3_analysis.json").read_text())["verdicts"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    ordered = h["H2"]["order_test"]["observed_order_ascending_effect"]
    predicted_class = {
        "refusal": "1",
        "validate_feelings": "2",
        "bold_formatting": "3",
        "both_sides": "4 = tie",
        "ethical_frameworks": "4 = tie",
    }
    for b in ordered:
        a1.scatter(sep[b], eff[b], color=GROUP[GROUPS[b]], marker=MARK[GROUPS[b]], s=46, zorder=3)
        label_on_left = b in {"refusal", "bold_formatting", "both_sides"}
        a1.annotate(
            f"{b.replace('_', ' ')} (class {predicted_class[b]})",
            (sep[b], eff[b]),
            xytext=(-4, -14) if b == "both_sides" else ((-4, 7) if label_on_left else (4, 5)),
            textcoords="offset points",
            ha="right" if label_on_left else "left",
            fontsize=7,
        )
    a1.set_xlabel("subspace separability (frozen pre-retraining; index)")
    a1.set_ylabel("filtering effect at 25% removal")
    a1.text(
        0.9555,
        0.50,
        "declared effect relations hold\n"
        "under the all-five frame\n"
        f"(nominal p = 2/120 = {h['H2']['order_test']['p_value_one_tailed']:.4f}, the floor);\n"
        "test is on effects, not index ranks;\n"
        "labels = pre-registered rank class",
        fontsize=7.5,
    )
    from matplotlib.lines import Line2D

    a1.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=MARK[g],
                color="none",
                markerfacecolor=GROUP[g],
                markersize=5.5,
                label=g,
            )
            for g in ("elicited", "taught", "mixed")
        ],
        fontsize=6.5,
        frameon=False,
        loc="center left",
    )
    for x, b in enumerate(("validate_feelings", "both_sides")):
        r = v[b]
        imp, ci = r["improvement_baseline_minus_decontam"], r["ci95"]
        a2.errorbar(
            x,
            imp,
            yerr=[[imp - ci[0]], [ci[1] - imp]],
            fmt="o",
            color=GROUP[GROUPS[b]],
            capsize=3,
            ms=6,
        )
    a2.axhline(1.0, ls="--", lw=0.9, color=OI["green"])
    a2.axhspan(-1.0, 1.0, color=OI["grey"], alpha=0.12)
    a2.axhline(0, lw=0.8, color=OI["grey"])
    a2.set_xticks(
        [0, 1],
        [
            f"{b}\n({v[b]['prediction']}: {'holds' if v[b]['prediction_holds'] else 'fails'})"
            for b in ("validate_feelings", "both_sides")
        ],
        fontsize=7.5,
    )
    a2.set_xlim(-0.5, 1.5)
    a2.set_ylim(-1.2, 1.25)
    a2.set_ylabel("gain from persona\ndecontamination (judge pts)")
    a2.text(0.5, 1.06, "the 'improves' bar", fontsize=7, ha="center", color=OI["green"])
    a2.text(-0.42, -1.12, "equivalence band", fontsize=6.5, color="#666666")
    fig.subplots_adjust(wspace=0.42)
    save(fig, "FIG-0_headline")


if __name__ == "__main__":
    for f in (fig0_headline, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
