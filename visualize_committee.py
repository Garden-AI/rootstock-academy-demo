# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8",
#     "ase>=3.22",
#     "numpy",
# ]
# ///
"""Render a summary figure from an academy_mlip_committee.py results dir.

    uv run visualize_committee.py committee_results_12345/

Reads committee_report.json (+ selected_structures.extxyz for the structure
renderings) and writes committee_summary.png/.svg alongside them. Three
panels: the campaign timeline (every scored structure vs round, committee
joins marked, selected structures ringed), the disagreement ranking by
seed/perturbation arm, and the top selected structures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (light mode): one working hue, one highlight hue,
# text/chrome inks. Validated blue/orange pair, CVD dE 24.7.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

ARM_CAPS = {"rattle": 0.35, "strain": 0.08, "vacancy": 3.0, "swap": 0.5}


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def panel_timeline(ax, report: dict, rng: np.random.Generator) -> None:
    history = report["history"]
    selected_ids = {c["id"] for c in report["selected"]}
    rounds = np.array([c["round"] for c in history], dtype=float)
    scores = np.array([c["qbc_force_std"] for c in history])
    # Marker area tracks how hard the arm was pushed (magnitude / its cap),
    # so the curator's escalation is visible as growing points.
    rel_mag = np.array([c["magnitude"] / ARM_CAPS.get(c["transform"], 1.0)
                        for c in history])
    sizes = 18 + 90 * np.clip(rel_mag, 0, 1)
    x = rounds + 1 + rng.uniform(-0.22, 0.22, len(rounds))  # rounds are 0-based

    ax.scatter(x, scores, s=sizes, color=BLUE, alpha=0.75, linewidths=0,
               label="evaluated structure", zorder=3)
    sel = [i for i, c in enumerate(history) if c["id"] in selected_ids]
    ax.scatter(x[sel], scores[sel], s=sizes[sel] + 55, facecolors="none",
               edgecolors=ORANGE, linewidths=1.6,
               label=f"selected top {len(report['selected'])}", zorder=4)

    # Committee joins: quorum members at round 0, stragglers mid-campaign.
    n_rounds_total = report["rounds_completed"]
    for name, joined in report["committee"].items():
        if joined > 0:
            ax.axvline(joined + 0.5, color=BASELINE, linewidth=1.0,
                       linestyle=(0, (4, 3)), zorder=2)
            # Label on the left of the line when it sits at the plot's right
            # edge, so the text never clips.
            at_edge = joined + 0.5 >= n_rounds_total + 0.4
            ax.annotate(f"{name} joined", xy=(joined + 0.5, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(-10 if at_edge else 4, -2),
                        textcoords="offset points",
                        rotation=90, va="top", ha="left",
                        fontsize=7.5, color=INK_2)

    n_rounds = report["rounds_completed"]
    ax.set_xticks(range(1, n_rounds + 1))
    ax.set_xlim(0.5, n_rounds + 0.5)
    ax.set_xlabel("campaign round", fontsize=9, color=INK_2)
    ax.set_ylabel("committee force disagreement\nmax per-atom σ(F)  (eV/Å)",
                  fontsize=9, color=INK_2)
    founding = sorted(n for n, j in report["committee"].items() if j == 0)
    quorum = (", ".join(founding) if len(", ".join(founding)) <= 55
              else f"{len(founding)} members")
    ax.set_title(
        f"Adaptive query-by-committee campaign — {len(history)} structures, "
        f"founding quorum: {quorum}",
        fontsize=10.5, color=INK, loc="left", pad=10)
    legend = ax.legend(loc="upper left", fontsize=8, frameon=False,
                       labelcolor=INK_2, scatterpoints=1)
    for handle in legend.legend_handles:
        handle.set_sizes([45])


MAX_ARM_ROWS = 12


def panel_arms(ax, report: dict) -> None:
    arms = [a for a in report["arms"] if a["n"] > 0]
    arms.sort(key=lambda a: a["mean_qbc"])
    n_total = len(arms)
    arms = arms[-MAX_ARM_ROWS:]
    labels = [f"{a['seed']} · {a['transform']}" for a in arms]
    values = [a["mean_qbc"] for a in arms]
    y = np.arange(len(arms))

    bars = ax.barh(y, values, height=0.62, color=BLUE, zorder=3)
    for bar, arm in zip(bars, arms):
        ax.annotate(f"{arm['mean_qbc']:.3f}  (n={arm['n']})",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, labels=labels, fontsize=8.5, color=INK)
    ax.set_xlabel("mean disagreement (eV/Å)", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(values) * 1.35)
    ax.grid(axis="y", visible=False)
    extra = f"  (top {len(arms)} of {n_total} arms)" if n_total > len(arms) else ""
    ax.set_title(f"Where does the committee disagree?{extra}",
                 fontsize=10.5, color=INK, loc="left", pad=10)


def panel_structures(fig, subspec, results_dir: Path, report: dict) -> None:
    from ase.io import read as ase_read
    from ase.visualize.plot import plot_atoms

    frames = ase_read(results_dir / "selected_structures.extxyz", index=":3")
    axes = subspec.subgridspec(1, len(frames), wspace=0.05)
    for i, atoms in enumerate(frames):
        ax = fig.add_subplot(axes[0, i])
        plot_atoms(atoms, ax, rotation="10x,15y,0z", radii=0.42)
        ax.set_axis_off()
        info = atoms.info
        ax.set_title(
            f"#{i + 1}  {info.get('seed', '?')} / {info.get('transform', '?')}"
            f"\nσ(F) {info.get('qbc_force_std', float('nan')):.2f} eV/Å",
            fontsize=8.5, color=INK_2, pad=4)
        if i == 0:
            ax.text(0.0, -0.06, "top structures to label with DFT "
                    "(element-colored, perturbed cells)",
                    transform=ax.transAxes, fontsize=8, color=MUTED,
                    va="top")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results_dir", type=Path, nargs="?",
                        default=Path("committee_results"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output image basename (default: "
                             "<results_dir>/committee_summary)")
    args = parser.parse_args()

    report = json.loads((args.results_dir / "committee_report.json").read_text())
    if not report.get("history"):
        raise SystemExit("report has no history — rerun the demo (older "
                         "reports predate the history field)")
    out = args.out or args.results_dir / "committee_summary"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })
    fig = plt.figure(figsize=(11, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42,
                            wspace=0.28, left=0.105, right=0.97,
                            top=0.85, bottom=0.07)

    ax_timeline = fig.add_subplot(grid[0, :])
    style_axes(ax_timeline)
    panel_timeline(ax_timeline, report, np.random.default_rng(0))

    ax_arms = fig.add_subplot(grid[1, 0])
    style_axes(ax_arms)
    panel_arms(ax_arms, report)

    panel_structures(fig, grid[1, 1], args.results_dir, report)

    seeds = report.get("seeds", {})
    provenance = ", ".join(
        f"{name} ({info['mp_id']})" for name, info in sorted(seeds.items())
        if isinstance(info, dict) and info.get("mp_id")) or "built-in seeds"
    fig.suptitle("Multi-MLIP committee via Academy agents + Rootstock",
                 fontsize=13, color=INK, x=0.09, y=0.98, ha="left")
    fig.text(0.09, 0.945, f"committee: {', '.join(report['committee'])}",
             fontsize=8.5, color=INK_2)
    fig.text(0.09, 0.92, f"seed structures: {provenance}",
             fontsize=8.5, color=INK_2)

    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"wrote {out}.png and {out}.svg")


if __name__ == "__main__":
    main()
