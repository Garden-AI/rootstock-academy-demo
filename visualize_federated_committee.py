# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy",
# ]
# ///
"""Render a summary figure from a federated_mlip_committee.py results dir.

    uv run visualize_federated_committee.py committee_results_delta/ --max-round 8

Reads committee_report.json and writes committee_summary.png/.svg alongside
it. Two panels: the campaign timeline (every scored structure by round, with
the committee's mid-campaign joins marked) and the disagreement ranking by
seed x perturbation arm.

    --max-round N     show rounds 1..N only (the arm panel is recomputed over
                      the shown rounds)
    --rename OLD=NEW  relabel a site in member names for display (repeatable)
    --note TEXT       footnote under the figure
    --out PATH        output basename (default <results_dir>/committee_summary)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ARM_COLORS = {
    "rattle": "#2a78d6",
    "swap": "#eb6834",
    "strain": "#3aa66b",
    "vacancy": "#9a5bd2",
}
ARM_CAPS = {"rattle": 0.35, "strain": 0.08, "swap": 0.5, "vacancy": 0.1}


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


def rename(name: str, renames: dict[str, str]) -> str:
    if "@" not in name:
        return name
    ckpt, site = name.rsplit("@", 1)
    return f"{ckpt}@{renames.get(site, site)}"


def panel_timeline(ax, history: list[dict], joined: dict[str, int], n_rounds: int,
                   rng: np.random.Generator) -> None:
    rounds = np.array([h["round"] for h in history], dtype=float)
    scores = np.array([h["force_std_max"] for h in history])
    arms = [h["arm"] for h in history]
    rel_mag = np.array([h["magnitude"] / ARM_CAPS.get(h["arm"], 1.0) for h in history])
    sizes = 16 + 80 * np.clip(rel_mag, 0, 1)
    x = rounds + rng.uniform(-0.22, 0.22, len(rounds))

    for arm, color in ARM_COLORS.items():
        idx = [i for i, a in enumerate(arms) if a == arm]
        if not idx:
            continue
        ax.scatter(x[idx], scores[idx], s=sizes[idx], color=color, alpha=0.8,
                   linewidths=0, label=arm, zorder=3)

    # Members voting per round, along the top edge.
    per_round = {}
    for h in history:
        per_round[h["round"]] = max(per_round.get(h["round"], 0), len(h["members"]))
    for rd in range(1, n_rounds + 1):
        if rd in per_round:
            ax.annotate(f"×{per_round[rd]}", xy=(rd, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)

    # Joins: group members by the round they first voted in.
    by_round: dict[int, list[str]] = defaultdict(list)
    for name, rd in joined.items():
        if rd > 0 and rd <= n_rounds:
            by_round[rd].append(name)
    for rd, names in sorted(by_round.items()):
        ax.axvline(rd - 0.5, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=2)
        if len(names) <= 2:
            label = ", ".join(sorted(names)) + " joined"
        else:
            sites = defaultdict(int)
            for n in names:
                sites[n.rsplit("@", 1)[-1]] += 1
            where = ", ".join(f"{k} ×{v}" for k, v in sorted(sites.items()))
            label = f"{len(names)} members joined ({where})"
        ax.annotate(label, xy=(rd - 0.5, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", rotation=90,
                    va="bottom", ha="left", fontsize=7.5, color=INK_2)

    ax.set_yscale("log")
    ax.set_ylim(bottom=max(5e-4, scores.min() * 0.5), top=scores.max() * 3)
    ax.set_xticks(range(1, n_rounds + 1))
    ax.set_xlim(0.5, n_rounds + 0.5)
    ax.set_xlabel("campaign round", fontsize=9, color=INK_2)
    ax.set_ylabel("committee disagreement\nmax per-atom σ(F)  (eV/Å)",
                  fontsize=9, color=INK_2)
    ax.set_title(
        f"Adaptive query-by-committee campaign — {len(history)} structures "
        f"over {n_rounds} rounds (×N = members voting)",
        fontsize=10.5, color=INK, loc="left", pad=16)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_2,
                       title="perturbation arm (marker size: how hard it was pushed)",
                       title_fontsize=7.5, scatterpoints=1, ncol=4)
    legend.get_title().set_color(MUTED)
    for handle in legend.legend_handles:
        handle.set_sizes([40])


MAX_ARM_ROWS = 12


def panel_arms(ax, history: list[dict]) -> None:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for h in history:
        groups[(h["seed"], h["arm"])].append(h["force_std_max"])
    rows = sorted(((s, a, float(np.mean(v)), len(v)) for (s, a), v in groups.items()),
                  key=lambda r: r[2])
    n_total = len(rows)
    rows = rows[-MAX_ARM_ROWS:]
    y = np.arange(len(rows))
    colors = [ARM_COLORS.get(a, INK_2) for _, a, _, _ in rows]
    bars = ax.barh(y, [r[2] for r in rows], height=0.62, color=colors, zorder=3)
    for bar, (_, _, mean, n) in zip(bars, rows):
        ax.annotate(f"{mean:.3f}  (n={n})",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK_2)
    ax.set_yticks(y, labels=[f"{s} · {a}" for s, a, _, _ in rows], fontsize=8.5,
                  color=INK)
    ax.set_xlabel("mean disagreement (eV/Å)", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(r[2] for r in rows) * 1.35)
    ax.grid(axis="y", visible=False)
    extra = f"  (top {len(rows)} of {n_total} arms)" if n_total > len(rows) else ""
    ax.set_title(f"Where does the committee disagree?{extra}", fontsize=10.5,
                 color=INK, loc="left", pad=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--note", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads((args.results_dir / "committee_report.json").read_text())
    if not report.get("history"):
        raise SystemExit("report has no history (written by runs from 2026-09-14 on)")
    renames = dict(item.split("=", 1) for item in args.rename)
    joined = {rename(n, renames): r for n, r in report["joined_round"].items()}
    history = report["history"]
    n_rounds = report["rounds_done"]
    if args.max_round:
        history = [h for h in history if h["round"] <= args.max_round]
        n_rounds = min(n_rounds, args.max_round)
    out = args.out or args.results_dir / "committee_summary"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    })
    fig = plt.figure(figsize=(11, 7.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.5,
                            left=0.19, right=0.97, top=0.82, bottom=0.1)
    ax_timeline = fig.add_subplot(grid[0])
    style_axes(ax_timeline)
    panel_timeline(ax_timeline, history, joined, n_rounds, np.random.default_rng(0))
    ax_arms = fig.add_subplot(grid[1])
    style_axes(ax_arms)
    panel_arms(ax_arms, history)

    founding = sorted(n for n, j in joined.items() if j == 0)
    late = defaultdict(list)
    for n, j in joined.items():
        if j:
            late[j].append(n)
    lines = [f"founding quorum: {', '.join(founding)}"]
    for j, names in sorted(late.items()):
        lines.append(f"joined round {j}: {', '.join(sorted(names))}")
    shown = (f"rounds 1–{n_rounds} of {report['rounds_done']} shown"
             if n_rounds < report["rounds_done"] else f"{n_rounds} rounds")
    lines.append(f"{shown}; seeds built in code: Cu, Au, Cu₃Au, CuAu")
    fig.suptitle("Federated MLIP committee via Academy agents, Globus Compute, and Rootstock",
                 fontsize=13, color=INK, x=0.06, y=0.98, ha="left")
    for i, line in enumerate(lines):
        fig.text(0.06, 0.945 - 0.024 * i, line, fontsize=8, color=INK_2)
    if args.note:
        fig.text(0.06, 0.015, args.note, fontsize=7.5, color=MUTED, wrap=True)

    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"wrote {out}.png and {out}.svg")


if __name__ == "__main__":
    main()
