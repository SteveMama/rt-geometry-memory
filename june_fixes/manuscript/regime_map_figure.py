"""Render the signal-conditioned regime map as a figure (review fix #8).

The regime map is the paper's central conceptual contribution and currently
exists only as prose. This renders it as a benchmark x budget grid colored by
winning signal family, with annotations transcribed from the tracked
checkpoint summaries (papers/rt_project_state_of_play.md and
papers/paper3_public_benchmark_checkpoint.md). Pass --from-json to override
cells from a JSON file instead.

    python june_fixes/manuscript/regime_map_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]

# (benchmark, budget) -> (winning signal family, annotation)
# Transcribed from tracked results; regenerate via --from-json after new runs.
DEFAULT_CELLS: dict[str, dict[str, tuple[str, str]]] = {
    "Hard stress set\n(constraint-critical)": {
        "0.20": ("geometry", "geometry_KCD\n$\\Delta$NLL +0.195*"),
        "0.35": ("geometry", "geometry_KCD\n$\\Delta$NLL +0.424*"),
        "0.50": ("geometry", "geometry_KCD\n$\\Delta$NLL +0.628*"),
    },
    "MSC valid\n(persona continuity)": {
        "0.20": ("semantic", "semantic /\nbudget-aware KCD"),
        "0.35": ("hybrid", "query-cond. geom.\ninside shortlist"),
        "0.50": ("semantic", "budget-aware\nsemantic KCD"),
    },
    "LoCoMo\n(event chains)": {
        "0.20": ("semantic", "semantic-led"),
        "0.35": ("semantic", "semantic-led,\nmixed"),
        "0.50": ("semantic", "mixed"),
    },
    "LongMemEval-S\n(episodic)": {
        "0.20": ("semantic", "semantic"),
        "0.35": ("hybrid", "semantic /\nsegment actions"),
        "0.50": ("geometry", "geometry_KCD"),
    },
}

COLORS = {
    "geometry": "#4C72B0",
    "semantic": "#DD8452",
    "hybrid": "#55A868",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "june_fixes" / "manuscript" / "figures",
    )
    args = parser.parse_args()

    cells = DEFAULT_CELLS
    if args.from_json is not None:
        raw = json.loads(args.from_json.read_text(encoding="utf-8"))
        cells = {
            benchmark: {
                budget: (payload[0], payload[1]) for budget, payload in budgets.items()
            }
            for benchmark, budgets in raw.items()
        }

    benchmarks = list(cells)
    budgets = sorted({budget for budgets in cells.values() for budget in budgets})

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    for row, benchmark in enumerate(benchmarks):
        for col, budget in enumerate(budgets):
            winner, annotation = cells[benchmark].get(budget, ("hybrid", "?"))
            ax.add_patch(
                plt.Rectangle(
                    (col, len(benchmarks) - 1 - row),
                    1,
                    1,
                    facecolor=COLORS[winner],
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.85,
                )
            )
            ax.text(
                col + 0.5,
                len(benchmarks) - 1 - row + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                fontweight="bold",
            )
    ax.set_xlim(0, len(budgets))
    ax.set_ylim(0, len(benchmarks))
    ax.set_xticks([i + 0.5 for i in range(len(budgets))])
    ax.set_xticklabels([f"budget {b}" for b in budgets], fontsize=9)
    ax.set_yticks([len(benchmarks) - 1 - i + 0.5 for i in range(len(benchmarks))])
    ax.set_yticklabels(benchmarks, fontsize=8.5)
    ax.set_title(
        "Signal-conditioned regime map: winning scoring signal by memory regime and budget",
        fontsize=10,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["geometry"], label="geometry-led"),
            Patch(facecolor=COLORS["semantic"], label="semantic-led"),
            Patch(facecolor=COLORS["hybrid"], label="hybrid / mixed"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.01,
        0.01,
        "* conversation-level p < 0.05 vs the semantic signal swap inside the same codec",
        fontsize=6.5,
        color="0.4",
    )
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = args.output_dir / f"regime_map.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[regime_map_figure] wrote {out}")
    acl_figures = REPO_ROOT / "ACL_manuscript" / "figures"
    if acl_figures.exists():
        fig.savefig(acl_figures / "regime_map.png", bbox_inches="tight")
        print(f"[regime_map_figure] wrote {acl_figures / 'regime_map.png'}")


if __name__ == "__main__":
    main()
