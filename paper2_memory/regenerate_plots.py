from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import load_study_rows, plot_budget_curves, plot_family_heatmap


DEFAULT_STUDY_ROOT = Path(__file__).resolve().parents[1] / "results" / "paper2" / "studies"


def _remove_old_composites(plots_dir: Path) -> None:
    for name in [
        "logit_budget_curves.png",
        "kl_budget_curves.png",
        "token_budget_curves.png",
    ]:
        target = plots_dir / name
        if target.exists():
            target.unlink()


def regenerate_study(study_dir: Path) -> None:
    rows = load_study_rows(study_dir / "evaluation_rows.csv")
    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _remove_old_composites(plots_dir)
    plot_budget_curves(rows, plots_dir / "logit_budget_curves.png", "logit_l2", "Mean logit L2", "Paper 2: Budget vs Logit Drift")
    plot_budget_curves(rows, plots_dir / "kl_budget_curves.png", "kl", "Mean KL", "Paper 2: Budget vs KL")
    plot_budget_curves(rows, plots_dir / "token_budget_curves.png", "token_fraction", "Actual token fraction", "Paper 2: Budget vs Token Fraction")

    budget_keys = sorted({row["budget_fraction"] for row in rows}, key=float)
    if budget_keys:
        mid_budget = budget_keys[min(len(budget_keys) // 2, len(budget_keys) - 1)]
        plot_family_heatmap(rows, plots_dir / "family_logit_heatmap.png", "logit_l2", mid_budget, "Mean logit L2 by Family")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate split Paper 2 plots from saved study CSVs.")
    parser.add_argument("--study-dir", type=Path, default=None)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.study_dir is not None:
        study_dirs = [args.study_dir]
    else:
        study_dirs = sorted(path for path in args.study_root.iterdir() if path.is_dir())

    for study_dir in study_dirs:
        if not (study_dir / "evaluation_rows.csv").exists():
            continue
        regenerate_study(study_dir)
        print(f"{study_dir}: regenerated Paper 2 plots")


if __name__ == "__main__":
    main()
