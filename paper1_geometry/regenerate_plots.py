from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .plotting import generate_baseline_plots, generate_study_plots


DEFAULT_STUDY_ROOT = Path(__file__).resolve().parents[1] / "results" / "paper1" / "studies"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _remove_old_composites(plots_dir: Path) -> None:
    for name in [
        "family_correlation_heatmap.png",
        "boundary_eval_heatmap.png",
        "geometry_vs_decoder.png",
        "baseline_eval_heatmap.png",
    ]:
        target = plots_dir / name
        if target.exists():
            target.unlink()


def regenerate_study(study_dir: Path) -> list[str]:
    conversation_rows = _read_csv(study_dir / "conversation_summary.csv")
    baseline_path = study_dir / "baseline_conversation_summary.csv"
    baseline_rows = _read_csv(baseline_path) if baseline_path.exists() else []
    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _remove_old_composites(plots_dir)
    written = generate_study_plots(conversation_rows, plots_dir)
    if baseline_rows:
        written.extend(generate_baseline_plots(baseline_rows, plots_dir))
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate split Paper 1 plots from saved study CSVs.")
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
        if not (study_dir / "conversation_summary.csv").exists():
            continue
        written = regenerate_study(study_dir)
        print(f"{study_dir}: regenerated {len(written)} plot files")


if __name__ == "__main__":
    main()
