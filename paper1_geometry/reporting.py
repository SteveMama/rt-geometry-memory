from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import aggregate_boundary_summary_rows


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_run_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run_directory(results_dir: Path) -> dict[str, object]:
    summary_path = results_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")

    run_summary = _load_run_summary(summary_path)
    conversations = run_summary.get("conversations", [])
    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in conversations:
        by_family[record["family"]].append(record["summary"])

    family_summary: dict[str, dict[str, float]] = {}
    for family, rows in sorted(by_family.items()):
        boundary_aggregate = aggregate_boundary_summary_rows(rows)
        family_summary[family] = {
            "num_conversations": len(rows),
            "mean_rank95": _mean([row["mean_rank95"] for row in rows]),
            "mean_curvature": _mean([row["mean_curvature"] for row in rows]),
            "mean_turning_angle": _mean([row.get("mean_turning_angle", 0.0) for row in rows]),
            "mean_rank_jump": _mean([row.get("mean_rank_jump", 0.0) for row in rows]),
            "mean_subspace_shift": _mean([row.get("mean_subspace_shift", 0.0) for row in rows]),
            "mean_boundary_score": _mean([row.get("mean_boundary_score", 0.0) for row in rows]),
            "mean_boundary_prominence": _mean([row.get("mean_boundary_prominence", 0.0) for row in rows]),
            "mean_state_geodesic_error": _mean([row["mean_state_geodesic_error"] for row in rows]),
            "mean_logit_l2": _mean([row["mean_logit_l2"] for row in rows]),
            "mean_kl": _mean([row["mean_kl"] for row in rows]),
            "mean_corr_geodesic_vs_logit_l2": _mean([row["corr_geodesic_vs_logit_l2"] for row in rows]),
            "mean_corr_geodesic_vs_kl": _mean([row["corr_geodesic_vs_kl"] for row in rows]),
            "mean_top1_agreement": _mean([row["top1_agreement"] for row in rows]),
            **boundary_aggregate,
        }

    return {
        "model_name": run_summary.get("model_name"),
        "device": run_summary.get("device"),
        "dtype": run_summary.get("dtype"),
        "state_layer": run_summary.get("state_layer"),
        "transformers_version": run_summary.get("transformers_version"),
        "aggregate": run_summary.get("aggregate", {}),
        "family_summary": family_summary,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Paper 1 Run Summary",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Device: `{summary['device']}`",
        f"- Dtype: `{summary['dtype']}`",
        f"- State layer: `{summary['state_layer']}`",
        f"- Transformers: `{summary['transformers_version']}`",
        "",
        "## Aggregate",
        "",
    ]
    aggregate = summary["aggregate"]
    lines.extend(
        [
            f"- Conversations: {aggregate.get('num_conversations', 0)}",
            f"- Mean rank95: {aggregate.get('mean_rank95', 0.0):.3f}",
            f"- Mean curvature: {aggregate.get('mean_curvature', 0.0):.3f}",
            f"- Mean corr(geodesic, logit L2): {aggregate.get('mean_corr_geodesic_vs_logit_l2', 0.0):.3f}",
            f"- Mean rank-jump score: {aggregate.get('mean_rank_jump', 0.0):.3f}",
            f"- Mean subspace-shift score: {aggregate.get('mean_subspace_shift', 0.0):.3f}",
            f"- Macro boundary F1 exact: {aggregate.get('macro_boundary_f1_exact', 0.0):.3f}",
            f"- Micro boundary F1 exact: {aggregate.get('micro_boundary_f1_exact', 0.0):.3f}",
            f"- Macro boundary F1 tol1: {aggregate.get('macro_boundary_f1_tol1', 0.0):.3f}",
            f"- Micro boundary F1 tol1: {aggregate.get('micro_boundary_f1_tol1', 0.0):.3f}",
            f"- Macro boundary F1 tol2: {aggregate.get('macro_boundary_f1_tol2', 0.0):.3f}",
            f"- Micro boundary F1 tol2: {aggregate.get('micro_boundary_f1_tol2', 0.0):.3f}",
            f"- Macro boundary F1 tol3: {aggregate.get('macro_boundary_f1_tol3', 0.0):.3f}",
            f"- Micro boundary F1 tol3: {aggregate.get('micro_boundary_f1_tol3', 0.0):.3f}",
            f"- Mean nearest boundary distance: {aggregate.get('mean_boundary_nearest_distance', 0.0):.3f}",
            f"- Mean WindowDiff: {aggregate.get('mean_boundary_windowdiff', 0.0):.3f}",
            f"- Mean Pk: {aggregate.get('mean_boundary_pk', 0.0):.3f}",
            f"- Mean boundary AUPRC: {aggregate.get('mean_boundary_auprc', 0.0):.3f}",
            f"- Mean candidate boundaries / conversation: {aggregate.get('mean_num_candidate_boundaries', 0.0):.3f}",
            f"- Mean gold boundary density: {aggregate.get('mean_gold_boundary_density', 0.0):.3f}",
            f"- Zero-gold conversations: {aggregate.get('zero_gold_boundary_fraction', 0.0):.3f}",
            f"- Mean logit L2: {aggregate.get('mean_logit_l2', 0.0):.3f}",
            f"- Mean KL: {aggregate.get('mean_kl', 0.0):.3f}",
            f"- Mean corr(geodesic, KL): {aggregate.get('mean_corr_geodesic_vs_kl', 0.0):.3f}",
            "",
            "## By Family",
            "",
        ]
    )

    for family, row in summary["family_summary"].items():
        lines.extend(
            [
                f"### {family}",
                f"- Conversations: {row['num_conversations']}",
                f"- Mean rank95: {row['mean_rank95']:.3f}",
                f"- Mean curvature: {row['mean_curvature']:.3f}",
                f"- Mean turning angle: {row['mean_turning_angle']:.3f}",
                f"- Mean rank-jump score: {row['mean_rank_jump']:.3f}",
                f"- Mean subspace-shift score: {row['mean_subspace_shift']:.3f}",
                f"- Mean boundary score: {row['mean_boundary_score']:.3f}",
                f"- Mean boundary prominence: {row['mean_boundary_prominence']:.3f}",
                f"- Macro boundary F1 exact: {row['macro_boundary_f1_exact']:.3f}",
                f"- Micro boundary F1 exact: {row['micro_boundary_f1_exact']:.3f}",
                f"- Macro boundary F1 tol1: {row['macro_boundary_f1_tol1']:.3f}",
                f"- Micro boundary F1 tol1: {row['micro_boundary_f1_tol1']:.3f}",
                f"- Macro boundary F1 tol2: {row['macro_boundary_f1_tol2']:.3f}",
                f"- Micro boundary F1 tol2: {row['micro_boundary_f1_tol2']:.3f}",
                f"- Macro boundary F1 tol3: {row['macro_boundary_f1_tol3']:.3f}",
                f"- Micro boundary F1 tol3: {row['micro_boundary_f1_tol3']:.3f}",
                f"- Mean nearest boundary distance: {row['mean_boundary_nearest_distance']:.3f}",
                f"- Mean WindowDiff: {row['mean_boundary_windowdiff']:.3f}",
                f"- Mean Pk: {row['mean_boundary_pk']:.3f}",
                f"- Mean boundary AUPRC: {row['mean_boundary_auprc']:.3f}",
                f"- Mean candidate boundaries / conversation: {row['mean_num_candidate_boundaries']:.3f}",
                f"- Mean gold boundary density: {row['mean_gold_boundary_density']:.3f}",
                f"- Mean state geodesic error: {row['mean_state_geodesic_error']:.3f}",
                f"- Mean logit L2: {row['mean_logit_l2']:.3f}",
                f"- Mean KL: {row['mean_kl']:.3f}",
                f"- Mean corr(geodesic, logit L2): {row['mean_corr_geodesic_vs_logit_l2']:.3f}",
                f"- Mean top-1 agreement: {row['mean_top1_agreement']:.3f}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Paper 1 result directories.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = summarize_run_directory(args.results_dir)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = render_markdown(summary)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
