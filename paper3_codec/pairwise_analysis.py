from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("geometry_keep_compress_drop", "geometry"),
    ("geometry_keep_compress_drop", "geometry_segment_actions"),
    ("geometry_segment_actions", "geometry"),
    ("semantic_keep_compress_drop", "semantic"),
    ("support_aware_semantic_keep_compress_drop", "semantic_keep_compress_drop"),
    ("budget_aware_semantic_keep_compress_drop", "support_aware_semantic_keep_compress_drop"),
    ("budget_aware_semantic_keep_compress_drop", "semantic"),
    ("budget_aware_semantic_keep_compress_drop", "semantic_filtered_geometry_keep_compress_drop"),
    ("semantic_keep_compress_drop", "geometry"),
    ("semantic_keep_compress_drop", "geometry_keep_compress_drop"),
    ("support_aware_geometry_keep_compress_drop", "geometry_keep_compress_drop"),
    ("support_aware_geometry_keep_compress_drop", "geometry"),
    ("semantic_filtered_geometry_keep_compress_drop", "support_aware_geometry_keep_compress_drop"),
    ("semantic_filtered_geometry_keep_compress_drop", "semantic"),
    ("semantic_filtered_geometry_keep_compress_drop", "semantic_keep_compress_drop"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bootstrap_mean_ci(
    values: list[float],
    *,
    rng: np.random.Generator,
    num_bootstrap: int = 2000,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return {"mean": value, "std": 0.0, "ci_low": value, "ci_high": value}
    samples = rng.choice(array, size=(num_bootstrap, array.size), replace=True)
    means = samples.mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def _paired_signflip_test(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    num_samples: int = 4000,
) -> float:
    if deltas.size == 0:
        return 1.0
    observed = float(abs(np.mean(deltas)))
    if observed < 1e-12:
        return 1.0
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(num_samples, deltas.size), replace=True)
    null_means = np.abs((signs * deltas[None, :]).mean(axis=1))
    return float(np.mean(null_means >= observed))


def _pairwise_metric_summary(
    rows: list[dict[str, str]],
    *,
    metric_key: str,
    policy_pairs: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    rng = np.random.default_rng(20260410 if metric_key == "logit_l2" else 20260411)
    summary: dict[str, Any] = {}
    model_keys = sorted({str(row["model_key"]) for row in rows})
    for model_key in model_keys:
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            key_maps = {
                policy_name: {
                    (str(row["conversation_id"]), int(row["target_turn"])): float(row[metric_key])
                    for row in budget_rows
                    if str(row["policy_name"]) == policy_name
                }
                for policy_name in sorted({str(row["policy_name"]) for row in budget_rows})
            }
            pair_payload: dict[str, Any] = {}
            for left, right in policy_pairs:
                if left not in key_maps or right not in key_maps:
                    continue
                keys = sorted(set(key_maps[left]) & set(key_maps[right]))
                deltas = np.asarray([key_maps[left][key] - key_maps[right][key] for key in keys], dtype=np.float64)
                pair_payload[f"{left}__vs__{right}"] = {
                    "num_pairs": int(deltas.size),
                    f"delta_{metric_key}": {
                        **_bootstrap_mean_ci(deltas.tolist(), rng=rng),
                        "p_value": _paired_signflip_test(deltas, rng=rng),
                    },
                }
            budget_payload[f"{budget:.2f}"] = pair_payload
        summary[model_key] = budget_payload
    return summary


def _format_report(
    logit_summary: dict[str, Any],
    behavior_summary: dict[str, Any],
) -> str:
    lines = [
        "# Paper 3 Pairwise Policy Analysis",
        "",
        "Negative deltas mean the left policy is better than the right policy.",
        "",
    ]
    for model_key in sorted(logit_summary):
        lines.append(f"## {model_key}")
        lines.append("")
        for budget_key in sorted(logit_summary[model_key], key=float):
            lines.append(f"- budget {budget_key}:")
            for pair_name, payload in logit_summary[model_key][budget_key].items():
                delta = payload["delta_logit_l2"]
                lines.append(
                    f"  {pair_name}: Δ logit L2 {delta['mean']:.3f} "
                    f"[{delta['ci_low']:.3f}, {delta['ci_high']:.3f}], p={delta['p_value']:.4f}"
                )
            if behavior_summary.get(model_key, {}).get(budget_key):
                lines.append("  behavior:")
                for pair_name, payload in behavior_summary[model_key][budget_key].items():
                    delta = payload["delta_answer_avg_neg_logprob"]
                    lines.append(
                        f"    {pair_name}: Δ answer NLL {delta['mean']:.4f} "
                        f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}], p={delta['p_value']:.4f}"
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute pairwise Paper 3 policy significance summaries.")
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_rows = _read_csv(args.evaluation_csv)
    behavior_rows = _read_csv(args.behavior_csv)
    logit_summary = _pairwise_metric_summary(eval_rows, metric_key="logit_l2", policy_pairs=DEFAULT_PAIRS)
    behavior_summary = _pairwise_metric_summary(
        behavior_rows,
        metric_key="answer_avg_neg_logprob",
        policy_pairs=DEFAULT_PAIRS,
    )
    payload = {
        "policy_pairs": [list(item) for item in DEFAULT_PAIRS],
        "logit_pairwise": logit_summary,
        "behavior_pairwise": behavior_summary,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(_format_report(logit_summary, behavior_summary), encoding="utf-8")
    print(f"Wrote pairwise Paper 3 analysis to {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
