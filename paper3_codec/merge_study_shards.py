from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .pairwise_analysis import _format_report as _format_pairwise_report
from .pairwise_analysis import _pairwise_metric_summary, DEFAULT_PAIRS
from .run_paper3 import (
    _aggregate_behavior_rows,
    _aggregate_rows,
    _behavior_improvement_vs_uniform,
    _improvement_vs_uniform,
)
from .study import (
    _behavior_confidence_summary,
    _confidence_summary,
    _format_report,
    _paired_delta_summary,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded Paper 3 study outputs.")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-dirs", required=True, help="Comma-separated shard output directories.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    shard_dirs = [Path(item.strip()) for item in args.shard_dirs.split(",") if item.strip()]
    if not shard_dirs:
        raise RuntimeError("No shard directories provided.")

    combined_rows: list[dict[str, Any]] = []
    combined_behavior_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        summary_path = shard_dir / "study_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing shard study summary: {summary_path}")
        shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        combined_rows.extend(_read_csv(shard_dir / "evaluation_rows.csv"))
        combined_behavior_rows.extend(_read_csv(shard_dir / "behavior_rows.csv"))

    if not combined_rows:
        raise RuntimeError("No evaluation rows found across study shards.")

    first_summary = shard_summaries[0]
    model_keys = sorted({str(row["model_key"]) for row in combined_rows})
    policies = sorted({str(row["policy_name"]) for row in combined_rows})
    budgets = sorted({float(row["budget_fraction"]) for row in combined_rows})

    model_name_by_key: dict[str, str] = {}
    model_meta_by_key: dict[str, dict[str, Any]] = {}
    for shard_summary in shard_summaries:
        for model_key, payload in shard_summary.get("models", {}).items():
            model_name_by_key.setdefault(model_key, str(payload.get("model_name", model_key)))
            model_meta_by_key.setdefault(model_key, payload)

    models_payload: dict[str, Any] = {}
    for model_key in model_keys:
        model_rows = [row for row in combined_rows if str(row["model_key"]) == model_key]
        model_behavior_rows = [row for row in combined_behavior_rows if str(row["model_key"]) == model_key]
        meta = model_meta_by_key.get(model_key, {})
        models_payload[model_key] = {
            "model_name": model_name_by_key.get(model_key, model_key),
            "num_conversations": len({str(row["conversation_id"]) for row in model_rows}),
            "num_evaluations": len(model_rows),
            "num_behavior_evaluations": len(model_behavior_rows),
            "segment_span": meta.get("segment_span"),
            "target_turn_stride": meta.get("target_turn_stride", first_summary.get("target_turn_stride")),
            "max_target_turns": meta.get("max_target_turns", first_summary.get("max_target_turns")),
            "max_turns_per_conversation": meta.get("max_turns_per_conversation", first_summary.get("max_turns_per_conversation")),
            "aggregate": _aggregate_rows(model_rows),
            "behavior_aggregate": _aggregate_behavior_rows(model_behavior_rows),
            "improvement_vs_uniform": _improvement_vs_uniform(model_rows),
            "behavior_improvement_vs_uniform": _behavior_improvement_vs_uniform(model_behavior_rows),
        }

    output_dir = args.output_root / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_keys": model_keys,
        "families": first_summary.get("families"),
        "budgets": budgets,
        "policies": policies,
        "target_turn_stride": first_summary.get("target_turn_stride"),
        "max_target_turns": first_summary.get("max_target_turns"),
        "max_turns_per_conversation": first_summary.get("max_turns_per_conversation"),
        "skip_conversations": 0,
        "models": models_payload,
    }

    confidence_summary = _confidence_summary(combined_rows)
    significance_summary = _paired_delta_summary(combined_rows, "logit_l2")
    behavior_confidence_summary = _behavior_confidence_summary(combined_behavior_rows)
    behavior_significance_summary = _paired_delta_summary(combined_behavior_rows, "answer_avg_neg_logprob")

    _write_csv(output_dir / "evaluation_rows.csv", combined_rows)
    _write_csv(output_dir / "behavior_rows.csv", combined_behavior_rows)
    (output_dir / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "confidence_summary.json").write_text(json.dumps(confidence_summary, indent=2), encoding="utf-8")
    (output_dir / "significance_summary.json").write_text(json.dumps(significance_summary, indent=2), encoding="utf-8")
    (output_dir / "behavior_confidence_summary.json").write_text(json.dumps(behavior_confidence_summary, indent=2), encoding="utf-8")
    (output_dir / "behavior_significance_summary.json").write_text(json.dumps(behavior_significance_summary, indent=2), encoding="utf-8")
    (output_dir / "study_report.md").write_text(
        _format_report(
            summary,
            confidence_summary,
            behavior_confidence_summary,
            significance_summary,
            behavior_significance_summary,
        ),
        encoding="utf-8",
    )

    pairwise_logit = _pairwise_metric_summary(combined_rows, metric_key="logit_l2", policy_pairs=DEFAULT_PAIRS)
    pairwise_behavior = _pairwise_metric_summary(
        combined_behavior_rows,
        metric_key="answer_avg_neg_logprob",
        policy_pairs=DEFAULT_PAIRS,
    )
    (output_dir / "pairwise_summary.json").write_text(
        json.dumps(
            {
                "policy_pairs": [list(item) for item in DEFAULT_PAIRS],
                "logit_pairwise": pairwise_logit,
                "behavior_pairwise": pairwise_behavior,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "pairwise_report.md").write_text(
        _format_pairwise_report(pairwise_logit, pairwise_behavior),
        encoding="utf-8",
    )
    print(f"Wrote merged study outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
