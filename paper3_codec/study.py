from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .run_paper3 import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    _parse_families,
    _parse_float_list,
    _parse_policies,
    run_codec_pilot,
)
from .stats import bootstrap_mean_ci, collapse_rows_by_keys, paired_signflip_test


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _study_summary(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for result in model_results:
        summary = result["summary"]
        payload[summary["model_key"]] = {
            "model_name": summary["model_name"],
            "num_conversations": summary["num_conversations"],
            "num_evaluations": summary["num_evaluations"],
            "num_behavior_evaluations": summary.get("num_behavior_evaluations", 0),
            "segment_span": summary["segment_span"],
            "target_turn_stride": summary.get("target_turn_stride", 1),
            "max_target_turns": summary.get("max_target_turns"),
            "max_turns_per_conversation": summary.get("max_turns_per_conversation"),
            "aggregate": summary["aggregate"],
            "behavior_aggregate": summary.get("behavior_aggregate", {}),
            "improvement_vs_uniform": summary["improvement_vs_uniform"],
            "behavior_improvement_vs_uniform": summary.get("behavior_improvement_vs_uniform", {}),
        }
    return payload


def _conversation_metric_values(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
) -> list[float]:
    collapsed = collapse_rows_by_keys(
        rows,
        metric_keys=[metric_key],
        group_keys=["conversation_id"],
    )
    return [float(row[metric_key]) for row in collapsed]


def _confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20260330)
    summary: dict[str, Any] = {}
    for model_key in sorted({str(row["model_key"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            policy_payload: dict[str, Any] = {}
            for policy_name in sorted({str(row["policy_name"]) for row in budget_rows}):
                policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
                policy_payload[policy_name] = {
                    "logit_l2": {
                        "row_level": bootstrap_mean_ci([float(row["logit_l2"]) for row in policy_rows], rng=rng),
                        "conversation_level": bootstrap_mean_ci(
                            _conversation_metric_values(policy_rows, metric_key="logit_l2"),
                            rng=rng,
                        ),
                    },
                    "kl": {
                        "row_level": bootstrap_mean_ci([float(row["kl"]) for row in policy_rows], rng=rng),
                        "conversation_level": bootstrap_mean_ci(
                            _conversation_metric_values(policy_rows, metric_key="kl"),
                            rng=rng,
                        ),
                    },
                    "token_fraction": {
                        "row_level": bootstrap_mean_ci([float(row["token_fraction"]) for row in policy_rows], rng=rng),
                        "conversation_level": bootstrap_mean_ci(
                            _conversation_metric_values(policy_rows, metric_key="token_fraction"),
                            rng=rng,
                        ),
                    },
                }
            budget_payload[f"{budget:.2f}"] = policy_payload
        summary[model_key] = budget_payload
    return summary


def _behavior_confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20260331)
    summary: dict[str, Any] = {}
    for model_key in sorted({str(row["model_key"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            policy_payload: dict[str, Any] = {}
            for policy_name in sorted({str(row["policy_name"]) for row in budget_rows}):
                policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
                policy_payload[policy_name] = {
                    "answer_avg_neg_logprob": {
                        "row_level": bootstrap_mean_ci(
                            [float(row["answer_avg_neg_logprob"]) for row in policy_rows],
                            rng=rng,
                        ),
                        "conversation_level": bootstrap_mean_ci(
                            _conversation_metric_values(policy_rows, metric_key="answer_avg_neg_logprob"),
                            rng=rng,
                        ),
                    },
                    "answer_avg_neg_logprob_delta": {
                        "row_level": bootstrap_mean_ci(
                            [float(row["answer_avg_neg_logprob_delta"]) for row in policy_rows],
                            rng=rng,
                        ),
                        "conversation_level": bootstrap_mean_ci(
                            _conversation_metric_values(policy_rows, metric_key="answer_avg_neg_logprob_delta"),
                            rng=rng,
                        ),
                    },
                }
            budget_payload[f"{budget:.2f}"] = policy_payload
        summary[model_key] = budget_payload
    return summary


def _paired_delta_summary(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260401 if metric_key == "logit_l2" else 20260402)
    summary: dict[str, Any] = {}
    for model_key in sorted({str(row["model_key"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            uniform_map = {
                (str(row["conversation_id"]), int(row["target_turn"])): row
                for row in budget_rows
                if str(row["policy_name"]) == "uniform"
            }
            comparison_policies = sorted(
                {str(row["policy_name"]) for row in budget_rows if str(row["policy_name"]) != "uniform"}
            )
            comparison_payload: dict[str, Any] = {}
            for policy_name in comparison_policies:
                deltas: list[float] = []
                delta_rows: list[dict[str, Any]] = []
                for row in budget_rows:
                    if str(row["policy_name"]) != policy_name:
                        continue
                    key = (str(row["conversation_id"]), int(row["target_turn"]))
                    baseline = uniform_map.get(key)
                    if baseline is None:
                        continue
                    delta_value = float(row[metric_key]) - float(baseline[metric_key])
                    deltas.append(delta_value)
                    delta_rows.append(
                        {
                            "conversation_id": str(row["conversation_id"]),
                            metric_key: delta_value,
                        }
                    )
                delta_array = np.asarray(deltas, dtype=np.float64)
                conversation_deltas = _conversation_metric_values(delta_rows, metric_key=metric_key)
                conversation_delta_array = np.asarray(conversation_deltas, dtype=np.float64)
                comparison_payload[policy_name] = {
                    "num_pairs": int(delta_array.size),
                    f"delta_{metric_key}": {
                        "row_level": {
                            **bootstrap_mean_ci(deltas, rng=rng),
                            "p_value": paired_signflip_test(delta_array, rng=rng),
                        },
                        "conversation_level": {
                            **bootstrap_mean_ci(conversation_deltas, rng=rng),
                            "p_value": paired_signflip_test(conversation_delta_array, rng=rng),
                        },
                    },
                }
            budget_payload[f"{budget:.2f}"] = comparison_payload
        summary[model_key] = budget_payload
    return summary


def _format_report(
    summary: dict[str, Any],
    confidence_summary: dict[str, Any],
    behavior_confidence_summary: dict[str, Any],
    significance_summary: dict[str, Any],
    behavior_significance_summary: dict[str, Any],
) -> str:
    lines = [
        f"# Paper 3 Study: {summary['study_name']}",
        "",
        f"- Created: {summary['created_at']}",
        f"- Models: {', '.join(summary['model_keys'])}",
        f"- Families: {', '.join(summary['families']) if summary['families'] else 'all'}",
        f"- Budgets: {', '.join(f'{float(item):.2f}' for item in summary['budgets'])}",
        f"- Policies: {', '.join(summary['policies'])}",
        "",
    ]
    for model_key, payload in summary["models"].items():
        lines.extend(
            [
                f"## {model_key}",
                "",
                f"- Model name: `{payload['model_name']}`",
                f"- Conversations: {payload['num_conversations']}",
                f"- Evaluations: {payload['num_evaluations']}",
                f"- Behavior evaluations: {payload['num_behavior_evaluations']}",
                f"- Segment span: {payload['segment_span']}",
                f"- Target-turn stride: {payload.get('target_turn_stride', 1)}",
                f"- Max target turns / conversation: {payload.get('max_target_turns')}",
                f"- Max turns / conversation: {payload.get('max_turns_per_conversation')}",
                "",
            ]
        )
        for budget_key, budget_payload in payload["improvement_vs_uniform"].items():
            lines.append(f"- Improvement vs uniform @ {budget_key}:")
            for policy_name, metrics in budget_payload.items():
                lines.append(
                    f"  {policy_name}: delta logit L2 {metrics['delta_logit_l2']:.3f}, "
                    f"relative logit L2 {metrics['relative_logit_l2']:.3f}"
                )
        if payload["behavior_improvement_vs_uniform"]:
            for budget_key, budget_payload in payload["behavior_improvement_vs_uniform"].items():
                lines.append(f"- Behavior improvement vs uniform @ {budget_key}:")
                for policy_name, metrics in budget_payload.items():
                    lines.append(
                        f"  {policy_name}: delta answer avg NLL {metrics['delta_answer_avg_neg_logprob']:.4f}, "
                        f"delta answer-loss increase {metrics['delta_answer_avg_neg_logprob_delta']:.4f}"
                    )
        lines.append("")
        lines.append("- Confidence and significance:")
        for budget_key, budget_payload in significance_summary.get(model_key, {}).items():
            lines.append(f"  budget {budget_key}:")
            for policy_name, metrics in budget_payload.items():
                row_delta = metrics["delta_logit_l2"]["row_level"]
                conv_delta = metrics["delta_logit_l2"]["conversation_level"]
                lines.append(
                    f"    {policy_name}: row Δ logit L2 {row_delta['mean']:.3f} "
                    f"[{row_delta['ci_low']:.3f}, {row_delta['ci_high']:.3f}], p={row_delta['p_value']:.4f}; "
                    f"conversation Δ {conv_delta['mean']:.3f} "
                    f"[{conv_delta['ci_low']:.3f}, {conv_delta['ci_high']:.3f}], p={conv_delta['p_value']:.4f}"
                )
        if behavior_significance_summary.get(model_key):
            lines.append("  behavior:")
            for budget_key, budget_payload in behavior_significance_summary[model_key].items():
                lines.append(f"    budget {budget_key}:")
                for policy_name, metrics in budget_payload.items():
                    row_delta = metrics["delta_answer_avg_neg_logprob"]["row_level"]
                    conv_delta = metrics["delta_answer_avg_neg_logprob"]["conversation_level"]
                    lines.append(
                        f"      {policy_name}: row Δ answer avg NLL {row_delta['mean']:.4f} "
                        f"[{row_delta['ci_low']:.4f}, {row_delta['ci_high']:.4f}], p={row_delta['p_value']:.4f}; "
                        f"conversation Δ {conv_delta['mean']:.4f} "
                        f"[{conv_delta['ci_low']:.4f}, {conv_delta['ci_high']:.4f}], p={conv_delta['p_value']:.4f}"
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Paper 3 pilot across multiple models.")
    parser.add_argument("--study-name", default="paper3_study_v1")
    parser.add_argument("--model-keys", default="qwen25_05b,qwen25_15b,smollm2_17b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default="long_dependency,retrieval_heavy,code_conversation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "studies")
    parser.add_argument("--budgets", default="0.20,0.35,0.50")
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--skip-conversations", type=int, default=0)
    parser.add_argument("--segment-span", type=int, default=2)
    parser.add_argument("--target-turn-stride", type=int, default=1)
    parser.add_argument("--max-target-turns", type=int, default=None)
    parser.add_argument("--max-turns-per-conversation", type=int, default=None)
    parser.add_argument("--harm-predictor-path", type=Path, default=None)
    parser.add_argument(
        "--policies",
        default="uniform,semantic,geometry,geometry_segment_actions,geometry_keep_compress_drop",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    model_keys = [item.strip() for item in args.model_keys.split(",") if item.strip()]
    policies = _parse_policies(args.policies)
    study_dir = args.output_root / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[study] Starting Paper 3 study '{args.study_name}' in {study_dir}",
        flush=True,
    )
    print(
        f"[study] Models={','.join(model_keys)} budgets={args.budgets} policies={','.join(policies)}",
        flush=True,
    )

    model_results: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    combined_behavior_rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        print(f"[study] Running model {model_key}...", flush=True)
        result = run_codec_pilot(
            model_key=model_key,
            input_paths=input_paths,
            families=_parse_families(args.families),
            budgets=_parse_float_list(args.budgets),
            recent_window=args.recent_window,
            min_history=args.min_history,
            max_input_tokens=args.max_input_tokens,
            dtype=args.dtype,
            state_layer=args.state_layer,
            device=args.device,
            limit_conversations=args.limit_conversations,
            skip_conversations=args.skip_conversations,
            output_dir=study_dir / model_key,
            segment_span=args.segment_span,
            policies=policies,
            target_turn_stride=args.target_turn_stride,
            max_target_turns=args.max_target_turns,
            max_turns_per_conversation=args.max_turns_per_conversation,
            harm_predictor_path=args.harm_predictor_path,
        )
        model_results.append(result)
        combined_rows.extend(result["rows"])
        combined_behavior_rows.extend(result["behavior_rows"])
        print(
            f"[study] Completed model {model_key}: "
            f"{result['summary']['num_evaluations']} eval rows, "
            f"{result['summary']['num_behavior_evaluations']} behavior rows",
            flush=True,
        )

    summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_keys": model_keys,
        "families": _parse_families(args.families),
        "budgets": _parse_float_list(args.budgets),
        "policies": list(policies),
        "target_turn_stride": args.target_turn_stride,
        "max_target_turns": args.max_target_turns,
        "max_turns_per_conversation": args.max_turns_per_conversation,
        "skip_conversations": args.skip_conversations,
        "models": _study_summary(model_results),
    }

    confidence_summary = _confidence_summary(combined_rows)
    significance_summary = _paired_delta_summary(combined_rows, "logit_l2")
    behavior_confidence_summary = _behavior_confidence_summary(combined_behavior_rows)
    behavior_significance_summary = _paired_delta_summary(combined_behavior_rows, "answer_avg_neg_logprob")

    _write_csv(study_dir / "evaluation_rows.csv", combined_rows)
    _write_csv(study_dir / "behavior_rows.csv", combined_behavior_rows)
    (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (study_dir / "confidence_summary.json").write_text(json.dumps(confidence_summary, indent=2), encoding="utf-8")
    (study_dir / "significance_summary.json").write_text(json.dumps(significance_summary, indent=2), encoding="utf-8")
    (study_dir / "behavior_confidence_summary.json").write_text(json.dumps(behavior_confidence_summary, indent=2), encoding="utf-8")
    (study_dir / "behavior_significance_summary.json").write_text(json.dumps(behavior_significance_summary, indent=2), encoding="utf-8")
    (study_dir / "study_report.md").write_text(
        _format_report(
            summary,
            confidence_summary,
            behavior_confidence_summary,
            significance_summary,
            behavior_significance_summary,
        ),
        encoding="utf-8",
    )
    print(f"[study] Wrote Paper 3 study outputs to {study_dir}", flush=True)


if __name__ == "__main__":
    main()
