from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .plotting import plot_budget_curves, plot_family_heatmap
from .run_paper2 import DEFAULT_INPUT, DEFAULT_OUTPUT, _parse_families, _parse_float_list, _parse_policies, run_controller


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_policy_budget_rows(model_key: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, budget_payload in summary["aggregate"].items():
        for budget_key, metrics in budget_payload.items():
            rows.append(
                {
                    "model_key": model_key,
                    "policy_name": policy_name,
                    "budget_fraction": budget_key,
                    **metrics,
                }
            )
    return rows


def _flatten_behavior_policy_budget_rows(model_key: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, budget_payload in summary.get("behavior_aggregate", {}).items():
        for budget_key, metrics in budget_payload.items():
            rows.append(
                {
                    "model_key": model_key,
                    "policy_name": policy_name,
                    "budget_fraction": budget_key,
                    **metrics,
                }
            )
    return rows


def _flatten_family_rows(model_key: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "model_key": model_key} for row in rows]


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


def _policy_confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20260327)
    summary: dict[str, Any] = {}
    model_keys = sorted({str(row["model_key"]) for row in rows})
    for model_key in model_keys:
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            policy_payload: dict[str, Any] = {}
            for policy_name in sorted({str(row["policy_name"]) for row in budget_rows}):
                policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
                policy_payload[policy_name] = {
                    "logit_l2": _bootstrap_mean_ci([float(row["logit_l2"]) for row in policy_rows], rng=rng),
                    "kl": _bootstrap_mean_ci([float(row["kl"]) for row in policy_rows], rng=rng),
                    "token_fraction": _bootstrap_mean_ci([float(row["token_fraction"]) for row in policy_rows], rng=rng),
                }
            budget_payload[f"{budget:.2f}"] = policy_payload
        summary[model_key] = budget_payload
    return summary


def _behavior_confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20260329)
    summary: dict[str, Any] = {}
    model_keys = sorted({str(row["model_key"]) for row in rows})
    for model_key in model_keys:
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            policy_payload: dict[str, Any] = {}
            for policy_name in sorted({str(row["policy_name"]) for row in budget_rows}):
                policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
                policy_payload[policy_name] = {
                    "answer_avg_neg_logprob": _bootstrap_mean_ci(
                        [float(row["answer_avg_neg_logprob"]) for row in policy_rows],
                        rng=rng,
                    ),
                    "answer_avg_neg_logprob_delta": _bootstrap_mean_ci(
                        [float(row["answer_avg_neg_logprob_delta"]) for row in policy_rows],
                        rng=rng,
                    ),
                }
            budget_payload[f"{budget:.2f}"] = policy_payload
        summary[model_key] = budget_payload
    return summary


def _paired_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20260328)
    summary: dict[str, Any] = {}
    model_keys = sorted({str(row["model_key"]) for row in rows})
    for model_key in model_keys:
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            comparison_policies = sorted({str(row["policy_name"]) for row in budget_rows if str(row["policy_name"]) != "uniform"})
            uniform_map = {
                (str(row["conversation_id"]), int(row["target_turn"])): row
                for row in budget_rows
                if str(row["policy_name"]) == "uniform"
            }
            comparison_payload: dict[str, Any] = {}
            for policy_name in comparison_policies:
                deltas_logit: list[float] = []
                deltas_kl: list[float] = []
                for row in budget_rows:
                    if str(row["policy_name"]) != policy_name:
                        continue
                    key = (str(row["conversation_id"]), int(row["target_turn"]))
                    baseline = uniform_map.get(key)
                    if baseline is None:
                        continue
                    deltas_logit.append(float(row["logit_l2"]) - float(baseline["logit_l2"]))
                    deltas_kl.append(float(row["kl"]) - float(baseline["kl"]))
                logit_array = np.asarray(deltas_logit, dtype=np.float64)
                kl_array = np.asarray(deltas_kl, dtype=np.float64)
                comparison_payload[policy_name] = {
                    "num_pairs": int(logit_array.size),
                    "delta_logit_l2": {
                        **_bootstrap_mean_ci(deltas_logit, rng=rng),
                        "p_value": _paired_signflip_test(logit_array, rng=rng),
                    },
                    "delta_kl": {
                        **_bootstrap_mean_ci(deltas_kl, rng=rng),
                        "p_value": _paired_signflip_test(kl_array, rng=rng),
                    },
                }
            budget_payload[f"{budget:.2f}"] = comparison_payload
        summary[model_key] = budget_payload
    return summary


def _paired_behavior_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20260330)
    summary: dict[str, Any] = {}
    model_keys = sorted({str(row["model_key"]) for row in rows})
    for model_key in model_keys:
        model_rows = [row for row in rows if str(row["model_key"]) == model_key]
        budget_payload: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in model_rows}):
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            comparison_policies = sorted({str(row["policy_name"]) for row in budget_rows if str(row["policy_name"]) != "uniform"})
            uniform_map = {
                (str(row["conversation_id"]), int(row["target_turn"])): row
                for row in budget_rows
                if str(row["policy_name"]) == "uniform"
            }
            comparison_payload: dict[str, Any] = {}
            for policy_name in comparison_policies:
                deltas_answer: list[float] = []
                for row in budget_rows:
                    if str(row["policy_name"]) != policy_name:
                        continue
                    key = (str(row["conversation_id"]), int(row["target_turn"]))
                    baseline = uniform_map.get(key)
                    if baseline is None:
                        continue
                    deltas_answer.append(float(row["answer_avg_neg_logprob"]) - float(baseline["answer_avg_neg_logprob"]))
                answer_array = np.asarray(deltas_answer, dtype=np.float64)
                comparison_payload[policy_name] = {
                    "num_pairs": int(answer_array.size),
                    "delta_answer_avg_neg_logprob": {
                        **_bootstrap_mean_ci(deltas_answer, rng=rng),
                        "p_value": _paired_signflip_test(answer_array, rng=rng),
                    },
                }
            budget_payload[f"{budget:.2f}"] = comparison_payload
        summary[model_key] = budget_payload
    return summary


def _study_summary(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for result in model_results:
        summary = result["summary"]
        payload[summary["model_key"]] = {
            "model_name": summary["model_name"],
            "num_conversations": summary["num_conversations"],
            "num_evaluations": summary["num_evaluations"],
            "num_behavior_evaluations": summary.get("num_behavior_evaluations", 0),
            "aggregate": summary["aggregate"],
            "behavior_aggregate": summary.get("behavior_aggregate", {}),
            "improvement_vs_uniform": summary["improvement_vs_uniform"],
            "behavior_improvement_vs_uniform": summary.get("behavior_improvement_vs_uniform", {}),
        }
    return payload


def _format_study_report(
    summary: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    behavior_policy_rows: list[dict[str, Any]],
    confidence_summary: dict[str, Any],
    behavior_confidence_summary: dict[str, Any],
    significance_summary: dict[str, Any],
    behavior_significance_summary: dict[str, Any],
) -> str:
    lines = [
        f"# Paper 2 Study: {summary['study_name']}",
        "",
        f"- Created: {summary['created_at']}",
        f"- Models: {', '.join(summary['model_keys'])}",
        f"- Families: {', '.join(summary['families'])}",
        f"- Budgets: {', '.join(f'{budget:.2f}' for budget in summary['budgets'])}",
        f"- Policies: {', '.join(summary['policies'])}",
        "",
        "## By Model",
        "",
    ]
    for model_key in summary["model_keys"]:
        model_payload = summary["models"][model_key]
        lines.append(f"### {model_key}")
        lines.append("")
        lines.append(f"- Model name: `{model_payload['model_name']}`")
        lines.append(f"- Conversations: {model_payload['num_conversations']}")
        lines.append(f"- Evaluations: {model_payload['num_evaluations']}")
        lines.append(f"- Behavior evaluations: {model_payload['num_behavior_evaluations']}")
        for budget_key, improvement in model_payload["improvement_vs_uniform"].items():
            lines.append(f"- Improvement vs uniform @ {budget_key}:")
            for policy_name, metrics in improvement.items():
                lines.append(
                    f"  {policy_name}: delta logit L2 {metrics['delta_logit_l2']:.3f}, "
                    f"relative logit L2 {metrics['relative_logit_l2']:.3f}"
                )
        for budget_key, improvement in model_payload["behavior_improvement_vs_uniform"].items():
            lines.append(f"- Behavior improvement vs uniform @ {budget_key}:")
            for policy_name, metrics in improvement.items():
                lines.append(
                    f"  {policy_name}: delta answer avg NLL {metrics['delta_answer_avg_neg_logprob']:.4f}, "
                    f"delta answer-loss increase {metrics['delta_answer_avg_neg_logprob_delta']:.4f}"
                )
        lines.append("")

    lines.append("## Confidence And Significance")
    lines.append("")
    for model_key in summary["model_keys"]:
        lines.append(f"### {model_key}")
        lines.append("")
        for budget_key, comparisons in significance_summary[model_key].items():
            lines.append(f"- budget {budget_key}:")
            for policy_name, payload in comparisons.items():
                delta = payload["delta_logit_l2"]
                lines.append(
                    f"  {policy_name}: mean delta logit L2 {delta['mean']:.3f} "
                    f"[{delta['ci_low']:.3f}, {delta['ci_high']:.3f}], p={delta['p_value']:.4f}"
                )
        lines.append("- Behavior:")
        for budget_key, comparisons in behavior_significance_summary[model_key].items():
            lines.append(f"  budget {budget_key}:")
            for policy_name, payload in comparisons.items():
                delta = payload["delta_answer_avg_neg_logprob"]
                lines.append(
                    f"    {policy_name}: mean delta answer avg NLL {delta['mean']:.4f} "
                    f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}], p={delta['p_value']:.4f}"
                )
        lines.append("")

    lines.append("## Aggregate Policy Means")
    lines.append("")
    for row in policy_rows:
        confidence = confidence_summary[row["model_key"]][row["budget_fraction"]][row["policy_name"]]["logit_l2"]
        lines.append(
            f"- {row['model_key']} | {row['policy_name']} | budget {row['budget_fraction']}: "
            f"logit L2 {float(row['mean_logit_l2']):.3f} "
            f"[{confidence['ci_low']:.3f}, {confidence['ci_high']:.3f}], "
            f"KL {float(row['mean_kl']):.6f}, "
            f"top1 {float(row['mean_top1_agreement']):.3f}, token fraction {float(row['mean_token_fraction']):.3f}"
        )
    if behavior_policy_rows:
        lines.append("")
        lines.append("## Behavior Policy Means")
        lines.append("")
        for row in behavior_policy_rows:
            confidence = behavior_confidence_summary[row["model_key"]][row["budget_fraction"]][row["policy_name"]]["answer_avg_neg_logprob"]
            lines.append(
                f"- {row['model_key']} | {row['policy_name']} | budget {row['budget_fraction']}: "
                f"answer avg NLL {float(row['mean_answer_avg_neg_logprob']):.4f} "
                f"[{confidence['ci_low']:.4f}, {confidence['ci_high']:.4f}], "
                f"answer-loss increase {float(row['mean_answer_avg_neg_logprob_delta']):.4f}"
            )
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Paper 2 multi-model study.")
    parser.add_argument("--study-name", default="blazing_study_v4_segment_behavior")
    parser.add_argument("--model-keys", default="qwen25_05b,qwen25_15b,smollm2_17b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default="long_dependency,retrieval_heavy,code_conversation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "studies")
    parser.add_argument("--budgets", default="0.20,0.35,0.50,0.65")
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--policies", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_keys = [item.strip() for item in args.model_keys.split(",") if item.strip()]
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    families = _parse_families(args.families) or []
    budgets = _parse_float_list(args.budgets)
    policies = _parse_policies(args.policies)

    study_dir = args.output_root / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_results: list[dict[str, Any]] = []
    policy_budget_rows: list[dict[str, Any]] = []
    behavior_policy_budget_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        result = run_controller(
            model_key=model_key,
            input_paths=input_paths,
            families=families,
            budgets=budgets,
            recent_window=args.recent_window,
            min_history=args.min_history,
            max_input_tokens=args.max_input_tokens,
            dtype=args.dtype,
            state_layer=args.state_layer,
            device=args.device,
            limit_conversations=args.limit_conversations,
            policies=policies,
            output_dir=study_dir / model_key,
        )
        model_results.append(result)
        policy_budget_rows.extend(_flatten_policy_budget_rows(model_key, result["summary"]))
        behavior_policy_budget_rows.extend(_flatten_behavior_policy_budget_rows(model_key, result["summary"]))
        all_rows.extend(_flatten_family_rows(model_key, result["rows"]))
        behavior_rows.extend(_flatten_family_rows(model_key, result["behavior_rows"]))

    study_summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_keys": model_keys,
        "families": families,
        "budgets": budgets,
        "recent_window": args.recent_window,
        "min_history": args.min_history,
        "policies": list(policies),
        "models": _study_summary(model_results),
    }
    confidence_summary = _policy_confidence_summary(all_rows)
    behavior_confidence_summary = _behavior_confidence_summary(behavior_rows)
    significance_summary = _paired_delta_summary(all_rows)
    behavior_significance_summary = _paired_behavior_delta_summary(behavior_rows)

    _write_csv(study_dir / "policy_budget_summary.csv", policy_budget_rows)
    if behavior_policy_budget_rows:
        _write_csv(study_dir / "behavior_policy_budget_summary.csv", behavior_policy_budget_rows)
    _write_csv(study_dir / "evaluation_rows.csv", all_rows)
    if behavior_rows:
        _write_csv(study_dir / "behavior_rows.csv", behavior_rows)
    (study_dir / "study_summary.json").write_text(json.dumps(study_summary, indent=2), encoding="utf-8")
    (study_dir / "confidence_summary.json").write_text(json.dumps(confidence_summary, indent=2), encoding="utf-8")
    (study_dir / "behavior_confidence_summary.json").write_text(json.dumps(behavior_confidence_summary, indent=2), encoding="utf-8")
    (study_dir / "significance_summary.json").write_text(json.dumps(significance_summary, indent=2), encoding="utf-8")
    (study_dir / "behavior_significance_summary.json").write_text(json.dumps(behavior_significance_summary, indent=2), encoding="utf-8")
    (study_dir / "study_report.md").write_text(
        _format_study_report(
            study_summary,
            policy_budget_rows,
            behavior_policy_budget_rows,
            confidence_summary,
            behavior_confidence_summary,
            significance_summary,
            behavior_significance_summary,
        ),
        encoding="utf-8",
    )

    plot_budget_curves(all_rows, plots_dir / "logit_budget_curves.png", "logit_l2", "Mean logit L2", "Paper 2: Budget vs Logit Drift")
    plot_budget_curves(all_rows, plots_dir / "kl_budget_curves.png", "kl", "Mean KL", "Paper 2: Budget vs KL")
    plot_budget_curves(all_rows, plots_dir / "token_budget_curves.png", "token_fraction", "Actual token fraction", "Paper 2: Budget vs Token Fraction")
    if budgets:
        mid_budget = f"{budgets[min(len(budgets) - 1, 1)]:.2f}"
        plot_family_heatmap(all_rows, plots_dir / "family_logit_heatmap.png", "logit_l2", mid_budget, "Mean logit L2 by Family")

    print(f"Wrote Paper 2 study outputs to {study_dir}")


if __name__ == "__main__":
    main()
