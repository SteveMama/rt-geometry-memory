from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from paper1_geometry.analysis import analyze_trajectory
from paper1_geometry.boundary_features import lexical_shift_scores
from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths
from paper1_geometry.modeling import ConversationStateExtractor, resolve_model_spec

from .policies import (
    select_segment_actions,
    select_turns,
    turn_geometry_risk,
    turn_hybrid_risk,
    turn_lexical_risk,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "paper2"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "paper1_geometry" / "assets" / "paper1_study_conversations.jsonl"
DEFAULT_POLICIES = (
    "uniform",
    "lexical",
    "geometry",
    "geometry_lexical",
    "uniform_segment_actions",
    "geometry_segment_actions",
)


def _kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    p_log = logits_p - np.logaddexp.reduce(logits_p)
    q_log = logits_q - np.logaddexp.reduce(logits_q)
    p = np.exp(p_log)
    return float(max(np.sum(p * (p_log - q_log)), 0.0))


def _policy_messages(
    conversation: ConversationRecord,
    target_turn: int,
    retained_prior_indices: list[int],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    for idx in retained_prior_indices:
        turn = conversation.turns[idx]
        messages.append({"role": turn.role, "content": turn.content})
    current_turn = conversation.turns[target_turn]
    messages.append({"role": current_turn.role, "content": current_turn.content})
    return messages


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_families(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy_name in sorted({str(row["policy_name"]) for row in rows}):
        policy_rows = [row for row in rows if str(row["policy_name"]) == policy_name]
        budget_summary: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in policy_rows}):
            budget_rows = [row for row in policy_rows if float(row["budget_fraction"]) == budget]
            budget_summary[f"{budget:.2f}"] = {
                "num_evaluations": len(budget_rows),
                "mean_logit_l2": float(np.mean([float(row["logit_l2"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_kl": float(np.mean([float(row["kl"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_top1_agreement": float(np.mean([float(row["top1_match"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_token_fraction": float(np.mean([float(row["token_fraction"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_budget_token_fraction": float(np.mean([float(row["budget_token_fraction"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_retained_turn_fraction": float(np.mean([float(row["retained_turn_fraction"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_kept_segments": float(np.mean([float(row.get("kept_segment_count", 0.0)) for row in budget_rows])) if budget_rows else 0.0,
                "mean_compressed_segments": float(np.mean([float(row.get("compressed_segment_count", 0.0)) for row in budget_rows])) if budget_rows else 0.0,
                "mean_evicted_segments": float(np.mean([float(row.get("evicted_segment_count", 0.0)) for row in budget_rows])) if budget_rows else 0.0,
            }
        summary[policy_name] = budget_summary
    return summary


def _aggregate_behavior_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy_name in sorted({str(row["policy_name"]) for row in rows}):
        policy_rows = [row for row in rows if str(row["policy_name"]) == policy_name]
        budget_summary: dict[str, Any] = {}
        for budget in sorted({float(row["budget_fraction"]) for row in policy_rows}):
            budget_rows = [row for row in policy_rows if float(row["budget_fraction"]) == budget]
            budget_summary[f"{budget:.2f}"] = {
                "num_evaluations": len(budget_rows),
                "mean_answer_avg_neg_logprob": float(np.mean([float(row["answer_avg_neg_logprob"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_answer_total_neg_logprob": float(np.mean([float(row["answer_total_neg_logprob"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_answer_avg_neg_logprob_delta": float(np.mean([float(row["answer_avg_neg_logprob_delta"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_answer_total_neg_logprob_delta": float(np.mean([float(row["answer_total_neg_logprob_delta"]) for row in budget_rows])) if budget_rows else 0.0,
            }
        summary[policy_name] = budget_summary
    return summary


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if str(row["family"]) == family]
        family_payload: dict[str, Any] = {}
        for policy_name in sorted({str(row["policy_name"]) for row in family_rows}):
            policy_rows = [row for row in family_rows if str(row["policy_name"]) == policy_name]
            budget_payload: dict[str, Any] = {}
            for budget in sorted({float(row["budget_fraction"]) for row in policy_rows}):
                budget_rows = [row for row in policy_rows if float(row["budget_fraction"]) == budget]
                budget_payload[f"{budget:.2f}"] = {
                    "mean_logit_l2": float(np.mean([float(row["logit_l2"]) for row in budget_rows])) if budget_rows else 0.0,
                    "mean_kl": float(np.mean([float(row["kl"]) for row in budget_rows])) if budget_rows else 0.0,
                    "mean_top1_agreement": float(np.mean([float(row["top1_match"]) for row in budget_rows])) if budget_rows else 0.0,
                    "mean_token_fraction": float(np.mean([float(row["token_fraction"]) for row in budget_rows])) if budget_rows else 0.0,
                }
            family_payload[policy_name] = budget_payload
        payload[family] = family_payload
    return payload


def _improvement_vs_uniform(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for budget in sorted({float(row["budget_fraction"]) for row in rows}):
        budget_rows = [row for row in rows if float(row["budget_fraction"]) == budget]
        uniform_rows = [row for row in budget_rows if str(row["policy_name"]) == "uniform"]
        if not uniform_rows:
            continue
        uniform_logit = float(np.mean([float(row["logit_l2"]) for row in uniform_rows]))
        uniform_kl = float(np.mean([float(row["kl"]) for row in uniform_rows]))
        budget_payload: dict[str, Any] = {}
        for policy_name in sorted({str(row["policy_name"]) for row in budget_rows if str(row["policy_name"]) != "uniform"}):
            policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
            mean_logit = float(np.mean([float(row["logit_l2"]) for row in policy_rows])) if policy_rows else 0.0
            mean_kl = float(np.mean([float(row["kl"]) for row in policy_rows])) if policy_rows else 0.0
            budget_payload[policy_name] = {
                "delta_logit_l2": mean_logit - uniform_logit,
                "delta_kl": mean_kl - uniform_kl,
                "relative_logit_l2": (mean_logit / uniform_logit) if uniform_logit > 0 else 0.0,
                "relative_kl": (mean_kl / uniform_kl) if uniform_kl > 0 else 0.0,
            }
        payload[f"{budget:.2f}"] = budget_payload
    return payload


def _behavior_improvement_vs_uniform(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for budget in sorted({float(row["budget_fraction"]) for row in rows}):
        budget_rows = [row for row in rows if float(row["budget_fraction"]) == budget]
        uniform_rows = [row for row in budget_rows if str(row["policy_name"]) == "uniform"]
        if not uniform_rows:
            continue
        uniform_answer_avg = float(np.mean([float(row["answer_avg_neg_logprob"]) for row in uniform_rows]))
        uniform_answer_delta = float(np.mean([float(row["answer_avg_neg_logprob_delta"]) for row in uniform_rows]))
        budget_payload: dict[str, Any] = {}
        for policy_name in sorted({str(row["policy_name"]) for row in budget_rows if str(row["policy_name"]) != "uniform"}):
            policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
            mean_answer_avg = float(np.mean([float(row["answer_avg_neg_logprob"]) for row in policy_rows])) if policy_rows else 0.0
            mean_answer_delta = float(np.mean([float(row["answer_avg_neg_logprob_delta"]) for row in policy_rows])) if policy_rows else 0.0
            budget_payload[policy_name] = {
                "delta_answer_avg_neg_logprob": mean_answer_avg - uniform_answer_avg,
                "delta_answer_avg_neg_logprob_delta": mean_answer_delta - uniform_answer_delta,
            }
        payload[f"{budget:.2f}"] = budget_payload
    return payload


def _prefix_turn_costs(prefix_token_counts: np.ndarray) -> np.ndarray:
    if prefix_token_counts.size == 0:
        return np.zeros(0, dtype=np.int32)
    costs = prefix_token_counts.astype(np.int32).copy()
    costs[1:] = np.maximum(prefix_token_counts[1:] - prefix_token_counts[:-1], 1)
    costs[0] = max(int(prefix_token_counts[0]), 1)
    return costs


def run_controller(
    *,
    model_key: str,
    input_paths: list[Path],
    families: list[str] | None,
    budgets: list[float],
    recent_window: int,
    min_history: int,
    max_input_tokens: int,
    dtype: str,
    state_layer: int,
    device: str | None,
    limit_conversations: int | None,
    output_dir: Path | None,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
) -> dict[str, Any]:
    spec = resolve_model_spec(model_key)
    if spec is None:
        raise RuntimeError(f"Unknown model key: {model_key}")

    conversations = load_conversations_from_paths(input_paths)
    if families is not None:
        conversations = [conversation for conversation in conversations if conversation.family in families]
    if limit_conversations is not None:
        conversations = conversations[:limit_conversations]
    if not conversations:
        raise RuntimeError("No conversations selected for Paper 2.")

    extractor = ConversationStateExtractor(
        model_name=spec.model_name,
        device=device,
        dtype=dtype,
        state_layer=state_layer,
    )

    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    for conversation in conversations:
        full_batch = extractor.extract_conversation(
            conversation,
            max_turns=None,
            max_input_tokens=max_input_tokens,
        )
        analysis = analyze_trajectory(
            states=full_batch.states,
            logits=full_batch.logits,
            reconstructed_logits=full_batch.logits,
            gold_boundaries=conversation.boundary_indices,
            lexical_boundary_scores=lexical_shift_scores(conversation),
        )
        geometry_risk = turn_geometry_risk(analysis)
        lexical_risk = turn_lexical_risk(analysis)
        hybrid_risk = turn_hybrid_risk(analysis)
        risk_map = {
            "uniform": np.zeros(len(conversation.turns), dtype=np.float32),
            "lexical": lexical_risk,
            "geometry": geometry_risk,
            "geometry_lexical": hybrid_risk,
            "uniform_segment_actions": np.zeros(len(conversation.turns), dtype=np.float32),
            "geometry_segment_actions": geometry_risk,
        }
        turn_costs = _prefix_turn_costs(full_batch.token_counts)

        for target_turn in range(min_history, len(conversation.turns)):
            full_logits = full_batch.logits[target_turn]
            full_tokens = int(full_batch.token_counts[target_turn])
            prefix_turn_count = target_turn
            prefix_turn_costs = turn_costs[:prefix_turn_count]
            is_behavior_turn = (
                conversation.turns[target_turn].role == "user"
                and target_turn + 1 < len(conversation.turns)
                and conversation.turns[target_turn + 1].role == "assistant"
            )
            full_behavior_score = None
            if is_behavior_turn:
                full_messages = _policy_messages(
                    conversation=conversation,
                    target_turn=target_turn,
                    retained_prior_indices=list(range(prefix_turn_count)),
                )
                full_behavior_score = extractor.score_assistant_response(
                    full_messages,
                    conversation.turns[target_turn + 1].content,
                    max_input_tokens=max_input_tokens,
                )
            for budget in budgets:
                for policy_name in policies:
                    if policy_name.endswith("segment_actions"):
                        selection = select_segment_actions(
                            policy_name=policy_name,
                            risk_scores=risk_map[policy_name][:prefix_turn_count],
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    else:
                        selection = select_turns(
                            policy_name=policy_name,
                            risk_scores=risk_map[policy_name][:prefix_turn_count],
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    messages = _policy_messages(
                        conversation=conversation,
                        target_turn=target_turn,
                        retained_prior_indices=selection.retained_turn_indices,
                    )
                    compressed = extractor.score_messages(messages, max_input_tokens=max_input_tokens)
                    evaluation_rows.append(
                        {
                            "conversation_id": conversation.conversation_id,
                            "family": conversation.family,
                            "target_turn": target_turn,
                            "policy_name": policy_name,
                            "budget_fraction": float(budget),
                            "budget_token_fraction": selection.retained_cost_fraction,
                            "retained_turn_fraction": selection.retained_fraction,
                            "retained_turn_indices": ",".join(str(index) for index in selection.retained_turn_indices),
                            "dropped_turn_indices": ",".join(
                                str(index) for index in range(prefix_turn_count) if index not in set(selection.retained_turn_indices)
                            ),
                            "full_token_count": full_tokens,
                            "compressed_token_count": compressed.token_count,
                            "token_fraction": float(compressed.token_count / max(full_tokens, 1)),
                            "logit_l2": float(np.linalg.norm(full_logits - compressed.logits)),
                            "kl": _kl_divergence(full_logits, compressed.logits),
                            "top1_match": float(np.argmax(full_logits) == np.argmax(compressed.logits)),
                            "kept_segment_count": selection.kept_segment_count,
                            "compressed_segment_count": selection.compressed_segment_count,
                            "evicted_segment_count": selection.evicted_segment_count,
                        }
                    )
                    if is_behavior_turn and full_behavior_score is not None:
                        behavior_score = extractor.score_assistant_response(
                            messages,
                            conversation.turns[target_turn + 1].content,
                            max_input_tokens=max_input_tokens,
                        )
                        behavior_rows.append(
                            {
                                "conversation_id": conversation.conversation_id,
                                "family": conversation.family,
                                "target_turn": target_turn,
                                "policy_name": policy_name,
                                "budget_fraction": float(budget),
                                "answer_token_count": behavior_score.token_count,
                                "answer_avg_neg_logprob": behavior_score.avg_neg_logprob,
                                "answer_total_neg_logprob": behavior_score.total_neg_logprob,
                                "full_answer_avg_neg_logprob": full_behavior_score.avg_neg_logprob,
                                "full_answer_total_neg_logprob": full_behavior_score.total_neg_logprob,
                                "answer_avg_neg_logprob_delta": behavior_score.avg_neg_logprob - full_behavior_score.avg_neg_logprob,
                                "answer_total_neg_logprob_delta": behavior_score.total_neg_logprob - full_behavior_score.total_neg_logprob,
                            }
                        )

    aggregate = _aggregate_rows(evaluation_rows)
    behavior_aggregate = _aggregate_behavior_rows(behavior_rows)
    family_summary = _family_summary(evaluation_rows)
    improvement = _improvement_vs_uniform(evaluation_rows)
    behavior_improvement = _behavior_improvement_vs_uniform(behavior_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "model_name": spec.model_name,
        "families": families,
        "budgets": budgets,
        "recent_window": recent_window,
        "min_history": min_history,
        "num_conversations": len(conversations),
        "num_evaluations": len(evaluation_rows),
        "num_behavior_evaluations": len(behavior_rows),
        "aggregate": aggregate,
        "behavior_aggregate": behavior_aggregate,
        "family_summary": family_summary,
        "improvement_vs_uniform": improvement,
        "behavior_improvement_vs_uniform": behavior_improvement,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "evaluation_rows.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(evaluation_rows[0].keys()) if evaluation_rows else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_rows)
        if behavior_rows:
            behavior_csv = output_dir / "behavior_rows.csv"
            with behavior_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(behavior_rows[0].keys()))
                writer.writeheader()
                writer.writerows(behavior_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(_format_run_report(summary), encoding="utf-8")

    return {
        "summary": summary,
        "rows": evaluation_rows,
        "behavior_rows": behavior_rows,
    }


def _format_run_report(summary: dict[str, Any]) -> str:
    budgets = [float(budget) for budget in summary["budgets"]]
    families = summary.get("families") or []
    lines = [
        f"# Paper 2 Run: {summary['model_key']}",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Families: {', '.join(families)}",
        f"- Budgets: {', '.join(f'{budget:.2f}' for budget in budgets)}",
        f"- Recent window: {summary['recent_window']}",
        f"- Conversations: {summary['num_conversations']}",
        f"- Evaluations: {summary['num_evaluations']}",
        f"- Behavior evaluations: {summary['num_behavior_evaluations']}",
        "",
        "## Aggregate",
        "",
    ]
    for policy_name, payload in summary["aggregate"].items():
        lines.append(f"### {policy_name}")
        lines.append("")
        for budget_key, budget_payload in payload.items():
            lines.append(
                f"- budget {budget_key}: logit L2 {budget_payload['mean_logit_l2']:.3f}, "
                f"KL {budget_payload['mean_kl']:.6f}, top1 {budget_payload['mean_top1_agreement']:.3f}, "
                f"actual token fraction {budget_payload['mean_token_fraction']:.3f}, "
                f"budget token fraction {budget_payload['mean_budget_token_fraction']:.3f}"
            )
        lines.append("")
    if summary["behavior_aggregate"]:
        lines.append("## Behavior Aggregate")
        lines.append("")
        for policy_name, payload in summary["behavior_aggregate"].items():
            lines.append(f"### {policy_name}")
            lines.append("")
            for budget_key, budget_payload in payload.items():
                lines.append(
                    f"- budget {budget_key}: answer avg NLL {budget_payload['mean_answer_avg_neg_logprob']:.4f}, "
                    f"answer delta {budget_payload['mean_answer_avg_neg_logprob_delta']:.4f}"
                )
            lines.append("")
    lines.append("## Improvement Vs Uniform")
    lines.append("")
    for budget_key, payload in summary["improvement_vs_uniform"].items():
        lines.append(f"### budget {budget_key}")
        lines.append("")
        for policy_name, policy_payload in payload.items():
            lines.append(
                f"- {policy_name}: delta logit L2 {policy_payload['delta_logit_l2']:.3f}, "
                f"delta KL {policy_payload['delta_kl']:.6f}, relative logit L2 {policy_payload['relative_logit_l2']:.3f}"
            )
        lines.append("")
    if summary["behavior_improvement_vs_uniform"]:
        lines.append("## Behavior Improvement Vs Uniform")
        lines.append("")
        for budget_key, payload in summary["behavior_improvement_vs_uniform"].items():
            lines.append(f"### budget {budget_key}")
            lines.append("")
            for policy_name, policy_payload in payload.items():
                lines.append(
                    f"- {policy_name}: delta answer avg NLL {policy_payload['delta_answer_avg_neg_logprob']:.4f}, "
                    f"delta answer-loss increase {policy_payload['delta_answer_avg_neg_logprob_delta']:.4f}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Paper 2 geometry-aware memory controller study.")
    parser.add_argument("--run-name", default="blazing_v1")
    parser.add_argument("--model-key", default="qwen25_05b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default="long_dependency,retrieval_heavy,code_conversation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default="0.20,0.35,0.50,0.65")
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())

    result = run_controller(
        model_key=args.model_key,
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
        output_dir=args.output_root / args.run_name,
    )
    print(f"Wrote Paper 2 outputs to {args.output_root / args.run_name}")
    print(
        f"Completed {result['summary']['num_evaluations']} evaluations across "
        f"{result['summary']['num_conversations']} conversations."
    )


if __name__ == "__main__":
    main()
