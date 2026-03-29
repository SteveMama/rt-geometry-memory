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
from paper2_memory.policies import (
    select_segment_actions,
    select_turns,
    turn_geometry_risk,
    turn_semantic_risk,
)

from .policies import CodecSelection, SparseSegmentMemory, select_sparse_segment_memory


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "paper3"
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "paper1_geometry" / "assets" / "paper2_behavior_stress_conversations.jsonl"
)
DEFAULT_POLICIES = (
    "uniform",
    "semantic",
    "geometry",
    "geometry_segment_actions",
    "geometry_keep_compress_drop",
)


def _progress(iterable: Any, *, total: int | None = None, desc: str = "", leave: bool = True) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)
    except Exception:
        return iterable


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_families(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_policies(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_POLICIES
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _sample_target_turns(
    *,
    num_turns: int,
    min_history: int,
    stride: int,
    max_target_turns: int | None,
) -> list[int]:
    if num_turns <= min_history:
        return []
    target_turns = list(range(min_history, num_turns, max(stride, 1)))
    if max_target_turns is None or len(target_turns) <= max_target_turns:
        return target_turns
    if max_target_turns <= 1:
        return [target_turns[-1]]
    positions = np.linspace(0, len(target_turns) - 1, num=max_target_turns)
    sampled = sorted({target_turns[int(round(position))] for position in positions})
    if sampled[-1] != target_turns[-1]:
        sampled[-1] = target_turns[-1]
    return sampled


def _prefix_turn_costs(prefix_token_counts: np.ndarray) -> np.ndarray:
    if prefix_token_counts.size == 0:
        return np.zeros(0, dtype=np.int32)
    costs = prefix_token_counts.astype(np.int32).copy()
    costs[1:] = np.maximum(prefix_token_counts[1:] - prefix_token_counts[:-1], 1)
    costs[0] = max(int(prefix_token_counts[0]), 1)
    return costs


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
    messages.append({"role": conversation.turns[target_turn].role, "content": conversation.turns[target_turn].content})
    return messages


def _kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    p_log = logits_p - np.logaddexp.reduce(logits_p)
    q_log = logits_q - np.logaddexp.reduce(logits_q)
    p = np.exp(p_log)
    return float(max(np.sum(p * (p_log - q_log)), 0.0))


def _memory_objects_payload(memory_objects: list[SparseSegmentMemory]) -> str:
    payload = []
    for item in memory_objects:
        payload.append(
            {
                "segment_start": item.segment_start,
                "segment_end": item.segment_end,
                "anchor_turn_index": item.anchor_turn_index,
                "support_turn_indices": item.support_turn_indices,
                "retained_turn_indices": item.retained_turn_indices,
                "risk": item.risk,
                "action": item.action,
            }
        )
    return json.dumps(payload, separators=(",", ":"))


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
                "mean_kept_segments": float(np.mean([float(row["kept_segment_count"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_compressed_segments": float(np.mean([float(row["compressed_segment_count"]) for row in budget_rows])) if budget_rows else 0.0,
                "mean_evicted_segments": float(np.mean([float(row["evicted_segment_count"]) for row in budget_rows])) if budget_rows else 0.0,
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


def _improvement_vs_uniform(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for budget in sorted({float(row["budget_fraction"]) for row in rows}):
        budget_rows = [row for row in rows if float(row["budget_fraction"]) == budget]
        uniform_rows = [row for row in budget_rows if str(row["policy_name"]) == "uniform"]
        if not uniform_rows:
            continue
        uniform_logit = float(np.mean([float(row["logit_l2"]) for row in uniform_rows]))
        budget_payload: dict[str, Any] = {}
        for policy_name in sorted({str(row["policy_name"]) for row in budget_rows if str(row["policy_name"]) != "uniform"}):
            policy_rows = [row for row in budget_rows if str(row["policy_name"]) == policy_name]
            mean_logit = float(np.mean([float(row["logit_l2"]) for row in policy_rows])) if policy_rows else 0.0
            budget_payload[policy_name] = {
                "delta_logit_l2": mean_logit - uniform_logit,
                "relative_logit_l2": (mean_logit / uniform_logit) if uniform_logit > 0 else 0.0,
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


def run_codec_pilot(
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
    segment_span: int = 2,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    target_turn_stride: int = 1,
    max_target_turns: int | None = None,
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
        raise RuntimeError("No conversations selected for Paper 3.")

    extractor = ConversationStateExtractor(
        model_name=spec.model_name,
        device=device,
        dtype=dtype,
        state_layer=state_layer,
    )
    conversation_target_turns = {
        conversation.conversation_id: _sample_target_turns(
            num_turns=len(conversation.turns),
            min_history=min_history,
            stride=target_turn_stride,
            max_target_turns=max_target_turns,
        )
        for conversation in conversations
    }
    total_target_turns = sum(len(target_turns) for target_turns in conversation_target_turns.values())
    total_policy_evals = total_target_turns * len(budgets) * len(policies)
    print(
        f"[{model_key}] Starting Paper 3 run with {len(conversations)} conversations, "
        f"{total_target_turns} target turns, {len(policies)} policies, {len(budgets)} budgets "
        f"(~{total_policy_evals} policy evaluations).",
        flush=True,
    )
    print(
        f"[{model_key}] Model={spec.model_name} device={extractor.device} "
        f"max_input_tokens={max_input_tokens} segment_span={segment_span} "
        f"target_turn_stride={target_turn_stride} max_target_turns={max_target_turns}",
        flush=True,
    )

    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    conversation_iterator = _progress(
        enumerate(conversations, start=1),
        total=len(conversations),
        desc=f"{model_key} conversations",
    )
    for conversation_index, conversation in conversation_iterator:
        full_batch = extractor.extract_conversation(
            conversation,
            max_turns=None,
            max_input_tokens=max_input_tokens,
        )
        target_turns = conversation_target_turns[conversation.conversation_id]
        if hasattr(conversation_iterator, "set_postfix_str"):
            conversation_iterator.set_postfix_str(
                f"{conversation.conversation_id} turns={len(conversation.turns)} sampled_targets={len(target_turns)} rows={len(evaluation_rows)}"
            )
        elif conversation_index == 1 or conversation_index % 10 == 0:
            print(
                f"[{model_key}] Conversation {conversation_index}/{len(conversations)} "
                f"{conversation.conversation_id} turns={len(conversation.turns)} "
                f"sampled_targets={len(target_turns)} rows={len(evaluation_rows)}",
                flush=True,
            )
        analysis = analyze_trajectory(
            states=full_batch.states,
            logits=full_batch.logits,
            reconstructed_logits=full_batch.logits,
            gold_boundaries=conversation.boundary_indices,
            lexical_boundary_scores=lexical_shift_scores(conversation),
        )
        geometry_risk = turn_geometry_risk(analysis)
        turn_costs = _prefix_turn_costs(full_batch.token_counts)

        for target_turn in target_turns:
            full_logits = full_batch.logits[target_turn]
            full_tokens = int(full_batch.token_counts[target_turn])
            prefix_turn_count = target_turn
            prefix_turn_costs = turn_costs[:prefix_turn_count]
            semantic_risk = turn_semantic_risk(full_batch.states, target_turn)[:prefix_turn_count]
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
                    memory_objects: list[SparseSegmentMemory] = []
                    if policy_name == "uniform":
                        selection = select_turns(
                            policy_name=policy_name,
                            risk_scores=np.zeros(prefix_turn_count, dtype=np.float32),
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    elif policy_name == "semantic":
                        selection = select_turns(
                            policy_name=policy_name,
                            risk_scores=semantic_risk,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    elif policy_name == "geometry":
                        selection = select_turns(
                            policy_name=policy_name,
                            risk_scores=geometry_risk[:prefix_turn_count],
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    elif policy_name == "geometry_segment_actions":
                        selection = select_segment_actions(
                            policy_name=policy_name,
                            risk_scores=geometry_risk[:prefix_turn_count],
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    elif policy_name == "geometry_keep_compress_drop":
                        selection = select_sparse_segment_memory(
                            risk_scores=geometry_risk[:prefix_turn_count],
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=segment_span,
                        )
                        memory_objects = selection.memory_objects
                    else:
                        raise ValueError(f"Unknown policy: {policy_name}")

                    messages = _policy_messages(
                        conversation=conversation,
                        target_turn=target_turn,
                        retained_prior_indices=selection.retained_turn_indices,
                    )
                    compressed = extractor.score_messages(messages, max_input_tokens=max_input_tokens)
                    evaluation_rows.append(
                        {
                            "model_key": model_key,
                            "conversation_id": conversation.conversation_id,
                            "family": conversation.family,
                            "target_turn": target_turn,
                            "policy_name": policy_name,
                            "budget_fraction": float(budget),
                            "budget_token_fraction": selection.retained_cost_fraction,
                            "retained_turn_fraction": selection.retained_fraction,
                            "retained_turn_indices": ",".join(str(index) for index in selection.retained_turn_indices),
                            "full_token_count": full_tokens,
                            "compressed_token_count": compressed.token_count,
                            "token_fraction": float(compressed.token_count / max(full_tokens, 1)),
                            "logit_l2": float(np.linalg.norm(full_logits - compressed.logits)),
                            "kl": _kl_divergence(full_logits, compressed.logits),
                            "top1_match": float(np.argmax(full_logits) == np.argmax(compressed.logits)),
                            "kept_segment_count": selection.kept_segment_count,
                            "compressed_segment_count": selection.compressed_segment_count,
                            "evicted_segment_count": selection.evicted_segment_count,
                            "memory_objects": _memory_objects_payload(memory_objects),
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
                                "model_key": model_key,
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
        if conversation_index == len(conversations) or conversation_index % 10 == 0:
            print(
                f"[{model_key}] Completed {conversation_index}/{len(conversations)} conversations; "
                f"rows={len(evaluation_rows)} behavior_rows={len(behavior_rows)}",
                flush=True,
            )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "model_name": spec.model_name,
        "families": families,
        "budgets": budgets,
        "recent_window": recent_window,
        "min_history": min_history,
        "segment_span": segment_span,
        "policies": list(policies),
        "target_turn_stride": target_turn_stride,
        "max_target_turns": max_target_turns,
        "num_conversations": len(conversations),
        "num_evaluations": len(evaluation_rows),
        "num_behavior_evaluations": len(behavior_rows),
        "aggregate": _aggregate_rows(evaluation_rows),
        "behavior_aggregate": _aggregate_behavior_rows(behavior_rows),
        "improvement_vs_uniform": _improvement_vs_uniform(evaluation_rows),
        "behavior_improvement_vs_uniform": _behavior_improvement_vs_uniform(behavior_rows),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "evaluation_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(evaluation_rows)
        if behavior_rows:
            with (output_dir / "behavior_rows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(behavior_rows[0].keys()))
                writer.writeheader()
                writer.writerows(behavior_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(_format_report(summary), encoding="utf-8")

    return {"summary": summary, "rows": evaluation_rows, "behavior_rows": behavior_rows}


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Paper 3 Pilot: {summary['model_key']}",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Budgets: {', '.join(f'{float(item):.2f}' for item in summary['budgets'])}",
        f"- Policies: {', '.join(summary.get('policies', []))}",
        f"- Segment span: {summary['segment_span']}",
        f"- Target-turn stride: {summary.get('target_turn_stride', 1)}",
        f"- Max target turns / conversation: {summary.get('max_target_turns')}",
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
        for budget_key, metrics in payload.items():
            lines.append(
                f"- budget {budget_key}: logit L2 {metrics['mean_logit_l2']:.3f}, "
                f"KL {metrics['mean_kl']:.6f}, token fraction {metrics['mean_token_fraction']:.3f}, "
                f"kept/compressed/evicted segments "
                f"{metrics['mean_kept_segments']:.2f}/{metrics['mean_compressed_segments']:.2f}/{metrics['mean_evicted_segments']:.2f}"
            )
        lines.append("")
    if summary["behavior_aggregate"]:
        lines.append("## Behavior Aggregate")
        lines.append("")
        for policy_name, payload in summary["behavior_aggregate"].items():
            lines.append(f"### {policy_name}")
            lines.append("")
            for budget_key, metrics in payload.items():
                lines.append(
                    f"- budget {budget_key}: answer avg NLL {metrics['mean_answer_avg_neg_logprob']:.4f}, "
                    f"answer delta {metrics['mean_answer_avg_neg_logprob_delta']:.4f}"
                )
            lines.append("")
    lines.append("## Improvement Vs Uniform")
    lines.append("")
    for budget_key, payload in summary["improvement_vs_uniform"].items():
        lines.append(f"### budget {budget_key}")
        lines.append("")
        for policy_name, metrics in payload.items():
            lines.append(
                f"- {policy_name}: delta logit L2 {metrics['delta_logit_l2']:.3f}, relative logit L2 {metrics['relative_logit_l2']:.3f}"
            )
        lines.append("")
    if summary["behavior_improvement_vs_uniform"]:
        lines.append("## Behavior Improvement Vs Uniform")
        lines.append("")
        for budget_key, payload in summary["behavior_improvement_vs_uniform"].items():
            lines.append(f"### budget {budget_key}")
            lines.append("")
            for policy_name, metrics in payload.items():
                lines.append(
                    f"- {policy_name}: delta answer avg NLL {metrics['delta_answer_avg_neg_logprob']:.4f}, "
                    f"delta answer-loss increase {metrics['delta_answer_avg_neg_logprob_delta']:.4f}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Paper 3 sparse memory pilot.")
    parser.add_argument("--run-name", default="paper3_pilot_v1")
    parser.add_argument("--model-key", default="qwen25_05b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default="long_dependency,retrieval_heavy,code_conversation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default="0.20,0.35,0.50")
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--segment-span", type=int, default=2)
    parser.add_argument("--target-turn-stride", type=int, default=1)
    parser.add_argument("--max-target-turns", type=int, default=None)
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated Paper 3 policies to evaluate.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    result = run_codec_pilot(
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
        segment_span=args.segment_span,
        policies=_parse_policies(args.policies),
        target_turn_stride=args.target_turn_stride,
        max_target_turns=args.max_target_turns,
    )
    print(f"Wrote Paper 3 outputs to {args.output_root / args.run_name}")
    print(
        f"Completed {result['summary']['num_evaluations']} evaluations across "
        f"{result['summary']['num_conversations']} conversations."
    )


if __name__ == "__main__":
    main()
