from __future__ import annotations

import argparse
import csv
import json
import re
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

from .policies import (
    CodecSelection,
    SparseSegmentMemory,
    select_masked_sparse_segment_memory,
    select_semantic_object_sparse_memory,
    select_semantic_filtered_sparse_segment_memory,
    select_sparse_segment_memory,
    select_support_aware_sparse_segment_memory,
    semantic_shortlist_mask,
)
from .query_geometry import query_conditioned_turn_risk, query_conditioned_turn_risk_v2
from .harm_predictor import HarmPredictorBundle
from .memory_objects import build_semantic_object_bundle


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "paper3"
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "paper1_geometry" / "assets" / "paper2_behavior_stress_conversations.jsonl"
)
DEFAULT_POLICIES = (
    "uniform",
    "semantic",
    "geometry",
    "query_conditioned_geometry",
    "geometry_segment_actions",
    "geometry_keep_compress_drop",
    "query_conditioned_geometry_keep_compress_drop",
    "semantic_keep_compress_drop",
)

CONSTRAINT_MARKERS: tuple[str, ...] = (
    "must",
    "should",
    "exactly",
    "only",
    "never",
    "always",
    "remember",
    "constraint",
    "format",
    "schema",
    "json",
    "yaml",
    "sql",
    "python",
    "javascript",
    "regex",
    "csv",
    "xml",
    "api",
    "fetch",
    "query",
    "retrieve",
    "retrieval",
    "search",
    "sort",
    "order by",
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


def _load_conversation_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"No conversation IDs found in {path}")
    return ids


def _select_conversations(
    *,
    conversations: list[ConversationRecord],
    families: list[str] | None,
    conversation_ids_path: Path | None,
    skip_conversations: int,
    limit_conversations: int | None,
) -> list[ConversationRecord]:
    selected = conversations
    if families is not None:
        selected = [conversation for conversation in selected if conversation.family in families]
    conversation_ids = _load_conversation_ids(conversation_ids_path)
    if conversation_ids is not None:
        conversation_map = {conversation.conversation_id: conversation for conversation in selected}
        missing_ids = [conversation_id for conversation_id in conversation_ids if conversation_id not in conversation_map]
        if missing_ids:
            raise RuntimeError(
                "Conversation ID manifest references missing conversations: "
                + ", ".join(missing_ids[:10])
                + ("..." if len(missing_ids) > 10 else "")
            )
        selected = [conversation_map[conversation_id] for conversation_id in conversation_ids]
    if skip_conversations > 0:
        selected = selected[skip_conversations:]
    if limit_conversations is not None:
        selected = selected[:limit_conversations]
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _topk_indices(values: np.ndarray, k: int) -> list[int]:
    if values.size == 0 or k <= 0:
        return []
    order = np.argsort(-np.asarray(values, dtype=np.float32), kind="stable")
    return [int(item) for item in order[: min(k, order.size)].tolist()]


def _constraint_marker_score(text: str) -> float:
    lowered = text.lower()
    marker_hits = sum(1.0 for marker in CONSTRAINT_MARKERS if marker in lowered)
    numeric_hits = float(len(re.findall(r"\d+", text)))
    structural_hits = 0.0
    if any(token in text for token in ("`", "{", "}", "[", "]", "(", ")", "->", "::")):
        structural_hits += 1.0
    if ":" in text:
        structural_hits += 0.5
    return marker_hits + min(numeric_hits, 3.0) * 0.35 + structural_hits


def _latest_user_index(conversation: ConversationRecord, prefix_turn_count: int) -> int | None:
    for idx in range(prefix_turn_count - 1, -1, -1):
        if conversation.turns[idx].role == "user":
            return idx
    return None


def _support_scores(
    *,
    conversation: ConversationRecord,
    prefix_turn_count: int,
    geometry_risk: np.ndarray,
    semantic_risk: np.ndarray,
) -> np.ndarray:
    if prefix_turn_count <= 0:
        return np.zeros(0, dtype=np.float32)
    user_bonus = np.asarray(
        [1.0 if conversation.turns[idx].role == "user" else 0.0 for idx in range(prefix_turn_count)],
        dtype=np.float32,
    )
    marker_scores = np.asarray(
        [_constraint_marker_score(conversation.turns[idx].content) for idx in range(prefix_turn_count)],
        dtype=np.float32,
    )
    latest_user = _latest_user_index(conversation, prefix_turn_count)
    latest_user_bonus = np.zeros(prefix_turn_count, dtype=np.float32)
    if latest_user is not None:
        latest_user_bonus[latest_user] = 1.0

    recency = np.linspace(0.0, 1.0, num=prefix_turn_count, dtype=np.float32)
    return _normalize(
        0.32 * _normalize(semantic_risk[:prefix_turn_count])
        + 0.18 * _normalize(geometry_risk[:prefix_turn_count])
        + 0.24 * user_bonus
        + 0.18 * _normalize(marker_scores)
        + 0.08 * recency
        + 0.40 * latest_user_bonus
    )


def _harm_proxy_scores(
    *,
    geometry_risk: np.ndarray,
    semantic_risk: np.ndarray,
    support_scores: np.ndarray,
    prefix_turn_count: int,
) -> np.ndarray:
    return _normalize(
        0.45 * _normalize(geometry_risk[:prefix_turn_count])
        + 0.30 * _normalize(semantic_risk[:prefix_turn_count])
        + 0.25 * _normalize(support_scores[:prefix_turn_count])
    )


def _semantic_support_proxy_scores(
    *,
    geometry_risk: np.ndarray,
    semantic_risk: np.ndarray,
    support_scores: np.ndarray,
    prefix_turn_count: int,
) -> np.ndarray:
    return _normalize(
        0.72 * _normalize(semantic_risk[:prefix_turn_count])
        + 0.20 * _normalize(support_scores[:prefix_turn_count])
        + 0.08 * _normalize(geometry_risk[:prefix_turn_count])
    )


def _budget_aware_semantic_params(budget_fraction: float) -> dict[str, float | int]:
    if budget_fraction <= 0.20 + 1e-8:
        return {"expansion_factor": 1.25, "segment_span": 2}
    if budget_fraction <= 0.35 + 1e-8:
        return {"expansion_factor": 1.60, "segment_span": 3}
    return {"expansion_factor": 2.10, "segment_span": 4}


def _requires_attention_features(feature_names: tuple[str, ...]) -> bool:
    return "attention_raw" in feature_names or "attention_sink_corrected" in feature_names


def _semantic_shortlist_candidate_mask(
    *,
    semantic_scores: np.ndarray,
    turn_costs: np.ndarray,
    budget_fraction: float,
    latest_user_index: int | None,
    expansion_factor: float,
) -> np.ndarray:
    return semantic_shortlist_mask(
        semantic_scores=semantic_scores,
        turn_costs=turn_costs,
        budget_fraction=budget_fraction,
        expansion_factor=expansion_factor,
        latest_user_index=latest_user_index,
    )


def _hybrid_query_support_semantic_scores(
    *,
    query_scores: np.ndarray,
    support_scores: np.ndarray,
    semantic_scores: np.ndarray,
    prefix_turn_count: int,
    include_query: bool = True,
    include_support: bool = True,
) -> np.ndarray:
    query_weight = 0.60 if include_query else 0.0
    support_weight = 0.25 if include_support else 0.0
    semantic_weight = 0.15
    return _normalize(
        query_weight * _normalize(query_scores[:prefix_turn_count])
        + support_weight * _normalize(support_scores[:prefix_turn_count])
        + semantic_weight * _normalize(semantic_scores[:prefix_turn_count])
    )


def _prefix_turn_costs(prefix_token_counts: np.ndarray) -> np.ndarray:
    if prefix_token_counts.size == 0:
        return np.zeros(0, dtype=np.int32)
    costs = prefix_token_counts.astype(np.int32).copy()
    costs[1:] = np.maximum(prefix_token_counts[1:] - prefix_token_counts[:-1], 1)
    costs[0] = max(int(prefix_token_counts[0]), 1)
    return costs


def _turn_attention_vectors(
    attention_summary: Any | None,
    *,
    prefix_turn_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if attention_summary is None:
        zeros = np.zeros(prefix_turn_count, dtype=np.float32)
        return zeros, zeros
    return (
        np.asarray(attention_summary.raw_turn_weights[:prefix_turn_count], dtype=np.float32),
        np.asarray(attention_summary.sink_corrected_turn_weights[:prefix_turn_count], dtype=np.float32),
    )


def _harm_predictor_feature_rows(
    *,
    conversation: ConversationRecord,
    prefix_turn_count: int,
    semantic_scores: np.ndarray,
    geometry_scores: np.ndarray,
    support_scores: np.ndarray,
    query_v2: Any,
    turn_costs: np.ndarray,
    latest_user_index: int | None,
    attention_raw: np.ndarray,
    attention_sink: np.ndarray,
    object_feature_rows: list[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(prefix_turn_count):
        row = {
            "candidate_type": "turn",
            "conversation_id": conversation.conversation_id,
            "semantic_score": float(semantic_scores[idx]),
            "geometry_score": float(geometry_scores[idx]),
            "support_score": float(support_scores[idx]),
            "query_geom_v2_risk": float(query_v2.risk[idx]),
            "query_geom_v2_curvature": float(query_v2.projected_curvature[idx]),
            "query_geom_v2_energy": float(query_v2.projected_subspace_energy[idx]),
            "query_geom_v2_alignment": float(query_v2.query_alignment[idx]),
            "query_geom_v2_local_projection": float(query_v2.local_projection[idx]),
            "segment_rank95": float(query_v2.segment_rank95[idx]),
            "segment_mean_step_norm": float(query_v2.segment_mean_step_norm[idx]),
            "segment_mean_stabilized_curvature": float(query_v2.segment_mean_stabilized_curvature[idx]),
            "role_user": float(conversation.turns[idx].role == "user"),
            "is_latest_user": float(latest_user_index is not None and idx == latest_user_index),
            "recency": float(idx / max(prefix_turn_count - 1, 1)),
            "token_cost": int(turn_costs[idx]),
            "constraint_score": float(_constraint_marker_score(conversation.turns[idx].content)),
            "attention_raw": float(attention_raw[idx]) if attention_raw.size else 0.0,
            "attention_sink_corrected": float(attention_sink[idx]) if attention_sink.size else 0.0,
        }
        if object_feature_rows is not None and idx < len(object_feature_rows):
            row.update(object_feature_rows[idx])
        rows.append(row)
    return rows


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
    skip_conversations: int = 0,
    conversation_ids_path: Path | None = None,
    output_dir: Path | None = None,
    segment_span: int = 2,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    target_turn_stride: int = 1,
    max_target_turns: int | None = None,
    max_turns_per_conversation: int | None = None,
    harm_predictor_path: Path | None = None,
) -> dict[str, Any]:
    spec = resolve_model_spec(model_key)
    if spec is None:
        raise RuntimeError(f"Unknown model key: {model_key}")

    conversations = _select_conversations(
        conversations=load_conversations_from_paths(input_paths),
        families=families,
        conversation_ids_path=conversation_ids_path,
        skip_conversations=skip_conversations,
        limit_conversations=limit_conversations,
    )
    if not conversations:
        raise RuntimeError("No conversations selected for Paper 3.")

    harm_predictor_bundle: HarmPredictorBundle | None = None
    harm_predictor_uses_attention = False
    if "semantic_harm_keep_compress_drop" in policies or "semantic_object_harm_keep_compress_drop" in policies:
        if harm_predictor_path is None:
            raise RuntimeError(
                "semantic_harm_keep_compress_drop and semantic_object_harm_keep_compress_drop require --harm-predictor-path."
            )
        harm_predictor_bundle = HarmPredictorBundle.load(harm_predictor_path)
        harm_predictor_uses_attention = _requires_attention_features(harm_predictor_bundle.feature_names)

    extractor = ConversationStateExtractor(
        model_name=spec.model_name,
        device=device,
        dtype=dtype,
        state_layer=state_layer,
    )
    conversation_target_turns = {
        conversation.conversation_id: _sample_target_turns(
            num_turns=min(len(conversation.turns), max_turns_per_conversation)
            if max_turns_per_conversation is not None
            else len(conversation.turns),
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
        f"target_turn_stride={target_turn_stride} max_target_turns={max_target_turns} "
        f"max_turns_per_conversation={max_turns_per_conversation} "
        f"skip_conversations={skip_conversations} "
        f"conversation_ids_path={conversation_ids_path}",
        flush=True,
    )

    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    completed_conversation_ids: list[str] = []
    if output_dir is not None:
        progress_path = output_dir / "progress.json"
        if progress_path.exists():
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_payload.get("status") == "complete" and (output_dir / "summary.json").exists():
                summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
                return {
                    "summary": summary,
                    "rows": _read_csv(output_dir / "evaluation_rows.csv"),
                    "behavior_rows": _read_csv(output_dir / "behavior_rows.csv"),
                }
            completed_conversation_ids = [
                str(item) for item in progress_payload.get("completed_conversation_ids", []) if str(item)
            ]
            evaluation_rows = _read_csv(output_dir / "evaluation_rows.partial.csv")
            behavior_rows = _read_csv(output_dir / "behavior_rows.partial.csv")
    completed_conversation_id_set = set(completed_conversation_ids)
    if completed_conversation_id_set:
        conversations = [
            conversation for conversation in conversations
            if conversation.conversation_id not in completed_conversation_id_set
        ]
        if not conversations:
            if output_dir is None:
                raise RuntimeError("All selected conversations already completed, but no output_dir was provided.")
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
                "max_turns_per_conversation": max_turns_per_conversation,
                "skip_conversations": skip_conversations,
                "conversation_ids_path": str(conversation_ids_path) if conversation_ids_path is not None else None,
                "num_conversations": len(completed_conversation_ids),
                "num_evaluations": len(evaluation_rows),
                "num_behavior_evaluations": len(behavior_rows),
                "aggregate": _aggregate_rows(evaluation_rows),
                "behavior_aggregate": _aggregate_behavior_rows(behavior_rows),
                "improvement_vs_uniform": _improvement_vs_uniform(evaluation_rows),
                "behavior_improvement_vs_uniform": _behavior_improvement_vs_uniform(behavior_rows),
            }
            _write_json(output_dir / "summary.json", summary)
            (output_dir / "report.md").write_text(_format_report(summary), encoding="utf-8")
            _write_json(
                output_dir / "progress.json",
                {
                    "status": "complete",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "model_key": model_key,
                    "num_conversations_total": len(completed_conversation_ids),
                    "num_conversations_completed": len(completed_conversation_ids),
                    "completed_conversation_ids": completed_conversation_ids,
                    "num_evaluation_rows": len(evaluation_rows),
                    "num_behavior_rows": len(behavior_rows),
                    "summary_path": str(output_dir / "summary.json"),
                    "report_path": str(output_dir / "report.md"),
                },
            )
            return {"summary": summary, "rows": evaluation_rows, "behavior_rows": behavior_rows}
        print(
            f"[{model_key}] resuming from partial outputs: completed={len(completed_conversation_ids)} "
            f"remaining={len(conversations)} rows={len(evaluation_rows)} behavior_rows={len(behavior_rows)}",
            flush=True,
        )
    conversation_iterator = _progress(
        enumerate(conversations, start=1),
        total=len(conversations),
        desc=f"{model_key} conversations",
    )
    for conversation_index, conversation in conversation_iterator:
        print(
            f"[{model_key}] starting conversation {conversation_index}/{len(conversations)} "
            f"id={conversation.conversation_id} turns={len(conversation.turns)} "
            f"sampled_targets={len(conversation_target_turns[conversation.conversation_id])}",
            flush=True,
        )
        full_batch = extractor.extract_conversation(
            conversation,
            max_turns=max_turns_per_conversation,
            max_input_tokens=max_input_tokens,
            progress_label=f"study {model_key} {conversation.conversation_id}",
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
            query_conditioned = query_conditioned_turn_risk(
                full_batch.states,
                target_turn,
                ambient_geometry=geometry_risk,
            )
            query_geometry_risk = query_conditioned.risk[:prefix_turn_count]
            query_conditioned_v2 = query_conditioned_turn_risk_v2(
                full_batch.states,
                target_turn,
                segment_span=max(segment_span, 3),
                ambient_geometry=geometry_risk,
            )
            query_geometry_risk_v2 = query_conditioned_v2.risk[:prefix_turn_count]
            support_scores = _support_scores(
                conversation=conversation,
                prefix_turn_count=prefix_turn_count,
                geometry_risk=geometry_risk,
                semantic_risk=semantic_risk,
            )
            harm_proxy = _harm_proxy_scores(
                geometry_risk=geometry_risk,
                semantic_risk=semantic_risk,
                support_scores=support_scores,
                prefix_turn_count=prefix_turn_count,
            )
            semantic_support_proxy = _semantic_support_proxy_scores(
                geometry_risk=geometry_risk,
                semantic_risk=semantic_risk,
                support_scores=support_scores,
                prefix_turn_count=prefix_turn_count,
            )
            latest_user_index = _latest_user_index(conversation, prefix_turn_count)
            constraint_scores = np.asarray(
                [_constraint_marker_score(conversation.turns[idx].content) for idx in range(prefix_turn_count)],
                dtype=np.float32,
            )
            full_messages = _policy_messages(
                conversation=conversation,
                target_turn=target_turn,
                retained_prior_indices=list(range(prefix_turn_count)),
            )
            attention_raw = np.zeros(prefix_turn_count, dtype=np.float32)
            attention_sink = np.zeros(prefix_turn_count, dtype=np.float32)
            if harm_predictor_bundle is not None and harm_predictor_uses_attention:
                full_prompt_score = extractor.score_messages(
                    full_messages,
                    max_input_tokens=max_input_tokens,
                    return_attention_summary=True,
                    cumulative_turn_token_counts=full_batch.token_counts[:prefix_turn_count],
                )
                attention_raw, attention_sink = _turn_attention_vectors(
                    full_prompt_score.attention_summary,
                    prefix_turn_count=prefix_turn_count,
                )
            is_behavior_turn = (
                conversation.turns[target_turn].role == "user"
                and target_turn + 1 < len(conversation.turns)
                and conversation.turns[target_turn + 1].role == "assistant"
            )
            full_behavior_score = None
            if is_behavior_turn:
                full_behavior_score = extractor.score_assistant_response(
                    full_messages,
                    conversation.turns[target_turn + 1].content,
                    max_input_tokens=max_input_tokens,
                )
            for budget in budgets:
                semantic_budget_params = _budget_aware_semantic_params(budget)
                shortlist_mask = _semantic_shortlist_candidate_mask(
                    semantic_scores=semantic_risk,
                    turn_costs=prefix_turn_costs,
                    budget_fraction=budget,
                    latest_user_index=latest_user_index,
                    expansion_factor=float(semantic_budget_params["expansion_factor"]),
                )
                hybrid_query_support = _hybrid_query_support_semantic_scores(
                    query_scores=query_geometry_risk_v2,
                    support_scores=support_scores,
                    semantic_scores=semantic_risk,
                    prefix_turn_count=prefix_turn_count,
                    include_query=True,
                    include_support=True,
                )
                hybrid_ambient_support = _hybrid_query_support_semantic_scores(
                    query_scores=geometry_risk[:prefix_turn_count],
                    support_scores=support_scores,
                    semantic_scores=semantic_risk,
                    prefix_turn_count=prefix_turn_count,
                    include_query=True,
                    include_support=True,
                )
                hybrid_no_query = _hybrid_query_support_semantic_scores(
                    query_scores=query_geometry_risk_v2,
                    support_scores=support_scores,
                    semantic_scores=semantic_risk,
                    prefix_turn_count=prefix_turn_count,
                    include_query=False,
                    include_support=True,
                )
                hybrid_no_support = _hybrid_query_support_semantic_scores(
                    query_scores=query_geometry_risk_v2,
                    support_scores=support_scores,
                    semantic_scores=semantic_risk,
                    prefix_turn_count=prefix_turn_count,
                    include_query=True,
                    include_support=False,
                )
                object_bundle = build_semantic_object_bundle(
                    conversation=conversation,
                    prefix_turn_count=prefix_turn_count,
                    semantic_scores=semantic_risk,
                    support_scores=support_scores,
                    query_scores=query_geometry_risk_v2,
                    candidate_mask=shortlist_mask,
                    constraint_scores=constraint_scores,
                    gap_tolerance=1,
                )
                predictor_turn_rows: list[dict[str, Any]] | None = None
                predicted_harm_scores = None
                if harm_predictor_bundle is not None:
                    object_feature_rows = object_bundle.per_turn_feature_rows(prefix_turn_count)
                    predictor_turn_rows = _harm_predictor_feature_rows(
                        conversation=conversation,
                        prefix_turn_count=prefix_turn_count,
                        semantic_scores=semantic_risk,
                        geometry_scores=geometry_risk[:prefix_turn_count],
                        support_scores=support_scores,
                        query_v2=query_conditioned_v2,
                        turn_costs=prefix_turn_costs,
                        latest_user_index=latest_user_index,
                        attention_raw=attention_raw,
                        attention_sink=attention_sink,
                        object_feature_rows=object_feature_rows,
                    )
                if harm_predictor_bundle is not None and predictor_turn_rows is not None:
                    predicted_harm_scores = harm_predictor_bundle.predict_rows(predictor_turn_rows)
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
                    elif policy_name == "query_conditioned_geometry":
                        selection = select_turns(
                            policy_name=policy_name,
                            risk_scores=query_geometry_risk,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                    elif policy_name == "query_conditioned_geometry_v2":
                        selection = select_turns(
                            policy_name=policy_name,
                            risk_scores=query_geometry_risk_v2,
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
                    elif policy_name == "query_conditioned_geometry_keep_compress_drop":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=query_geometry_risk,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=max(segment_span, 3),
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "query_conditioned_geometry_keep_compress_drop_v2":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=query_geometry_risk_v2,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_keep_compress_drop":
                        selection = select_sparse_segment_memory(
                            risk_scores=semantic_risk,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=segment_span,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "support_aware_semantic_keep_compress_drop":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=semantic_support_proxy,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=max(segment_span, 3),
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "budget_aware_semantic_keep_compress_drop":
                        selection = select_semantic_filtered_sparse_segment_memory(
                            geometry_like_scores=semantic_support_proxy,
                            support_scores=support_scores,
                            semantic_scores=semantic_risk,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                            latest_user_index=latest_user_index,
                            expansion_factor=float(semantic_budget_params["expansion_factor"]),
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_query_conditioned_geometry_keep_compress_drop":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=hybrid_query_support,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                            candidate_mask=shortlist_mask,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_ambient_geometry_keep_compress_drop":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=hybrid_ambient_support,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                            candidate_mask=shortlist_mask,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_query_conditioned_geometry_keep_compress_drop_no_query":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=hybrid_no_query,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                            candidate_mask=shortlist_mask,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_query_conditioned_geometry_keep_compress_drop_no_support":
                        selection = select_masked_sparse_segment_memory(
                            risk_scores=hybrid_no_support,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                            candidate_mask=shortlist_mask,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_harm_keep_compress_drop":
                        if predicted_harm_scores is None:
                            raise RuntimeError("semantic_harm_keep_compress_drop requires a loaded harm predictor.")
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=predicted_harm_scores[:prefix_turn_count],
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=int(semantic_budget_params["segment_span"]),
                            candidate_mask=shortlist_mask,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_object_keep_compress_drop":
                        selection = select_semantic_object_sparse_memory(
                            bundle=object_bundle,
                            risk_scores=semantic_support_proxy,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_object_harm_keep_compress_drop":
                        if predicted_harm_scores is None:
                            raise RuntimeError("semantic_object_harm_keep_compress_drop requires a loaded harm predictor.")
                        selection = select_semantic_object_sparse_memory(
                            bundle=object_bundle,
                            risk_scores=predicted_harm_scores[:prefix_turn_count],
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "support_aware_geometry_keep_compress_drop":
                        selection = select_support_aware_sparse_segment_memory(
                            risk_scores=harm_proxy,
                            support_scores=support_scores,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=max(segment_span, 3),
                        )
                        memory_objects = selection.memory_objects
                    elif policy_name == "semantic_filtered_geometry_keep_compress_drop":
                        selection = select_semantic_filtered_sparse_segment_memory(
                            geometry_like_scores=harm_proxy,
                            support_scores=support_scores,
                            semantic_scores=semantic_risk,
                            turn_costs=prefix_turn_costs,
                            prefix_turn_count=prefix_turn_count,
                            budget_fraction=budget,
                            recent_window=recent_window,
                            segment_span=max(segment_span, 3),
                            latest_user_index=latest_user_index,
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
        completed_conversation_ids.append(conversation.conversation_id)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(output_dir / "evaluation_rows.partial.csv", evaluation_rows)
            if behavior_rows:
                _write_csv(output_dir / "behavior_rows.partial.csv", behavior_rows)
            _write_json(
                output_dir / "progress.json",
                {
                    "status": "running",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "model_key": model_key,
                    "num_conversations_total": len(conversations),
                    "num_conversations_completed": len(completed_conversation_ids),
                    "completed_conversation_ids": completed_conversation_ids,
                    "num_evaluation_rows": len(evaluation_rows),
                    "num_behavior_rows": len(behavior_rows),
                    "target_turn_stride": target_turn_stride,
                    "max_target_turns": max_target_turns,
                    "max_turns_per_conversation": max_turns_per_conversation,
                    "skip_conversations": skip_conversations,
                    "conversation_ids_path": str(conversation_ids_path) if conversation_ids_path is not None else None,
                },
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
        "max_turns_per_conversation": max_turns_per_conversation,
        "skip_conversations": skip_conversations,
        "conversation_ids_path": str(conversation_ids_path) if conversation_ids_path is not None else None,
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
        _write_csv(output_dir / "evaluation_rows.csv", evaluation_rows)
        if behavior_rows:
            _write_csv(output_dir / "behavior_rows.csv", behavior_rows)
        _write_json(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(_format_report(summary), encoding="utf-8")
        _write_json(
            output_dir / "progress.json",
            {
                "status": "complete",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "model_key": model_key,
                "num_conversations_total": len(conversations),
                "num_conversations_completed": len(completed_conversation_ids),
                "completed_conversation_ids": completed_conversation_ids,
                "num_evaluation_rows": len(evaluation_rows),
                "num_behavior_rows": len(behavior_rows),
                "summary_path": str(output_dir / "summary.json"),
                "report_path": str(output_dir / "report.md"),
            },
        )

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
        f"- Max turns / conversation: {summary.get('max_turns_per_conversation')}",
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
    parser.add_argument("--skip-conversations", type=int, default=0)
    parser.add_argument("--conversation-ids-path", type=Path, default=None)
    parser.add_argument("--segment-span", type=int, default=2)
    parser.add_argument("--target-turn-stride", type=int, default=1)
    parser.add_argument("--max-target-turns", type=int, default=None)
    parser.add_argument("--max-turns-per-conversation", type=int, default=None)
    parser.add_argument("--harm-predictor-path", type=Path, default=None)
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
        skip_conversations=args.skip_conversations,
        conversation_ids_path=args.conversation_ids_path,
        output_dir=args.output_root / args.run_name,
        segment_span=args.segment_span,
        policies=_parse_policies(args.policies),
        target_turn_stride=args.target_turn_stride,
        max_target_turns=args.max_target_turns,
        max_turns_per_conversation=args.max_turns_per_conversation,
        harm_predictor_path=args.harm_predictor_path,
    )
    print(f"Wrote Paper 3 outputs to {args.output_root / args.run_name}")
    print(
        f"Completed {result['summary']['num_evaluations']} evaluations across "
        f"{result['summary']['num_conversations']} conversations."
    )


if __name__ == "__main__":
    main()
