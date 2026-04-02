from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from paper1_geometry.analysis import analyze_trajectory
from paper1_geometry.boundary_features import lexical_shift_scores
from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths
from paper1_geometry.modeling import ConversationStateExtractor, resolve_model_spec
from paper2_memory.policies import turn_geometry_risk, turn_semantic_risk

from .harm_predictor import scalarize_harm
from .memory_objects import build_semantic_object_bundle
from .policies import support_aware_segment_retained_indices, semantic_shortlist_mask
from .query_geometry import query_conditioned_turn_risk_v2
from .run_paper3 import (
    _constraint_marker_score,
    _harm_proxy_scores,
    _latest_user_index,
    _parse_families,
    _parse_float_list,
    _policy_messages,
    _prefix_turn_costs,
    _progress,
    _semantic_support_proxy_scores,
    _support_scores,
    _topk_indices,
    _kl_divergence,
    _sample_target_turns,
)
from .stats import (
    bootstrap_mean_ci,
    collapse_rows_by_keys,
    kendall_tau,
    ndcg_score,
    paired_signflip_test,
    spearman,
    topk_recall,
    zscore,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "paper3" / "harm_oracle"
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "benchmarks" / "msc_valid_normalized.jsonl"
DEFAULT_FEATURE_KEYS: tuple[str, ...] = (
    "semantic_score",
    "geometry_score",
    "support_score",
    "query_geom_v2_risk",
    "combined_structural_score",
    "object_memory_score",
)
SEGMENT_SPANS: tuple[int, ...] = (2, 3, 4)


def _budget_segment_span(budget_fraction: float) -> int:
    if budget_fraction <= 0.20 + 1e-8:
        return 2
    if budget_fraction <= 0.35 + 1e-8:
        return 3
    return 4


def _candidate_rng(conversation_id: str, target_turn: int, budget_fraction: float) -> np.random.Generator:
    seed_source = f"{conversation_id}:{target_turn}:{budget_fraction:.2f}"
    digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def _ordered_unique(items: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _random_indices(
    *,
    prefix_turn_count: int,
    recent_start: int,
    already_selected: set[int],
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = [idx for idx in range(0, recent_start) if idx not in already_selected]
    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    sampled = rng.choice(np.asarray(candidates, dtype=np.int32), size=count, replace=False)
    return [int(item) for item in sampled.tolist()]


def _turn_attention_features(
    attention_summary: Any | None,
    *,
    prefix_turn_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if attention_summary is None:
        empty = np.zeros(prefix_turn_count, dtype=np.float32)
        return empty, empty
    return (
        np.asarray(attention_summary.raw_turn_weights[:prefix_turn_count], dtype=np.float32),
        np.asarray(attention_summary.sink_corrected_turn_weights[:prefix_turn_count], dtype=np.float32),
    )


def _segment_feature_mean(values: np.ndarray, start: int, end_inclusive: int) -> float:
    if values.size == 0:
        return 0.0
    end = min(end_inclusive + 1, values.size)
    start = min(max(start, 0), end)
    if end <= start:
        return 0.0
    return float(np.mean(values[start:end]))


def _candidate_feature_payload(
    *,
    conversation: ConversationRecord,
    prefix_turn_count: int,
    candidate_start: int,
    candidate_end: int,
    candidate_type: str,
    semantic_risk: np.ndarray,
    geometry_risk: np.ndarray,
    support_scores: np.ndarray,
    semantic_support_proxy: np.ndarray,
    query_v2: Any,
    turn_costs: np.ndarray,
    attention_raw: np.ndarray,
    attention_sink: np.ndarray,
    latest_user_index: int | None,
    shortlist_flags: dict[str, set[int]],
    object_feature_rows: list[dict[str, float]],
) -> dict[str, Any]:
    turn_slice = slice(candidate_start, candidate_end + 1)
    is_turn = candidate_type == "turn"
    role = conversation.turns[candidate_start].role if is_turn else "segment"
    role_user = float(role == "user") if is_turn else float(
        np.mean([1.0 if conversation.turns[idx].role == "user" else 0.0 for idx in range(candidate_start, candidate_end + 1)])
    )
    token_cost = int(np.sum(turn_costs[turn_slice]))
    recency = float(candidate_end / max(prefix_turn_count - 1, 1))
    payload = {
        "role": role,
        "role_user": role_user,
        "is_latest_user": float(
            latest_user_index is not None and candidate_start <= latest_user_index <= candidate_end
        ),
        "recency": recency,
        "token_cost": token_cost,
        "semantic_score": _segment_feature_mean(semantic_risk, candidate_start, candidate_end),
        "geometry_score": _segment_feature_mean(geometry_risk, candidate_start, candidate_end),
        "support_score": _segment_feature_mean(support_scores, candidate_start, candidate_end),
        "semantic_support_proxy": _segment_feature_mean(semantic_support_proxy, candidate_start, candidate_end),
        "query_geom_v2_risk": _segment_feature_mean(query_v2.risk, candidate_start, candidate_end),
        "query_geom_v2_curvature": _segment_feature_mean(query_v2.projected_curvature, candidate_start, candidate_end),
        "query_geom_v2_energy": _segment_feature_mean(query_v2.projected_subspace_energy, candidate_start, candidate_end),
        "query_geom_v2_alignment": _segment_feature_mean(query_v2.query_alignment, candidate_start, candidate_end),
        "query_geom_v2_local_projection": _segment_feature_mean(query_v2.local_projection, candidate_start, candidate_end),
        "segment_rank95": _segment_feature_mean(query_v2.segment_rank95, candidate_start, candidate_end),
        "segment_mean_step_norm": _segment_feature_mean(query_v2.segment_mean_step_norm, candidate_start, candidate_end),
        "segment_mean_stabilized_curvature": _segment_feature_mean(query_v2.segment_mean_stabilized_curvature, candidate_start, candidate_end),
        "constraint_score": float(
            np.mean([
                _constraint_marker_score(conversation.turns[idx].content)
                for idx in range(candidate_start, candidate_end + 1)
            ])
        ),
        "attention_raw": float(np.sum(attention_raw[turn_slice])) if attention_raw.size else 0.0,
        "attention_sink_corrected": float(np.sum(attention_sink[turn_slice])) if attention_sink.size else 0.0,
        "in_semantic_topk": float(any(idx in shortlist_flags["semantic"] for idx in range(candidate_start, candidate_end + 1))),
        "in_support_topk": float(any(idx in shortlist_flags["support"] for idx in range(candidate_start, candidate_end + 1))),
        "in_query_topk": float(any(idx in shortlist_flags["query"] for idx in range(candidate_start, candidate_end + 1))),
        "combined_structural_score": float(
            0.60 * _segment_feature_mean(query_v2.risk, candidate_start, candidate_end)
            + 0.25 * _segment_feature_mean(support_scores, candidate_start, candidate_end)
            + 0.15 * _segment_feature_mean(semantic_risk, candidate_start, candidate_end)
        ),
    }
    for feature_key in (
        "object_size",
        "object_recency",
        "object_memory_score",
        "is_object_anchor",
        "is_object_freshest",
        "object_type_persona",
        "object_type_event",
        "object_type_constraint",
        "object_type_update",
        "object_type_generic",
    ):
        feature_values = [
            float(object_feature_rows[idx].get(feature_key, 0.0))
            for idx in range(candidate_start, candidate_end + 1)
            if idx < len(object_feature_rows)
        ]
        payload[feature_key] = float(np.mean(feature_values)) if feature_values else 0.0
    return payload


def _oracle_ablation_rows_for_target(
    *,
    conversation: ConversationRecord,
    benchmark_name: str,
    model_key: str,
    extractor: ConversationStateExtractor,
    full_batch: Any,
    full_logits: np.ndarray,
    full_behavior_score: Any | None,
    geometry_risk: np.ndarray,
    target_turn: int,
    budget_fraction: float,
    recent_window: int,
    max_input_tokens: int,
    semantic_risk: np.ndarray,
    support_scores: np.ndarray,
    semantic_support_proxy: np.ndarray,
    query_v2: Any,
    attention_raw: np.ndarray,
    attention_sink: np.ndarray,
) -> list[dict[str, Any]]:
    prefix_turn_count = target_turn
    if prefix_turn_count <= 0:
        return []
    turn_costs = _prefix_turn_costs(full_batch.token_counts)[:prefix_turn_count]
    latest_user_index = _latest_user_index(conversation, prefix_turn_count)
    recent_start = max(prefix_turn_count - recent_window, 0)
    eligible_prefix_count = recent_start
    if eligible_prefix_count <= 0:
        return []

    top_semantic = _topk_indices(semantic_risk[:eligible_prefix_count], 8)
    top_support = _topk_indices(semantic_support_proxy[:eligible_prefix_count], 8)
    top_query = _topk_indices(query_v2.risk[:eligible_prefix_count], 8)
    rng = _candidate_rng(conversation.conversation_id, target_turn, budget_fraction)
    random_turns = _random_indices(
        prefix_turn_count=prefix_turn_count,
        recent_start=recent_start,
        already_selected=set(top_semantic + top_support + top_query),
        count=8,
        rng=rng,
    )
    turn_candidates = _ordered_unique(top_semantic + top_support + top_query + random_turns)[:24]
    shortlist_flags = {
        "semantic": set(top_semantic),
        "support": set(top_support),
        "query": set(top_query),
    }
    shortlist_mask = semantic_shortlist_mask(
        semantic_scores=semantic_risk[:prefix_turn_count],
        turn_costs=turn_costs,
        budget_fraction=budget_fraction,
        expansion_factor=2.0,
        latest_user_index=latest_user_index,
    )
    constraint_scores = np.asarray(
        [_constraint_marker_score(conversation.turns[idx].content) for idx in range(prefix_turn_count)],
        dtype=np.float32,
    )
    object_bundle = build_semantic_object_bundle(
        conversation=conversation,
        prefix_turn_count=prefix_turn_count,
        semantic_scores=semantic_risk,
        support_scores=support_scores,
        query_scores=query_v2.risk[:prefix_turn_count],
        candidate_mask=shortlist_mask,
        constraint_scores=constraint_scores,
        gap_tolerance=1,
    )
    object_feature_rows = object_bundle.per_turn_feature_rows(prefix_turn_count)

    rows: list[dict[str, Any]] = []
    for turn_index in turn_candidates:
        retained_prior = [idx for idx in range(prefix_turn_count) if idx != turn_index]
        messages = _policy_messages(
            conversation=conversation,
            target_turn=target_turn,
            retained_prior_indices=retained_prior,
        )
        compressed = extractor.score_messages(messages, max_input_tokens=max_input_tokens)
        behavior_delta = None
        if full_behavior_score is not None:
            behavior_score = extractor.score_assistant_response(
                messages,
                conversation.turns[target_turn + 1].content,
                max_input_tokens=max_input_tokens,
            )
            behavior_delta = behavior_score.avg_neg_logprob - full_behavior_score.avg_neg_logprob
        feature_payload = _candidate_feature_payload(
            conversation=conversation,
            prefix_turn_count=prefix_turn_count,
            candidate_start=turn_index,
            candidate_end=turn_index,
            candidate_type="turn",
            semantic_risk=semantic_risk,
            geometry_risk=geometry_risk[:prefix_turn_count],
            support_scores=support_scores,
            semantic_support_proxy=semantic_support_proxy,
            query_v2=query_v2,
            turn_costs=turn_costs,
            attention_raw=attention_raw,
            attention_sink=attention_sink,
            latest_user_index=latest_user_index,
            shortlist_flags=shortlist_flags,
            object_feature_rows=object_feature_rows,
        )
        rows.append(
            {
                "benchmark": benchmark_name,
                "family": conversation.family,
                "model_key": model_key,
                "conversation_id": conversation.conversation_id,
                "target_turn": target_turn,
                "budget_fraction": float(budget_fraction),
                "candidate_type": "turn",
                "candidate_start": turn_index,
                "candidate_end": turn_index,
                "action": "drop_turn",
                **feature_payload,
                "delta_logit_l2": float(np.linalg.norm(full_logits - compressed.logits)),
                "delta_answer_avg_neg_logprob_delta": float(behavior_delta) if behavior_delta is not None else 0.0,
                "has_behavior_label": 1 if behavior_delta is not None else 0,
            }
        )

    segment_starts = _ordered_unique(turn_candidates + ([latest_user_index] if latest_user_index is not None and latest_user_index < eligible_prefix_count else []))
    seen_segments: set[tuple[int, int]] = set()
    for start in segment_starts:
        for span in SEGMENT_SPANS:
            end = min(start + span - 1, eligible_prefix_count - 1)
            if end <= start:
                continue
            key = (start, end)
            if key in seen_segments:
                continue
            seen_segments.add(key)
            feature_payload = _candidate_feature_payload(
                conversation=conversation,
                prefix_turn_count=prefix_turn_count,
                candidate_start=start,
                candidate_end=end,
                candidate_type="segment",
                semantic_risk=semantic_risk,
                geometry_risk=geometry_risk[:prefix_turn_count],
                support_scores=support_scores,
                semantic_support_proxy=semantic_support_proxy,
                query_v2=query_v2,
                turn_costs=turn_costs,
                attention_raw=attention_raw,
                attention_sink=attention_sink,
                latest_user_index=latest_user_index,
                shortlist_flags=shortlist_flags,
                object_feature_rows=object_feature_rows,
            )

            for action in ("evict_segment", "compress_segment"):
                if action == "evict_segment":
                    retained_prior = [idx for idx in range(prefix_turn_count) if idx < start or idx > end]
                else:
                    retained_inside = support_aware_segment_retained_indices(
                        start=start,
                        end=end + 1,
                        risk_scores=semantic_support_proxy,
                        support_scores=support_scores,
                        candidate_mask=None,
                    )
                    retained_prior = [
                        idx for idx in range(prefix_turn_count)
                        if idx < start or idx > end or idx in retained_inside
                    ]
                messages = _policy_messages(
                    conversation=conversation,
                    target_turn=target_turn,
                    retained_prior_indices=retained_prior,
                )
                compressed = extractor.score_messages(messages, max_input_tokens=max_input_tokens)
                behavior_delta = None
                if full_behavior_score is not None:
                    behavior_score = extractor.score_assistant_response(
                        messages,
                        conversation.turns[target_turn + 1].content,
                        max_input_tokens=max_input_tokens,
                    )
                    behavior_delta = behavior_score.avg_neg_logprob - full_behavior_score.avg_neg_logprob
                rows.append(
                    {
                        "benchmark": benchmark_name,
                        "family": conversation.family,
                        "model_key": model_key,
                        "conversation_id": conversation.conversation_id,
                        "target_turn": target_turn,
                        "budget_fraction": float(budget_fraction),
                        "candidate_type": "segment",
                        "candidate_start": start,
                        "candidate_end": end,
                        "action": action,
                        **feature_payload,
                        "delta_logit_l2": float(np.linalg.norm(full_logits - compressed.logits)),
                        "delta_answer_avg_neg_logprob_delta": float(behavior_delta) if behavior_delta is not None else 0.0,
                        "has_behavior_label": 1 if behavior_delta is not None else 0,
                    }
                )

    return rows


def _apply_harm_scalar(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (
            str(row["benchmark"]),
            str(row["model_key"]),
            f"{float(row['budget_fraction']):.2f}",
            str(row["candidate_type"]),
        )
        grouped[key].append(idx)

    for indices in grouped.values():
        logit_values = np.asarray([float(rows[idx]["delta_logit_l2"]) for idx in indices], dtype=np.float32)
        behavior_values = np.asarray([float(rows[idx]["delta_answer_avg_neg_logprob_delta"]) for idx in indices], dtype=np.float32)
        masks = np.asarray([1.0 if int(rows[idx]["has_behavior_label"]) else 0.0 for idx in indices], dtype=np.float32)
        if np.any(masks > 0.0):
            score = scalarize_harm(
                logit_values=logit_values,
                behavior_values=np.where(masks > 0.0, behavior_values, 0.0),
            )
        else:
            score = zscore(logit_values)
        for local_idx, row_idx in enumerate(indices):
            rows[row_idx]["harm_scalar"] = float(score[local_idx])


def _turn_row_views(
    rows: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str, str, str], list[dict[str, Any]]]]:
    turn_rows = [row for row in rows if str(row["candidate_type"]) == "turn"]
    overall_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    shortlist_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        key = (
            str(row["benchmark"]),
            str(row["model_key"]),
            f"{float(row['budget_fraction']):.2f}",
            f"{row['conversation_id']}::{int(row['target_turn'])}",
        )
        overall_groups[key].append(row)
        if int(row.get("in_semantic_topk", 0)) == 1:
            shortlist_groups[key].append(row)
    return {"overall": overall_groups, "semantic_shortlist": shortlist_groups}


def _ranking_summary(
    rows: list[dict[str, Any]],
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
) -> dict[str, Any]:
    def _group_metrics(grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]]) -> dict[str, Any]:
        feature_metrics: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (benchmark, model_key, budget_key, group_id), group_rows in grouped.items():
            if len(group_rows) < 2:
                continue
            oracle = np.asarray([float(row["harm_scalar"]) for row in group_rows], dtype=np.float32)
            conversation_id = group_id.split("::", 1)[0]
            for feature_key in feature_keys:
                predicted = np.asarray([float(row.get(feature_key, 0.0) or 0.0) for row in group_rows], dtype=np.float32)
                feature_metrics[(benchmark, model_key, budget_key)][feature_key].append(
                    {
                        "spearman": spearman(predicted, oracle),
                        "kendall_tau": kendall_tau(predicted, oracle),
                        "top5_recall": topk_recall(predicted, oracle, k=5),
                        "ndcg5": ndcg_score(predicted, oracle, k=5),
                        "conversation_id": conversation_id,
                    }
                )

        summary: dict[str, Any] = {}
        rng = np.random.default_rng(20260413)
        for (benchmark, model_key, budget_key), metric_map in feature_metrics.items():
            benchmark_payload = summary.setdefault(benchmark, {})
            model_payload = benchmark_payload.setdefault(model_key, {})
            budget_payload = model_payload.setdefault(budget_key, {})
            semantic_rows = metric_map.get("semantic_score", [])
            for feature_key, rows_for_feature in metric_map.items():
                payload: dict[str, Any] = {}
                for metric_name in ("spearman", "kendall_tau", "top5_recall", "ndcg5"):
                    metric_values = [float(item[metric_name]) for item in rows_for_feature]
                    conversation_metric_values = [
                        float(value[metric_name])
                        for value in collapse_rows_by_keys(
                            [
                                {
                                    metric_name: float(item[metric_name]),
                                    "conversation_id": item["conversation_id"],
                                }
                                for item in rows_for_feature
                            ],
                            metric_keys=[metric_name],
                            group_keys=["conversation_id"],
                        )
                    ]
                    payload[metric_name] = {
                        "row_level": bootstrap_mean_ci(metric_values, rng=rng),
                        "conversation_level": bootstrap_mean_ci(conversation_metric_values, rng=rng),
                    }
                if feature_key != "semantic_score" and semantic_rows:
                    comparison_payload: dict[str, Any] = {}
                    for metric_name in ("spearman", "kendall_tau", "top5_recall", "ndcg5"):
                        semantic_by_conv = {item["conversation_id"]: float(item[metric_name]) for item in semantic_rows}
                        feature_by_conv = {item["conversation_id"]: float(item[metric_name]) for item in rows_for_feature}
                        common = sorted(set(semantic_by_conv) & set(feature_by_conv))
                        deltas = [feature_by_conv[item] - semantic_by_conv[item] for item in common]
                        delta_array = np.asarray(deltas, dtype=np.float64)
                        comparison_payload[metric_name] = {
                            "row_level": {
                                **bootstrap_mean_ci(deltas, rng=rng),
                                "p_value": paired_signflip_test(delta_array, rng=rng),
                            },
                            "conversation_level": {
                                **bootstrap_mean_ci(deltas, rng=rng),
                                "p_value": paired_signflip_test(delta_array, rng=rng),
                            },
                        }
                    payload["vs_semantic"] = comparison_payload
                budget_payload[feature_key] = payload
        return summary

    turn_row_views = _turn_row_views(rows)
    return {view_name: _group_metrics(grouped) for view_name, grouped in turn_row_views.items()}


def _topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.zeros(0, dtype=np.int32)
    k = min(k, scores.size)
    order = np.argsort(-scores, kind="stable")
    return order[:k].astype(np.int32)


def _oracle_topk_summary(
    rows: list[dict[str, Any]],
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    def _group_metrics(grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]]) -> dict[str, Any]:
        feature_metrics: dict[tuple[str, str, str], dict[str, list[dict[str, float | str]]]] = defaultdict(lambda: defaultdict(list))
        for (benchmark, model_key, budget_key, group_id), group_rows in grouped.items():
            if len(group_rows) < 2:
                continue
            oracle = np.asarray([float(row["harm_scalar"]) for row in group_rows], dtype=np.float32)
            conversation_id = group_id.split("::", 1)[0]
            for k in k_values:
                k_eff = min(k, len(group_rows))
                if k_eff <= 0:
                    continue
                oracle_top = _topk_indices_from_scores(oracle, k_eff)
                oracle_top_set = {int(item) for item in oracle_top.tolist()}
                oracle_top_mean_harm = float(np.mean(oracle[oracle_top]))
                for feature_key in feature_keys:
                    predicted = np.asarray([float(row.get(feature_key, 0.0) or 0.0) for row in group_rows], dtype=np.float32)
                    predicted_top = _topk_indices_from_scores(predicted, k_eff)
                    predicted_top_set = {int(item) for item in predicted_top.tolist()}
                    predicted_top_mean_harm = float(np.mean(oracle[predicted_top]))
                    bucket = feature_metrics[(benchmark, model_key, budget_key)][feature_key]
                    if not bucket or str(bucket[-1].get("conversation_id")) != conversation_id:
                        bucket.append({"conversation_id": conversation_id})
                    bucket[-1].update(
                        {
                            f"top{k}_recall": float(len(oracle_top_set & predicted_top_set) / max(k_eff, 1)),
                            f"top{k}_predicted_mean_harm": predicted_top_mean_harm,
                            f"top{k}_oracle_mean_harm": oracle_top_mean_harm,
                            f"top{k}_regret": oracle_top_mean_harm - predicted_top_mean_harm,
                        }
                    )

        summary: dict[str, Any] = {}
        rng = np.random.default_rng(20260414)
        for (benchmark, model_key, budget_key), metric_map in feature_metrics.items():
            benchmark_payload = summary.setdefault(benchmark, {})
            model_payload = benchmark_payload.setdefault(model_key, {})
            budget_payload = model_payload.setdefault(budget_key, {})
            semantic_rows = metric_map.get("semantic_score", [])
            for feature_key, rows_for_feature in metric_map.items():
                payload: dict[str, Any] = {}
                for k in k_values:
                    for metric_name in (
                        f"top{k}_recall",
                        f"top{k}_predicted_mean_harm",
                        f"top{k}_oracle_mean_harm",
                        f"top{k}_regret",
                    ):
                        metric_values = [float(item[metric_name]) for item in rows_for_feature]
                        conversation_metric_values = [
                            float(value[metric_name])
                            for value in collapse_rows_by_keys(
                                [
                                    {
                                        metric_name: float(item[metric_name]),
                                        "conversation_id": item["conversation_id"],
                                    }
                                    for item in rows_for_feature
                                ],
                                metric_keys=[metric_name],
                                group_keys=["conversation_id"],
                            )
                        ]
                        payload[metric_name] = {
                            "row_level": bootstrap_mean_ci(metric_values, rng=rng),
                            "conversation_level": bootstrap_mean_ci(conversation_metric_values, rng=rng),
                        }
                if feature_key != "semantic_score" and semantic_rows:
                    comparison_payload: dict[str, Any] = {}
                    for k in k_values:
                        for metric_name in (f"top{k}_recall", f"top{k}_predicted_mean_harm", f"top{k}_regret"):
                            semantic_by_conv = {item["conversation_id"]: float(item[metric_name]) for item in semantic_rows}
                            feature_by_conv = {item["conversation_id"]: float(item[metric_name]) for item in rows_for_feature}
                            common = sorted(set(semantic_by_conv) & set(feature_by_conv))
                            deltas = [feature_by_conv[item] - semantic_by_conv[item] for item in common]
                            delta_array = np.asarray(deltas, dtype=np.float64)
                            comparison_payload[metric_name] = {
                                "row_level": {
                                    **bootstrap_mean_ci(deltas, rng=rng),
                                    "p_value": paired_signflip_test(delta_array, rng=rng),
                                },
                                "conversation_level": {
                                    **bootstrap_mean_ci(deltas, rng=rng),
                                    "p_value": paired_signflip_test(delta_array, rng=rng),
                                },
                            }
                    payload["vs_semantic"] = comparison_payload
                budget_payload[feature_key] = payload
        return summary

    turn_row_views = _turn_row_views(rows)
    return {view_name: _group_metrics(grouped) for view_name, grouped in turn_row_views.items()}


def _gate_summary(ranking_summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view_name, view_payload in ranking_summary.items():
        gate_hits: list[dict[str, Any]] = []
        for benchmark in ("msc_valid", "longmemeval_s_cleaned"):
            for model_key, budget_payload in view_payload.get(benchmark, {}).items():
                for budget_key, feature_payload in budget_payload.items():
                    for feature_key in ("query_geom_v2_risk", "combined_structural_score"):
                        payload = feature_payload.get(feature_key, {})
                        semantic_delta = payload.get("vs_semantic", {})
                        kendall = semantic_delta.get("kendall_tau", {}).get("row_level", {}).get("mean", 0.0)
                        recall = semantic_delta.get("top5_recall", {}).get("row_level", {}).get("mean", 0.0)
                        if kendall >= 0.03 or recall >= 0.05:
                            gate_hits.append(
                                {
                                    "benchmark": benchmark,
                                    "model_key": model_key,
                                    "budget_fraction": budget_key,
                                    "feature": feature_key,
                                    "kendall_delta": kendall,
                                    "top5_recall_delta": recall,
                                }
                            )
        result[view_name] = {"passed": bool(gate_hits), "hits": gate_hits}
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Paper 3 Oracle Harm Headroom Study: {summary['study_name']}",
        "",
        f"- Created: {summary['created_at']}",
        f"- Benchmark: {summary['benchmark_name']}",
        f"- Models: {', '.join(summary['model_keys'])}",
        f"- Budgets: {', '.join(f'{float(item):.2f}' for item in summary['budgets'])}",
        f"- Conversations: {summary['num_conversations']}",
        f"- Candidate rows: {summary['num_candidate_rows']}",
        "",
        "## Gate 1",
        "",
    ]
    for view_name, gate_payload in summary["gate_summary"].items():
        lines.append(f"- {view_name}: {'PASS' if gate_payload['passed'] else 'FAIL'}")
        for item in gate_payload["hits"]:
            lines.append(
                f"  - {item['benchmark']} {item['model_key']} @ {item['budget_fraction']} "
                f"{item['feature']}: Δ kendall {item['kendall_delta']:.4f}, "
                f"Δ top5 recall {item['top5_recall_delta']:.4f}"
            )
    lines.append("")
    lines.append("## Ranking Summary (semantic shortlist)")
    lines.append("")
    for benchmark, benchmark_payload in summary["ranking_summary"]["semantic_shortlist"].items():
        lines.append(f"### {benchmark}")
        lines.append("")
        for model_key, budget_payload in benchmark_payload.items():
            lines.append(f"- {model_key}:")
            for budget_key, feature_payload in budget_payload.items():
                lines.append(f"  - budget {budget_key}:")
                for feature_key in ("semantic_score", "query_geom_v2_risk", "combined_structural_score"):
                    payload = feature_payload.get(feature_key)
                    if payload is None:
                        continue
                    kendall = payload["kendall_tau"]["row_level"]["mean"]
                    recall = payload["top5_recall"]["row_level"]["mean"]
                    lines.append(
                        f"    - {feature_key}: kendall {kendall:.4f}, top5 recall {recall:.4f}"
                    )
    lines.append("")
    lines.append("## Oracle Top-k Headroom (semantic shortlist)")
    lines.append("")
    for benchmark, benchmark_payload in summary["oracle_topk_summary"]["semantic_shortlist"].items():
        lines.append(f"### {benchmark}")
        lines.append("")
        for model_key, budget_payload in benchmark_payload.items():
            lines.append(f"- {model_key}:")
            for budget_key, feature_payload in budget_payload.items():
                lines.append(f"  - budget {budget_key}:")
                for feature_key in ("semantic_score", "query_geom_v2_risk", "combined_structural_score"):
                    payload = feature_payload.get(feature_key)
                    if payload is None:
                        continue
                    top5_recall = payload["top5_recall"]["row_level"]["mean"]
                    top5_regret = payload["top5_regret"]["row_level"]["mean"]
                    top5_mean_harm = payload["top5_predicted_mean_harm"]["row_level"]["mean"]
                    semantic_delta = payload.get("vs_semantic", {}).get("top5_recall", {}).get("row_level", {}).get("mean", 0.0)
                    lines.append(
                        f"    - {feature_key}: top5 recall {top5_recall:.4f}, "
                        f"top5 mean harm {top5_mean_harm:.4f}, top5 regret {top5_regret:.4f}, "
                        f"Δ recall vs semantic {semantic_delta:.4f}"
                    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_oracle_harm_probe(
    *,
    study_name: str,
    benchmark_name: str,
    model_keys: list[str],
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
    target_turn_stride: int,
    max_target_turns: int | None,
    max_turns_per_conversation: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    conversations = load_conversations_from_paths(input_paths)
    if families is not None:
        conversations = [conversation for conversation in conversations if conversation.family in families]
    if limit_conversations is not None:
        conversations = conversations[:limit_conversations]
    if not conversations:
        raise RuntimeError("No conversations selected for oracle harm study.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_payloads: dict[str, Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        spec = resolve_model_spec(model_key)
        if spec is None:
            raise RuntimeError(f"Unknown model key: {model_key}")
        extractor = ConversationStateExtractor(
            model_name=spec.model_name,
            device=device,
            dtype=dtype,
            state_layer=state_layer,
        )
        target_turns_by_conversation = {
            conversation.conversation_id: _sample_target_turns(
                num_turns=(
                    min(len(conversation.turns), max_turns_per_conversation)
                    if max_turns_per_conversation is not None
                    else len(conversation.turns)
                ),
                min_history=min_history,
                stride=target_turn_stride,
                max_target_turns=max_target_turns,
            )
            for conversation in conversations
        }

        iterator = _progress(
            enumerate(conversations, start=1),
            total=len(conversations),
            desc=f"{model_key} oracle",
        )
        for conversation_index, conversation in iterator:
            full_batch = extractor.extract_conversation(
                conversation,
                max_turns=max_turns_per_conversation,
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
            turn_costs = _prefix_turn_costs(full_batch.token_counts)
            target_turns = target_turns_by_conversation[conversation.conversation_id]

            for target_turn in target_turns:
                prefix_turn_count = target_turn
                if prefix_turn_count <= 0:
                    continue
                full_logits = full_batch.logits[target_turn]
                semantic_risk = turn_semantic_risk(full_batch.states, target_turn)[:prefix_turn_count]
                support_scores = _support_scores(
                    conversation=conversation,
                    prefix_turn_count=prefix_turn_count,
                    geometry_risk=geometry_risk,
                    semantic_risk=semantic_risk,
                )
                semantic_support_proxy = _semantic_support_proxy_scores(
                    geometry_risk=geometry_risk,
                    semantic_risk=semantic_risk,
                    support_scores=support_scores,
                    prefix_turn_count=prefix_turn_count,
                )
                full_messages = _policy_messages(
                    conversation=conversation,
                    target_turn=target_turn,
                    retained_prior_indices=list(range(prefix_turn_count)),
                )
                full_prompt_score = extractor.score_messages(
                    full_messages,
                    max_input_tokens=max_input_tokens,
                    return_attention_summary=True,
                    cumulative_turn_token_counts=full_batch.token_counts[:prefix_turn_count],
                )
                attention_raw, attention_sink = _turn_attention_features(
                    full_prompt_score.attention_summary,
                    prefix_turn_count=prefix_turn_count,
                )
                full_behavior_score = None
                if (
                    conversation.turns[target_turn].role == "user"
                    and target_turn + 1 < len(conversation.turns)
                    and conversation.turns[target_turn + 1].role == "assistant"
                ):
                    full_behavior_score = extractor.score_assistant_response(
                        full_messages,
                        conversation.turns[target_turn + 1].content,
                        max_input_tokens=max_input_tokens,
                    )

                for budget_fraction in budgets:
                    segment_span = _budget_segment_span(budget_fraction)
                    query_v2 = query_conditioned_turn_risk_v2(
                        full_batch.states,
                        target_turn,
                        segment_span=segment_span,
                        ambient_geometry=geometry_risk,
                    )
                    candidate_rows.extend(
                        _oracle_ablation_rows_for_target(
                            conversation=conversation,
                            benchmark_name=benchmark_name,
                            model_key=model_key,
                            extractor=extractor,
                            full_batch=full_batch,
                            full_logits=full_logits,
                            full_behavior_score=full_behavior_score,
                            geometry_risk=geometry_risk,
                            target_turn=target_turn,
                            budget_fraction=budget_fraction,
                            recent_window=recent_window,
                            max_input_tokens=max_input_tokens,
                            semantic_risk=semantic_risk,
                            support_scores=support_scores,
                            semantic_support_proxy=semantic_support_proxy,
                            query_v2=query_v2,
                            attention_raw=attention_raw,
                            attention_sink=attention_sink,
                        )
                    )

            model_payloads[model_key] = {"model_name": spec.model_name}

    _apply_harm_scalar(candidate_rows)
    ranking_summary = _ranking_summary(candidate_rows)
    oracle_topk_summary = _oracle_topk_summary(candidate_rows)
    gate_summary = _gate_summary(ranking_summary)
    summary = {
        "study_name": study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_name": benchmark_name,
        "model_keys": model_keys,
        "budgets": budgets,
        "families": families,
        "num_conversations": len(conversations),
        "num_candidate_rows": len(candidate_rows),
        "models": model_payloads,
        "ranking_summary": ranking_summary,
        "oracle_topk_summary": oracle_topk_summary,
        "gate_summary": gate_summary,
    }

    _write_csv(output_dir / "candidate_rows.csv", candidate_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_format_report(summary), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Paper 3 oracle harm headroom study.")
    parser.add_argument("--study-name", default="paper3_oracle_harm_v1")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--model-keys", default="qwen25_05b")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default="0.20,0.35,0.50")
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--target-turn-stride", type=int, default=4)
    parser.add_argument("--max-target-turns", type=int, default=16)
    parser.add_argument(
        "--max-turns-per-conversation",
        type=int,
        default=None,
        help="Truncate each conversation to this many turns before extraction. "
             "Critical for long-conversation benchmarks like LongMemEval to keep runtime manageable.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    model_keys = [item.strip() for item in args.model_keys.split(",") if item.strip()]
    output_dir = args.output_root / args.study_name
    summary = run_oracle_harm_probe(
        study_name=args.study_name,
        benchmark_name=args.benchmark_name,
        model_keys=model_keys,
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
        target_turn_stride=args.target_turn_stride,
        max_target_turns=args.max_target_turns,
        max_turns_per_conversation=args.max_turns_per_conversation,
        output_dir=output_dir,
    )
    print(
        f"Wrote oracle harm study to {output_dir} "
        f"({summary['num_candidate_rows']} candidate rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()
