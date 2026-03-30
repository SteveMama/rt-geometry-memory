from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SparseSegmentMemory:
    segment_start: int
    segment_end: int
    anchor_turn_index: int
    support_turn_indices: list[int]
    retained_turn_indices: list[int]
    risk: float
    action: str


@dataclass(frozen=True, slots=True)
class CodecSelection:
    retained_turn_indices: list[int]
    retained_fraction: float
    retained_cost_fraction: float
    kept_segment_count: int
    compressed_segment_count: int
    evicted_segment_count: int
    memory_objects: list[SparseSegmentMemory]


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _segment_score(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(0.65 * np.mean(values) + 0.35 * np.max(values))


def _support_candidates(start: int, end: int, risk_scores: np.ndarray) -> list[int]:
    if end <= start:
        return []
    local_scores = risk_scores[start:end]
    if local_scores.size == 0:
        return []
    support_index = start + int(np.argmax(local_scores))
    segment_length = end - start
    if segment_length <= 1:
        return [start]
    if segment_length == 2:
        # True compression for two-turn regions: keep only the higher-risk turn.
        return [support_index]

    anchor_index = start
    if support_index == anchor_index:
        support_index = end - 1
    return sorted({anchor_index, support_index})


def _support_aware_candidates(
    start: int,
    end: int,
    risk_scores: np.ndarray,
    support_scores: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
) -> list[int]:
    if end <= start:
        return []
    candidate_indices = list(range(start, end))
    if candidate_mask is not None:
        candidate_indices = [idx for idx in candidate_indices if bool(candidate_mask[idx])]
    if not candidate_indices:
        return []

    if support_scores is None:
        return _support_candidates(start, end, risk_scores)

    local_risk = np.asarray([risk_scores[idx] for idx in candidate_indices], dtype=np.float32)
    local_support = np.asarray([support_scores[idx] for idx in candidate_indices], dtype=np.float32)
    priority = local_risk + 0.85 * local_support
    best_index = candidate_indices[int(np.argmax(priority))]
    support_candidates = [idx for idx in candidate_indices if support_scores[idx] > 0.0]
    latest_support = support_candidates[-1] if support_candidates else None

    segment_length = end - start
    if segment_length <= 1:
        return [best_index]
    if segment_length == 2:
        return [latest_support if latest_support is not None else best_index]

    anchor_index = start if start in candidate_indices else candidate_indices[0]
    retained = [best_index]
    if latest_support is not None:
        retained.append(latest_support)
    retained.append(anchor_index)
    return sorted(set(retained))


def semantic_shortlist_mask(
    *,
    semantic_scores: np.ndarray,
    turn_costs: np.ndarray,
    budget_fraction: float,
    expansion_factor: float = 2.0,
    latest_user_index: int | None = None,
) -> np.ndarray:
    if semantic_scores.size == 0:
        return np.zeros(0, dtype=bool)
    costs = np.maximum(turn_costs.astype(np.int32), 1)
    budget_cost = int(np.ceil(budget_fraction * float(np.sum(costs))))
    shortlist_budget = max(int(np.ceil(expansion_factor * budget_cost)), int(np.min(costs)))
    density = semantic_scores / np.maximum(costs.astype(np.float32), 1.0)
    order = list(np.argsort(-density, kind="stable"))
    spent = 0
    selected: list[int] = []
    for idx in order:
        item_cost = int(costs[idx])
        if spent + item_cost > shortlist_budget and selected:
            continue
        selected.append(int(idx))
        spent += item_cost
        if spent >= shortlist_budget:
            break
    if latest_user_index is not None and 0 <= latest_user_index < semantic_scores.size:
        selected.append(int(latest_user_index))
    mask = np.zeros(semantic_scores.size, dtype=bool)
    if selected:
        mask[sorted(set(selected))] = True
    return mask


def _select_sparse_segment_memory_core(
    *,
    risk_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
    segment_span: int = 2,
    support_scores: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
) -> CodecSelection:
    if prefix_turn_count <= 1:
        retained = list(range(prefix_turn_count))
        return CodecSelection(
            retained_turn_indices=retained,
            retained_fraction=1.0,
            retained_cost_fraction=1.0,
            kept_segment_count=0,
            compressed_segment_count=0,
            evicted_segment_count=0,
            memory_objects=[],
        )

    recent_start = max(prefix_turn_count - recent_window, 0)
    older_count = recent_start
    recent_indices = list(range(recent_start, prefix_turn_count))
    total_cost = int(np.sum(turn_costs[:prefix_turn_count])) if prefix_turn_count > 0 else 0
    recent_cost = int(np.sum(turn_costs[recent_start:prefix_turn_count])) if prefix_turn_count > 0 else 0
    if older_count <= 0:
        return CodecSelection(
            retained_turn_indices=recent_indices,
            retained_fraction=1.0,
            retained_cost_fraction=float(recent_cost / max(total_cost, 1)),
            kept_segment_count=0,
            compressed_segment_count=0,
            evicted_segment_count=0,
            memory_objects=[],
        )

    budget_cost = int(np.ceil(budget_fraction * float(np.sum(turn_costs[:older_count]))))
    states: dict[int, tuple[float, list[tuple[str, SparseSegmentMemory]]]] = {0: (0.0, [])}

    for start in range(0, older_count, max(segment_span, 1)):
        end = min(start + max(segment_span, 1), older_count)
        full_indices = list(range(start, end))
        eligible_mask = candidate_mask[:older_count] if candidate_mask is not None else None
        segment_has_candidate = True
        if eligible_mask is not None:
            segment_has_candidate = bool(np.any(eligible_mask[start:end]))
        compressed_indices = _support_aware_candidates(
            start,
            end,
            risk_scores,
            support_scores=support_scores,
            candidate_mask=eligible_mask,
        )
        full_cost = int(np.sum(turn_costs[full_indices])) if full_indices else 0
        compressed_cost = int(np.sum(turn_costs[compressed_indices])) if compressed_indices else 0
        segment_values = np.asarray(risk_scores[start:end], dtype=np.float32)
        if support_scores is not None:
            segment_values = segment_values + 0.60 * np.asarray(support_scores[start:end], dtype=np.float32)
        segment_risk = _segment_score(segment_values)
        if full_cost <= 0:
            continue
        compression_ratio = float(compressed_cost / max(full_cost, 1))
        compress_preservation = 0.86 + 0.12 * (1.0 - compression_ratio)
        keep_preservation = 1.0
        memory_stub = SparseSegmentMemory(
            segment_start=start,
            segment_end=end,
            anchor_turn_index=compressed_indices[0] if compressed_indices else start,
            support_turn_indices=[
                idx for idx in compressed_indices if idx != (compressed_indices[0] if compressed_indices else start)
            ],
            retained_turn_indices=compressed_indices,
            risk=segment_risk,
            action="compress",
        )
        options = [("evict", 0, 0.0, memory_stub)]
        if segment_has_candidate and compressed_cost > 0:
            options.append(
                (
                    "compress",
                    compressed_cost,
                    compress_preservation * segment_risk,
                    memory_stub,
                )
            )
            options.append(
                (
                    "keep",
                    full_cost,
                    keep_preservation * segment_risk,
                    SparseSegmentMemory(
                        segment_start=start,
                        segment_end=end,
                        anchor_turn_index=start,
                        support_turn_indices=[idx for idx in full_indices if idx != start],
                        retained_turn_indices=full_indices,
                        risk=segment_risk,
                        action="keep",
                    ),
                )
            )

        next_states: dict[int, tuple[float, list[tuple[str, SparseSegmentMemory]]]] = {}
        for spent_cost, (utility, actions) in states.items():
            for action_name, action_cost, action_utility, memory in options:
                new_cost = spent_cost + action_cost
                if new_cost > budget_cost:
                    continue
                new_utility = utility + action_utility
                candidate_actions = actions + [(action_name, memory)]
                current = next_states.get(new_cost)
                if current is None or new_utility > current[0]:
                    next_states[new_cost] = (new_utility, candidate_actions)
        if next_states:
            states = next_states

    best_cost = max(states, key=lambda cost: (states[cost][0], cost))
    selected_actions = states[best_cost][1]
    retained_old: list[int] = []
    memory_objects: list[SparseSegmentMemory] = []
    kept = 0
    compressed = 0
    evicted = 0
    for action_name, memory in selected_actions:
        if action_name == "keep":
            retained_old.extend(memory.retained_turn_indices)
            kept += 1
            memory_objects.append(memory)
        elif action_name == "compress":
            retained_old.extend(memory.retained_turn_indices)
            compressed += 1
            memory_objects.append(memory)
        else:
            evicted += 1

    retained = sorted(set(retained_old + recent_indices))
    retained_fraction = len(retained) / max(prefix_turn_count, 1)
    retained_cost = int(np.sum(turn_costs[retained])) if retained else 0
    return CodecSelection(
        retained_turn_indices=retained,
        retained_fraction=float(retained_fraction),
        retained_cost_fraction=float(retained_cost / max(total_cost, 1)),
        kept_segment_count=kept,
        compressed_segment_count=compressed,
        evicted_segment_count=evicted,
        memory_objects=memory_objects,
    )


def select_sparse_segment_memory(
    *,
    risk_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
    segment_span: int = 2,
) -> CodecSelection:
    return _select_sparse_segment_memory_core(
        risk_scores=risk_scores,
        turn_costs=turn_costs,
        prefix_turn_count=prefix_turn_count,
        budget_fraction=budget_fraction,
        recent_window=recent_window,
        segment_span=segment_span,
        support_scores=None,
        candidate_mask=None,
    )


def select_support_aware_sparse_segment_memory(
    *,
    risk_scores: np.ndarray,
    support_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
    segment_span: int = 3,
) -> CodecSelection:
    return _select_sparse_segment_memory_core(
        risk_scores=_normalize(risk_scores + 0.75 * support_scores),
        turn_costs=turn_costs,
        prefix_turn_count=prefix_turn_count,
        budget_fraction=budget_fraction,
        recent_window=recent_window,
        segment_span=segment_span,
        support_scores=support_scores,
        candidate_mask=None,
    )


def select_semantic_filtered_sparse_segment_memory(
    *,
    geometry_like_scores: np.ndarray,
    support_scores: np.ndarray,
    semantic_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
    segment_span: int = 3,
    latest_user_index: int | None = None,
) -> CodecSelection:
    candidate_mask = semantic_shortlist_mask(
        semantic_scores=semantic_scores,
        turn_costs=turn_costs,
        budget_fraction=budget_fraction,
        latest_user_index=latest_user_index,
    )
    return _select_sparse_segment_memory_core(
        risk_scores=_normalize(geometry_like_scores + 0.60 * support_scores),
        turn_costs=turn_costs,
        prefix_turn_count=prefix_turn_count,
        budget_fraction=budget_fraction,
        recent_window=recent_window,
        segment_span=segment_span,
        support_scores=support_scores,
        candidate_mask=candidate_mask,
    )
