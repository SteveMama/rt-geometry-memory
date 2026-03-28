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


def select_sparse_segment_memory(
    *,
    risk_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
    segment_span: int = 2,
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
        compressed_indices = _support_candidates(start, end, risk_scores)
        full_cost = int(np.sum(turn_costs[full_indices])) if full_indices else 0
        compressed_cost = int(np.sum(turn_costs[compressed_indices])) if compressed_indices else 0
        segment_risk = _segment_score(np.asarray(risk_scores[start:end], dtype=np.float32))
        if full_cost <= 0:
            continue
        compression_ratio = float(compressed_cost / max(full_cost, 1))
        # Favor compression when it preserves most of a segment's value at materially lower cost.
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
        options = [
            ("evict", 0, 0.0, memory_stub),
            (
                "compress",
                compressed_cost,
                compress_preservation * segment_risk,
                memory_stub,
            ),
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
            ),
        ]

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
