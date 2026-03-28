from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def turn_geometry_risk(analysis: dict) -> np.ndarray:
    state_error = np.asarray(analysis["series"]["state_geodesic_errors"], dtype=np.float32)
    turning = np.asarray(analysis["series"]["turning_angles"], dtype=np.float32)
    subspace = np.asarray(analysis["series"].get("subspace_shifts", []), dtype=np.float32)
    risk = _normalize(state_error)
    for scores in [turning, subspace]:
        if scores.size == 0:
            continue
        expanded = np.zeros_like(risk, dtype=np.float32)
        normalized = _normalize(scores)
        for idx, value in enumerate(normalized, start=1):
            expanded[idx - 1] += 0.5 * value
            expanded[idx] += 0.5 * value
        risk = risk + expanded
    return _normalize(risk)


def turn_lexical_risk(analysis: dict) -> np.ndarray:
    lexical = np.asarray(analysis["series"].get("lexical_boundary_scores", []), dtype=np.float32)
    if lexical.size == 0:
        num_turns = len(analysis["series"]["state_geodesic_errors"])
        return np.zeros(num_turns, dtype=np.float32)
    num_turns = lexical.size + 2
    expanded = np.zeros(num_turns, dtype=np.float32)
    normalized = _normalize(lexical)
    for idx, value in enumerate(normalized, start=1):
        expanded[idx - 1] += 0.5 * value
        expanded[idx] += 0.5 * value
    return _normalize(expanded)


def turn_hybrid_risk(analysis: dict) -> np.ndarray:
    geometry = turn_geometry_risk(analysis)
    lexical = turn_lexical_risk(analysis)
    if geometry.size == 0:
        return lexical
    return _normalize(geometry + 0.75 * lexical)


def turn_semantic_risk(states: np.ndarray, target_turn: int) -> np.ndarray:
    if states.size == 0:
        return np.zeros(0, dtype=np.float32)
    target_state = np.asarray(states[target_turn], dtype=np.float32)
    state_array = np.asarray(states, dtype=np.float32)
    target_norm = float(np.linalg.norm(target_state))
    state_norms = np.linalg.norm(state_array, axis=1)
    denom = np.maximum(state_norms * max(target_norm, 1e-8), 1e-8)
    cosine = np.clip(np.sum(state_array * target_state[None, :], axis=1) / denom, -1.0, 1.0)
    return _normalize((cosine + 1.0) * 0.5)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    retained_turn_indices: list[int]
    retained_fraction: float
    retained_cost_fraction: float
    kept_segment_count: int = 0
    compressed_segment_count: int = 0
    evicted_segment_count: int = 0


def _top_k_indices(values: np.ndarray, k: int) -> list[int]:
    if k <= 0 or values.size == 0:
        return []
    order = np.argsort(-values, kind="stable")
    return sorted(int(idx) for idx in order[:k])


def _uniform_indices(num_items: int, k: int) -> list[int]:
    if k <= 0 or num_items <= 0:
        return []
    if k >= num_items:
        return list(range(num_items))
    positions = np.linspace(0, num_items - 1, num=k)
    return sorted({int(round(value)) for value in positions})


def _fill_under_budget(order: list[int], costs: np.ndarray, budget_cost: int) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    spent = 0
    for idx in order:
        item_cost = int(max(costs[idx], 1))
        if spent + item_cost > budget_cost:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        spent += item_cost
    remaining = [idx for idx in range(costs.size) if idx not in selected_set]
    for idx in sorted(remaining, key=lambda item: (int(max(costs[item], 1)), item)):
        item_cost = int(max(costs[idx], 1))
        if spent + item_cost > budget_cost:
            continue
        selected.append(int(idx))
        spent += item_cost
    return sorted(selected)


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    start: int
    end: int
    full_indices: list[int]
    compressed_indices: list[int]
    full_cost: int
    compressed_cost: int
    risk: float


def segment_risk_scores(
    risk_scores: np.ndarray,
    older_count: int,
    segment_span: int,
) -> list[tuple[int, int, float]]:
    if older_count <= 0:
        return []
    payload: list[tuple[int, int, float]] = []
    for start in range(0, older_count, max(segment_span, 1)):
        end = min(start + max(segment_span, 1), older_count)
        segment_values = np.asarray(risk_scores[start:end], dtype=np.float32)
        if segment_values.size == 0:
            continue
        score = float(0.65 * np.mean(segment_values) + 0.35 * np.max(segment_values))
        payload.append((start, end, score))
    return payload


def _segment_specs(
    risk_scores: np.ndarray,
    turn_costs: np.ndarray,
    older_count: int,
    segment_span: int,
) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []
    for start, end, score in segment_risk_scores(risk_scores, older_count, segment_span):
        full_indices = list(range(start, end))
        compressed_indices = [end - 1] if end - start > 1 else [start]
        full_cost = int(np.sum(turn_costs[full_indices])) if full_indices else 0
        compressed_cost = int(np.sum(turn_costs[compressed_indices])) if compressed_indices else 0
        specs.append(
            SegmentSpec(
                start=start,
                end=end,
                full_indices=full_indices,
                compressed_indices=compressed_indices,
                full_cost=max(full_cost, 0),
                compressed_cost=max(compressed_cost, 0),
                risk=float(score),
            )
        )
    return specs


def _segment_options(spec: SegmentSpec, policy_name: str) -> list[tuple[str, int, float, list[int]]]:
    if policy_name == "geometry_segment_actions":
        compress_utility = 0.58 * spec.risk
        keep_utility = 1.0 * spec.risk
    elif policy_name == "uniform_segment_actions":
        compress_utility = 0.58
        keep_utility = 1.0
    else:
        raise ValueError(f"Unknown segment policy: {policy_name}")
    return [
        ("evict", 0, 0.0, []),
        ("compress", spec.compressed_cost, compress_utility, spec.compressed_indices),
        ("keep", spec.full_cost, keep_utility, spec.full_indices),
    ]


def select_segment_actions(
    policy_name: str,
    risk_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
    segment_span: int = 2,
) -> PolicySelection:
    if prefix_turn_count <= 1:
        return PolicySelection(
            retained_turn_indices=list(range(prefix_turn_count)),
            retained_fraction=1.0,
            retained_cost_fraction=1.0,
        )

    recent_start = max(prefix_turn_count - recent_window, 0)
    older_count = recent_start
    recent_indices = list(range(recent_start, prefix_turn_count))
    total_cost = int(np.sum(turn_costs[:prefix_turn_count])) if prefix_turn_count > 0 else 0
    recent_cost = int(np.sum(turn_costs[recent_start:prefix_turn_count])) if prefix_turn_count > 0 else 0
    if older_count <= 0:
        return PolicySelection(
            retained_turn_indices=recent_indices,
            retained_fraction=1.0,
            retained_cost_fraction=float(recent_cost / max(total_cost, 1)),
        )

    older_costs = np.asarray(turn_costs[:older_count], dtype=np.int32)
    budget_cost = int(np.ceil(budget_fraction * float(np.sum(older_costs))))
    segment_specs = _segment_specs(risk_scores, older_costs, older_count, segment_span)
    # Dynamic programming over short dialogue budgets. Costs are small in this setup.
    states: dict[int, tuple[float, list[tuple[str, SegmentSpec]]]] = {0: (0.0, [])}
    for spec in segment_specs:
        next_states: dict[int, tuple[float, list[tuple[str, SegmentSpec]]]] = {}
        for spent_cost, (utility, actions) in states.items():
            for action_name, action_cost, action_utility, _ in _segment_options(spec, policy_name):
                new_cost = spent_cost + action_cost
                if new_cost > budget_cost:
                    continue
                new_utility = utility + action_utility
                current = next_states.get(new_cost)
                candidate_actions = actions + [(action_name, spec)]
                if current is None or new_utility > current[0]:
                    next_states[new_cost] = (new_utility, candidate_actions)
        if next_states:
            states = next_states

    best_cost = max(states, key=lambda cost: (states[cost][0], cost))
    selected_actions = states[best_cost][1]
    retained_old: list[int] = []
    kept_segment_count = 0
    compressed_segment_count = 0
    evicted_segment_count = 0
    for action_name, spec in selected_actions:
        if action_name == "keep":
            retained_old.extend(spec.full_indices)
            kept_segment_count += 1
        elif action_name == "compress":
            retained_old.extend(spec.compressed_indices)
            compressed_segment_count += 1
        else:
            evicted_segment_count += 1

    retained = sorted(set(retained_old + recent_indices))
    retained_fraction = len(retained) / max(prefix_turn_count, 1)
    retained_cost = int(np.sum(turn_costs[retained])) if retained else 0
    return PolicySelection(
        retained_turn_indices=retained,
        retained_fraction=float(retained_fraction),
        retained_cost_fraction=float(retained_cost / max(total_cost, 1)),
        kept_segment_count=kept_segment_count,
        compressed_segment_count=compressed_segment_count,
        evicted_segment_count=evicted_segment_count,
    )


def select_turns(
    policy_name: str,
    risk_scores: np.ndarray,
    turn_costs: np.ndarray,
    prefix_turn_count: int,
    budget_fraction: float,
    recent_window: int,
) -> PolicySelection:
    if prefix_turn_count <= 1:
        return PolicySelection(
            retained_turn_indices=list(range(prefix_turn_count)),
            retained_fraction=1.0,
            retained_cost_fraction=1.0,
        )

    recent_start = max(prefix_turn_count - recent_window, 0)
    older_count = recent_start
    older_costs = np.asarray(turn_costs[:older_count], dtype=np.int32)
    older_budget_cost = int(np.ceil(budget_fraction * float(np.sum(older_costs)))) if older_count > 0 else 0
    recent_indices = list(range(recent_start, prefix_turn_count))
    recent_cost = int(np.sum(turn_costs[recent_start:prefix_turn_count])) if prefix_turn_count > 0 else 0
    total_cost = int(np.sum(turn_costs[:prefix_turn_count])) if prefix_turn_count > 0 else 0
    if older_count <= 0:
        retained = recent_indices
        return PolicySelection(
            retained_turn_indices=retained,
            retained_fraction=1.0,
            retained_cost_fraction=float(recent_cost / max(total_cost, 1)),
        )

    older_scores = risk_scores[:older_count]
    if policy_name == "uniform":
        candidate_order = _uniform_indices(older_count, older_count)
    elif policy_name in {"lexical", "geometry", "geometry_lexical", "semantic"}:
        density = older_scores / np.maximum(older_costs.astype(np.float32), 1.0)
        candidate_order = list(np.argsort(-density, kind="stable"))
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    selected_old = _fill_under_budget(candidate_order, older_costs, older_budget_cost)
    retained = sorted(set(selected_old + recent_indices))
    retained_fraction = len(retained) / max(prefix_turn_count, 1)
    retained_cost = int(np.sum(turn_costs[retained])) if retained else 0
    return PolicySelection(
        retained_turn_indices=retained,
        retained_fraction=float(retained_fraction),
        retained_cost_fraction=float(retained_cost / max(total_cost, 1)),
    )
