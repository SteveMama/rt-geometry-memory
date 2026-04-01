from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OBJECT_TYPES: tuple[str, ...] = (
    "persona",
    "event",
    "constraint",
    "update",
    "generic",
)

_PERSONA_MARKERS: tuple[str, ...] = (
    "i am",
    "i'm",
    "i like",
    "i love",
    "i enjoy",
    "my favorite",
    "my favourite",
    "i work",
    "i live",
    "i have",
    "my hobby",
    "my hobbies",
    "my dog",
    "my cat",
)

_EVENT_MARKERS: tuple[str, ...] = (
    "today",
    "tomorrow",
    "yesterday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "week",
    "month",
    "meeting",
    "trip",
    "flight",
    "party",
    "review",
    "launch",
    "appointment",
    "concert",
    "dinner",
    "plan",
    "planned",
)

_UPDATE_MARKERS: tuple[str, ...] = (
    "actually",
    "instead",
    "changed",
    "now",
    "currently",
    "used to",
    "no longer",
    "switched",
    "left",
    "moved",
    "migrated",
    "started",
    "stopped",
    "finished",
    "updated",
    "correction",
)


@dataclass(frozen=True, slots=True)
class SemanticObjectSpec:
    object_id: int
    object_start: int
    object_end: int
    turn_indices: list[int]
    object_type: str
    anchor_turn_index: int
    freshest_turn_index: int
    latest_user_turn_index: int | None
    compressed_turn_indices: list[int]
    memory_score: float


@dataclass(frozen=True, slots=True)
class SemanticObjectBundle:
    objects: list[SemanticObjectSpec]
    turn_object_id: np.ndarray
    object_size: np.ndarray
    object_recency: np.ndarray
    object_memory_score: np.ndarray
    is_object_anchor: np.ndarray
    is_object_freshest: np.ndarray
    object_type_persona: np.ndarray
    object_type_event: np.ndarray
    object_type_constraint: np.ndarray
    object_type_update: np.ndarray
    object_type_generic: np.ndarray

    def per_turn_feature_rows(self, prefix_turn_count: int) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for idx in range(prefix_turn_count):
            rows.append(
                {
                    "object_size": float(self.object_size[idx]) if idx < self.object_size.size else 0.0,
                    "object_recency": float(self.object_recency[idx]) if idx < self.object_recency.size else 0.0,
                    "object_memory_score": float(self.object_memory_score[idx]) if idx < self.object_memory_score.size else 0.0,
                    "is_object_anchor": float(self.is_object_anchor[idx]) if idx < self.is_object_anchor.size else 0.0,
                    "is_object_freshest": float(self.is_object_freshest[idx]) if idx < self.is_object_freshest.size else 0.0,
                    "object_type_persona": float(self.object_type_persona[idx]) if idx < self.object_type_persona.size else 0.0,
                    "object_type_event": float(self.object_type_event[idx]) if idx < self.object_type_event.size else 0.0,
                    "object_type_constraint": float(self.object_type_constraint[idx]) if idx < self.object_type_constraint.size else 0.0,
                    "object_type_update": float(self.object_type_update[idx]) if idx < self.object_type_update.size else 0.0,
                    "object_type_generic": float(self.object_type_generic[idx]) if idx < self.object_type_generic.size else 0.0,
                }
            )
        return rows


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def classify_memory_object_type(*, text: str, constraint_score: float) -> str:
    lowered = text.lower()
    if constraint_score >= 1.0:
        return "constraint"
    if any(marker in lowered for marker in _UPDATE_MARKERS):
        return "update"
    if any(marker in lowered for marker in _PERSONA_MARKERS):
        return "persona"
    if any(marker in lowered for marker in _EVENT_MARKERS):
        return "event"
    return "generic"


def _merge_seed_indices(seed_indices: list[int], *, gap_tolerance: int = 1) -> list[tuple[int, int]]:
    if not seed_indices:
        return []
    merged: list[tuple[int, int]] = []
    start = seed_indices[0]
    end = seed_indices[0]
    for idx in seed_indices[1:]:
        if idx - end <= max(gap_tolerance, 0) + 1:
            end = idx
            continue
        merged.append((start, end))
        start = idx
        end = idx
    merged.append((start, end))
    return merged


def _choose_object_type(
    turn_types: list[str],
    *,
    turn_weights: np.ndarray,
) -> str:
    scores = {name: 0.0 for name in OBJECT_TYPES}
    for turn_type, weight in zip(turn_types, turn_weights.tolist(), strict=True):
        scores[turn_type] += float(weight)
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _compressed_indices_for_object(
    *,
    turn_indices: list[int],
    semantic_scores: np.ndarray,
    support_scores: np.ndarray,
    query_scores: np.ndarray,
    object_type: str,
    latest_user_turn_index: int | None,
) -> list[int]:
    if not turn_indices:
        return []
    freshest = turn_indices[-1]
    local_priority = (
        0.55 * _normalize(semantic_scores[turn_indices])
        + 0.25 * _normalize(support_scores[turn_indices])
        + 0.20 * _normalize(query_scores[turn_indices])
    )
    anchor = turn_indices[int(np.argmax(local_priority))]
    retained = [freshest, anchor]
    if latest_user_turn_index is not None:
        retained.append(latest_user_turn_index)
    if object_type == "constraint":
        retained.append(turn_indices[int(np.argmax(_normalize(support_scores[turn_indices])))] )
    elif object_type in {"event", "update"}:
        retained.append(turn_indices[0])
    return sorted(set(int(idx) for idx in retained))


def build_semantic_object_bundle(
    *,
    conversation: Any,
    prefix_turn_count: int,
    semantic_scores: np.ndarray,
    support_scores: np.ndarray,
    query_scores: np.ndarray,
    candidate_mask: np.ndarray,
    constraint_scores: np.ndarray,
    gap_tolerance: int = 1,
) -> SemanticObjectBundle:
    turn_object_id = np.full(prefix_turn_count, -1, dtype=np.int32)
    object_size = np.zeros(prefix_turn_count, dtype=np.float32)
    object_recency = np.zeros(prefix_turn_count, dtype=np.float32)
    object_memory_score = np.zeros(prefix_turn_count, dtype=np.float32)
    is_object_anchor = np.zeros(prefix_turn_count, dtype=np.float32)
    is_object_freshest = np.zeros(prefix_turn_count, dtype=np.float32)
    object_type_arrays = {
        name: np.zeros(prefix_turn_count, dtype=np.float32)
        for name in OBJECT_TYPES
    }

    if prefix_turn_count <= 0:
        return SemanticObjectBundle(
            objects=[],
            turn_object_id=turn_object_id,
            object_size=object_size,
            object_recency=object_recency,
            object_memory_score=object_memory_score,
            is_object_anchor=is_object_anchor,
            is_object_freshest=is_object_freshest,
            object_type_persona=object_type_arrays["persona"],
            object_type_event=object_type_arrays["event"],
            object_type_constraint=object_type_arrays["constraint"],
            object_type_update=object_type_arrays["update"],
            object_type_generic=object_type_arrays["generic"],
        )

    seed_indices = [int(idx) for idx in np.flatnonzero(candidate_mask[:prefix_turn_count]).tolist()]
    if not seed_indices:
        fallback_index = int(np.argmax(_normalize(semantic_scores[:prefix_turn_count])))
        seed_indices = [fallback_index]

    turn_base_score = (
        0.55 * _normalize(semantic_scores[:prefix_turn_count])
        + 0.25 * _normalize(support_scores[:prefix_turn_count])
        + 0.20 * _normalize(query_scores[:prefix_turn_count])
    ).astype(np.float32)
    merged_ranges = _merge_seed_indices(seed_indices, gap_tolerance=gap_tolerance)

    objects: list[SemanticObjectSpec] = []
    prev_end = -1
    for object_id, (seed_start, seed_end) in enumerate(merged_ranges):
        start = seed_start
        end = seed_end
        if start > 0 and conversation.turns[start - 1].role != conversation.turns[start].role:
            start -= 1
        if end + 1 < prefix_turn_count and conversation.turns[end + 1].role != conversation.turns[end].role:
            end += 1
        start = max(start, prev_end + 1)
        end = max(end, start)
        if start >= prefix_turn_count:
            break
        end = min(end, prefix_turn_count - 1)
        prev_end = end
        turn_indices = list(range(start, end + 1))
        if not turn_indices:
            continue
        turn_types = [
            classify_memory_object_type(
                text=conversation.turns[idx].content,
                constraint_score=float(constraint_scores[idx]),
            )
            for idx in turn_indices
        ]
        turn_weights = turn_base_score[turn_indices]
        object_type = _choose_object_type(turn_types, turn_weights=turn_weights)
        latest_user_turn_index = next(
            (idx for idx in reversed(turn_indices) if conversation.turns[idx].role == "user"),
            None,
        )
        compressed_turn_indices = _compressed_indices_for_object(
            turn_indices=turn_indices,
            semantic_scores=semantic_scores,
            support_scores=support_scores,
            query_scores=query_scores,
            object_type=object_type,
            latest_user_turn_index=latest_user_turn_index,
        )
        anchor_turn_index = compressed_turn_indices[0] if compressed_turn_indices else turn_indices[0]
        freshest_turn_index = turn_indices[-1]
        type_bonus = {
            "persona": 0.04,
            "event": 0.06,
            "constraint": 0.10,
            "update": 0.08,
            "generic": 0.0,
        }[object_type]
        memory_score = float(
            0.55 * float(np.mean(turn_base_score[turn_indices]))
            + 0.35 * float(np.max(turn_base_score[turn_indices]))
            + 0.10 * type_bonus
        )
        spec = SemanticObjectSpec(
            object_id=len(objects),
            object_start=start,
            object_end=end,
            turn_indices=turn_indices,
            object_type=object_type,
            anchor_turn_index=anchor_turn_index,
            freshest_turn_index=freshest_turn_index,
            latest_user_turn_index=latest_user_turn_index,
            compressed_turn_indices=compressed_turn_indices,
            memory_score=memory_score,
        )
        objects.append(spec)
        recency = float(end / max(prefix_turn_count - 1, 1))
        for idx in turn_indices:
            turn_object_id[idx] = spec.object_id
            object_size[idx] = float(len(turn_indices))
            object_recency[idx] = recency
            object_memory_score[idx] = memory_score
            object_type_arrays[object_type][idx] = 1.0
        is_object_anchor[anchor_turn_index] = 1.0
        is_object_freshest[freshest_turn_index] = 1.0

    return SemanticObjectBundle(
        objects=objects,
        turn_object_id=turn_object_id,
        object_size=object_size,
        object_recency=object_recency,
        object_memory_score=object_memory_score,
        is_object_anchor=is_object_anchor,
        is_object_freshest=is_object_freshest,
        object_type_persona=object_type_arrays["persona"],
        object_type_event=object_type_arrays["event"],
        object_type_constraint=object_type_arrays["constraint"],
        object_type_update=object_type_arrays["update"],
        object_type_generic=object_type_arrays["generic"],
    )
