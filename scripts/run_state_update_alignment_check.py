from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper1_geometry.conversations import ConversationRecord, TurnRecord, load_conversations
from paper1_geometry.geometry import EPS, normalize_rows, segment_reference, sphere_log_map, sphere_parallel_transport
from paper1_geometry.modeling import ConversationStateExtractor, resolve_model_spec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic state-update smoke check comparing semantic similarity "
            "against tangent-space directional alignment."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("benchmarks/state_update_synthetic_conversations.jsonl"),
        help="Synthetic state-update conversation JSONL.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("benchmarks/state_update_synthetic_labels.json"),
        help="JSON labels pointing to original/update/query turns.",
    )
    parser.add_argument(
        "--model-key",
        default="qwen25_05b",
        help="Model key or full model name to use for hidden-state extraction.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=-0.2,
        help="Directional-alignment threshold for counting a state-update reversal.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.5,
        help="Semantic similarity threshold for the original-update pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/state_update_alignment/state_update_synthetic_v1"),
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=0,
        help="Seed for the single sampled same-role non-update control turn.",
    )
    return parser


def _load_labels(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _conversation_map(path: Path) -> dict[str, ConversationRecord]:
    return {conversation.conversation_id: conversation for conversation in load_conversations(path)}


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < EPS:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def _isolated_turn_state(
    extractor: ConversationStateExtractor,
    turn: TurnRecord,
    *,
    max_input_tokens: int,
    cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    key = (turn.role, turn.content)
    cached = cache.get(key)
    if cached is not None:
        return cached
    state = extractor.score_messages(
        [{"role": turn.role, "content": turn.content}],
        max_input_tokens=max_input_tokens,
    ).state.astype(np.float32)
    isolated_state = state / max(float(np.linalg.norm(state)), EPS)
    cache[key] = isolated_state
    return isolated_state


def _turn_entry_vectors(unit_states: np.ndarray) -> dict[int, np.ndarray]:
    n_states = unit_states.shape[0]
    if n_states < 2:
        return {}
    reference = segment_reference(unit_states, 0, n_states - 1)
    vectors: dict[int, np.ndarray] = {}
    for turn_index in range(1, n_states):
        step = sphere_log_map(unit_states[turn_index - 1], unit_states[turn_index])
        transported = sphere_parallel_transport(unit_states[turn_index - 1], reference, step)
        vectors[turn_index] = transported.astype(np.float32)
    return vectors


def _turn_exit_vectors(unit_states: np.ndarray) -> dict[int, np.ndarray]:
    n_states = unit_states.shape[0]
    if n_states < 2:
        return {}
    reference = segment_reference(unit_states, 0, n_states - 1)
    vectors: dict[int, np.ndarray] = {}
    for turn_index in range(0, n_states - 1):
        step = sphere_log_map(unit_states[turn_index], unit_states[turn_index + 1])
        transported = sphere_parallel_transport(unit_states[turn_index], reference, step)
        vectors[turn_index] = transported.astype(np.float32)
    return vectors


def _state_position_vectors(unit_states: np.ndarray) -> dict[int, np.ndarray]:
    n_states = unit_states.shape[0]
    if n_states == 0:
        return {}
    reference = segment_reference(unit_states, 0, n_states - 1)
    vectors: dict[int, np.ndarray] = {}
    for turn_index in range(n_states):
        vectors[turn_index] = sphere_log_map(reference, unit_states[turn_index]).astype(np.float32)
    return vectors


def main() -> None:
    args = build_arg_parser().parse_args()

    labels_payload = _load_labels(args.labels_path)
    conversation_specs = labels_payload.get("conversations", [])
    if not isinstance(conversation_specs, list) or not conversation_specs:
        raise ValueError(f"No conversation labels found in {args.labels_path}")

    conversations = _conversation_map(args.input_path)
    missing = [
        str(spec.get("conversation_id"))
        for spec in conversation_specs
        if str(spec.get("conversation_id")) not in conversations
    ]
    if missing:
        raise ValueError(f"Missing labeled conversations in {args.input_path}: {missing}")

    model_spec = resolve_model_spec(args.model_key)
    model_name = model_spec.model_name if model_spec is not None else args.model_key
    extractor = ConversationStateExtractor(
        model_name,
        device=args.device,
        dtype=args.dtype,
    )
    print(
        f"[state_update_alignment] model={args.model_key} resolved={model_name} device={extractor.device} "
        f"max_input_tokens={args.max_input_tokens}"
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    isolated_cache: dict[tuple[str, str], np.ndarray] = {}
    rng = random.Random(args.control_seed)
    pair_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(conversation_specs, start=1):
        conversation_id = str(spec["conversation_id"])
        conversation = conversations[conversation_id]
        original_turn = int(spec["original_turn"])
        update_turn = int(spec["update_turn"])
        query_turn = int(spec["query_turn"])

        print(f"[state_update_alignment] ({index}/{len(conversation_specs)}) extracting {conversation_id}")
        batch = extractor.extract_conversation(
            conversation,
            max_input_tokens=args.max_input_tokens,
        )
        unit_states, _ = normalize_rows(np.asarray(batch.states, dtype=np.float32))
        entry_vectors = _turn_entry_vectors(unit_states)
        exit_vectors = _turn_exit_vectors(unit_states)
        state_positions = _state_position_vectors(unit_states)
        if original_turn not in exit_vectors or update_turn not in exit_vectors:
            raise ValueError(
                f"Conversation {conversation_id} needs labeled turns with a following turn for exit vectors, "
                f"got original={original_turn}, update={update_turn}"
            )

        original_exit = exit_vectors[original_turn]
        update_exit = exit_vectors[update_turn]
        original_entry = entry_vectors.get(original_turn)
        update_entry = entry_vectors.get(update_turn)
        original_state_position = state_positions[original_turn]

        control_candidates = [
            turn_index
            for turn_index, role in enumerate(batch.turn_roles)
            if role == conversation.turns[update_turn].role
            and turn_index in entry_vectors
            and turn_index not in {original_turn, update_turn, query_turn}
        ]
        if not control_candidates:
            raise ValueError(f"Conversation {conversation_id} has no same-role non-update control turns")
        sampled_control_turn = rng.choice(control_candidates)
        sampled_control_entry = entry_vectors[sampled_control_turn]
        control_cross_values = [_cosine(original_state_position, entry_vectors[turn_index]) for turn_index in control_candidates]
        state_update_entry_cross = _cosine(original_state_position, update_entry) if update_entry is not None else 0.0
        sampled_control_entry_cross = _cosine(original_state_position, sampled_control_entry)
        mean_control_entry_cross = statistics.mean(control_cross_values)
        min_control_entry_cross = min(control_cross_values)
        update_more_negative_than_mean_control = state_update_entry_cross < mean_control_entry_cross
        update_more_negative_than_all_controls = state_update_entry_cross < min_control_entry_cross

        original_isolated = _isolated_turn_state(
            extractor,
            conversation.turns[original_turn],
            max_input_tokens=args.max_input_tokens,
            cache=isolated_cache,
        )
        update_isolated = _isolated_turn_state(
            extractor,
            conversation.turns[update_turn],
            max_input_tokens=args.max_input_tokens,
            cache=isolated_cache,
        )
        query_isolated = _isolated_turn_state(
            extractor,
            conversation.turns[query_turn],
            max_input_tokens=args.max_input_tokens,
            cache=isolated_cache,
        )

        alignment = _cosine(original_exit, update_exit)
        entry_alignment = _cosine(original_entry, update_entry) if original_entry is not None and update_entry is not None else 0.0
        original_entry_update_exit_alignment = (
            _cosine(original_entry, update_exit) if original_entry is not None else 0.0
        )
        original_exit_update_entry_alignment = (
            _cosine(original_exit, update_entry) if update_entry is not None else 0.0
        )
        pair_semantic_similarity = _cosine(original_isolated, update_isolated)
        query_to_original_similarity = _cosine(query_isolated, original_isolated)
        query_to_update_similarity = _cosine(query_isolated, update_isolated)
        contextual_state_similarity = _cosine(unit_states[original_turn], unit_states[update_turn])

        pair_rows.append(
            {
                "conversation_id": conversation_id,
                "topic": str(spec.get("topic", "")),
                "original_turn": original_turn,
                "update_turn": update_turn,
                "query_turn": query_turn,
                "directional_alignment": alignment,
                "entry_alignment": entry_alignment,
                "original_entry_update_exit_alignment": original_entry_update_exit_alignment,
                "original_exit_update_entry_alignment": original_exit_update_entry_alignment,
                "state_update_entry_cross": state_update_entry_cross,
                "sampled_control_turn": sampled_control_turn,
                "sampled_control_entry_cross": sampled_control_entry_cross,
                "mean_control_entry_cross": mean_control_entry_cross,
                "min_control_entry_cross": min_control_entry_cross,
                "num_control_candidates": len(control_candidates),
                "update_more_negative_than_mean_control": update_more_negative_than_mean_control,
                "update_more_negative_than_all_controls": update_more_negative_than_all_controls,
                "original_exit_norm": float(np.linalg.norm(original_exit)),
                "update_exit_norm": float(np.linalg.norm(update_exit)),
                "original_entry_norm": float(np.linalg.norm(original_entry)) if original_entry is not None else 0.0,
                "update_entry_norm": float(np.linalg.norm(update_entry)) if update_entry is not None else 0.0,
                "pair_semantic_similarity": pair_semantic_similarity,
                "query_to_original_semantic_similarity": query_to_original_similarity,
                "query_to_update_semantic_similarity": query_to_update_similarity,
                "query_similarity_gap": query_to_update_similarity - query_to_original_similarity,
                "contextual_state_similarity": contextual_state_similarity,
                "alignment_lt_threshold": alignment < args.alignment_threshold,
                "semantic_gt_threshold": pair_semantic_similarity > args.semantic_threshold,
                "joint_success": alignment < args.alignment_threshold
                and pair_semantic_similarity > args.semantic_threshold,
                "original_text": conversation.turns[original_turn].content,
                "update_text": conversation.turns[update_turn].content,
                "query_text": conversation.turns[query_turn].content,
                "expected_current_answer": str(spec.get("expected_current_answer", "")),
            }
        )

    alignments = [row["directional_alignment"] for row in pair_rows]
    entry_alignments = [row["entry_alignment"] for row in pair_rows]
    entry_exit_alignments = [row["original_entry_update_exit_alignment"] for row in pair_rows]
    exit_entry_alignments = [row["original_exit_update_entry_alignment"] for row in pair_rows]
    state_update_crosses = [row["state_update_entry_cross"] for row in pair_rows]
    sampled_control_crosses = [row["sampled_control_entry_cross"] for row in pair_rows]
    mean_control_crosses = [row["mean_control_entry_cross"] for row in pair_rows]
    pair_semantics = [row["pair_semantic_similarity"] for row in pair_rows]
    query_gaps = [row["query_similarity_gap"] for row in pair_rows]

    aggregate = {
        "benchmark": labels_payload.get("benchmark", "state_update_synthetic_v1"),
        "description": labels_payload.get("description", ""),
        "model_key": args.model_key,
        "model_name": model_name,
        "device": extractor.device,
        "max_input_tokens": args.max_input_tokens,
        "num_conversations": len(pair_rows),
        "alignment_threshold": args.alignment_threshold,
        "semantic_threshold": args.semantic_threshold,
        "mean_directional_alignment": statistics.mean(alignments),
        "median_directional_alignment": statistics.median(alignments),
        "negative_alignment_count": sum(value < 0.0 for value in alignments),
        "alignment_lt_threshold_count": sum(value < args.alignment_threshold for value in alignments),
        "mean_entry_alignment": statistics.mean(entry_alignments),
        "mean_original_entry_update_exit_alignment": statistics.mean(entry_exit_alignments),
        "mean_original_exit_update_entry_alignment": statistics.mean(exit_entry_alignments),
        "negative_entry_exit_count": sum(value < 0.0 for value in entry_exit_alignments),
        "negative_exit_entry_count": sum(value < 0.0 for value in exit_entry_alignments),
        "mean_state_update_entry_cross": statistics.mean(state_update_crosses),
        "mean_sampled_control_entry_cross": statistics.mean(sampled_control_crosses),
        "mean_control_entry_cross": statistics.mean(mean_control_crosses),
        "negative_state_update_entry_cross_count": sum(value < 0.0 for value in state_update_crosses),
        "negative_sampled_control_cross_count": sum(value < 0.0 for value in sampled_control_crosses),
        "update_more_negative_than_mean_control_count": sum(bool(row["update_more_negative_than_mean_control"]) for row in pair_rows),
        "update_more_negative_than_all_controls_count": sum(bool(row["update_more_negative_than_all_controls"]) for row in pair_rows),
        "mean_pair_semantic_similarity": statistics.mean(pair_semantics),
        "median_pair_semantic_similarity": statistics.median(pair_semantics),
        "semantic_gt_threshold_count": sum(value > args.semantic_threshold for value in pair_semantics),
        "joint_success_count": sum(bool(row["joint_success"]) for row in pair_rows),
        "mean_query_similarity_gap": statistics.mean(query_gaps),
        "median_query_similarity_gap": statistics.median(query_gaps),
    }
    aggregate["passes_falsification_gate"] = bool(
        aggregate["alignment_lt_threshold_count"] >= 7
        and aggregate["semantic_gt_threshold_count"] >= 7
        and aggregate["joint_success_count"] >= 7
    )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "aggregate": aggregate,
                "pairs": pair_rows,
            },
            handle,
            indent=2,
        )

    with (output_dir / "pair_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "topic",
                "original_turn",
                "update_turn",
                "query_turn",
                "directional_alignment",
                "entry_alignment",
                "original_entry_update_exit_alignment",
                "original_exit_update_entry_alignment",
                "state_update_entry_cross",
                "sampled_control_turn",
                "sampled_control_entry_cross",
                "mean_control_entry_cross",
                "min_control_entry_cross",
                "num_control_candidates",
                "update_more_negative_than_mean_control",
                "update_more_negative_than_all_controls",
                "original_exit_norm",
                "update_exit_norm",
                "original_entry_norm",
                "update_entry_norm",
                "pair_semantic_similarity",
                "query_to_original_semantic_similarity",
                "query_to_update_semantic_similarity",
                "query_similarity_gap",
                "contextual_state_similarity",
                "alignment_lt_threshold",
                "semantic_gt_threshold",
                "joint_success",
                "original_text",
                "update_text",
                "query_text",
                "expected_current_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    lines = [
        "# Synthetic State-Update Alignment Check",
        "",
        f"- Benchmark: `{aggregate['benchmark']}`",
        f"- Model: `{aggregate['model_key']}` ({aggregate['model_name']})",
        f"- Device: `{aggregate['device']}`",
        f"- Conversations: {aggregate['num_conversations']}",
        f"- Alignment threshold: `{aggregate['alignment_threshold']}`",
        f"- Semantic threshold: `{aggregate['semantic_threshold']}`",
        f"- Mean directional alignment: {aggregate['mean_directional_alignment']:.4f}",
        f"- Median directional alignment: {aggregate['median_directional_alignment']:.4f}",
        f"- Negative alignments: {aggregate['negative_alignment_count']} / {aggregate['num_conversations']}",
        (
            f"- Alignments below threshold: {aggregate['alignment_lt_threshold_count']} / "
            f"{aggregate['num_conversations']}"
        ),
        f"- Mean entry alignment (diagnostic): {aggregate['mean_entry_alignment']:.4f}",
        f"- Mean original-entry / update-exit alignment (diagnostic): {aggregate['mean_original_entry_update_exit_alignment']:.4f}",
        f"- Mean original-exit / update-entry alignment (diagnostic): {aggregate['mean_original_exit_update_entry_alignment']:.4f}",
        f"- Negative mixed entry/exit counts: {aggregate['negative_entry_exit_count']} / {aggregate['num_conversations']} and {aggregate['negative_exit_entry_count']} / {aggregate['num_conversations']}",
        f"- Mean state-position / update-entry cross (diagnostic): {aggregate['mean_state_update_entry_cross']:.4f}",
        f"- Mean sampled non-update control cross (diagnostic): {aggregate['mean_sampled_control_entry_cross']:.4f}",
        f"- Mean all-control cross (diagnostic): {aggregate['mean_control_entry_cross']:.4f}",
        f"- Negative state-update cross count: {aggregate['negative_state_update_entry_cross_count']} / {aggregate['num_conversations']}",
        f"- Negative sampled-control cross count: {aggregate['negative_sampled_control_cross_count']} / {aggregate['num_conversations']}",
        f"- Update more negative than mean control: {aggregate['update_more_negative_than_mean_control_count']} / {aggregate['num_conversations']}",
        f"- Update more negative than all controls: {aggregate['update_more_negative_than_all_controls_count']} / {aggregate['num_conversations']}",
        f"- Mean original-update semantic similarity: {aggregate['mean_pair_semantic_similarity']:.4f}",
        f"- Semantic similarities above threshold: {aggregate['semantic_gt_threshold_count']} / {aggregate['num_conversations']}",
        f"- Joint threshold passes: {aggregate['joint_success_count']} / {aggregate['num_conversations']}",
        f"- Mean query-similarity gap (update - original): {aggregate['mean_query_similarity_gap']:.4f}",
        f"- Falsification gate pass: `{aggregate['passes_falsification_gate']}`",
        "",
        "## Interpretation",
        "",
        "Directional alignment is computed exactly from the stated formula: the transported step from each labeled turn to the following turn, all moved into one shared tangent space.",
        "Semantic similarity is computed from isolated-turn hidden states using the same base model.",
        "For debugging, the CSV also includes entry-step alignment and the two mixed entry/exit alignments.",
        "",
        "## Per-conversation results",
        "",
        "| Conversation | Exit align | Entry align | State/update cross | Mean control cross | Pair semantic | Joint pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['conversation_id']} | {row['directional_alignment']:.4f} | "
            f"{row['entry_alignment']:.4f} | "
            f"{row['state_update_entry_cross']:.4f} | "
            f"{row['mean_control_entry_cross']:.4f} | "
            f"{row['pair_semantic_similarity']:.4f} | "
            f"{'yes' if row['joint_success'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Labeled update pairs",
            "",
        ]
    )
    for row in pair_rows:
        lines.extend(
            [
                f"### {row['conversation_id']}",
                "",
                f"- Original turn `{row['original_turn']}`: {row['original_text']}",
                f"- Update turn `{row['update_turn']}`: {row['update_text']}",
                f"- Query turn `{row['query_turn']}`: {row['query_text']}",
                f"- Expected current answer: `{row['expected_current_answer']}`",
                f"- Sampled control turn `{row['sampled_control_turn']}` with cross `{row['sampled_control_entry_cross']:.4f}`",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[state_update_alignment] Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
