from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper1_geometry.conversations import ConversationRecord, load_conversations
from paper1_geometry.geometry import EPS, normalize_rows, sphere_distance
from paper1_geometry.modeling import ConversationStateExtractor, resolve_model_spec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a role-conditioned residual probe that compares parent-relative state "
            "residual norms for user and assistant turns."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Normalized conversation JSONL input path.")
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Optional benchmark label used in the output summary. Defaults to the input stem.",
    )
    parser.add_argument(
        "--model-key",
        default="qwen25_05b",
        help="Model key or full model name to use for hidden-state extraction.",
    )
    parser.add_argument("--limit-conversations", type=int, default=5, help="Number of conversations to analyze.")
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/role_residual_check/role_residual_check_v1"),
    )
    return parser


def _parent_relative_residual(parent: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    parallel_component = float(np.dot(current, parent))
    residual = current - parallel_component * parent
    residual_norm = float(np.linalg.norm(residual))
    angle = sphere_distance(parent, current)
    return residual_norm, angle


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _conversation_subset(conversations: list[ConversationRecord], limit: int) -> list[ConversationRecord]:
    if limit <= 0:
        return conversations
    return conversations[:limit]


def main() -> None:
    args = build_arg_parser().parse_args()
    conversations = _conversation_subset(load_conversations(args.input_path), args.limit_conversations)
    if not conversations:
        raise ValueError(f"No conversations found in {args.input_path}")

    benchmark_name = args.benchmark_name or args.input_path.stem
    model_spec = resolve_model_spec(args.model_key)
    model_name = model_spec.model_name if model_spec is not None else args.model_key
    extractor = ConversationStateExtractor(
        model_name,
        device=args.device,
        dtype=args.dtype,
    )
    print(
        f"[role_residual_check] benchmark={benchmark_name} model={args.model_key} "
        f"resolved={model_name} device={extractor.device} conversations={len(conversations)} "
        f"max_input_tokens={args.max_input_tokens}"
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    turn_rows: list[dict[str, Any]] = []
    conversation_rows: list[dict[str, Any]] = []

    aggregate_user_residuals: list[float] = []
    aggregate_assistant_residuals: list[float] = []
    aggregate_pair_deltas: list[float] = []

    for index, conversation in enumerate(conversations, start=1):
        print(
            f"[role_residual_check] ({index}/{len(conversations)}) extracting "
            f"{conversation.conversation_id}"
        )
        batch = extractor.extract_conversation(
            conversation,
            max_input_tokens=args.max_input_tokens,
        )
        unit_states, _ = normalize_rows(np.asarray(batch.states, dtype=np.float32))

        user_residuals: list[float] = []
        assistant_residuals: list[float] = []
        pair_deltas: list[float] = []

        for turn_index in range(1, len(conversation.turns)):
            parent = unit_states[turn_index - 1]
            current = unit_states[turn_index]
            residual_norm, angle = _parent_relative_residual(parent, current)
            role = conversation.turns[turn_index].role
            content = conversation.turns[turn_index].content
            parent_role = conversation.turns[turn_index - 1].role

            if role == "user":
                user_residuals.append(residual_norm)
                aggregate_user_residuals.append(residual_norm)
            elif role == "assistant":
                assistant_residuals.append(residual_norm)
                aggregate_assistant_residuals.append(residual_norm)

            turn_rows.append(
                {
                    "conversation_id": conversation.conversation_id,
                    "turn_index": turn_index,
                    "role": role,
                    "parent_role": parent_role,
                    "residual_norm": residual_norm,
                    "parent_angle": angle,
                    "content": content,
                }
            )

        for turn_index in range(2, len(conversation.turns)):
            if conversation.turns[turn_index].role != "assistant":
                continue
            if conversation.turns[turn_index - 1].role != "user":
                continue
            assistant_parent = unit_states[turn_index - 1]
            assistant_current = unit_states[turn_index]
            user_parent = unit_states[turn_index - 2]
            user_current = unit_states[turn_index - 1]
            assistant_residual_norm, _ = _parent_relative_residual(assistant_parent, assistant_current)
            user_residual_norm, _ = _parent_relative_residual(user_parent, user_current)
            delta = assistant_residual_norm - user_residual_norm
            pair_deltas.append(delta)
            aggregate_pair_deltas.append(delta)

        mean_user = _safe_mean(user_residuals)
        mean_assistant = _safe_mean(assistant_residuals)
        mean_pair_delta = _safe_mean(pair_deltas)
        conversation_rows.append(
            {
                "conversation_id": conversation.conversation_id,
                "num_turns": len(conversation.turns),
                "num_user_scored_turns": len(user_residuals),
                "num_assistant_scored_turns": len(assistant_residuals),
                "mean_user_residual": mean_user,
                "mean_assistant_residual": mean_assistant,
                "delta_user_minus_assistant": None
                if mean_user is None or mean_assistant is None
                else mean_user - mean_assistant,
                "pair_count": len(pair_deltas),
                "mean_pair_delta_assistant_minus_user": mean_pair_delta,
                "assistant_lower_pair_fraction": None
                if not pair_deltas
                else float(np.mean(np.asarray(pair_deltas, dtype=np.float32) < 0.0)),
            }
        )

    aggregate = {
        "benchmark": benchmark_name,
        "model_key": args.model_key,
        "model_name": model_name,
        "device": extractor.device,
        "num_conversations": len(conversation_rows),
        "num_scored_turns": len(turn_rows),
        "mean_user_residual": _safe_mean(aggregate_user_residuals),
        "mean_assistant_residual": _safe_mean(aggregate_assistant_residuals),
        "mean_user_minus_assistant": None
        if not aggregate_user_residuals or not aggregate_assistant_residuals
        else float(statistics.mean(aggregate_user_residuals) - statistics.mean(aggregate_assistant_residuals)),
        "conversation_user_higher_count": int(
            sum(
                (
                    row["delta_user_minus_assistant"] is not None
                    and row["delta_user_minus_assistant"] > 0.0
                )
                for row in conversation_rows
            )
        ),
        "conversation_assistant_higher_count": int(
            sum(
                (
                    row["delta_user_minus_assistant"] is not None
                    and row["delta_user_minus_assistant"] < 0.0
                )
                for row in conversation_rows
            )
        ),
        "mean_pair_delta_assistant_minus_user": _safe_mean(aggregate_pair_deltas),
        "assistant_lower_pair_count": int(sum(delta < 0.0 for delta in aggregate_pair_deltas)),
        "assistant_higher_pair_count": int(sum(delta > 0.0 for delta in aggregate_pair_deltas)),
        "pair_count": len(aggregate_pair_deltas),
        "assistant_lower_pair_fraction": None
        if not aggregate_pair_deltas
        else float(np.mean(np.asarray(aggregate_pair_deltas, dtype=np.float32) < 0.0)),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "aggregate": aggregate,
                "conversations": conversation_rows,
            },
            handle,
            indent=2,
        )

    with (output_dir / "turn_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "turn_index",
                "role",
                "parent_role",
                "residual_norm",
                "parent_angle",
                "content",
            ],
        )
        writer.writeheader()
        writer.writerows(turn_rows)

    with (output_dir / "conversation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "num_turns",
                "num_user_scored_turns",
                "num_assistant_scored_turns",
                "mean_user_residual",
                "mean_assistant_residual",
                "delta_user_minus_assistant",
                "pair_count",
                "mean_pair_delta_assistant_minus_user",
                "assistant_lower_pair_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(conversation_rows)

    lines = [
        "# Role Residual Check",
        "",
        f"- Benchmark: `{benchmark_name}`",
        f"- Model: `{args.model_key}` ({model_name})",
        f"- Device: `{extractor.device}`",
        f"- Conversations: {aggregate['num_conversations']}",
        f"- Scored turns: {aggregate['num_scored_turns']}",
        f"- Mean user residual: {_fmt(aggregate['mean_user_residual'])}",
        f"- Mean assistant residual: {_fmt(aggregate['mean_assistant_residual'])}",
        f"- Mean user-minus-assistant residual: {_fmt(aggregate['mean_user_minus_assistant'])}",
        f"- Conversations with higher user residual mean: {aggregate['conversation_user_higher_count']}",
        f"- Conversations with higher assistant residual mean: {aggregate['conversation_assistant_higher_count']}",
        f"- Paired assistant-after-user exchanges: {aggregate['pair_count']}",
        f"- Mean pair delta (assistant minus preceding user): {_fmt(aggregate['mean_pair_delta_assistant_minus_user'])}",
        f"- Assistant lower than preceding user fraction: {_fmt(aggregate['assistant_lower_pair_fraction'])}",
        "",
        "## Per-conversation summary",
        "",
        "| Conversation | User mean | Assistant mean | User-assistant delta | Pair count | Mean pair delta | Assistant-lower frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in conversation_rows:
        lines.append(
            f"| {row['conversation_id']} | {_fmt(row['mean_user_residual'])} | "
            f"{_fmt(row['mean_assistant_residual'])} | {_fmt(row['delta_user_minus_assistant'])} | "
            f"{row['pair_count']} | {_fmt(row['mean_pair_delta_assistant_minus_user'])} | "
            f"{_fmt(row['assistant_lower_pair_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "Residual norm is computed after projecting the current normalized turn state onto the orthogonal complement "
            "of the previous turn state. If assistant echo turns really add less new information than user injections, "
            "assistant residuals should be lower than user residuals, especially in adjacent user->assistant exchange pairs.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[role_residual_check] Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
