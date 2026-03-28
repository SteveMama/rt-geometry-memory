from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths


def _parse_indices(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(item) for item in raw.split(",") if item.strip()]


def _conversation_map(paths: list[Path]) -> dict[str, ConversationRecord]:
    conversations = load_conversations_from_paths(paths)
    return {conversation.conversation_id: conversation for conversation in conversations}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_turn(turn_index: int, role: str, content: str, *, uniform_kept: bool, geometry_kept: bool) -> str:
    marker = []
    marker.append("U" if uniform_kept else "-")
    marker.append("G" if geometry_kept else "-")
    return f"[{''.join(marker)}] t{turn_index} {role}: {content}"


def build_case_report(
    *,
    evaluation_rows: list[dict[str, str]],
    behavior_rows: list[dict[str, str]],
    conversations: dict[str, ConversationRecord],
    model_key: str,
    budget_fraction: str,
    output_path: Path,
    top_n: int,
) -> None:
    eval_subset = [
        row for row in evaluation_rows
        if row.get("model_key", model_key) == model_key and row["budget_fraction"] == budget_fraction
    ]
    behavior_subset = [
        row for row in behavior_rows
        if row.get("model_key", model_key) == model_key and row["budget_fraction"] == budget_fraction
    ]
    uniform_eval = {
        (row["conversation_id"], row["target_turn"]): row
        for row in eval_subset
        if row["policy_name"] == "uniform"
    }
    geometry_eval = {
        (row["conversation_id"], row["target_turn"]): row
        for row in eval_subset
        if row["policy_name"] == "geometry"
    }
    uniform_behavior = {
        (row["conversation_id"], row["target_turn"]): row
        for row in behavior_subset
        if row["policy_name"] == "uniform"
    }
    geometry_behavior = {
        (row["conversation_id"], row["target_turn"]): row
        for row in behavior_subset
        if row["policy_name"] == "geometry"
    }

    cases: list[dict[str, object]] = []
    for key, geometry_row in geometry_eval.items():
        baseline = uniform_eval.get(key)
        if baseline is None:
            continue
        conversation_id, target_turn = key
        behavior_delta = 0.0
        if key in geometry_behavior and key in uniform_behavior:
            behavior_delta = (
                float(geometry_behavior[key]["answer_avg_neg_logprob"])
                - float(uniform_behavior[key]["answer_avg_neg_logprob"])
            )
        cases.append(
            {
                "conversation_id": conversation_id,
                "target_turn": int(target_turn),
                "logit_delta": float(geometry_row["logit_l2"]) - float(baseline["logit_l2"]),
                "behavior_delta": behavior_delta,
                "geometry_row": geometry_row,
                "uniform_row": baseline,
            }
        )

    ranked = sorted(cases, key=lambda item: (float(item["logit_delta"]), float(item["behavior_delta"])))[:top_n]
    lines: list[str] = [
        f"# Paper 2 Case Analysis: {model_key} @ budget {budget_fraction}",
        "",
        "## Top Geometry Wins",
        "",
    ]
    for case in ranked:
        lines.append(
            f"- {case['conversation_id']} target_turn={case['target_turn']}: "
            f"delta logit L2 {case['logit_delta']:.3f}, delta answer avg NLL {case['behavior_delta']:.4f}"
        )
    lines.append("")

    for rank, case in enumerate(ranked, start=1):
        conversation_id = str(case["conversation_id"])
        target_turn = int(case["target_turn"])
        conversation = conversations[conversation_id]
        geometry_row = case["geometry_row"]
        uniform_row = case["uniform_row"]
        geometry_kept = set(_parse_indices(str(geometry_row["retained_turn_indices"])))
        uniform_kept = set(_parse_indices(str(uniform_row["retained_turn_indices"])))
        geometry_only = sorted(geometry_kept - uniform_kept)

        lines.extend(
            [
                f"## Case {rank}: {conversation_id}",
                "",
                f"- Family: {conversation.family}",
                f"- Target turn: {target_turn}",
                f"- Geometry vs uniform logit delta: {case['logit_delta']:.3f}",
                f"- Geometry vs uniform answer avg NLL delta: {case['behavior_delta']:.4f}",
                f"- Uniform retained turns: {sorted(uniform_kept)}",
                f"- Geometry retained turns: {sorted(geometry_kept)}",
                f"- Geometry-only retained turns: {geometry_only}",
                "",
                "### Prefix Turns",
                "",
            ]
        )
        for turn_index, turn in enumerate(conversation.turns[:target_turn]):
            lines.append(
                _format_turn(
                    turn_index,
                    turn.role,
                    turn.content,
                    uniform_kept=turn_index in uniform_kept,
                    geometry_kept=turn_index in geometry_kept,
                )
            )
        lines.extend(
            [
                "",
                "### Query And Gold Answer",
                "",
                f"user t{target_turn}: {conversation.turns[target_turn].content}",
            ]
        )
        if target_turn + 1 < len(conversation.turns):
            lines.append(f"assistant t{target_turn + 1}: {conversation.turns[target_turn + 1].content}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate qualitative Paper 2 case analysis.")
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--budget-fraction", default="0.35")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    build_case_report(
        evaluation_rows=_load_rows(args.evaluation_csv),
        behavior_rows=_load_rows(args.behavior_csv),
        conversations=_conversation_map(input_paths),
        model_key=args.model_key,
        budget_fraction=args.budget_fraction,
        output_path=args.output_path,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
