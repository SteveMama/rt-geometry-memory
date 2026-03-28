from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_indices(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(item) for item in raw.split(",") if item.strip()]


def _conversation_map(paths: list[Path]) -> dict[str, ConversationRecord]:
    conversations = load_conversations_from_paths(paths)
    return {conversation.conversation_id: conversation for conversation in conversations}


def _support_user_turns(conversation: ConversationRecord, target_turn: int) -> list[int]:
    return [
        index
        for index, turn in enumerate(conversation.turns[:target_turn])
        if turn.role == "user" and _turn_label(turn.content) != "query"
    ]


def _turn_label(content: str) -> str:
    lowered = content.lower()
    if lowered.startswith("for the rest of this chat"):
        return "format rule"
    if lowered.startswith("remember"):
        return "base memory"
    if lowered.startswith("add these") or lowered.startswith("also require") or lowered.startswith("add these launch") or lowered.startswith("add these appointments"):
        return "support constraint"
    if lowered.startswith("show only") or lowered.startswith("now give me"):
        return "query"
    return "user support"


def build_memory_critical_report(
    *,
    evaluation_rows: list[dict[str, str]],
    conversations: dict[str, ConversationRecord],
    model_key: str,
    budget_fraction: str,
    output_path: Path,
) -> None:
    relevant = [
        row for row in evaluation_rows
        if row.get("model_key", model_key) == model_key and row["budget_fraction"] == budget_fraction
    ]
    policy_rows = defaultdict(dict)
    for row in relevant:
        policy_rows[(row["conversation_id"], row["target_turn"])][row["policy_name"]] = row

    lines = [
        f"# Memory-Critical Support Analysis: {model_key} @ budget {budget_fraction}",
        "",
        "This report treats earlier user turns as the support memory units on the hard Paper 2 stress set.",
        "The main question is whether geometry keeps the support turns that the final query depends on while uniform drops them.",
        "",
    ]

    aggregate = {
        "cases": 0,
        "geometry_better_cases": 0,
        "uniform_better_cases": 0,
        "ties": 0,
        "geometry_kept_all_support": 0,
        "uniform_kept_all_support": 0,
        "geometry_preserved_latest_support_only": 0,
    }
    family_stats = defaultdict(lambda: Counter())
    rescued_turn_labels: Counter[str] = Counter()
    rescued_examples: list[dict[str, object]] = []

    for key, rows in sorted(policy_rows.items()):
        if "uniform" not in rows or "geometry" not in rows:
            continue
        conversation_id, target_turn_raw = key
        target_turn = int(target_turn_raw)
        conversation = conversations[conversation_id]
        support_turns = _support_user_turns(conversation, target_turn)
        if not support_turns:
            continue

        uniform_kept = set(_parse_indices(rows["uniform"]["retained_turn_indices"]))
        geometry_kept = set(_parse_indices(rows["geometry"]["retained_turn_indices"]))
        uniform_support = sorted(index for index in support_turns if index in uniform_kept)
        geometry_support = sorted(index for index in support_turns if index in geometry_kept)
        geometry_only_support = sorted(set(geometry_support) - set(uniform_support))
        latest_support = support_turns[-1]
        geometry_logit = float(rows["geometry"]["logit_l2"])
        uniform_logit = float(rows["uniform"]["logit_l2"])
        delta_logit = geometry_logit - uniform_logit

        aggregate["cases"] += 1
        family_stats[conversation.family]["cases"] += 1
        if len(geometry_support) > len(uniform_support):
            aggregate["geometry_better_cases"] += 1
            family_stats[conversation.family]["geometry_better_cases"] += 1
        elif len(geometry_support) < len(uniform_support):
            aggregate["uniform_better_cases"] += 1
            family_stats[conversation.family]["uniform_better_cases"] += 1
        else:
            aggregate["ties"] += 1
            family_stats[conversation.family]["ties"] += 1

        if len(geometry_support) == len(support_turns):
            aggregate["geometry_kept_all_support"] += 1
            family_stats[conversation.family]["geometry_kept_all_support"] += 1
        if len(uniform_support) == len(support_turns):
            aggregate["uniform_kept_all_support"] += 1
            family_stats[conversation.family]["uniform_kept_all_support"] += 1
        if latest_support in geometry_kept and latest_support not in uniform_kept:
            aggregate["geometry_preserved_latest_support_only"] += 1
            family_stats[conversation.family]["geometry_preserved_latest_support_only"] += 1

        for index in geometry_only_support:
            rescued_turn_labels[_turn_label(conversation.turns[index].content)] += 1
            rescued_examples.append(
                {
                    "conversation_id": conversation_id,
                    "family": conversation.family,
                    "target_turn": target_turn,
                    "rescued_turn": index,
                    "rescued_label": _turn_label(conversation.turns[index].content),
                    "delta_logit": delta_logit,
                    "content": conversation.turns[index].content,
                }
            )

    if aggregate["cases"] == 0:
        lines.append("No comparable geometry/uniform cases were found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "## Aggregate",
            "",
            f"- Cases compared: {aggregate['cases']}",
            f"- Geometry retains more support user turns than uniform: {aggregate['geometry_better_cases']} cases",
            f"- Uniform retains more support user turns than geometry: {aggregate['uniform_better_cases']} cases",
            f"- Ties on support-turn retention: {aggregate['ties']} cases",
            f"- Geometry keeps all support user turns: {aggregate['geometry_kept_all_support']} cases",
            f"- Uniform keeps all support user turns: {aggregate['uniform_kept_all_support']} cases",
            f"- Geometry keeps the latest support user turn while uniform drops it: {aggregate['geometry_preserved_latest_support_only']} cases",
            "",
            "## By Family",
            "",
        ]
    )
    for family, stats in sorted(family_stats.items()):
        lines.extend(
            [
                f"### {family}",
                "",
                f"- Cases: {stats['cases']}",
                f"- Geometry better on support-turn retention: {stats['geometry_better_cases']}",
                f"- Uniform better on support-turn retention: {stats['uniform_better_cases']}",
                f"- Ties: {stats['ties']}",
                f"- Geometry keeps all support user turns: {stats['geometry_kept_all_support']}",
                f"- Uniform keeps all support user turns: {stats['uniform_kept_all_support']}",
                f"- Geometry keeps latest support user turn while uniform drops it: {stats['geometry_preserved_latest_support_only']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Geometry-Only Rescued Support Types",
            "",
        ]
    )
    for label, count in rescued_turn_labels.most_common():
        lines.append(f"- {label}: {count}")
    lines.append("")

    lines.extend(
        [
            "## Top Rescued Support Turns",
            "",
        ]
    )
    rescued_examples.sort(key=lambda item: float(item["delta_logit"]))
    for example in rescued_examples[:8]:
        lines.extend(
            [
                (
                    f"- {example['conversation_id']} t{example['rescued_turn']} "
                    f"({example['family']}, target t{example['target_turn']}): "
                    f"{example['rescued_label']}, delta logit L2 {example['delta_logit']:.3f}"
                ),
                f"  content: {example['content']}",
            ]
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze memory-critical support turn retention.")
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--budget-fraction", default="0.35")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    build_memory_critical_report(
        evaluation_rows=_load_rows(args.evaluation_csv),
        conversations=_conversation_map(input_paths),
        model_key=args.model_key,
        budget_fraction=args.budget_fraction,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
