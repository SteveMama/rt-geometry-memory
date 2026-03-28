from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths
from paper2_memory.memory_critical_analysis import _parse_indices, _support_user_turns, _turn_label


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _conversation_map(paths: list[Path]) -> dict[str, ConversationRecord]:
    conversations = load_conversations_from_paths(paths)
    return {conversation.conversation_id: conversation for conversation in conversations}


def build_memory_critical_report(
    *,
    evaluation_rows: list[dict[str, str]],
    conversations: dict[str, ConversationRecord],
    model_key: str,
    budget_fraction: str,
    policy_name: str,
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
        f"# Paper 3 Memory-Critical Analysis: {model_key} | {policy_name} @ budget {budget_fraction}",
        "",
        "This report compares the selected Paper 3 policy against uniform on support-turn retention.",
        "",
    ]

    aggregate = Counter()
    rescued_turn_labels: Counter[str] = Counter()
    rescued_examples: list[dict[str, object]] = []

    for key, rows in sorted(policy_rows.items()):
        if "uniform" not in rows or policy_name not in rows:
            continue
        conversation_id, target_turn_raw = key
        target_turn = int(target_turn_raw)
        conversation = conversations[conversation_id]
        support_turns = _support_user_turns(conversation, target_turn)
        if not support_turns:
            continue

        baseline = rows["uniform"]
        candidate = rows[policy_name]
        uniform_kept = set(_parse_indices(baseline["retained_turn_indices"]))
        candidate_kept = set(_parse_indices(candidate["retained_turn_indices"]))
        uniform_support = sorted(index for index in support_turns if index in uniform_kept)
        candidate_support = sorted(index for index in support_turns if index in candidate_kept)
        candidate_only_support = sorted(set(candidate_support) - set(uniform_support))
        latest_support = support_turns[-1]
        delta_logit = float(candidate["logit_l2"]) - float(baseline["logit_l2"])

        aggregate["cases"] += 1
        if len(candidate_support) > len(uniform_support):
            aggregate["policy_better_cases"] += 1
        elif len(candidate_support) < len(uniform_support):
            aggregate["uniform_better_cases"] += 1
        else:
            aggregate["ties"] += 1
        if latest_support in candidate_kept and latest_support not in uniform_kept:
            aggregate["policy_preserved_latest_support_only"] += 1
        if int(candidate.get("compressed_segment_count", "0")) > 0:
            aggregate["compressed_cases"] += 1
        if int(candidate.get("compressed_segment_count", "0")) > 0 and len(candidate_support) >= len(uniform_support):
            aggregate["compressed_nonworse_cases"] += 1

        for index in candidate_only_support:
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
                    "compressed_segments": int(candidate.get("compressed_segment_count", "0")),
                }
            )

    if aggregate["cases"] == 0:
        lines.append("No comparable policy/uniform cases were found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "## Aggregate",
            "",
            f"- Cases compared: {aggregate['cases']}",
            f"- {policy_name} retains more support user turns than uniform: {aggregate['policy_better_cases']} cases",
            f"- Uniform retains more support user turns than {policy_name}: {aggregate['uniform_better_cases']} cases",
            f"- Ties: {aggregate['ties']}",
            f"- {policy_name} keeps the latest support turn while uniform drops it: {aggregate['policy_preserved_latest_support_only']} cases",
            f"- Cases with nonzero compressed segments: {aggregate['compressed_cases']}",
            f"- Compressed cases that are not worse than uniform on support retention: {aggregate['compressed_nonworse_cases']}",
            "",
            "## Rescued Support Types",
            "",
        ]
    )
    for label, count in rescued_turn_labels.most_common():
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.extend(["## Top Rescued Support Turns", ""])
    rescued_examples.sort(key=lambda item: float(item["delta_logit"]))
    for example in rescued_examples[:8]:
        lines.extend(
            [
                (
                    f"- {example['conversation_id']} t{example['rescued_turn']} "
                    f"({example['family']}, target t{example['target_turn']}): "
                    f"{example['rescued_label']}, delta logit L2 {example['delta_logit']:.3f}, "
                    f"compressed segments {example['compressed_segments']}"
                ),
                f"  content: {example['content']}",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Paper 3 support-turn retention against uniform.")
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--budget-fraction", default="0.35")
    parser.add_argument("--policy-name", default="geometry_keep_compress_drop")
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
        policy_name=args.policy_name,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
