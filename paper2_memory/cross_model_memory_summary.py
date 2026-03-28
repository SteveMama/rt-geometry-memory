from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths
from paper2_memory.memory_critical_analysis import _parse_indices, _turn_label


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _conversation_map(paths: list[Path]) -> dict[str, ConversationRecord]:
    conversations = load_conversations_from_paths(paths)
    return {conversation.conversation_id: conversation for conversation in conversations}


def _support_user_turns(conversation: ConversationRecord, target_turn: int) -> list[int]:
    return [
        index
        for index, turn in enumerate(conversation.turns[:target_turn])
        if turn.role == "user" and _turn_label(turn.content) != "query"
    ]


def build_cross_model_summary(
    *,
    evaluation_csvs: list[Path],
    conversations: dict[str, ConversationRecord],
    budget_fraction: str,
    output_md: Path,
    output_csv: Path,
) -> None:
    model_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for csv_path in evaluation_csvs:
        for row in _load_rows(csv_path):
            if row["budget_fraction"] != budget_fraction:
                continue
            model_key = row.get("model_key", "")
            if model_key:
                model_rows[model_key].append(row)

    summary_rows: list[dict[str, object]] = []
    lines = [
        f"# Cross-Model Memory-Critical Summary @ budget {budget_fraction}",
        "",
        "This report compares geometry vs uniform on support-turn retention across models.",
        "",
    ]

    for model_key in sorted(model_rows):
        rows = model_rows[model_key]
        policy_rows = defaultdict(dict)
        for row in rows:
            policy_rows[(row["conversation_id"], row["target_turn"])][row["policy_name"]] = row

        cases = 0
        geometry_better = 0
        uniform_better = 0
        ties = 0
        geometry_latest_support_only = 0
        rescued_type_counts: Counter[str] = Counter()

        for key, rowset in sorted(policy_rows.items()):
            if "uniform" not in rowset or "geometry" not in rowset:
                continue
            conversation_id, target_turn_raw = key
            target_turn = int(target_turn_raw)
            conversation = conversations[conversation_id]
            support_turns = _support_user_turns(conversation, target_turn)
            if not support_turns:
                continue

            uniform_kept = set(_parse_indices(rowset["uniform"]["retained_turn_indices"]))
            geometry_kept = set(_parse_indices(rowset["geometry"]["retained_turn_indices"]))
            uniform_support = sorted(index for index in support_turns if index in uniform_kept)
            geometry_support = sorted(index for index in support_turns if index in geometry_kept)
            geometry_only_support = sorted(set(geometry_support) - set(uniform_support))

            cases += 1
            if len(geometry_support) > len(uniform_support):
                geometry_better += 1
            elif len(geometry_support) < len(uniform_support):
                uniform_better += 1
            else:
                ties += 1

            latest_support = support_turns[-1]
            if latest_support in geometry_kept and latest_support not in uniform_kept:
                geometry_latest_support_only += 1

            for index in geometry_only_support:
                rescued_type_counts[_turn_label(conversation.turns[index].content)] += 1

        summary_rows.append(
            {
                "model_key": model_key,
                "cases": cases,
                "geometry_better_cases": geometry_better,
                "uniform_better_cases": uniform_better,
                "ties": ties,
                "geometry_better_rate": geometry_better / cases if cases else 0.0,
                "uniform_better_rate": uniform_better / cases if cases else 0.0,
                "geometry_latest_support_only_cases": geometry_latest_support_only,
                "top_rescued_type": rescued_type_counts.most_common(1)[0][0] if rescued_type_counts else "",
                "top_rescued_type_count": rescued_type_counts.most_common(1)[0][1] if rescued_type_counts else 0,
                "rescued_type_counts": dict(rescued_type_counts),
            }
        )

        lines.extend(
            [
                f"## {model_key}",
                "",
                f"- Cases compared: {cases}",
                f"- Geometry better on support-turn retention: {geometry_better} ({geometry_better / cases:.3f})" if cases else "- Geometry better on support-turn retention: 0",
                f"- Uniform better on support-turn retention: {uniform_better} ({uniform_better / cases:.3f})" if cases else "- Uniform better on support-turn retention: 0",
                f"- Ties: {ties}",
                f"- Geometry keeps the latest support turn while uniform drops it: {geometry_latest_support_only}",
                "",
                "Rescued support types:",
            ]
        )
        for label, count in rescued_type_counts.most_common():
            lines.append(f"- {label}: {count}")
        lines.append("")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_key",
            "cases",
            "geometry_better_cases",
            "uniform_better_cases",
            "ties",
            "geometry_better_rate",
            "uniform_better_rate",
            "geometry_latest_support_only_cases",
            "top_rescued_type",
            "top_rescued_type_count",
            "rescued_type_counts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            normalized = dict(row)
            normalized["rescued_type_counts"] = str(normalized["rescued_type_counts"])
            writer.writerow(normalized)

    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize memory-critical support retention across models.")
    parser.add_argument("--evaluation-csvs", required=True, help="Comma-separated evaluation_rows.csv paths.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--budget-fraction", default="0.35")
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluation_csvs = [Path(item.strip()) for item in args.evaluation_csvs.split(",") if item.strip()]
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())
    build_cross_model_summary(
        evaluation_csvs=evaluation_csvs,
        conversations=_conversation_map(input_paths),
        budget_fraction=args.budget_fraction,
        output_md=args.output_md,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
