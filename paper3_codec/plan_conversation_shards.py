from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from paper1_geometry.conversations import load_conversations_from_paths

from .run_paper3 import _parse_families, _select_conversations, _sample_target_turns


def _estimated_cost(
    *,
    turn_count: int,
    target_turn_count: int,
) -> int:
    if target_turn_count <= 0:
        return max(turn_count, 1)
    return max(turn_count, 1) * max(target_turn_count, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan cost-balanced conversation shards for Paper 3 runs.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", default=None)
    parser.add_argument("--families", default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--skip-conversations", type=int, default=0)
    parser.add_argument("--target-turn-stride", type=int, default=4)
    parser.add_argument("--max-target-turns", type=int, default=16)
    parser.add_argument("--max-turns-per-conversation", type=int, default=None)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item.strip()) for item in args.extra_input_paths.split(",") if item.strip())

    conversations = _select_conversations(
        conversations=load_conversations_from_paths(input_paths),
        families=_parse_families(args.families),
        conversation_ids_path=None,
        skip_conversations=args.skip_conversations,
        limit_conversations=args.limit_conversations,
    )
    if not conversations:
        raise RuntimeError("No conversations available for shard planning.")

    plans = []
    for conversation in conversations:
        truncated_turn_count = (
            min(len(conversation.turns), args.max_turns_per_conversation)
            if args.max_turns_per_conversation is not None
            else len(conversation.turns)
        )
        target_turn_count = len(
            _sample_target_turns(
                num_turns=truncated_turn_count,
                min_history=args.min_history,
                stride=args.target_turn_stride,
                max_target_turns=args.max_target_turns,
            )
        )
        plans.append(
            {
                "conversation_id": conversation.conversation_id,
                "family": conversation.family,
                "turn_count": len(conversation.turns),
                "truncated_turn_count": truncated_turn_count,
                "target_turn_count": target_turn_count,
                "estimated_cost": _estimated_cost(
                    turn_count=truncated_turn_count,
                    target_turn_count=target_turn_count,
                ),
            }
        )

    shard_payloads = [
        {
            "shard_index": shard_index,
            "estimated_cost": 0,
            "conversation_count": 0,
            "conversation_ids": [],
            "items": [],
        }
        for shard_index in range(args.shard_count)
    ]
    for item in sorted(plans, key=lambda payload: (-payload["estimated_cost"], -payload["truncated_turn_count"], payload["conversation_id"])):
        shard = min(shard_payloads, key=lambda payload: (payload["estimated_cost"], payload["conversation_count"], payload["shard_index"]))
        shard["estimated_cost"] += int(item["estimated_cost"])
        shard["conversation_count"] += 1
        shard["conversation_ids"].append(item["conversation_id"])
        shard["items"].append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard in shard_payloads:
        shard_path = args.output_dir / f"shard_{shard['shard_index']}_ids.txt"
        shard_path.write_text("\n".join(shard["conversation_ids"]) + ("\n" if shard["conversation_ids"] else ""), encoding="utf-8")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_conversations": len(conversations),
        "shard_count": args.shard_count,
        "families": _parse_families(args.families),
        "limit_conversations": args.limit_conversations,
        "skip_conversations": args.skip_conversations,
        "target_turn_stride": args.target_turn_stride,
        "max_target_turns": args.max_target_turns,
        "max_turns_per_conversation": args.max_turns_per_conversation,
        "shards": shard_payloads,
    }
    (args.output_dir / "plan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
