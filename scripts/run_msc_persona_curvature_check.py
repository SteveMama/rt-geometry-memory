from __future__ import annotations

import argparse
import csv
import json
import sys
import statistics
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper1_geometry.conversations import ConversationRecord, TurnRecord, load_conversations
from paper1_geometry.geometry import normalize_rows, stabilized_curvature_series
from paper1_geometry.modeling import ConversationStateExtractor, resolve_model_spec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a targeted MSC support-vs-filler curvature check on manually labeled conversations."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Normalized MSC JSONL input path.")
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("benchmarks/msc_persona_curvature_labels.json"),
        help="Manual label JSON file.",
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
        "--output-dir",
        type=Path,
        default=Path("results/msc_persona_curvature/msc_persona_curvature_v1"),
    )
    return parser


def _load_labels(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _conversation_map(path: Path) -> dict[str, ConversationRecord]:
    conversations = load_conversations(path)
    return {conversation.conversation_id: conversation for conversation in conversations}


def _percentile_rank(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= value))


def main() -> None:
    args = build_arg_parser().parse_args()
    labels_payload = _load_labels(args.labels_path)
    conversation_specs = labels_payload.get("conversations", [])
    if not isinstance(conversation_specs, list) or not conversation_specs:
        raise ValueError(f"No labeled conversations found in {args.labels_path}")

    conversations = _conversation_map(args.input_path)
    missing = [
        spec.get("conversation_id")
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
        f"[msc_persona_curvature] model={args.model_key} resolved={model_name} device={extractor.device} "
        f"max_input_tokens={args.max_input_tokens}"
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(conversation_specs, start=1):
        conversation_id = str(spec["conversation_id"])
        support_turns = [int(value) for value in spec.get("support_turns", [])]
        filler_turns = [int(value) for value in spec.get("filler_turns", [])]
        conversation = conversations[conversation_id]

        print(f"[msc_persona_curvature] ({index}/{len(conversation_specs)}) extracting {conversation_id}")
        batch = extractor.extract_conversation(
            conversation,
            max_input_tokens=args.max_input_tokens,
        )
        unit_states, _ = normalize_rows(np.asarray(batch.states, dtype=np.float32))
        curvatures = stabilized_curvature_series(unit_states)
        turn_to_curvature = {turn_index + 1: float(value) for turn_index, value in enumerate(curvatures)}
        interior_values = np.asarray(list(turn_to_curvature.values()), dtype=np.float32)

        def collect(turn_indices: list[int], label: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for turn_index in turn_indices:
                curvature = turn_to_curvature.get(turn_index)
                if curvature is None:
                    continue
                turn = conversation.turns[turn_index]
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "label": label,
                        "turn_index": turn_index,
                        "role": turn.role,
                        "curvature": curvature,
                        "percentile_rank": _percentile_rank(interior_values, curvature),
                        "content": turn.content,
                    }
                )
            return rows

        support_rows = collect(support_turns, "support")
        filler_rows = collect(filler_turns, "filler")
        if not support_rows or not filler_rows:
            raise ValueError(f"Conversation {conversation_id} produced empty support or filler rows")

        selected_rows.extend(support_rows)
        selected_rows.extend(filler_rows)

        support_mean = statistics.mean(row["curvature"] for row in support_rows)
        filler_mean = statistics.mean(row["curvature"] for row in filler_rows)
        support_pct_mean = statistics.mean(row["percentile_rank"] for row in support_rows)
        filler_pct_mean = statistics.mean(row["percentile_rank"] for row in filler_rows)

        summary_rows.append(
            {
                "conversation_id": conversation_id,
                "support_turns": ",".join(str(value) for value in support_turns),
                "filler_turns": ",".join(str(value) for value in filler_turns),
                "support_mean_curvature": support_mean,
                "filler_mean_curvature": filler_mean,
                "delta_curvature": support_mean - filler_mean,
                "support_mean_percentile": support_pct_mean,
                "filler_mean_percentile": filler_pct_mean,
                "support_note": str(spec.get("support_note", "")),
                "filler_note": str(spec.get("filler_note", "")),
            }
        )

    deltas = [row["delta_curvature"] for row in summary_rows]
    support_pct = [row["support_mean_percentile"] for row in summary_rows]
    filler_pct = [row["filler_mean_percentile"] for row in summary_rows]
    aggregate = {
        "benchmark": labels_payload.get("benchmark", "msc_valid"),
        "model_key": args.model_key,
        "model_name": model_name,
        "device": extractor.device,
        "max_input_tokens": args.max_input_tokens,
        "num_conversations": len(summary_rows),
        "mean_delta_curvature": statistics.mean(deltas),
        "median_delta_curvature": statistics.median(deltas),
        "positive_delta_count": sum(delta > 0.0 for delta in deltas),
        "negative_delta_count": sum(delta < 0.0 for delta in deltas),
        "mean_support_percentile": statistics.mean(support_pct),
        "mean_filler_percentile": statistics.mean(filler_pct),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "aggregate": aggregate,
                "conversations": summary_rows,
            },
            handle,
            indent=2,
        )

    with (output_dir / "selected_turns.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "label",
                "turn_index",
                "role",
                "curvature",
                "percentile_rank",
                "content",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    with (output_dir / "conversation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "conversation_id",
                "support_turns",
                "filler_turns",
                "support_mean_curvature",
                "filler_mean_curvature",
                "delta_curvature",
                "support_mean_percentile",
                "filler_mean_percentile",
                "support_note",
                "filler_note",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# MSC Persona Curvature Check",
        "",
        f"- Benchmark: `{aggregate['benchmark']}`",
        f"- Model: `{aggregate['model_key']}` ({aggregate['model_name']})",
        f"- Device: `{aggregate['device']}`",
        f"- Conversations: {aggregate['num_conversations']}",
        f"- Mean support-filler curvature delta: {aggregate['mean_delta_curvature']:.4f}",
        f"- Median support-filler curvature delta: {aggregate['median_delta_curvature']:.4f}",
        f"- Positive deltas: {aggregate['positive_delta_count']}",
        f"- Negative deltas: {aggregate['negative_delta_count']}",
        f"- Mean support percentile: {aggregate['mean_support_percentile']:.4f}",
        f"- Mean filler percentile: {aggregate['mean_filler_percentile']:.4f}",
        "",
        "## Per-conversation summary",
        "",
        "| Conversation | Support mean curvature | Filler mean curvature | Delta | Support pct | Filler pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['conversation_id']} | {row['support_mean_curvature']:.4f} | "
            f"{row['filler_mean_curvature']:.4f} | {row['delta_curvature']:.4f} | "
            f"{row['support_mean_percentile']:.4f} | {row['filler_mean_percentile']:.4f} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Positive delta means the manually labeled support/persona turns had higher stabilized curvature "
        "than the chosen filler turns in that conversation."
    )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[msc_persona_curvature] Wrote outputs to {output_dir}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
