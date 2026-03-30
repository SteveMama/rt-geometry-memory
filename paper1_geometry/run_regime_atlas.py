from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .conversations import load_conversations_from_paths
from .modeling import ConversationStateExtractor, list_default_models, resolve_model_spec
from .regime_diagnostics import (
    build_conversation_series_diagnostic,
    build_saturation_report,
    plot_representative_series,
    select_representative_diagnostics,
    write_conversation_series_summary,
)
from .regime_atlas import (
    build_atlas_report,
    cluster_segment_rows,
    extract_segment_rows,
    family_regime_counts,
    write_atlas_outputs,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "regime_atlas"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a geometric regime atlas over conversation segments.")
    parser.add_argument("--study-name", default="regime_atlas_v1")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--rank-energy", type=float, default=0.95)
    parser.add_argument("--max-segment-len", type=int, default=8)
    parser.add_argument("--min-segment-len", type=int, default=3)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-models", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        for spec in list_default_models():
            print(f"{spec.key:12s}  {spec.model_name:36s}  {spec.notes}")
        return

    if args.model_key is not None:
        spec = resolve_model_spec(args.model_key)
        if spec is None:
            valid = ", ".join(item.key for item in list_default_models())
            parser.error(f"Unknown --model-key '{args.model_key}'. Valid keys: {valid}")
        args.model_name = spec.model_name

    input_paths = [args.input_path]
    if args.extra_input_paths.strip():
        input_paths.extend(Path(item) for item in args.extra_input_paths.split(",") if item.strip())

    conversations = load_conversations_from_paths(input_paths)
    if args.limit_conversations is not None:
        conversations = conversations[: args.limit_conversations]
    if not conversations:
        parser.error("No conversations loaded.")

    family_counts = Counter(conversation.family for conversation in conversations)
    output_dir = args.output_root / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[regime_atlas] study={args.study_name} model={args.model_key or args.model_name}")
    print(f"[regime_atlas] output={output_dir}")
    print(f"[regime_atlas] loaded {len(conversations)} conversations across families={dict(sorted(family_counts.items()))}")

    extractor = ConversationStateExtractor(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        state_layer=args.state_layer,
    )
    print(
        f"[regime_atlas] extractor device={extractor.device} max_input_tokens={args.max_input_tokens} "
        f"cluster_count={args.cluster_count}"
    )

    rows = []
    series_diagnostics = []
    for index, conversation in enumerate(conversations, start=1):
        print(f"[regime_atlas] ({index}/{len(conversations)}) extracting {conversation.family}:{conversation.conversation_id}")
        batch = extractor.extract_conversation(
            conversation,
            max_turns=args.max_turns,
            max_input_tokens=args.max_input_tokens,
        )
        series_diagnostics.append(build_conversation_series_diagnostic(conversation, batch))
        segment_rows = extract_segment_rows(
            conversation,
            batch,
            rank_energy=args.rank_energy,
            max_segment_len=args.max_segment_len,
            min_segment_len=args.min_segment_len,
        )
        rows.extend(segment_rows)
        print(
            f"[regime_atlas] ({index}/{len(conversations)}) turns_used={batch.states.shape[0]} "
            f"segments={len(segment_rows)} running_total={len(rows)}"
        )

    _, cluster_summaries = cluster_segment_rows(
        rows,
        cluster_count=args.cluster_count,
        seed=args.seed,
    )
    family_rows = family_regime_counts(rows)
    report_text = build_atlas_report(
        rows,
        cluster_summaries,
        family_rows,
        model_key=args.model_key or args.model_name,
        input_paths=[str(path) for path in input_paths],
        cluster_count=args.cluster_count,
        max_segment_len=args.max_segment_len,
        min_segment_len=args.min_segment_len,
    )
    metadata = {
        "study_name": args.study_name,
        "model_name": args.model_name,
        "model_key": args.model_key,
        "device": extractor.device,
        "dtype": args.dtype,
        "state_layer": args.state_layer,
        "input_paths": [str(path) for path in input_paths],
        "limit_conversations": args.limit_conversations,
        "max_input_tokens": args.max_input_tokens,
        "max_turns": args.max_turns,
        "rank_energy": args.rank_energy,
        "max_segment_len": args.max_segment_len,
        "min_segment_len": args.min_segment_len,
        "cluster_count": args.cluster_count,
        "seed": args.seed,
        "num_conversations": len(conversations),
        "num_segments": len(rows),
    }
    write_atlas_outputs(
        output_dir,
        rows,
        cluster_summaries,
        family_rows,
        report_text,
        metadata=metadata,
    )
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    representatives = select_representative_diagnostics(series_diagnostics)
    plot_representative_series(
        representatives,
        diagnostics_dir / "representative_turn_series.png",
        log_curvature=False,
    )
    plot_representative_series(
        representatives,
        diagnostics_dir / "representative_turn_series_log.png",
        log_curvature=True,
    )
    write_conversation_series_summary(
        series_diagnostics,
        diagnostics_dir / "conversation_series_summary.csv",
    )
    (diagnostics_dir / "curvature_saturation_report.md").write_text(
        build_saturation_report(rows, series_diagnostics),
        encoding="utf-8",
    )
    print(f"[regime_atlas] wrote atlas outputs to {output_dir}")
    print(json.dumps({"families": dict(sorted(family_counts.items())), "segments": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
