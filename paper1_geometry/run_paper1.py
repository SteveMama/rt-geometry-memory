from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import (
    aggregate_boundary_summary_rows,
    analyze_trajectory,
    save_analysis_json,
    save_analysis_npz,
)
from .boundary_features import lexical_shift_scores
from .conversations import load_conversations, load_conversations_from_paths
from .modeling import (
    ConversationStateExtractor,
    list_default_models,
    resolve_model_spec,
)


DEFAULT_INPUT = Path(__file__).resolve().parent / "assets" / "sample_conversations.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "paper1"


def run_model_experiment(
    model_name: str,
    device: str | None,
    dtype: str,
    state_layer: int,
    input_path: Path,
    input_paths: list[Path] | None,
    output_dir: Path,
    limit_conversations: int | None,
    max_turns: int | None,
    max_input_tokens: int,
    rank_energy: float,
    max_segment_len: int,
    min_segment_len: int,
    model_key: str | None = None,
) -> dict[str, object]:
    conversations = load_conversations_from_paths(input_paths) if input_paths else load_conversations(input_path)
    if limit_conversations is not None:
        conversations = conversations[:limit_conversations]

    extractor = ConversationStateExtractor(
        model_name=model_name,
        device=device,
        dtype=dtype,
        state_layer=state_layer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, object] = {
        "model_name": model_name,
        "model_key": model_key,
        "device": extractor.device,
        "dtype": dtype,
        "state_layer": state_layer,
        "transformers_version": extractor.transformers_version,
        "input_path": str(input_path),
        "input_paths": [str(path) for path in input_paths] if input_paths else [str(input_path)],
        "conversations": [],
    }

    for conversation in conversations:
        batch = extractor.extract_conversation(
            conversation,
            max_turns=max_turns,
            max_input_tokens=max_input_tokens,
        )
        initial_analysis = analyze_trajectory(
            states=batch.states,
            logits=batch.logits,
            reconstructed_logits=batch.logits,
            gold_boundaries=conversation.boundary_indices,
            lexical_boundary_scores=lexical_shift_scores(conversation),
            rank_energy=rank_energy,
            max_segment_len=max_segment_len,
            min_segment_len=min_segment_len,
        )
        reconstructed_logits = extractor.project_logits(initial_analysis["artifacts"]["reconstructed_states"])
        analysis = analyze_trajectory(
            states=batch.states,
            logits=batch.logits,
            reconstructed_logits=reconstructed_logits,
            gold_boundaries=conversation.boundary_indices,
            lexical_boundary_scores=lexical_shift_scores(conversation),
            rank_energy=rank_energy,
            max_segment_len=max_segment_len,
            min_segment_len=min_segment_len,
        )

        base_name = f"{conversation.family}_{conversation.conversation_id}_{model_name.split('/')[-1]}".replace("/", "_")
        json_path = output_dir / f"{base_name}.json"
        npz_path = output_dir / f"{base_name}.npz"
        save_analysis_json(analysis, json_path)
        save_analysis_npz(analysis, batch.states, batch.logits, npz_path)

        run_summary["conversations"].append(
            {
                "conversation_id": conversation.conversation_id,
                "family": conversation.family,
                "num_turns": int(batch.states.shape[0]),
                "gold_boundaries": conversation.boundary_indices or [],
                "output_json": str(json_path),
                "output_npz": str(npz_path),
                "summary": analysis["summary"],
                "boundary_variants": analysis.get("boundary_variants", {}),
            }
        )

    aggregate = [row["summary"] for row in run_summary["conversations"]]
    boundary_aggregate = aggregate_boundary_summary_rows(aggregate)
    run_summary["aggregate"] = {
        "num_conversations": len(aggregate),
        "mean_rank95": float(np.mean([row["mean_rank95"] for row in aggregate])) if aggregate else 0.0,
        "mean_curvature": float(np.mean([row["mean_curvature"] for row in aggregate])) if aggregate else 0.0,
        "mean_turning_angle": float(np.mean([row.get("mean_turning_angle", 0.0) for row in aggregate])) if aggregate else 0.0,
        "mean_rank_jump": float(np.mean([row.get("mean_rank_jump", 0.0) for row in aggregate])) if aggregate else 0.0,
        "mean_subspace_shift": float(np.mean([row.get("mean_subspace_shift", 0.0) for row in aggregate])) if aggregate else 0.0,
        "mean_boundary_score": float(np.mean([row.get("mean_boundary_score", 0.0) for row in aggregate])) if aggregate else 0.0,
        "mean_boundary_prominence": float(np.mean([row.get("mean_boundary_prominence", 0.0) for row in aggregate])) if aggregate else 0.0,
        "mean_logit_l2": float(np.mean([row["mean_logit_l2"] for row in aggregate])) if aggregate else 0.0,
        "mean_kl": float(np.mean([row["mean_kl"] for row in aggregate])) if aggregate else 0.0,
        "mean_corr_geodesic_vs_logit_l2": float(
            np.mean([row["corr_geodesic_vs_logit_l2"] for row in aggregate])
        )
        if aggregate
        else 0.0,
        "mean_corr_geodesic_vs_kl": float(
            np.mean([row["corr_geodesic_vs_kl"] for row in aggregate])
        )
        if aggregate
        else 0.0,
        **boundary_aggregate,
    }

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper 1 geometry bootstrap runner.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--rank-energy", type=float, default=0.95)
    parser.add_argument("--max-segment-len", type=int, default=6)
    parser.add_argument("--min-segment-len", type=int, default=3)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--detailed-models", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        for spec in list_default_models():
            if args.detailed_models:
                print(
                    f"{spec.key:12s}  {spec.model_name:36s}  min_tf={spec.min_transformers_version:6s}  "
                    f"params={spec.parameter_size:6s}  ctx={spec.context_length:5d}  {spec.mac_notes}"
                )
            else:
                print(f"{spec.key:12s}  {spec.model_name:36s}  {spec.notes}")
        return

    if args.model_key is not None:
        spec = resolve_model_spec(args.model_key)
        if spec is None:
            valid = ", ".join(item.key for item in list_default_models())
            parser.error(f"Unknown --model-key '{args.model_key}'. Valid keys: {valid}")
        args.model_name = spec.model_name

    run_model_experiment(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        state_layer=args.state_layer,
        input_path=args.input_path,
        input_paths=None,
        output_dir=args.output_dir,
        limit_conversations=args.limit_conversations,
        max_turns=args.max_turns,
        max_input_tokens=args.max_input_tokens,
        rank_energy=args.rank_energy,
        max_segment_len=args.max_segment_len,
        min_segment_len=args.min_segment_len,
        model_key=args.model_key,
    )
    print(f"Wrote Paper 1 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
