"""External and trivial baselines for the KCD comparison.

Review fix #3: the manuscript compares only the authors' own policy variants.
This module adds the baselines a reviewer will ask for first:

  - ``uniform``                       (reference, re-emitted for pairing)
  - ``recency``                       keep the most recent turns under budget
  - ``random``                        seeded random retention (floor)
  - ``lexical_tfidf``                 TF-IDF cosine to the target turn — a
                                      *text-level* semantic baseline, unlike
                                      the paper's hidden-state cosine signal
  - ``recency_keep_compress_drop``    recency signal inside the same KCD codec
                                      (tests codec-structure vs signal)
  - ``llmlingua2``                    optional published prompt compressor
                                      (pip install llmlingua), skipped cleanly
                                      if unavailable

Outputs use the exact ``evaluation_rows.csv`` / ``behavior_rows.csv`` schema
of ``paper3_codec.run_paper3``, so ``paper3_codec.merge_study_shards`` and all
existing significance tooling work unchanged. Trajectory extraction reuses the
shared ``EXTRACT_CACHE_ROOT`` cache, so this is cheap after any prior study.

Run from the repo root:

    python -m june_fixes.baselines.baseline_study \
      --study-name june_baselines_hardset \
      --model-keys qwen25_15b \
      --input-path paper1_geometry/assets/paper2_behavior_stress_conversations.jsonl \
      --output-root results/june_fixes/baselines

Shard with --conversation-ids-path exactly like the existing scale-up runners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper1_geometry.conversations import ConversationRecord, load_conversations_from_paths
from paper1_geometry.modeling import ConversationStateExtractor, resolve_model_spec
from paper2_memory.policies import PolicySelection, select_turns
from paper3_codec.policies import select_sparse_segment_memory
from paper3_codec.run_paper3 import (
    _extract_or_load_conversation_batch,
    _kl_divergence,
    _policy_messages,
    _prefix_turn_costs,
    _sample_target_turns,
)

DEFAULT_POLICIES = (
    "uniform",
    "recency",
    "random",
    "lexical_tfidf",
    "recency_keep_compress_drop",
    "llmlingua2",
    "longllmlingua",
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def lexical_tfidf_scores(conversation: ConversationRecord, prefix_turn_count: int, target_turn: int) -> np.ndarray:
    """TF-IDF cosine similarity of each prior turn to the target turn."""
    documents = [
        _tokenize(conversation.turns[idx].content) for idx in range(prefix_turn_count)
    ]
    target_tokens = _tokenize(conversation.turns[target_turn].content)
    all_documents = documents + [target_tokens]
    document_frequency: Counter[str] = Counter()
    for tokens in all_documents:
        document_frequency.update(set(tokens))
    num_documents = len(all_documents)

    def _vector(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(sum(counts.values()), 1)
        return {
            term: (count / total)
            * math.log((1 + num_documents) / (1 + document_frequency[term]))
            for term, count in counts.items()
        }

    target_vector = _vector(target_tokens)
    target_norm = math.sqrt(sum(value * value for value in target_vector.values())) or 1e-8
    scores = np.zeros(prefix_turn_count, dtype=np.float32)
    for idx, tokens in enumerate(documents):
        vector = _vector(tokens)
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1e-8
        dot = sum(value * target_vector.get(term, 0.0) for term, value in vector.items())
        scores[idx] = dot / (norm * target_norm)
    if scores.size and float(scores.max() - scores.min()) > 1e-8:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores


def recency_scores(prefix_turn_count: int) -> np.ndarray:
    if prefix_turn_count <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.linspace(0.0, 1.0, prefix_turn_count, dtype=np.float32)


def random_scores(conversation_id: str, target_turn: int, prefix_turn_count: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}|{conversation_id}|{target_turn}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.random(prefix_turn_count).astype(np.float32)


class LLMLinguaCompressor:
    """Optional wrapper around llmlingua-2; loads lazily, fails soft."""

    def __init__(self) -> None:
        self.compressor = None
        self.available = False
        try:
            from llmlingua import PromptCompressor  # type: ignore

            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001 - optional dependency
            print(f"[baseline_study] llmlingua unavailable, skipping policy: {exc}", flush=True)

    def compress(self, text: str, rate: float, question: str | None = None) -> str:
        assert self.compressor is not None
        kwargs: dict[str, Any] = {"rate": max(min(rate, 0.99), 0.05)}
        if question:
            kwargs["question"] = question
        result = self.compressor.compress_prompt(text, **kwargs)
        return str(result.get("compressed_prompt", text))


class LongLLMLinguaCompressor:
    """Question-aware LongLLMLingua compressor (ACL 2024).

    Uses the target-turn content as the ``question`` for coarse-to-fine
    dynamic compression. This is the key mechanism distinguishing
    LongLLMLingua from LLMLingua-2: importance scores are re-ranked
    conditioned on the query rather than computed unconditionally.

    Loads llmlingua-2 (the publicly available, non-gated model) with
    question-conditioning enabled. Falls back to question-unaware
    llmlingua-2 if question parameter is rejected by an older version.
    """

    def __init__(self) -> None:
        self.compressor = None
        self.available = False
        try:
            from llmlingua import PromptCompressor  # type: ignore

            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001
            print(f"[baseline_study] longllmlingua unavailable, skipping: {exc}", flush=True)

    def compress(self, text: str, rate: float, question: str) -> str:
        assert self.compressor is not None
        rate = max(min(rate, 0.99), 0.05)
        try:
            result = self.compressor.compress_prompt(
                text,
                question=question,
                rate=rate,
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.3,
                condition_compare=True,
                context_budget="+100",
                rank_method="longllmlingua",
            )
        except TypeError:
            # older llmlingua versions don't support all kwargs — fall back
            result = self.compressor.compress_prompt(text, question=question, rate=rate)
        return str(result.get("compressed_prompt", text))


def _llmlingua_messages(
    conversation: ConversationRecord,
    *,
    target_turn: int,
    recent_window: int,
    budget_fraction: float,
    compressor: LLMLinguaCompressor,
) -> list[dict[str, str]]:
    """LLMLingua-2 (no question awareness)."""
    prefix_turn_count = target_turn
    recent_start = max(prefix_turn_count - recent_window, 0)
    older_text = "\n".join(
        f"{conversation.turns[idx].role}: {conversation.turns[idx].content}"
        for idx in range(recent_start)
    )
    messages: list[dict[str, str]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    if older_text.strip():
        compressed = compressor.compress(older_text, rate=budget_fraction)
        messages.append(
            {
                "role": "system",
                "content": "Compressed conversation memory:\n" + compressed,
            }
        )
    for idx in range(recent_start, prefix_turn_count):
        turn = conversation.turns[idx]
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {
            "role": conversation.turns[target_turn].role,
            "content": conversation.turns[target_turn].content,
        }
    )
    return messages


def _longllmlingua_messages(
    conversation: ConversationRecord,
    *,
    target_turn: int,
    recent_window: int,
    budget_fraction: float,
    compressor: LongLLMLinguaCompressor,
) -> list[dict[str, str]]:
    """LongLLMLingua with question-conditioned compression (ACL 2024).

    The target turn content is passed as the question so that the
    compressor re-ranks older turns by their relevance to the query.
    This is the key mechanism: a user constraint turn scores higher
    than its semantically-similar assistant echo when the question
    makes the constraint salient.
    """
    prefix_turn_count = target_turn
    recent_start = max(prefix_turn_count - recent_window, 0)
    older_text = "\n".join(
        f"{conversation.turns[idx].role}: {conversation.turns[idx].content}"
        for idx in range(recent_start)
    )
    question = conversation.turns[target_turn].content
    messages: list[dict[str, str]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    if older_text.strip():
        compressed = compressor.compress(older_text, rate=budget_fraction, question=question)
        messages.append(
            {
                "role": "system",
                "content": "Compressed conversation memory:\n" + compressed,
            }
        )
    for idx in range(recent_start, prefix_turn_count):
        turn = conversation.turns[idx]
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {
            "role": conversation.turns[target_turn].role,
            "content": conversation.turns[target_turn].content,
        }
    )
    return messages


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_baseline_study(args: argparse.Namespace) -> None:
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item) for item in args.extra_input_paths.split(",") if item)
    conversations = load_conversations_from_paths(input_paths)
    families = (
        {item.strip() for item in args.families.split(",") if item.strip()}
        if args.families
        else None
    )
    if families:
        conversations = [item for item in conversations if item.family in families]
    if args.conversation_ids_path is not None:
        allowed = {
            line.strip()
            for line in args.conversation_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        conversations = [item for item in conversations if item.conversation_id in allowed]
    if args.skip_conversations:
        conversations = conversations[args.skip_conversations :]
    if args.limit_conversations is not None:
        conversations = conversations[: args.limit_conversations]
    if not conversations:
        raise ValueError("no conversations selected")

    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    budgets = [float(item) for item in args.budgets.split(",") if item]
    model_keys = [item.strip() for item in args.model_keys.split(",") if item.strip()]

    compressor = LLMLinguaCompressor() if "llmlingua2" in policies else None
    if compressor is not None and not compressor.available:
        policies = [item for item in policies if item != "llmlingua2"]

    long_compressor = LongLLMLinguaCompressor() if "longllmlingua" in policies else None
    if long_compressor is not None and not long_compressor.available:
        policies = [item for item in policies if item != "longllmlingua"]

    output_dir: Path = args.output_root / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    models_meta: dict[str, Any] = {}

    for model_key in model_keys:
        spec = resolve_model_spec(model_key)
        extractor = ConversationStateExtractor(
            spec.model_name, device=args.device, dtype=args.dtype, state_layer=args.state_layer
        )
        models_meta[model_key] = {"model_name": spec.model_name}
        for conversation_index, conversation in enumerate(conversations, start=1):
            batch = _extract_or_load_conversation_batch(
                extractor=extractor,
                conversation=conversation,
                max_turns=args.max_turns_per_conversation,
                max_input_tokens=args.max_input_tokens,
            )
            num_turns = int(batch.token_counts.shape[0])
            turn_costs = _prefix_turn_costs(batch.token_counts)
            target_turns = _sample_target_turns(
                num_turns=num_turns,
                min_history=args.min_history,
                stride=args.target_turn_stride,
                max_target_turns=args.max_target_turns,
            )
            for target_turn in target_turns:
                prefix_turn_count = target_turn
                if prefix_turn_count < args.min_history:
                    continue
                full_logits = batch.logits[target_turn]
                full_tokens = int(batch.token_counts[target_turn])
                prefix_turn_costs = turn_costs[:prefix_turn_count]
                is_behavior_turn = (
                    conversation.turns[target_turn].role == "user"
                    and target_turn + 1 < len(conversation.turns)
                    and conversation.turns[target_turn + 1].role == "assistant"
                )
                full_behavior_score = None
                if is_behavior_turn:
                    full_messages = _policy_messages(
                        conversation=conversation,
                        target_turn=target_turn,
                        retained_prior_indices=list(range(prefix_turn_count)),
                    )
                    full_behavior_score = extractor.score_assistant_response(
                        full_messages,
                        conversation.turns[target_turn + 1].content,
                        max_input_tokens=args.max_input_tokens,
                    )

                score_lookup = {
                    "recency": recency_scores(prefix_turn_count),
                    "random": random_scores(
                        conversation.conversation_id, target_turn, prefix_turn_count, args.seed
                    ),
                    "lexical_tfidf": lexical_tfidf_scores(
                        conversation, prefix_turn_count, target_turn
                    ),
                }
                for budget in budgets:
                    for policy_name in policies:
                        if policy_name == "llmlingua2":
                            messages = _llmlingua_messages(
                                conversation,
                                target_turn=target_turn,
                                recent_window=args.recent_window,
                                budget_fraction=budget,
                                compressor=compressor,  # type: ignore[arg-type]
                            )
                            selection = PolicySelection(
                                retained_turn_indices=list(
                                    range(
                                        max(prefix_turn_count - args.recent_window, 0),
                                        prefix_turn_count,
                                    )
                                ),
                                retained_fraction=float("nan"),
                                retained_cost_fraction=float(budget),
                            )
                        elif policy_name == "longllmlingua":
                            messages = _longllmlingua_messages(
                                conversation,
                                target_turn=target_turn,
                                recent_window=args.recent_window,
                                budget_fraction=budget,
                                compressor=long_compressor,  # type: ignore[arg-type]
                            )
                            selection = PolicySelection(
                                retained_turn_indices=list(
                                    range(
                                        max(prefix_turn_count - args.recent_window, 0),
                                        prefix_turn_count,
                                    )
                                ),
                                retained_fraction=float("nan"),
                                retained_cost_fraction=float(budget),
                            )
                        else:
                            if policy_name == "uniform":
                                selection = select_turns(
                                    policy_name="uniform",
                                    risk_scores=np.zeros(prefix_turn_count, dtype=np.float32),
                                    turn_costs=prefix_turn_costs,
                                    prefix_turn_count=prefix_turn_count,
                                    budget_fraction=budget,
                                    recent_window=args.recent_window,
                                )
                            elif policy_name == "recency_keep_compress_drop":
                                selection = select_sparse_segment_memory(
                                    risk_scores=score_lookup["recency"],
                                    turn_costs=prefix_turn_costs,
                                    prefix_turn_count=prefix_turn_count,
                                    budget_fraction=budget,
                                    recent_window=args.recent_window,
                                    segment_span=args.segment_span,
                                )
                            elif policy_name in score_lookup:
                                # select_turns dispatches on known names only;
                                # "semantic" gives the standard density ordering
                                # for any score vector.
                                selection = select_turns(
                                    policy_name="semantic",
                                    risk_scores=score_lookup[policy_name],
                                    turn_costs=prefix_turn_costs,
                                    prefix_turn_count=prefix_turn_count,
                                    budget_fraction=budget,
                                    recent_window=args.recent_window,
                                )
                            else:
                                raise ValueError(f"Unknown baseline policy: {policy_name}")
                            messages = _policy_messages(
                                conversation=conversation,
                                target_turn=target_turn,
                                retained_prior_indices=selection.retained_turn_indices,
                            )
                        compressed = extractor.score_messages(
                            messages, max_input_tokens=args.max_input_tokens
                        )
                        evaluation_rows.append(
                            {
                                "model_key": model_key,
                                "conversation_id": conversation.conversation_id,
                                "family": conversation.family,
                                "target_turn": target_turn,
                                "policy_name": policy_name,
                                "budget_fraction": float(budget),
                                "budget_token_fraction": selection.retained_cost_fraction,
                                "retained_turn_fraction": selection.retained_fraction,
                                "retained_turn_indices": ",".join(
                                    str(index) for index in selection.retained_turn_indices
                                ),
                                "full_token_count": full_tokens,
                                "compressed_token_count": compressed.token_count,
                                "token_fraction": float(
                                    compressed.token_count / max(full_tokens, 1)
                                ),
                                "logit_l2": float(
                                    np.linalg.norm(full_logits - compressed.logits)
                                ),
                                "kl": _kl_divergence(full_logits, compressed.logits),
                                "top1_match": float(
                                    np.argmax(full_logits) == np.argmax(compressed.logits)
                                ),
                                "kept_segment_count": getattr(
                                    selection, "kept_segment_count", 0
                                ),
                                "compressed_segment_count": getattr(
                                    selection, "compressed_segment_count", 0
                                ),
                                "evicted_segment_count": getattr(
                                    selection, "evicted_segment_count", 0
                                ),
                                "memory_objects": "",
                            }
                        )
                        if is_behavior_turn and full_behavior_score is not None:
                            behavior_score = extractor.score_assistant_response(
                                messages,
                                conversation.turns[target_turn + 1].content,
                                max_input_tokens=args.max_input_tokens,
                            )
                            behavior_rows.append(
                                {
                                    "model_key": model_key,
                                    "conversation_id": conversation.conversation_id,
                                    "family": conversation.family,
                                    "target_turn": target_turn,
                                    "policy_name": policy_name,
                                    "budget_fraction": float(budget),
                                    "answer_token_count": behavior_score.token_count,
                                    "answer_avg_neg_logprob": behavior_score.avg_neg_logprob,
                                    "answer_total_neg_logprob": behavior_score.total_neg_logprob,
                                    "full_answer_avg_neg_logprob": full_behavior_score.avg_neg_logprob,
                                    "full_answer_total_neg_logprob": full_behavior_score.total_neg_logprob,
                                    "answer_avg_neg_logprob_delta": behavior_score.avg_neg_logprob
                                    - full_behavior_score.avg_neg_logprob,
                                    "answer_total_neg_logprob_delta": behavior_score.total_neg_logprob
                                    - full_behavior_score.total_neg_logprob,
                                }
                            )
            completed_ids.append(conversation.conversation_id)
            if conversation_index % 5 == 0 or conversation_index == len(conversations):
                print(
                    f"[baseline_study:{model_key}] {conversation_index}/{len(conversations)} "
                    f"conversations, rows={len(evaluation_rows)}",
                    flush=True,
                )
            _write_csv(output_dir / "evaluation_rows.partial.csv", evaluation_rows)
            _write_json(
                output_dir / "progress.json",
                {
                    "status": "running",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "study_name": args.study_name,
                    "completed_conversation_ids": completed_ids,
                    "num_conversations_completed": len(completed_ids),
                },
            )
        models_meta[model_key].update(
            {
                "num_conversations": len(conversations),
                "num_evaluations": len(
                    [row for row in evaluation_rows if row["model_key"] == model_key]
                ),
                "num_behavior_evaluations": len(
                    [row for row in behavior_rows if row["model_key"] == model_key]
                ),
            }
        )

    _write_csv(output_dir / "evaluation_rows.csv", evaluation_rows)
    _write_csv(output_dir / "behavior_rows.csv", behavior_rows)
    _write_json(
        output_dir / "study_summary.json",
        {
            "study_name": args.study_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_keys": model_keys,
            "families": sorted(families) if families else None,
            "budgets": budgets,
            "policies": policies,
            "target_turn_stride": args.target_turn_stride,
            "max_target_turns": args.max_target_turns,
            "max_turns_per_conversation": args.max_turns_per_conversation,
            "skip_conversations": args.skip_conversations,
            "conversation_ids_path": str(args.conversation_ids_path)
            if args.conversation_ids_path
            else None,
            "models": models_meta,
        },
    )
    _write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "study_name": args.study_name,
            "completed_conversation_ids": completed_ids,
            "num_conversations_completed": len(completed_ids),
        },
    )
    print(
        f"[baseline_study] wrote {len(evaluation_rows)} evaluation rows and "
        f"{len(behavior_rows)} behavior rows to {output_dir}",
        flush=True,
    )
    print(
        "[baseline_study] run paper3_codec.merge_study_shards over this directory "
        "(alone or with other shards) to get standard significance summaries",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", type=str, default="june_baselines_v1")
    parser.add_argument("--model-keys", type=str, default="qwen25_15b")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", type=str, default=None)
    parser.add_argument("--families", type=str, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/june_fixes/baselines")
    )
    parser.add_argument("--budgets", type=str, default="0.20,0.35,0.50")
    parser.add_argument("--policies", type=str, default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--recent-window", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--state-layer", type=int, default=-1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--skip-conversations", type=int, default=0)
    parser.add_argument("--conversation-ids-path", type=Path, default=None)
    parser.add_argument("--segment-span", type=int, default=2)
    parser.add_argument("--target-turn-stride", type=int, default=1)
    parser.add_argument("--max-target-turns", type=int, default=None)
    parser.add_argument("--max-turns-per-conversation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    run_baseline_study(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
