"""Generation-based QA accuracy for completed Paper 3 studies.

Review fix #1: the manuscript reports only logit L2 and answer NLL. This
module replays the exact turn selections recorded in an existing study's
``evaluation_rows.csv``, generates an answer under each compressed context,
and scores it against the gold assistant turn with exact match, token F1,
and normalized containment. A full-context reference generation is scored
per (conversation, target turn) so every policy gets a delta against the
uncompressed model.

The module never re-runs policy selection, so QA numbers describe the same
runs the paper already reports.

Run from the repo root, e.g.:

    python -m june_fixes.qa_accuracy.qa_accuracy_study \
      --study-dir results/paper3/studies/paper3_gate1_refinement_msc_valid \
      --input-path benchmarks/msc_valid_normalized.jsonl \
      --model-key qwen25_15b \
      --output-dir results/june_fixes/qa_accuracy/msc_valid

Sharding mirrors the existing scale-up pattern: pass
``--conversation-ids-path shard_N_ids.txt`` per worker and merge with
``june_fixes.qa_accuracy.merge_qa_shards``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
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
from paper1_geometry.modeling import ConversationStateExtractor
from paper3_codec.run_paper3 import _policy_messages
from paper3_codec.stats import bootstrap_mean_ci, collapse_rows_by_keys, paired_signflip_test

GENERATION_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", "8"))

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def contains_answer(prediction: str, gold: str) -> float:
    norm_gold = normalize_answer(gold)
    if not norm_gold:
        return 0.0
    return float(norm_gold in normalize_answer(prediction))


def _score_prediction(prediction: str, gold: str) -> dict[str, float]:
    return {
        "qa_exact_match": exact_match(prediction, gold),
        "qa_token_f1": token_f1(prediction, gold),
        "qa_contains": contains_answer(prediction, gold),
    }


def generate_batch(
    extractor: ConversationStateExtractor,
    message_lists: list[list[dict[str, str]]],
    *,
    max_input_tokens: int | None,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Greedy batched generation through the extractor's model/tokenizer."""
    tokenizer = extractor.tokenizer
    model = extractor.model
    torch = extractor.torch
    rendered = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in message_lists
    ]
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: list[str] = []
    try:
        for start in range(0, len(rendered), max(batch_size, 1)):
            chunk = rendered[start : start + max(batch_size, 1)]
            batch = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=max_input_tokens is not None,
                max_length=max_input_tokens,
            )
            batch = {key: value.to(extractor.device) for key, value in batch.items()}
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = batch["input_ids"].shape[1]
            for row_index in range(len(chunk)):
                text = tokenizer.decode(
                    generated[row_index][prompt_length:], skip_special_tokens=True
                )
                outputs.append(text.strip())
    finally:
        tokenizer.padding_side = original_side
    return outputs


def _load_evaluation_rows(study_dir: Path, model_key: str) -> list[dict[str, Any]]:
    csv_path = study_dir / "evaluation_rows.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing evaluation_rows.csv under {study_dir}")
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_key") != model_key:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows for model_key={model_key} in {csv_path}")
    return rows


def _behavior_eligible(conversation: ConversationRecord, target_turn: int) -> bool:
    return (
        0 <= target_turn < len(conversation.turns)
        and conversation.turns[target_turn].role == "user"
        and target_turn + 1 < len(conversation.turns)
        and conversation.turns[target_turn + 1].role == "assistant"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_qa_rows(
    qa_rows: list[dict[str, Any]],
    *,
    reference_policy: str = "uniform",
    rng_seed: int = 1234,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate means and paired significance vs the reference policy."""
    rng = np.random.default_rng(rng_seed)
    metrics = ["qa_exact_match", "qa_token_f1", "qa_contains", "qa_full_token_f1"]
    summary: dict[str, Any] = {}
    significance: dict[str, Any] = {}

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in qa_rows:
        by_model.setdefault(str(row["model_key"]), []).append(row)

    for model_key, model_rows in by_model.items():
        summary[model_key] = {}
        significance[model_key] = {}
        budgets = sorted({float(row["budget_fraction"]) for row in model_rows})
        policies = sorted({str(row["policy_name"]) for row in model_rows})
        for budget in budgets:
            budget_key = f"{budget:.2f}"
            summary[model_key][budget_key] = {}
            significance[model_key][budget_key] = {}
            budget_rows = [row for row in model_rows if float(row["budget_fraction"]) == budget]
            reference_lookup = {
                (row["conversation_id"], row["target_turn"]): row
                for row in budget_rows
                if row["policy_name"] == reference_policy
            }
            for policy in policies:
                policy_rows = [row for row in budget_rows if row["policy_name"] == policy]
                if not policy_rows:
                    continue
                summary[model_key][budget_key][policy] = {
                    metric: bootstrap_mean_ci(
                        [float(row[metric]) for row in policy_rows], rng=rng
                    )
                    for metric in metrics
                    if metric in policy_rows[0]
                }
                if policy == reference_policy or not reference_lookup:
                    continue
                paired = []
                for row in policy_rows:
                    ref = reference_lookup.get((row["conversation_id"], row["target_turn"]))
                    if ref is None:
                        continue
                    paired.append(
                        {
                            "conversation_id": row["conversation_id"],
                            "delta_qa_exact_match": float(row["qa_exact_match"])
                            - float(ref["qa_exact_match"]),
                            "delta_qa_token_f1": float(row["qa_token_f1"])
                            - float(ref["qa_token_f1"]),
                            "delta_qa_contains": float(row["qa_contains"])
                            - float(ref["qa_contains"]),
                        }
                    )
                if not paired:
                    continue
                policy_sig: dict[str, Any] = {"num_pairs": len(paired)}
                for metric in ("delta_qa_exact_match", "delta_qa_token_f1", "delta_qa_contains"):
                    row_values = np.asarray([item[metric] for item in paired], dtype=np.float64)
                    conversation_rows = collapse_rows_by_keys(
                        paired, metric_keys=[metric], group_keys=["conversation_id"]
                    )
                    conv_values = np.asarray(
                        [item[metric] for item in conversation_rows], dtype=np.float64
                    )
                    policy_sig[metric] = {
                        "row_level": {
                            **bootstrap_mean_ci(row_values.tolist(), rng=rng),
                            "p_value": paired_signflip_test(row_values, rng=rng),
                        },
                        "conversation_level": {
                            **bootstrap_mean_ci(conv_values.tolist(), rng=rng),
                            "p_value": paired_signflip_test(conv_values, rng=rng),
                        },
                    }
                significance[model_key][budget_key][policy] = policy_sig
    return summary, significance


def write_qa_report(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    significance: dict[str, Any],
    num_rows: int,
    study_name: str,
) -> None:
    lines = [f"# QA Accuracy Report: {study_name}", "", f"Scored generations: {num_rows}", ""]
    for model_key, budgets in sorted(summary.items()):
        lines.append(f"## {model_key}")
        for budget_key, policies in sorted(budgets.items()):
            lines.append(f"\n### budget {budget_key}\n")
            lines.append("| policy | EM | token F1 | contains | vs-uniform ΔF1 (conv p) |")
            lines.append("|---|---|---|---|---|")
            for policy, stats in sorted(policies.items()):
                em = stats.get("qa_exact_match", {}).get("mean", float("nan"))
                f1 = stats.get("qa_token_f1", {}).get("mean", float("nan"))
                contains = stats.get("qa_contains", {}).get("mean", float("nan"))
                sig = (
                    significance.get(model_key, {})
                    .get(budget_key, {})
                    .get(policy, {})
                    .get("delta_qa_token_f1", {})
                    .get("conversation_level", {})
                )
                sig_text = (
                    f"{sig.get('mean', float('nan')):+.4f} (p={sig.get('p_value', float('nan')):.4f})"
                    if sig
                    else "—"
                )
                lines.append(
                    f"| {policy} | {em:.4f} | {f1:.4f} | {contains:.4f} | {sig_text} |"
                )
        lines.append("")
    (output_dir / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_qa_accuracy(
    *,
    study_dir: Path,
    input_paths: list[Path],
    model_key: str,
    output_dir: Path,
    policies: list[str] | None,
    budgets: list[float] | None,
    max_input_tokens: int,
    max_new_tokens: int,
    dtype: str,
    device: str | None,
    limit_conversations: int | None,
    conversation_ids_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations = {
        record.conversation_id: record for record in load_conversations_from_paths(input_paths)
    }
    allowed_ids: set[str] | None = None
    if conversation_ids_path is not None:
        allowed_ids = {
            line.strip()
            for line in conversation_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    evaluation_rows = _load_evaluation_rows(study_dir, model_key)
    tasks: list[dict[str, Any]] = []
    reference_keys: set[tuple[str, int]] = set()
    seen_conversations: list[str] = []
    for row in evaluation_rows:
        conversation_id = row["conversation_id"]
        if allowed_ids is not None and conversation_id not in allowed_ids:
            continue
        conversation = conversations.get(conversation_id)
        if conversation is None:
            continue
        target_turn = int(row["target_turn"])
        if not _behavior_eligible(conversation, target_turn):
            continue
        policy_name = row["policy_name"]
        budget_fraction = float(row["budget_fraction"])
        if policies is not None and policy_name not in policies:
            continue
        if budgets is not None and not any(
            abs(budget_fraction - budget) < 1e-6 for budget in budgets
        ):
            continue
        if conversation_id not in seen_conversations:
            if (
                limit_conversations is not None
                and len(seen_conversations) >= limit_conversations
            ):
                continue
            seen_conversations.append(conversation_id)
        retained_raw = str(row.get("retained_turn_indices", "")).strip()
        retained = [int(item) for item in retained_raw.split(",") if item != ""]
        tasks.append(
            {
                "conversation_id": conversation_id,
                "family": row.get("family", conversation.family),
                "target_turn": target_turn,
                "policy_name": policy_name,
                "budget_fraction": budget_fraction,
                "retained_prior_indices": retained,
            }
        )
        reference_keys.add((conversation_id, target_turn))

    if not tasks:
        raise ValueError(
            "no QA-eligible rows matched the filters; check --model-key/--policies/--budgets"
        )

    extractor = ConversationStateExtractor(model_key, device=device, dtype=dtype)

    # Full-context reference generations, one per (conversation, target turn).
    reference_messages: list[list[dict[str, str]]] = []
    reference_order = sorted(reference_keys)
    for conversation_id, target_turn in reference_order:
        conversation = conversations[conversation_id]
        reference_messages.append(
            _policy_messages(
                conversation=conversation,
                target_turn=target_turn,
                retained_prior_indices=list(range(target_turn)),
            )
        )
    print(
        f"[qa_accuracy] generating {len(reference_order)} full-context references "
        f"and {len(tasks)} policy answers ({model_key})",
        flush=True,
    )
    reference_predictions = generate_batch(
        extractor,
        reference_messages,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        batch_size=GENERATION_BATCH_SIZE,
    )
    reference_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for (conversation_id, target_turn), prediction in zip(
        reference_order, reference_predictions
    ):
        gold = conversations[conversation_id].turns[target_turn + 1].content
        reference_by_key[(conversation_id, target_turn)] = {
            "prediction": prediction,
            **{f"full_{k}": v for k, v in _score_prediction(prediction, gold).items()},
        }

    policy_messages = [
        _policy_messages(
            conversation=conversations[task["conversation_id"]],
            target_turn=task["target_turn"],
            retained_prior_indices=task["retained_prior_indices"],
        )
        for task in tasks
    ]
    policy_predictions = generate_batch(
        extractor,
        policy_messages,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        batch_size=GENERATION_BATCH_SIZE,
    )

    qa_rows: list[dict[str, Any]] = []
    for task, prediction in zip(tasks, policy_predictions):
        conversation = conversations[task["conversation_id"]]
        gold = conversation.turns[task["target_turn"] + 1].content
        scores = _score_prediction(prediction, gold)
        reference = reference_by_key[(task["conversation_id"], task["target_turn"])]
        qa_rows.append(
            {
                "model_key": model_key,
                "conversation_id": task["conversation_id"],
                "family": task["family"],
                "target_turn": task["target_turn"],
                "policy_name": task["policy_name"],
                "budget_fraction": task["budget_fraction"],
                "prediction": prediction,
                "gold_answer": gold,
                **scores,
                "qa_full_exact_match": reference["full_qa_exact_match"],
                "qa_full_token_f1": reference["full_qa_token_f1"],
                "qa_full_contains": reference["full_qa_contains"],
                "qa_token_f1_vs_full": scores["qa_token_f1"] - reference["full_qa_token_f1"],
            }
        )

    summary, significance = summarize_qa_rows(qa_rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "model_key": model_key,
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "num_conversations": len({row["conversation_id"] for row in qa_rows}),
        "num_rows": len(qa_rows),
        "qa_summary": summary,
    }
    _write_csv(output_dir / "qa_rows.csv", qa_rows)
    _write_json(output_dir / "qa_summary.json", payload)
    _write_json(output_dir / "qa_significance_summary.json", significance)
    write_qa_report(
        output_dir,
        summary=summary,
        significance=significance,
        num_rows=len(qa_rows),
        study_name=study_dir.name,
    )
    _write_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "study_name": study_dir.name,
            "num_rows": len(qa_rows),
        },
    )
    print(f"[qa_accuracy] wrote {len(qa_rows)} rows to {output_dir}", flush=True)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--extra-input-paths", type=str, default=None)
    parser.add_argument("--model-key", type=str, default="qwen25_15b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--budgets", type=str, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--conversation-ids-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [args.input_path]
    if args.extra_input_paths:
        input_paths.extend(Path(item) for item in args.extra_input_paths.split(",") if item)
    run_qa_accuracy(
        study_dir=args.study_dir,
        input_paths=input_paths,
        model_key=args.model_key,
        output_dir=args.output_dir,
        policies=args.policies.split(",") if args.policies else None,
        budgets=[float(item) for item in args.budgets.split(",")] if args.budgets else None,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        limit_conversations=args.limit_conversations,
        conversation_ids_path=args.conversation_ids_path,
    )


if __name__ == "__main__":
    main()
