from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ["records", "data", "examples", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported benchmark payload shape in {path}")


def _normalize_role(raw_role: str) -> str:
    lowered = raw_role.strip().lower()
    if lowered in {"assistant", "model", "bot", "agent"}:
        return "assistant"
    if lowered in {"system", "developer"}:
        return "system"
    return "user"


def _coerce_turns(raw_turns: list[Any]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for item in raw_turns:
        if isinstance(item, str):
            role = "user" if len(turns) % 2 == 0 else "assistant"
            text = item.strip()
        elif isinstance(item, dict):
            role = _normalize_role(
                str(
                    item.get("role")
                    or item.get("speaker")
                    or item.get("from")
                    or item.get("author")
                    or "user"
                )
            )
            text = str(
                item.get("content")
                or item.get("text")
                or item.get("utterance")
                or item.get("message")
                or item.get("value")
                or ""
            ).strip()
        else:
            continue
        if not text:
            continue
        turns.append({"role": role, "content": text})
    return turns


def _coerce_session_history(raw_sessions: list[Any]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for session in raw_sessions:
        if isinstance(session, list):
            turns.extend(_coerce_turns(session))
            continue
        if isinstance(session, dict):
            for key in ["messages", "dialogue", "conversation", "turns", "history", "session"]:
                value = session.get(key)
                if isinstance(value, list):
                    turns.extend(_coerce_turns(value))
                    break
            else:
                turns.extend(_coerce_turns([session]))
    return turns


def _normalized_record(
    record: dict[str, Any],
    *,
    record_id: str,
    family: str,
) -> dict[str, Any]:
    turns = _coerce_turns(list(record.get("turns", [])))
    if not turns:
        raise ValueError(f"Record {record_id} has no turns")
    if len(turns) < 2 or turns[-2]["role"] != "user" or turns[-1]["role"] != "assistant":
        raise ValueError(
            f"Record {record_id} must end with a user turn followed by a gold assistant answer"
        )
    payload: dict[str, Any] = {
        "conversation_id": record_id,
        "family": record.get("family") or family,
        "turns": turns,
    }
    system_prompt = record.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        payload["system_prompt"] = system_prompt.strip()
    return payload


def _adapt_locomo(record: dict[str, Any], index: int, family: str) -> dict[str, Any]:
    record_id = str(record.get("conversation_id") or record.get("id") or f"locomo-{index:05d}")
    dialogue = record.get("dialogue") or record.get("conversation") or record.get("messages") or record.get("turns")
    if not isinstance(dialogue, list):
        raise ValueError(f"LoCoMo record {record_id} has no dialogue list")
    turns = _coerce_turns(dialogue)
    question = str(record.get("question") or record.get("query") or record.get("prompt") or "").strip()
    answer = str(record.get("answer") or record.get("response") or record.get("gold_answer") or "").strip()
    if question:
        turns.append({"role": "user", "content": question})
    if answer:
        turns.append({"role": "assistant", "content": answer})
    return _normalized_record(
        {"turns": turns, "system_prompt": record.get("system_prompt"), "family": family},
        record_id=record_id,
        family=family,
    )


def _adapt_msc(record: dict[str, Any], index: int, family: str) -> dict[str, Any]:
    record_id = str(record.get("conversation_id") or record.get("id") or f"msc-{index:05d}")
    raw_sessions = (
        record.get("sessions")
        or record.get("session_history")
        or record.get("dialogues")
        or record.get("dialogs")
        or record.get("history")
    )
    turns: list[dict[str, str]] = []
    if isinstance(raw_sessions, list):
        turns = _coerce_session_history(raw_sessions)
    else:
        single = (
            record.get("dialog")
            or record.get("dialogue")
            or record.get("conversation")
            or record.get("messages")
            or record.get("turns")
        )
        if isinstance(single, list):
            turns = _coerce_turns(single)
    question = str(record.get("question") or record.get("query") or record.get("prompt") or "").strip()
    answer = str(record.get("answer") or record.get("response") or record.get("gold_answer") or "").strip()
    if question:
        turns.append({"role": "user", "content": question})
    if answer:
        turns.append({"role": "assistant", "content": answer})
    return _normalized_record(
        {"turns": turns, "system_prompt": record.get("system_prompt"), "family": family},
        record_id=record_id,
        family=family,
    )


def _adapt_longmemeval(record: dict[str, Any], index: int, family: str) -> dict[str, Any]:
    record_id = str(
        record.get("question_id")
        or record.get("conversation_id")
        or record.get("id")
        or f"longmemeval-{index:05d}"
    )
    history = (
        record.get("history")
        or record.get("dialogue")
        or record.get("messages")
        or record.get("turns")
        or record.get("haystack_sessions")
    )
    if not isinstance(history, list):
        raise ValueError(f"LongMemEval record {record_id} has no history list")
    if history and isinstance(history[0], list):
        turns = _coerce_session_history(history)
    else:
        turns = _coerce_session_history(history)
    question = str(record.get("question") or record.get("query") or record.get("prompt") or "").strip()
    answer = str(record.get("answer") or record.get("response") or record.get("gold_answer") or "").strip()
    if question:
        turns.append({"role": "user", "content": question})
    if answer:
        turns.append({"role": "assistant", "content": answer})
    return _normalized_record(
        {"turns": turns, "system_prompt": record.get("system_prompt"), "family": family},
        record_id=record_id,
        family=family,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a public benchmark into RT conversation JSONL.")
    parser.add_argument("--format", choices=["normalized", "locomo", "msc", "longmemeval"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", default="public_benchmark")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = _load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if args.format == "normalized":
            record_id = str(record.get("conversation_id") or record.get("id") or f"record-{index:05d}")
            payload = _normalized_record(record, record_id=record_id, family=args.family)
        elif args.format == "locomo":
            payload = _adapt_locomo(record, index, args.family)
        elif args.format == "msc":
            payload = _adapt_msc(record, index, args.family)
        else:
            payload = _adapt_longmemeval(record, index, args.family)
        normalized.append(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in normalized:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(f"Wrote {len(normalized)} normalized benchmark conversations to {args.output}")


if __name__ == "__main__":
    main()
