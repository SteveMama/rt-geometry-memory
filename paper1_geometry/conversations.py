from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class TurnRecord:
    role: str
    content: str


@dataclass(slots=True)
class ConversationRecord:
    conversation_id: str
    family: str
    turns: list[TurnRecord]
    system_prompt: str | None = None
    boundary_indices: list[int] | None = None
    boundary_labels: list[str] | None = None


def load_conversations(path: str | Path) -> list[ConversationRecord]:
    return load_conversations_from_paths([Path(path)])


def load_conversations_from_paths(paths: list[str | Path]) -> list[ConversationRecord]:
    conversations: list[ConversationRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            turns = [TurnRecord(role=turn["role"], content=turn["content"]) for turn in payload["turns"]]
            conversations.append(
                ConversationRecord(
                    conversation_id=payload["conversation_id"],
                    family=payload["family"],
                    system_prompt=payload.get("system_prompt"),
                    turns=turns,
                    boundary_indices=payload.get("boundary_indices"),
                    boundary_labels=payload.get("boundary_labels"),
                )
            )
    return conversations


def conversation_prefix_messages(
    conversation: ConversationRecord,
    turn_index: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    messages.extend(
        {"role": turn.role, "content": turn.content}
        for turn in conversation.turns[: turn_index + 1]
    )
    return messages
