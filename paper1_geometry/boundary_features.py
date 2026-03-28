from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from .conversations import ConversationRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _token_counter(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text))


def _cosine_distance(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 1.0
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    cosine = dot / (norm_a * norm_b)
    return float(1.0 - max(min(cosine, 1.0), -1.0))


def _window_text(conversation: ConversationRecord, start: int, end: int) -> str:
    return " ".join(turn.content for turn in conversation.turns[start:end])


def lexical_shift_scores(conversation: ConversationRecord) -> np.ndarray:
    n_turns = len(conversation.turns)
    if n_turns < 3:
        return np.zeros(0, dtype=np.float32)
    values: list[float] = []
    for idx in range(1, n_turns - 1):
        left_start = max(0, idx - 1)
        left_text = _window_text(conversation, left_start, idx + 1)
        right_end = min(n_turns, idx + 3)
        right_text = _window_text(conversation, idx + 1, right_end)
        values.append(_cosine_distance(_token_counter(left_text), _token_counter(right_text)))
    return np.asarray(values, dtype=np.float32)


def _style_features(text: str) -> np.ndarray:
    chars = np.asarray([ord(ch) for ch in text], dtype=np.int32) if text else np.zeros(0, dtype=np.int32)
    alpha_count = int(sum(ch.isalpha() for ch in text))
    upper_count = int(sum(ch.isupper() for ch in text))
    return np.asarray(
        [
            len(text),
            text.count("\n") + 1 if text else 0,
            text.count("- "),
            sum(ch.isdigit() for ch in text),
            sum(ch in ":;,.?!" for ch in text),
            sum(ch in "()[]{}" for ch in text),
            sum(ch in "`*_#/" for ch in text),
            upper_count / max(alpha_count, 1),
            float(chars.mean()) if chars.size else 0.0,
        ],
        dtype=np.float32,
    )


def style_shift_scores(conversation: ConversationRecord) -> np.ndarray:
    n_turns = len(conversation.turns)
    if n_turns < 3:
        return np.zeros(0, dtype=np.float32)
    features = [_style_features(turn.content) for turn in conversation.turns]
    values: list[float] = []
    for idx in range(1, n_turns - 1):
        values.append(float(np.linalg.norm(features[idx] - features[idx + 1])))
    return np.asarray(values, dtype=np.float32)
