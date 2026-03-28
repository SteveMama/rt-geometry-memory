from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .conversations import ConversationRecord, conversation_prefix_messages


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    model_name: str
    notes: str
    min_transformers_version: str
    parameter_size: str
    context_length: int
    mac_notes: str


DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen25_05b",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        notes="Smallest existing repo-compatible baseline. Best first-pass local model on 16 GB RAM.",
        min_transformers_version="4.37.0",
        parameter_size="0.49B",
        context_length=32768,
        mac_notes="Best default start on Apple Silicon with MPS or CPU fallback.",
    ),
    ModelSpec(
        key="qwen3_06b",
        model_name="Qwen/Qwen3-0.6B",
        notes="Current lightweight Qwen family member with official local support across MLX-LM, Ollama, and llama.cpp.",
        min_transformers_version="4.51.0",
        parameter_size="0.6B",
        context_length=32768,
        mac_notes="Good Mac candidate after upgrading transformers to a Qwen3-capable release.",
    ),
    ModelSpec(
        key="qwen25_15b",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        notes="Still realistic on Apple Silicon for slower comparison runs.",
        min_transformers_version="4.37.0",
        parameter_size="1.54B",
        context_length=32768,
        mac_notes="Useful second-pass comparison on 16 GB+ Apple Silicon.",
    ),
    ModelSpec(
        key="qwen25_3b",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        notes="First larger-model checkpoint for Colab or stronger local hardware.",
        min_transformers_version="4.37.0",
        parameter_size="3.09B",
        context_length=32768,
        mac_notes="Prefer Colab or discrete GPU; usually too slow for comfortable Mac iteration.",
    ),
    ModelSpec(
        key="smollm2_17b",
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        notes="Cross-family compact baseline for checking whether the geometry story generalizes beyond Qwen.",
        min_transformers_version="4.48.0",
        parameter_size="1.7B",
        context_length=8192,
        mac_notes="Reasonable non-Qwen comparison model for local Paper 1 validation.",
    ),
)


def list_default_models() -> tuple[ModelSpec, ...]:
    return DEFAULT_MODELS


def resolve_model_spec(model_name_or_key: str) -> ModelSpec | None:
    for spec in DEFAULT_MODELS:
        if model_name_or_key in {spec.key, spec.model_name}:
            return spec
    return None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in version.split("."):
        digits = "".join(char for char in item if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def transformers_version_ok(installed: str, minimum: str) -> bool:
    return _version_tuple(installed) >= _version_tuple(minimum)


@dataclass(slots=True)
class TrajectoryBatch:
    states: np.ndarray
    logits: np.ndarray
    token_counts: np.ndarray
    turn_roles: list[str]


@dataclass(slots=True)
class MessageScore:
    state: np.ndarray
    logits: np.ndarray
    token_count: int


@dataclass(slots=True)
class CompletionScore:
    token_count: int
    total_logprob: float
    avg_logprob: float
    total_neg_logprob: float
    avg_neg_logprob: float


class ConversationStateExtractor:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: str = "auto",
        state_layer: int = -1,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or self._choose_device()
        self.dtype = self._choose_dtype(dtype)
        self.state_layer = state_layer
        self.transformers_version = transformers.__version__
        self.model_spec = resolve_model_spec(model_name)
        if self.model_spec is not None and not transformers_version_ok(
            self.transformers_version,
            self.model_spec.min_transformers_version,
        ):
            raise RuntimeError(
                f"{self.model_spec.model_name} requires transformers>={self.model_spec.min_transformers_version}, "
                f"but the current environment has {self.transformers_version}. "
                "Upgrade transformers or choose a compatible model preset."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def _choose_device(self) -> str:
        if self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _choose_dtype(self, dtype: str) -> Any:
        if dtype == "float16":
            return self.torch.float16
        if dtype == "float32":
            return self.torch.float32
        if self.device == "mps":
            return self.torch.float16
        return self.torch.float32

    def _tokenize_messages(
        self,
        messages: list[dict[str, str]],
        max_input_tokens: int | None,
    ) -> dict[str, Any]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        )
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    def extract_conversation(
        self,
        conversation: ConversationRecord,
        max_turns: int | None = None,
        max_input_tokens: int | None = None,
    ) -> TrajectoryBatch:
        turn_limit = len(conversation.turns) if max_turns is None else min(max_turns, len(conversation.turns))
        states: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        token_counts: list[int] = []
        turn_roles: list[str] = []

        with self.torch.no_grad():
            for turn_index in range(turn_limit):
                messages = conversation_prefix_messages(conversation, turn_index)
                score = self.score_messages(messages, max_input_tokens=max_input_tokens)
                states.append(score.state)
                logits.append(score.logits)
                token_counts.append(score.token_count)
                turn_roles.append(conversation.turns[turn_index].role)

        return TrajectoryBatch(
            states=np.stack(states, axis=0),
            logits=np.stack(logits, axis=0),
            token_counts=np.asarray(token_counts, dtype=np.int32),
            turn_roles=turn_roles,
        )

    def score_messages(
        self,
        messages: list[dict[str, str]],
        max_input_tokens: int | None = None,
    ) -> MessageScore:
        with self.torch.no_grad():
            encoded = self._tokenize_messages(messages, max_input_tokens=max_input_tokens)
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_state = outputs.hidden_states[self.state_layer][0, -1].detach().to(self.torch.float32).cpu().numpy()
            next_logits = outputs.logits[0, -1].detach().to(self.torch.float32).cpu().numpy()
        return MessageScore(
            state=hidden_state.astype(np.float32),
            logits=next_logits.astype(np.float32),
            token_count=int(encoded["input_ids"].shape[1]),
        )

    def project_logits(self, states: np.ndarray) -> np.ndarray:
        output_head = self.model.get_output_embeddings()
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(states).to(self.device, dtype=self.torch.float32)
            projected = output_head(tensor).detach().to(self.torch.float32).cpu().numpy()
        return projected.astype(np.float32)

    def score_assistant_response(
        self,
        prompt_messages: list[dict[str, str]],
        target_text: str,
        max_input_tokens: int | None = None,
    ) -> CompletionScore:
        prompt_rendered = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_rendered = self.tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": target_text}],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_batch = self.tokenizer(
            prompt_rendered,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        )
        full_batch = self.tokenizer(
            full_rendered,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        )
        prompt_ids = prompt_batch["input_ids"][0]
        full_ids = full_batch["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
        full_len = int(full_ids.shape[0])
        if full_len <= prompt_len:
            return CompletionScore(
                token_count=0,
                total_logprob=0.0,
                avg_logprob=0.0,
                total_neg_logprob=0.0,
                avg_neg_logprob=0.0,
            )

        encoded = {name: tensor.to(self.device) for name, tensor in full_batch.items()}
        with self.torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            log_probs = self.torch.log_softmax(outputs.logits[0, :-1], dim=-1)
            target_ids = encoded["input_ids"][0, 1:]
            target_start = max(prompt_len - 1, 0)
            suffix_log_probs = log_probs[target_start:]
            suffix_target_ids = target_ids[target_start:]
            if suffix_target_ids.numel() == 0:
                return CompletionScore(
                    token_count=0,
                    total_logprob=0.0,
                    avg_logprob=0.0,
                    total_neg_logprob=0.0,
                    avg_neg_logprob=0.0,
                )
            gathered = suffix_log_probs.gather(1, suffix_target_ids.unsqueeze(-1)).squeeze(-1)
            total_logprob = float(gathered.sum().detach().to(self.torch.float32).cpu().item())
            token_count = int(suffix_target_ids.shape[0])
        avg_logprob = total_logprob / max(token_count, 1)
        return CompletionScore(
            token_count=token_count,
            total_logprob=total_logprob,
            avg_logprob=avg_logprob,
            total_neg_logprob=-total_logprob,
            avg_neg_logprob=-avg_logprob,
        )

    def save_local_metadata(self, output_path: str | Path) -> None:
        payload = {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "state_layer": self.state_layer,
            "transformers_version": self.transformers_version,
        }
        Path(output_path).write_text(str(payload), encoding="utf-8")
