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
        key="llama32_3b",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        notes="Non-Qwen 3B validation target for Paper 3 generalization checks.",
        min_transformers_version="4.45.0",
        parameter_size="3B",
        context_length=131072,
        mac_notes="Prefer Colab or discrete GPU. Hugging Face gated-access approval and token may be required.",
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
    attention_summary: "TurnAttentionSummary | None" = None


@dataclass(slots=True)
class TurnAttentionSummary:
    raw_turn_weights: np.ndarray
    sink_corrected_turn_weights: np.ndarray
    sink_baseline: float
    query_position: int


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
        if self.torch.cuda.is_available():
            return "cuda"
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
        *,
        progress_label: str | None = None,
        progress_every: int | None = None,
    ) -> TrajectoryBatch:
        turn_limit = len(conversation.turns) if max_turns is None else min(max_turns, len(conversation.turns))
        states: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        token_counts: list[int] = []
        turn_roles: list[str] = []

        if progress_label:
            every = progress_every if progress_every is not None else max(1, min(8, turn_limit))
            print(
                f"[extract_conversation] start {progress_label} turns={turn_limit}",
                flush=True,
            )
        else:
            every = None

        with self.torch.no_grad():
            for turn_index in range(turn_limit):
                messages = conversation_prefix_messages(conversation, turn_index)
                score = self.score_messages(messages, max_input_tokens=max_input_tokens)
                states.append(score.state)
                logits.append(score.logits)
                token_counts.append(score.token_count)
                turn_roles.append(conversation.turns[turn_index].role)
                if progress_label and every is not None:
                    completed = turn_index + 1
                    if completed == 1 or completed == turn_limit or completed % every == 0:
                        print(
                            f"[extract_conversation] {progress_label} turn {completed}/{turn_limit}",
                            flush=True,
                        )

        batch = TrajectoryBatch(
            states=np.stack(states, axis=0),
            logits=np.stack(logits, axis=0),
            token_counts=np.asarray(token_counts, dtype=np.int32),
            turn_roles=turn_roles,
        )
        if progress_label:
            print(
                f"[extract_conversation] done {progress_label}",
                flush=True,
            )
        return batch

    def score_messages(
        self,
        messages: list[dict[str, str]],
        max_input_tokens: int | None = None,
        *,
        return_attention_summary: bool = False,
        cumulative_turn_token_counts: np.ndarray | None = None,
        attention_layers: int = 4,
        sink_token_count: int = 32,
    ) -> MessageScore:
        with self.torch.no_grad():
            encoded = self._tokenize_messages(messages, max_input_tokens=max_input_tokens)
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                output_attentions=return_attention_summary,
                use_cache=False,
                return_dict=True,
            )
            hidden_state = outputs.hidden_states[self.state_layer][0, -1].detach().to(self.torch.float32).cpu().numpy()
            next_logits = outputs.logits[0, -1].detach().to(self.torch.float32).cpu().numpy()
            attention_summary = None
            if return_attention_summary and cumulative_turn_token_counts is not None:
                attention_summary = self._summarize_turn_attention(
                    attentions=outputs.attentions,
                    cumulative_turn_token_counts=np.asarray(cumulative_turn_token_counts, dtype=np.int32),
                    total_tokens=int(encoded["input_ids"].shape[1]),
                    attention_layers=attention_layers,
                    sink_token_count=sink_token_count,
                )
        return MessageScore(
            state=hidden_state.astype(np.float32),
            logits=next_logits.astype(np.float32),
            token_count=int(encoded["input_ids"].shape[1]),
            attention_summary=attention_summary,
        )

    def _summarize_turn_attention(
        self,
        *,
        attentions: Any,
        cumulative_turn_token_counts: np.ndarray,
        total_tokens: int,
        attention_layers: int,
        sink_token_count: int,
    ) -> TurnAttentionSummary:
        if attentions is None or cumulative_turn_token_counts.size == 0 or total_tokens <= 0:
            empty = np.zeros(cumulative_turn_token_counts.size, dtype=np.float32)
            return TurnAttentionSummary(
                raw_turn_weights=empty,
                sink_corrected_turn_weights=empty,
                sink_baseline=0.0,
                query_position=max(total_tokens - 1, 0),
            )

        layer_count = len(attentions)
        use_layers = attentions[max(layer_count - max(attention_layers, 1), 0) :]
        token_weights = np.zeros(total_tokens, dtype=np.float64)
        query_position = total_tokens - 1
        for layer_attention in use_layers:
            # shape: [batch, heads, seq, seq]
            matrix = layer_attention[0, :, query_position, :].detach().to(self.torch.float32).cpu().numpy()
            token_weights += matrix.mean(axis=0)
        token_weights /= max(len(use_layers), 1)

        raw_turn_weights = np.zeros(cumulative_turn_token_counts.size, dtype=np.float32)
        sink_corrected = np.zeros(cumulative_turn_token_counts.size, dtype=np.float32)
        sink_window = min(max(sink_token_count, 0), total_tokens)
        sink_baseline = float(np.mean(token_weights[:sink_window])) if sink_window > 0 else 0.0

        prev_end = 0
        for idx, raw_end in enumerate(cumulative_turn_token_counts.tolist()):
            end = min(max(int(raw_end), 0), total_tokens)
            start = min(max(prev_end, 0), total_tokens)
            if end <= start:
                prev_end = end
                continue
            turn_mass = float(np.sum(token_weights[start:end]))
            turn_len = end - start
            raw_turn_weights[idx] = turn_mass
            sink_corrected[idx] = max(turn_mass - sink_baseline * float(turn_len), 0.0)
            prev_end = end

        return TurnAttentionSummary(
            raw_turn_weights=raw_turn_weights,
            sink_corrected_turn_weights=sink_corrected,
            sink_baseline=sink_baseline,
            query_position=query_position,
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
