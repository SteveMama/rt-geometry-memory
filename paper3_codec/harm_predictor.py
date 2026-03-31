from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "semantic_score",
    "geometry_score",
    "support_score",
    "query_geom_v2_risk",
    "query_geom_v2_curvature",
    "query_geom_v2_energy",
    "query_geom_v2_alignment",
    "query_geom_v2_local_projection",
    "segment_rank95",
    "segment_mean_step_norm",
    "segment_mean_stabilized_curvature",
    "role_user",
    "is_latest_user",
    "recency",
    "token_cost",
    "constraint_score",
    "attention_raw",
    "attention_sink_corrected",
)


@dataclass(frozen=True, slots=True)
class HarmPredictorConfig:
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    hidden_dims: tuple[int, int] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 128
    seed: int = 20260412


class HarmPredictor(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.logit_head = torch.nn.Linear(h2, 1)
        self.behavior_head = torch.nn.Linear(h2, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(features)
        return self.logit_head(hidden).squeeze(-1), self.behavior_head(hidden).squeeze(-1)


def feature_vector_from_row(
    row: dict[str, Any],
    *,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> np.ndarray:
    return np.asarray([float(row.get(name, 0.0) or 0.0) for name in feature_names], dtype=np.float32)


def scalarize_harm(
    *,
    logit_values: np.ndarray,
    behavior_values: np.ndarray | None = None,
) -> np.ndarray:
    def _z(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - mean) / std).astype(np.float32)

    score = _z(logit_values)
    if behavior_values is not None and behavior_values.size:
        score = score + 0.5 * _z(np.maximum(behavior_values, 0.0))
    return score.astype(np.float32)


def stable_conversation_split(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict[str, set[str]]:
    conversation_ids = sorted({str(row["conversation_id"]) for row in rows})
    scored = []
    for conversation_id in conversation_ids:
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        scored.append((int(digest[:12], 16), conversation_id))
    scored.sort()
    ordered = [conversation_id for _, conversation_id in scored]
    n_total = len(ordered)
    if n_total <= 1:
        return {"train": set(ordered), "val": set(), "test": set()}
    if n_total == 2:
        return {"train": {ordered[0]}, "val": {ordered[1]}, "test": set()}
    n_train = int(np.floor(train_fraction * n_total))
    n_val = int(np.floor(val_fraction * n_total))
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    if n_train + n_val >= n_total:
        n_train = max(n_total - 1, 1)
        n_val = max(n_total - n_train, 0)
    train_ids = set(ordered[:n_train])
    val_ids = set(ordered[n_train : n_train + n_val])
    test_ids = set(ordered[n_train + n_val :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def prepare_training_arrays(
    rows: list[dict[str, Any]],
    *,
    config: HarmPredictorConfig,
) -> dict[str, Any]:
    turn_rows = [row for row in rows if str(row.get("candidate_type", "")) == "turn"]
    if not turn_rows:
        raise RuntimeError("No turn-level oracle rows available for harm predictor training.")

    split = stable_conversation_split(turn_rows)
    arrays: dict[str, Any] = {"split": split, "feature_names": list(config.feature_names)}
    for split_name, conversation_ids in split.items():
        split_rows = [row for row in turn_rows if str(row["conversation_id"]) in conversation_ids]
        if not split_rows:
            arrays[split_name] = None
            continue
        features = np.stack([feature_vector_from_row(row, feature_names=config.feature_names) for row in split_rows], axis=0)
        logit_targets = np.asarray([float(row["delta_logit_l2"]) for row in split_rows], dtype=np.float32)
        behavior_targets = np.asarray(
            [float(row.get("delta_answer_avg_neg_logprob_delta", 0.0) or 0.0) for row in split_rows],
            dtype=np.float32,
        )
        behavior_mask = np.asarray(
            [1.0 if str(row.get("has_behavior_label", "0")) == "1" else 0.0 for row in split_rows],
            dtype=np.float32,
        )
        arrays[split_name] = {
            "rows": split_rows,
            "features": features,
            "logit_targets": logit_targets,
            "behavior_targets": behavior_targets,
            "behavior_mask": behavior_mask,
            "harm_scalar": scalarize_harm(logit_values=logit_targets, behavior_values=behavior_targets),
        }
    return arrays


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


def train_harm_predictor(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: HarmPredictorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or HarmPredictorConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = prepare_training_arrays(rows, config=cfg)
    train_split = arrays["train"]
    val_split = arrays["val"]
    test_split = arrays["test"]
    if train_split is None or val_split is None:
        raise RuntimeError("Need non-empty train and validation splits for harm predictor training.")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    feature_mean = train_split["features"].mean(axis=0)
    feature_std = train_split["features"].std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    def _normalize_features(features: np.ndarray) -> np.ndarray:
        return ((features - feature_mean) / feature_std).astype(np.float32)

    model = HarmPredictor(
        input_dim=len(cfg.feature_names),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    mse = torch.nn.MSELoss(reduction="none")

    train_x = _normalize_features(train_split["features"])
    val_x = _normalize_features(val_split["features"])

    train_features = _to_tensor(train_x)
    train_logit = _to_tensor(train_split["harm_scalar"])
    train_behavior = _to_tensor(train_split["behavior_targets"])
    train_behavior_mask = _to_tensor(train_split["behavior_mask"])

    val_features = _to_tensor(val_x)
    val_logit = _to_tensor(val_split["harm_scalar"])
    val_behavior = _to_tensor(val_split["behavior_targets"])
    val_behavior_mask = _to_tensor(val_split["behavior_mask"])

    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    patience_left = cfg.patience

    for _ in range(cfg.max_epochs):
        model.train()
        permutation = np.random.permutation(train_features.shape[0])
        for start in range(0, train_features.shape[0], cfg.batch_size):
            batch_indices = permutation[start : start + cfg.batch_size]
            features = train_features[batch_indices]
            logit_targets = train_logit[batch_indices]
            behavior_targets = train_behavior[batch_indices]
            behavior_mask = train_behavior_mask[batch_indices]

            pred_logit, pred_behavior = model(features)
            logit_loss = mse(pred_logit, logit_targets).mean()
            behavior_losses = mse(pred_behavior, behavior_targets)
            behavior_denom = torch.clamp(behavior_mask.sum(), min=1.0)
            behavior_loss = (behavior_losses * behavior_mask).sum() / behavior_denom
            loss = logit_loss + 0.5 * behavior_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_logit, pred_behavior = model(val_features)
            logit_loss = mse(pred_logit, val_logit).mean()
            behavior_losses = mse(pred_behavior, val_behavior)
            behavior_denom = torch.clamp(val_behavior_mask.sum(), min=1.0)
            behavior_loss = (behavior_losses * val_behavior_mask).sum() / behavior_denom
            val_loss = float((logit_loss + 0.5 * behavior_loss).item())

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            patience_left = cfg.patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Harm predictor training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    payload = {
        "config": {
            "feature_names": list(cfg.feature_names),
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "state_dict": model.state_dict(),
        "split": arrays["split"],
        "best_val_loss": best_val,
    }
    torch.save(payload, output_dir / "harm_predictor.pt")
    summary = {
        "best_val_loss": best_val,
        "num_train_rows": 0 if train_split is None else int(train_split["features"].shape[0]),
        "num_val_rows": 0 if val_split is None else int(val_split["features"].shape[0]),
        "num_test_rows": 0 if test_split is None else int(test_split["features"].shape[0]),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass(slots=True)
class HarmPredictorBundle:
    model: HarmPredictor
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "HarmPredictorBundle":
        payload = torch.load(Path(path), map_location="cpu")
        feature_names = tuple(payload["config"]["feature_names"])
        hidden_dims = tuple(int(value) for value in payload["config"]["hidden_dims"])
        dropout = float(payload["config"]["dropout"])
        model = HarmPredictor(
            input_dim=len(feature_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(
            model=model,
            feature_names=feature_names,
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
        )

    def predict_rows(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.zeros(0, dtype=np.float32)
        features = np.stack(
            [feature_vector_from_row(row, feature_names=self.feature_names) for row in rows],
            axis=0,
        ).astype(np.float32)
        normalized = (features - self.feature_mean) / np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        with torch.no_grad():
            pred_logit, pred_behavior = self.model(torch.from_numpy(normalized))
        pred_logit_np = pred_logit.detach().cpu().numpy().astype(np.float32)
        pred_behavior_np = pred_behavior.detach().cpu().numpy().astype(np.float32)
        logit_score = scalarize_harm(logit_values=pred_logit_np)
        behavior_score = scalarize_harm(logit_values=np.maximum(pred_behavior_np, 0.0))
        return (logit_score + 0.5 * behavior_score).astype(np.float32)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Paper 3 harm predictor from oracle rows.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = _read_csv_rows(args.candidate_csv)
    summary = train_harm_predictor(rows, output_dir=args.output_dir)
    print(
        f"Trained harm predictor at {args.output_dir} "
        f"(train={summary['num_train_rows']} val={summary['num_val_rows']} test={summary['num_test_rows']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
