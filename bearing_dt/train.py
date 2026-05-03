from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bearing_dt.config import read_yaml
from bearing_dt.data.dataset import BearingDataset, context_columns, feature_columns, load_processed, normalize_context_mode
from bearing_dt.data.features import extract_feature_row
from bearing_dt.data.splits import SplitIndices, make_split
from bearing_dt.metrics import interval_metrics, regression_metrics
from bearing_dt.models import build_model
from bearing_dt.physics import damage_monotonic_violation_rate, monotonic_violation_rate, physics_regularization
from bearing_dt.table import Rows, group_by, read_rows_csv, take, write_rows_csv
from bearing_dt.uncertainty import ConformalInterval
from bearing_dt.utils import available_device, ensure_dir, set_seed, utc_timestamp, write_json


SKLEARN_MODELS = {"random_forest", "gradient_boosting"}


def _context_mode(config: dict[str, Any]) -> str:
    context_cfg = config.get("context", {})
    if isinstance(context_cfg, dict) and "mode" in context_cfg:
        return normalize_context_mode(context_cfg.get("mode"))
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and "context_mode" in model_cfg:
        return normalize_context_mode(model_cfg.get("context_mode"))
    return normalize_context_mode("load_speed")


def _context_info(context_mode: str, split_description: str) -> dict[str, Any]:
    cols = context_columns(context_mode)
    return {
        "context_mode": context_mode,
        "context_columns": cols,
        "context_dim": len(cols),
        "regime_used_as_input": "regime_code" in cols,
        "condition_used_as_input": "condition_code" in cols,
        "regime_used_for_split": split_description.startswith("leave_one_operating_regime") or "regime:" in split_description,
    }


def _assert_context_is_clean(context_mode: str) -> None:
    cols = context_columns(context_mode)
    if context_mode in {"none", "load_speed"} and any("regime" in col or "condition" in col for col in cols):
        raise AssertionError(f"context_mode={context_mode} unexpectedly includes categorical context columns: {cols}")


class RandomSubspaceRidgeRegressor:
    """Dependency-free fallback when scikit-learn is unavailable."""

    def __init__(self, n_estimators: int = 80, alpha: float = 1e-2, feature_fraction: float = 0.65, seed: int = 7) -> None:
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.feature_fraction = feature_fraction
        self.seed = seed
        self.models: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomSubspaceRidgeRegressor":
        rng = np.random.default_rng(self.seed)
        n_features = x.shape[1]
        subset_size = max(1, int(n_features * self.feature_fraction))
        self.models = []
        for _ in range(self.n_estimators):
            cols = np.sort(rng.choice(n_features, size=subset_size, replace=False))
            rows = rng.choice(len(x), size=len(x), replace=True)
            xb = np.c_[np.ones(len(rows)), x[rows][:, cols]]
            yb = y[rows]
            reg = self.alpha * np.eye(xb.shape[1])
            reg[0, 0] = 0.0
            coef = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ yb
            self.models.append((cols, coef))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model has not been fitted")
        preds = []
        for cols, coef in self.models:
            xb = np.c_[np.ones(len(x)), x[:, cols]]
            preds.append(xb @ coef)
        return np.mean(np.stack(preds, axis=0), axis=0)


class GradientBoostedStumpRegressor:
    """Small dependency-free squared-loss gradient boosting baseline."""

    def __init__(
        self,
        n_estimators: int = 240,
        learning_rate: float = 0.035,
        max_thresholds: int = 32,
        subsample_features: float = 1.0,
        seed: int = 7,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_thresholds = max_thresholds
        self.subsample_features = subsample_features
        self.seed = seed
        self.init_: float = 0.0
        self.stumps: list[tuple[int, float, float, float]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostedStumpRegressor":
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.init_ = float(np.mean(y))
        pred = np.full(len(y), self.init_, dtype=np.float64)
        self.stumps = []
        n_features = x.shape[1]
        feature_count = max(1, int(np.ceil(n_features * self.subsample_features)))
        for _ in range(self.n_estimators):
            residual = y - pred
            best = None
            feature_indices = np.sort(rng.choice(n_features, size=feature_count, replace=False))
            for col in feature_indices:
                values = x[:, col]
                finite = values[np.isfinite(values)]
                if len(np.unique(finite)) < 2:
                    continue
                quantiles = np.linspace(0.05, 0.95, self.max_thresholds)
                thresholds = np.unique(np.quantile(finite, quantiles))
                for threshold in thresholds:
                    left = values <= threshold
                    right = ~left
                    if not left.any() or not right.any():
                        continue
                    left_value = float(np.mean(residual[left]))
                    right_value = float(np.mean(residual[right]))
                    update = np.where(left, left_value, right_value)
                    loss = float(np.mean((residual - update) ** 2))
                    if best is None or loss < best[0]:
                        best = (loss, int(col), float(threshold), left_value, right_value)
            if best is None:
                break
            _, col, threshold, left_value, right_value = best
            pred += self.learning_rate * np.where(x[:, col] <= threshold, left_value, right_value)
            self.stumps.append((col, threshold, left_value, right_value))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.stumps:
            raise RuntimeError("Model has not been fitted")
        x = np.asarray(x, dtype=np.float64)
        pred = np.full(x.shape[0], self.init_, dtype=np.float64)
        for col, threshold, left_value, right_value in self.stumps:
            pred += self.learning_rate * np.where(x[:, col] <= threshold, left_value, right_value)
        return pred


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name)


def _run_dir(config: dict[str, Any]) -> Path:
    name = _safe_name(str(config.get("experiment_name", "experiment")))
    runs_dir = ensure_dir(config.get("run", {}).get("runs_dir", "runs"))
    return ensure_dir(runs_dir / f"{utc_timestamp()}_{name}")


def _design_matrix(frame: Rows, feat_cols: list[str], indices: np.ndarray, context_mode: str = "load_speed", mean=None, std=None):
    cols = feat_cols + context_columns(context_mode)
    if context_mode in {"none", "load_speed"} and any("regime" in col or "condition" in col for col in cols):
        raise AssertionError(f"context_mode={context_mode} includes forbidden categorical columns in design matrix")
    values = np.array([[float(frame[int(i)][c]) for c in cols] for i in indices], dtype=np.float32)
    if mean is None:
        mean = np.nanmean(values, axis=0, keepdims=True)
    if std is None:
        std = np.nanstd(values, axis=0, keepdims=True) + 1e-6
    return np.nan_to_num((values - mean) / std), mean, std


def _targets(frame: Rows, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = take(frame, indices)
    return (
        np.array([row["rul_norm"] for row in subset], dtype=np.float32),
        np.array([row["rul"] for row in subset], dtype=np.float32),
        np.array([row["max_step"] for row in subset], dtype=np.float32),
    )


def _apply_train_fraction(frame: Rows, split: SplitIndices, fraction: float, seed: int) -> SplitIndices:
    if fraction >= 0.999:
        return split
    if fraction <= 0:
        raise ValueError("train_fraction must be positive")
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = {}
    for idx in split.train:
        grouped.setdefault(str(frame[int(idx)]["bearing_id"]), []).append(int(idx))
    selected = []
    for group_indices in grouped.values():
        group_indices = np.array(group_indices, dtype=np.int64)
        take = max(1, int(np.ceil(len(group_indices) * fraction)))
        selected.extend(rng.choice(group_indices, size=take, replace=False).tolist())
    return SplitIndices(
        train=np.array(sorted(selected), dtype=np.int64),
        val=split.val,
        test=split.test,
        description=f"{split.description}|train_fraction:{fraction:g}",
        cal=split.cal,
    )


def _split_groups(split: SplitIndices) -> dict[str, np.ndarray]:
    groups = {"train": split.train, "val": split.val, "test": split.test}
    if split.cal is not None:
        groups["cal"] = split.cal
    return groups


def _calibration_split_name(split: SplitIndices) -> str:
    return "cal" if split.cal is not None else "val"


def _append_eval_subsets(preds: Rows, config: dict[str, Any]) -> Rows:
    subsets = []
    test_rows = [row for row in preds if row["split"] == "test"]
    for subset in config.get("eval_subsets", []):
        name = subset["name"]
        rows = [dict(row) for row in test_rows]
        if "max_life_fraction" in subset:
            rows = [row for row in rows if float(row["life_fraction"]) <= float(subset["max_life_fraction"])]
        if "min_life_fraction" in subset:
            rows = [row for row in rows if float(row["life_fraction"]) >= float(subset["min_life_fraction"])]
        if rows:
            for row in rows:
                row["split"] = f"test_{name}"
            subsets.extend(rows)
    if subsets:
        return [*preds, *subsets]
    return preds


def _prediction_frame(
    frame: Rows,
    indices: np.ndarray,
    split_name: str,
    pred_norm: np.ndarray,
    lower_norm: np.ndarray | None = None,
    upper_norm: np.ndarray | None = None,
    std_norm: np.ndarray | None = None,
    damage: np.ndarray | None = None,
    damage_rate: np.ndarray | None = None,
) -> Rows:
    subset = take(frame, indices)
    max_step = np.array([row["max_step"] for row in subset], dtype=np.float64)
    pred_norm = np.clip(np.asarray(pred_norm, dtype=np.float64), 0.0, None)
    out: Rows = []
    for i, row in enumerate(subset):
        out.append(
            {
                "split": split_name,
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "bearing_id": row["bearing_id"],
                "condition_id": row["condition_id"],
                "step": row["step"],
                "life_fraction": row["life_fraction"],
                "rul": row["rul"],
                "rul_norm": row["rul_norm"],
                "max_step": row["max_step"],
                "y_pred_norm": float(pred_norm[i]),
                "y_pred_rul": float(pred_norm[i] * max_step[i]),
            }
        )
    if lower_norm is not None and upper_norm is not None:
        lower_norm = np.clip(lower_norm, 0.0, None)
        upper_norm = np.clip(upper_norm, 0.0, None)
        for i, row in enumerate(out):
            row["lower_norm"] = float(lower_norm[i])
            row["upper_norm"] = float(upper_norm[i])
            row["lower_rul"] = float(lower_norm[i] * max_step[i])
            row["upper_rul"] = float(upper_norm[i] * max_step[i])
    if std_norm is not None:
        for i, row in enumerate(out):
            row["ensemble_std_norm"] = float(std_norm[i])
    if damage is not None:
        for i, row in enumerate(out):
            row["damage_pred"] = float(damage[i])
    if damage_rate is not None:
        for i, row in enumerate(out):
            row["damage_rate_pred"] = float(damage_rate[i])
    return out


def _col(rows: Rows, key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=np.float64)


def _compute_split_metrics(preds: Rows, alpha: float = 0.1) -> dict[str, float]:
    metrics = regression_metrics(_col(preds, "rul"), _col(preds, "y_pred_rul"))
    norm_metrics = regression_metrics(_col(preds, "rul_norm"), _col(preds, "y_pred_norm"))
    metrics["normalized_mae"] = norm_metrics["mae"]
    metrics["normalized_rmse"] = norm_metrics["rmse"]
    if (
        preds
        and {"lower_rul", "upper_rul"}.issubset(preds[0])
        and all(str(row.get("lower_rul", "")) != "" and str(row.get("upper_rul", "")) != "" for row in preds)
    ):
        metrics.update(interval_metrics(_col(preds, "rul"), _col(preds, "lower_rul"), _col(preds, "upper_rul"), alpha=alpha))
    if preds and all(str(row.get("damage_pred", "")) != "" for row in preds):
        metrics["monotonic_violation_rate"] = damage_monotonic_violation_rate(preds, _col(preds, "damage_pred"))
        metrics["monotonic_metric_source"] = "damage_pred"
    else:
        metrics["monotonic_violation_rate"] = monotonic_violation_rate(preds, _col(preds, "y_pred_norm"))
        metrics["monotonic_metric_source"] = "rul_proxy"
    return metrics


def _fit_sklearn(config: dict[str, Any], frame: Rows, split: SplitIndices, feat_cols: list[str], run_dir: Path) -> Rows:
    model_type = config.get("model", {}).get("type", "random_forest")
    model_cfg = config.get("model", {})
    context_mode = _context_mode(config)
    _assert_context_is_clean(context_mode)
    backend = str(model_cfg.get("backend", "numpy")).lower()
    if backend == "sklearn":
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=int(model_cfg.get("n_estimators", 200)),
                min_samples_leaf=int(model_cfg.get("min_samples_leaf", 2)),
                random_state=int(config.get("seed", 7)),
                n_jobs=int(model_cfg.get("n_jobs", 1)),
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=int(model_cfg.get("n_estimators", 300)),
                learning_rate=float(model_cfg.get("learning_rate", 0.05)),
                max_depth=int(model_cfg.get("max_depth", 3)),
                min_samples_leaf=int(model_cfg.get("min_samples_leaf", 2)),
                random_state=int(config.get("seed", 7)),
            )
        else:
            raise ValueError(model_type)
    else:
        if model_type == "gradient_boosting":
            model = GradientBoostedStumpRegressor(
                n_estimators=int(model_cfg.get("n_estimators", 240)),
                learning_rate=float(model_cfg.get("learning_rate", 0.035)),
                max_thresholds=int(model_cfg.get("max_thresholds", 32)),
                subsample_features=float(model_cfg.get("subsample_features", 1.0)),
                seed=int(config.get("seed", 7)),
            )
        else:
            model = RandomSubspaceRidgeRegressor(
                n_estimators=int(model_cfg.get("n_estimators", 80)),
                seed=int(config.get("seed", 7)),
            )
    x_train, mean, std = _design_matrix(frame, feat_cols, split.train, context_mode)
    y_train, _, _ = _targets(frame, split.train)
    model.fit(x_train, y_train)
    with (run_dir / "model.pkl").open("wb") as f:
        pickle.dump({"model": model, "mean": mean, "std": std, "feature_columns": feat_cols, **_context_info(context_mode, split.description)}, f)
    predictions: Rows = []
    cal_pred = None
    cal_true = None
    cal_name = _calibration_split_name(split)
    for name, indices in _split_groups(split).items():
        x, _, _ = _design_matrix(frame, feat_cols, indices, context_mode, mean, std)
        pred = model.predict(x)
        if name == cal_name:
            cal_pred = pred
            cal_true, _, _ = _targets(frame, indices)
        predictions.extend(_prediction_frame(frame, indices, name, pred))
    conformal = None
    use_conformal = bool(config.get("uncertainty", {}).get("enabled", True))
    if use_conformal and cal_pred is not None and cal_true is not None and len(cal_true):
        conformal = ConformalInterval(alpha=float(config.get("uncertainty", {}).get("alpha", 0.1))).fit(cal_true, cal_pred)
        predictions = []
        for name, indices in _split_groups(split).items():
            x, _, _ = _design_matrix(frame, feat_cols, indices, context_mode, mean, std)
            pred = model.predict(x)
            lo, hi = conformal.predict(pred)
            predictions.extend(_prediction_frame(frame, indices, name, pred, lo, hi))
        rng = np.random.default_rng(int(config.get("seed", 7)))
        for scenario in config.get("stress_tests", []):
            x, _, _ = _design_matrix(frame, feat_cols, split.test, context_mode, mean, std)
            stressed = x.copy()
            if "feature_noise_std" in scenario:
                stressed += rng.normal(0, float(scenario["feature_noise_std"]), size=stressed.shape)
            if scenario.get("zero_engineered_features"):
                stressed[:, : len(feat_cols)] = 0.0
            if "missing_feature_fraction" in scenario:
                frac = float(scenario["missing_feature_fraction"])
                n = max(1, int(stressed.shape[1] * frac))
                cols = rng.choice(stressed.shape[1], size=n, replace=False)
                stressed[:, cols] = 0.0
            pred = model.predict(stressed)
            lo, hi = conformal.predict(pred)
            predictions.extend(_prediction_frame(frame, split.test, f"test_stress_{scenario['name']}", pred, lo, hi))
    return predictions


def _prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    input_ablation: dict[str, Any] | None = None,
    *,
    training: bool = False,
    context_dropout: float = 0.0,
) -> dict[str, torch.Tensor]:
    out = {k: v.to(device) for k, v in batch.items()}
    ablation = input_ablation or {}
    if ablation.get("zero_raw_signal"):
        out["signal"] = torch.zeros_like(out["signal"])
    if ablation.get("zero_engineered_features"):
        out["features"] = torch.zeros_like(out["features"])
    if ablation.get("zero_condition_metadata"):
        context = out["context"].clone()
        context[:, :] = 0.0
        out["context"] = context
    elif training and context_dropout > 0:
        keep = torch.rand((out["context"].shape[0], 1), device=out["context"].device) >= float(context_dropout)
        out["context"] = out["context"] * keep.to(out["context"].dtype)
    if "context_scale" in ablation:
        context = out["context"].clone()
        scale = torch.tensor(ablation["context_scale"], dtype=context.dtype, device=context.device)
        n = min(context.shape[1], len(scale))
        if n > 0:
            context[:, :n] = context[:, :n] * scale[:n]
        out["context"] = context
    return out


def _predict_torch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_ablation: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    preds, targets, sample_ids, damage, damage_rate = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _prepare_batch(batch, device, input_ablation)
            out = model(batch)
            preds.append(out["rul"].detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            sample_ids.append(batch["sample_id"].detach().cpu().numpy())
            if "damage" in out:
                damage.append(out["damage"].detach().cpu().numpy())
            if "damage_rate" in out:
                damage_rate.append(out["damage_rate"].detach().cpu().numpy())
    result = {
        "pred": np.concatenate(preds) if preds else np.array([]),
        "target": np.concatenate(targets) if targets else np.array([]),
        "sample_id": np.concatenate(sample_ids) if sample_ids else np.array([]),
    }
    if damage:
        result["damage"] = np.concatenate(damage)
    if damage_rate:
        result["damage_rate"] = np.concatenate(damage_rate)
    return result


def _train_one_torch_member(
    config: dict[str, Any],
    signals: np.ndarray,
    frame: Rows,
    split: SplitIndices,
    feat_cols: list[str],
    seed: int,
    member_idx: int,
    run_dir: Path,
) -> tuple[torch.nn.Module, dict[str, np.ndarray], dict[str, Any]]:
    set_seed(seed)
    device = available_device(bool(config.get("training", {}).get("cuda", True)))
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    input_ablation = config.get("input_ablation", {})
    context_mode = _context_mode(config)
    _assert_context_is_clean(context_mode)
    context_dim = len(context_columns(context_mode))
    model = build_model(
        model_type=model_cfg.get("type", "digital_twin"),
        channels=int(signals.shape[-1]),
        feature_dim=len(feat_cols),
        hidden=int(model_cfg.get("hidden", 64)),
        context_dim=context_dim,
    ).to(device)
    train_ds = BearingDataset(signals, frame, split.train, feat_cols, context_mode=context_mode)
    val_ds = BearingDataset(
        signals,
        frame,
        split.val,
        feat_cols,
        train_ds.feature_mean,
        train_ds.feature_std,
        train_ds.context_mean,
        train_ds.context_std,
        context_mode,
    )
    physics_weight = float(train_cfg.get("physics_weight", 0.1))
    sequence_batches = bool(train_cfg.get("sequence_batches", False))
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 32)), shuffle=not sequence_batches)
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    best_state = None
    best_val = float("inf")
    patience = int(train_cfg.get("patience", 12))
    stale = 0
    weights = train_cfg.get("physics_weights", {})
    epochs = int(train_cfg.get("epochs", 80))
    best_epoch = 0
    epochs_completed = 0
    for epoch in range(epochs):
        epochs_completed = epoch + 1
        model.train()
        for batch in train_loader:
            batch = _prepare_batch(
                batch,
                device,
                input_ablation,
                training=True,
                context_dropout=float(train_cfg.get("context_dropout", 0.0)),
            )
            out = model(batch)
            loss = F.mse_loss(out["rul"], batch["target"].float())
            if physics_weight > 0:
                components = physics_regularization(out, batch)
                phys = sum(float(weights.get(k, 1.0)) * v for k, v in components.items())
                loss = loss + physics_weight * phys
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        val_pred = _predict_torch(model, val_loader, device, input_ablation)
        val_loss = float(np.mean((val_pred["pred"] - val_pred["target"]) ** 2)) if len(val_pred["pred"]) else 0.0
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / f"model_member{member_idx}.pt")
    loaders = {}
    train_stats = (train_ds.feature_mean, train_ds.feature_std, train_ds.context_mean, train_ds.context_std)
    for name, indices in _split_groups(split).items():
        ds = BearingDataset(signals, frame, indices, feat_cols, *train_stats, context_mode)
        loaders[name] = DataLoader(ds, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False)
    preds = {name: _predict_torch(model, loader, device, input_ablation) for name, loader in loaders.items()}
    training_record = {
        "member": int(member_idx),
        "seed": int(seed),
        "selection_metric": "validation_mse",
        "reported_state": "best_validation_loss",
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_completed": int(epochs_completed),
        "max_epochs": int(epochs),
        "patience": int(patience),
        "stopped_early": bool(epochs_completed < epochs),
    }
    return model, preds, training_record


def _stress_signals(signals: np.ndarray, scenario: dict[str, Any], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stressed = signals.copy()
    if "signal_noise_std" in scenario:
        stressed += rng.normal(0, float(scenario["signal_noise_std"]), size=stressed.shape).astype(np.float32)
    if "zero_channel" in scenario and stressed.shape[-1] > int(scenario["zero_channel"]):
        stressed[:, :, int(scenario["zero_channel"])] = 0.0
    if "missing_channel_fraction" in scenario:
        n = max(1, int(stressed.shape[-1] * float(scenario["missing_channel_fraction"])))
        channels = rng.choice(stressed.shape[-1], size=n, replace=False)
        stressed[:, :, channels] = 0.0
    return stressed


def _scenario_input_ablation(base_ablation: dict[str, Any] | None, scenario: dict[str, Any]) -> dict[str, Any]:
    ablation = dict(base_ablation or {})
    for key in ("zero_raw_signal", "zero_engineered_features", "zero_condition_metadata"):
        if key in scenario:
            ablation[key] = bool(scenario[key])
    if "context_scale" in scenario:
        ablation["context_scale"] = scenario["context_scale"]
    return ablation


def _frame_with_recomputed_features(
    frame: Rows,
    feat_cols: list[str],
    indices: np.ndarray,
    stressed_signals: np.ndarray,
    sample_rate: float,
) -> Rows:
    updated = [dict(row) for row in frame]
    for idx in indices:
        i = int(idx)
        recomputed = extract_feature_row(stressed_signals[i], sample_rate=sample_rate)
        for col in feat_cols:
            if col in recomputed:
                updated[i][col] = recomputed[col]
    return updated


def _fit_torch(
    config: dict[str, Any],
    signals: np.ndarray,
    frame: Rows,
    split: SplitIndices,
    feat_cols: list[str],
    run_dir: Path,
) -> tuple[Rows, dict[str, Any]]:
    model_cfg = config.get("model", {})
    ensemble_size = int(model_cfg.get("ensemble_size", 1))
    seed = int(config.get("seed", 7))
    context_mode = _context_mode(config)
    member_preds = []
    models = []
    training_records = []
    for member in range(ensemble_size):
        model, preds, training_record = _train_one_torch_member(config, signals, frame, split, feat_cols, seed + member, member, run_dir)
        models.append(model)
        member_preds.append(preds)
        training_records.append(training_record)
    predictions: Rows = []
    cal_name = _calibration_split_name(split)
    cal_stack = np.stack([p[cal_name]["pred"] for p in member_preds], axis=0)
    cal_mean = cal_stack.mean(axis=0)
    cal_true = member_preds[0][cal_name]["target"]
    use_conformal = bool(config.get("uncertainty", {}).get("enabled", True))
    conformal = None
    if use_conformal:
        conformal = ConformalInterval(alpha=float(config.get("uncertainty", {}).get("alpha", 0.1))).fit(cal_true, cal_mean)
    for name, indices in _split_groups(split).items():
        stack = np.stack([p[name]["pred"] for p in member_preds], axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        lo, hi = (None, None)
        if conformal is not None:
            lo, hi = conformal.predict(mean, std if ensemble_size > 1 else None)
        damage = None
        if "damage" in member_preds[0][name]:
            damage = np.stack([p[name]["damage"] for p in member_preds], axis=0).mean(axis=0)
        damage_rate = None
        if "damage_rate" in member_preds[0][name]:
            damage_rate = np.stack([p[name]["damage_rate"] for p in member_preds], axis=0).mean(axis=0)
        predictions.extend(_prediction_frame(frame, indices, name, mean, lo, hi, std, damage, damage_rate))
    if config.get("stress_tests"):
        device = available_device(bool(config.get("training", {}).get("cuda", True)))
        train_ds = BearingDataset(signals, frame, split.train, feat_cols, context_mode=context_mode)
        train_stats = (train_ds.feature_mean, train_ds.feature_std, train_ds.context_mean, train_ds.context_std)
        batch_size = int(config.get("training", {}).get("batch_size", 64))
        for scenario_idx, scenario in enumerate(config.get("stress_tests", [])):
            stressed_signals = _stress_signals(signals, scenario, seed + 10_000 + scenario_idx)
            scenario_frame = frame
            if scenario.get("recompute_engineered_features"):
                scenario_frame = _frame_with_recomputed_features(
                    frame,
                    feat_cols,
                    split.test,
                    stressed_signals,
                    float(config.get("data", {}).get("sample_rate", 25_600.0)),
                )
            ds = BearingDataset(stressed_signals, scenario_frame, split.test, feat_cols, *train_stats, context_mode)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            scenario_ablation = _scenario_input_ablation(config.get("input_ablation", {}), scenario)
            scenario_preds = []
            for model in models:
                scenario_preds.append(_predict_torch(model.to(device), loader, device, scenario_ablation))
            stack = np.stack([p["pred"] for p in scenario_preds], axis=0)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            lo, hi = (None, None)
            if conformal is not None:
                lo, hi = conformal.predict(mean, std if ensemble_size > 1 else None)
            damage = None
            if "damage" in scenario_preds[0]:
                damage = np.stack([p["damage"] for p in scenario_preds], axis=0).mean(axis=0)
            damage_rate = None
            if "damage_rate" in scenario_preds[0]:
                damage_rate = np.stack([p["damage_rate"] for p in scenario_preds], axis=0).mean(axis=0)
            predictions.extend(_prediction_frame(frame, split.test, f"test_stress_{scenario['name']}", mean, lo, hi, std, damage, damage_rate))
    if config.get("condition_response_tests"):
        device = available_device(bool(config.get("training", {}).get("cuda", True)))
        train_ds = BearingDataset(signals, frame, split.train, feat_cols, context_mode=context_mode)
        ds = BearingDataset(
            signals,
            frame,
            split.test,
            feat_cols,
            train_ds.feature_mean,
            train_ds.feature_std,
            train_ds.context_mean,
            train_ds.context_std,
            context_mode,
        )
        loader = DataLoader(ds, batch_size=int(config.get("training", {}).get("batch_size", 64)), shuffle=False)
        for response in config.get("condition_response_tests", []):
            ablation = dict(config.get("input_ablation", {}))
            ablation["context_scale"] = response.get("context_scale", [1.25, 1.25, 1.0])
            response_preds = [_predict_torch(model.to(device), loader, device, ablation) for model in models]
            stack = np.stack([p["pred"] for p in response_preds], axis=0)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            damage = None
            if "damage" in response_preds[0]:
                damage = np.stack([p["damage"] for p in response_preds], axis=0).mean(axis=0)
            damage_rate = None
            if "damage_rate" in response_preds[0]:
                damage_rate = np.stack([p["damage_rate"] for p in response_preds], axis=0).mean(axis=0)
            predictions.extend(
                _prediction_frame(
                    frame,
                    split.test,
                    f"test_condition_response_{response['name']}",
                    mean,
                    None,
                    None,
                    std,
                    damage,
                    damage_rate,
                )
            )
    training_selection = {
        "reported_state": "best_validation_loss",
        "selection_metric": "validation_mse",
        "members": training_records,
    }
    write_json(run_dir / "training_selection.json", training_selection)
    return predictions, training_selection


def train_experiment(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    run_dir = _run_dir(config)
    shutil.copy2(config_path, run_dir / "config.yaml")
    signals, frame, manifest = load_processed(config.get("data", {}).get("processed_dir", "data/processed/synthetic_tiny"))
    device = available_device(bool(config.get("training", {}).get("cuda", True)))
    print(f"Training device: {device}")
    feat_cols = manifest.get("feature_columns") or feature_columns(frame)
    split_cfg = config.get("split", {})
    split_seed = int(split_cfg.get("seed", seed)) if isinstance(split_cfg, dict) else seed
    split = make_split(frame, split_cfg, seed=seed)
    split = _apply_train_fraction(frame, split, float(config.get("data_efficiency", {}).get("train_fraction", 1.0)), seed)
    if len(split.train) == 0 or len(split.val) == 0 or len(split.test) == 0 or (split.cal is not None and len(split.cal) == 0):
        raise ValueError(f"Empty split generated: {split}")
    model_type = str(config.get("model", {}).get("type", "digital_twin")).lower()
    context_mode = _context_mode(config)
    _assert_context_is_clean(context_mode)
    context_meta = _context_info(context_mode, split.description)
    training_selection = None
    if model_type in SKLEARN_MODELS:
        preds = _fit_sklearn(config, frame, split, feat_cols, run_dir)
    else:
        preds, training_selection = _fit_torch(config, signals, frame, split, feat_cols, run_dir)
    preds = _append_eval_subsets(preds, config)
    for row in preds:
        row["model_type"] = model_type
    write_rows_csv(run_dir / "predictions.csv", preds)
    alpha = float(config.get("uncertainty", {}).get("alpha", 0.1))
    metrics = {str(name): _compute_split_metrics(rows, alpha=alpha) for name, rows in group_by(preds, "split").items()}
    metrics["run"] = {
        "experiment_name": config.get("experiment_name", run_dir.name),
        "model_type": model_type,
        "split": split.description,
        "split_seed": split_seed,
        "samples": int(len(frame)),
        "calibration_split": _calibration_split_name(split),
        "features": int(len(feat_cols)),
        **context_meta,
    }
    if training_selection is not None:
        metrics["run"]["training_selection"] = training_selection
    write_json(run_dir / "metrics.json", metrics)
    run_manifest = {
        "config": str(config_path),
        "processed_manifest": manifest,
        "split": {
            "description": split.description,
            "seed": split_seed,
            "train": int(len(split.train)),
            "val": int(len(split.val)),
            "cal": int(len(split.cal)) if split.cal is not None else None,
            "test": int(len(split.test)),
        },
        "context": context_meta,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "deterministic_cuda_kernels": False,
        },
    }
    if training_selection is not None:
        run_manifest["training_selection"] = training_selection
    write_json(run_dir / "run_manifest.json", run_manifest)
    print(f"Run complete: {run_dir}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f}, MAE: {metrics['test']['mae']:.4f}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.train")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    train_experiment(args.config)


if __name__ == "__main__":
    main()
