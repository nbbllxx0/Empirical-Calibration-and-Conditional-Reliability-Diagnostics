from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalInterval:
    alpha: float = 0.1
    qhat: float | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalInterval":
        residual = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        n = len(residual)
        if n == 0:
            raise ValueError("Cannot calibrate conformal interval with empty validation set")
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(1.0, float(q))
        self.qhat = float(np.quantile(residual, q, method="higher"))
        return self

    def predict(self, y_pred: np.ndarray, epistemic_std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.qhat is None:
            raise RuntimeError("ConformalInterval.fit must be called before predict")
        pred = np.asarray(y_pred, dtype=np.float64)
        width = np.full_like(pred, self.qhat, dtype=np.float64)
        if epistemic_std is not None:
            width = width + np.asarray(epistemic_std, dtype=np.float64)
        return pred - width, pred + width
