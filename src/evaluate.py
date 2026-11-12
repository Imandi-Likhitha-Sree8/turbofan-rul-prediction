"""Regression metrics and uncertainty helpers for turbofan RUL prediction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric C-MAPSS score penalising RUL overestimation more than underestimation.

    Definition
    ----------
    For each engine:
        d = y_pred − y_true
        s = exp(−d/13) − 1   if d < 0  (under-prediction)
        s = exp(d/10)  − 1   if d ≥ 0  (over-prediction)
    Score = sum(s)

    The exponent is clipped at 50 to prevent numeric overflow.
    Overestimating remaining life (d > 0) is penalised more steeply.

    Parameters
    ----------
    y_true: array of actual RUL values.
    y_pred: array of predicted RUL values.

    Returns
    -------
    Total asymmetric score (lower is better; 0 is perfect).
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    return float(np.expm1(np.clip(exponent, 0, 50)).sum())


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, R², and NASA asymmetric score.

    Parameters
    ----------
    y_true: ground-truth RUL values.
    y_pred: model predictions.

    Returns
    -------
    Dictionary with keys ``'RMSE'``, ``'MAE'``, ``'R2'``, and
    ``'NASA_score'``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA_score": nasa_score(y_true, y_pred),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repeats: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """95% percentile confidence intervals by resampling test engines.

    Each bootstrap iteration draws *n* engines (with replacement) from the
    *n* test predictions and re-computes RMSE and MAE.

    Parameters
    ----------
    y_true: 1-D array of ground-truth RUL labels (one per engine).
    y_pred: 1-D array of predictions in the same order.
    repeats: number of bootstrap replications.
    seed: random seed for reproducibility.

    Returns
    -------
    Dictionary mapping metric name → [lower_2.5%, upper_97.5%] bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    samples: dict[str, list[float]] = {"RMSE": [], "MAE": []}
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(y_true[idx], y_pred[idx])
        for key in samples:
            samples[key].append(m[key])
    return {
        key: [
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
        ]
        for key, vals in samples.items()
    }
