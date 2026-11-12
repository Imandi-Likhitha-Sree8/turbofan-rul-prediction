"""Leakage-safe, per-engine rolling feature engineering for C-MAPSS data.

Design principles
-----------------
* All rolling statistics use only current and previous observations — no
  future-cycle values are ever used to predict an earlier cycle.
* Grouped rolling is computed independently for each engine.
* engine_id is retained for grouping/splitting but excluded from model inputs
  by ``model_columns()``.
* NaN rolling standard deviations (early cycles with < 2 observations) are
  *intentionally* left as NaN so that scikit-learn pipeline imputation handles
  them correctly and uniformly across CV folds.
"""
from __future__ import annotations

import pandas as pd

from src.data import BASE_COLUMNS, validate_trajectory

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]

# Features that form the raw (non-rolling) baseline.
RAW_FEATURE_COLS = ["cycle"] + SETTING_COLS + SENSOR_COLS


def build_features(frame: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Return a copy of *frame* augmented with trailing rolling sensor statistics.

    Parameters
    ----------
    frame:
        A trajectory DataFrame with at least the ``BASE_COLUMNS`` columns.
        May optionally contain a ``'RUL'`` column (preserved if present).
    rolling_window:
        Number of consecutive cycles to include in each rolling window.
        Must be ≥ 1.

    Returns
    -------
    A new DataFrame, sorted by (engine_id, cycle), containing the original
    columns plus ``sensor_<i>_rollmean_<w>`` and ``sensor_<i>_rollstd_<w>``
    for every sensor *i* and window size *w*.

    Notes
    -----
    Rolling statistics are computed with ``min_periods=1`` for means (so the
    very first cycle of each engine has a valid mean) and ``min_periods=2`` for
    standard deviations (so the first cycle deliberately produces NaN, which
    the downstream pipeline imputes).
    """
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1.")
    validate_trajectory(frame)

    target_cols = BASE_COLUMNS + (["RUL"] if "RUL" in frame.columns else [])
    result = frame[target_cols].copy()
    result = result.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    grouped = result.groupby("engine_id", sort=False)[SENSOR_COLS]

    # Rolling means — available from the very first cycle.
    means = (
        grouped.rolling(rolling_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    means.index = result.index
    means.columns = [f"{c}_rollmean_{rolling_window}" for c in SENSOR_COLS]

    # Rolling std — NaN when fewer than min(2, window) observations are present.
    # min_periods cannot exceed window, so we cap it at the window size.
    std_min_periods = min(2, rolling_window)
    stds = (
        grouped.rolling(rolling_window, min_periods=std_min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )
    stds.index = result.index
    stds.columns = [f"{c}_rollstd_{rolling_window}" for c in SENSOR_COLS]

    return pd.concat([result, means, stds], axis=1)


def model_columns(feature_frame: pd.DataFrame) -> list[str]:
    """Return the list of model input columns from a feature-engineered frame.

    Explicitly excluded: ``engine_id`` (identifier) and ``RUL`` (target).
    All other columns — raw sensors, settings, cycle, and rolling statistics —
    are returned.
    """
    return [
        col
        for col in feature_frame.columns
        if col not in {"engine_id", "RUL"}
    ]
