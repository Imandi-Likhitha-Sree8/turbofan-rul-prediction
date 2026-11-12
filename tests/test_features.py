"""Tests for feature engineering: rolling statistics, leakage, and column exclusion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import BASE_COLUMNS
from src.features import build_features, model_columns
from tests.test_data import make_sample_frame


class TestBuildFeatures:
    def test_output_has_more_columns_than_input(self):
        """build_features must add rolling mean and std columns."""
        frame = make_sample_frame()
        result = build_features(frame)
        # 2 × 21 sensors additional columns (mean + std)
        assert result.shape[1] == frame.shape[1] + 2 * 21

    def test_sorted_by_engine_then_cycle(self):
        """Output must be sorted by (engine_id, cycle)."""
        frame = make_sample_frame(
            engine_ids=[2, 1, 2, 1], cycles=[2, 3, 1, 1]
        )
        result = build_features(frame)
        pairs = list(zip(result["engine_id"], result["cycle"]))
        assert pairs == sorted(pairs)

    def test_rolling_does_not_cross_engines(self):
        """Rolling statistics must not bleed across engine boundaries."""
        frame = make_sample_frame()
        result = build_features(frame, rolling_window=2)
        # First row of engine 2 must have NaN std (only one obs per engine so far).
        engine2_first = result[result["engine_id"] == 2].iloc[0]
        assert np.isnan(engine2_first["sensor_1_rollstd_2"])

    def test_first_cycle_std_is_nan(self):
        """Standard deviation for a window needing ≥ 2 observations must be NaN on the first cycle."""
        frame = make_sample_frame()
        result = build_features(frame, rolling_window=5)
        first_per_engine = result.groupby("engine_id").head(1)
        assert first_per_engine["sensor_1_rollstd_5"].isna().all()

    def test_rollmean_correct_value(self):
        """Rolling mean over window=2 for engine 1 cycles [1, 3] should average correctly."""
        frame = make_sample_frame(engine_ids=[1, 1], cycles=[1, 2])
        # Set sensor_1 explicitly.
        frame["sensor_1"] = [4.0, 6.0]
        result = build_features(frame, rolling_window=2)
        # Cycle 1: mean = 4.0 (only one obs). Cycle 2: mean = (4+6)/2 = 5.0.
        means = result.sort_values("cycle")["sensor_1_rollmean_2"].tolist()
        assert means[0] == pytest.approx(4.0)
        assert means[1] == pytest.approx(5.0)

    def test_window_one_std_is_all_nan(self):
        """With window=1, rolling std is always NaN because std of one value is undefined."""
        frame = make_sample_frame()
        result = build_features(frame, rolling_window=1)
        # A window of 1 observation always produces NaN for std (requires ≥ 2 values).
        assert result["sensor_1_rollstd_1"].isna().all()

    def test_invalid_window_raises(self):
        """rolling_window < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="rolling_window"):
            build_features(make_sample_frame(), rolling_window=0)

    def test_rul_column_preserved(self):
        """If the input has a 'RUL' column it should appear in the output."""
        from src.data import add_rul
        frame = add_rul(make_sample_frame())
        result = build_features(frame)
        assert "RUL" in result.columns

    def test_no_future_data_leakage(self):
        """Rolling statistics at cycle t must not use values from cycle t+1."""
        engine_data = pd.DataFrame({col: [1] * 4 for col in BASE_COLUMNS})
        engine_data["engine_id"] = [1, 1, 1, 1]
        engine_data["cycle"] = [1, 2, 3, 4]
        # Set sensor_1 to known values to verify no look-ahead.
        engine_data["sensor_1"] = [1.0, 2.0, 3.0, 4.0]
        result = build_features(engine_data, rolling_window=3)
        result = result.sort_values("cycle").reset_index(drop=True)
        # At cycle 3 (index 2), window includes cycles 1,2,3 → mean = 2.0
        assert result.loc[2, "sensor_1_rollmean_3"] == pytest.approx(2.0)
        # At cycle 4 (index 3), window includes cycles 2,3,4 → mean = 3.0
        assert result.loc[3, "sensor_1_rollmean_3"] == pytest.approx(3.0)


class TestModelColumns:
    def test_engine_id_excluded(self):
        """engine_id must not appear in model_columns output."""
        frame = build_features(make_sample_frame())
        assert "engine_id" not in model_columns(frame)

    def test_rul_excluded_when_present(self):
        """RUL must not appear in model_columns output."""
        from src.data import add_rul
        frame = build_features(add_rul(make_sample_frame()))
        assert "RUL" not in model_columns(frame)

    def test_all_sensor_rollmean_included(self):
        """model_columns must include all 21 rolling mean features."""
        frame = build_features(make_sample_frame())
        cols = model_columns(frame)
        for i in range(1, 22):
            assert f"sensor_{i}_rollmean_5" in cols

    def test_cycle_included(self):
        """cycle must be a model feature."""
        frame = build_features(make_sample_frame())
        assert "cycle" in model_columns(frame)
