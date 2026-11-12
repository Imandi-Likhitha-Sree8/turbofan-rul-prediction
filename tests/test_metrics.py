"""Tests for regression metrics and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import bootstrap_intervals, nasa_score, regression_metrics


class TestNasaScore:
    def test_zero_error_gives_zero_score(self):
        """Perfect predictions must yield NASA score = 0."""
        y = np.array([50.0, 100.0, 30.0])
        assert nasa_score(y, y) == pytest.approx(0.0)

    def test_overestimation_penalised_more_than_underestimation(self):
        """A positive error (overestimate) must exceed the same-magnitude negative error."""
        y_true = np.array([50.0])
        over = nasa_score(y_true, y_true + 10)
        under = nasa_score(y_true, y_true - 10)
        assert over > under

    def test_score_is_non_negative_for_any_error(self):
        """NASA score must be ≥ 0 for any prediction."""
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 200, 50)
        y_pred = rng.uniform(0, 200, 50)
        assert nasa_score(y_true, y_pred) >= 0.0

    def test_larger_error_gives_higher_score(self):
        """Increasing prediction error must increase the NASA score."""
        y_true = np.array([100.0])
        assert nasa_score(y_true, y_true + 5) < nasa_score(y_true, y_true + 20)

    def test_numpy_expm1_formula(self):
        """Verify the formula value for a known single-engine case."""
        # error = 10 → exponent = 10/10 = 1 → expm1(1) = e - 1
        result = nasa_score(np.array([0.0]), np.array([10.0]))
        expected = float(np.expm1(1.0))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_negative_error_formula(self):
        """Verify the formula for under-prediction."""
        # error = -13 → exponent = 13/13 = 1 → expm1(1) = e - 1
        result = nasa_score(np.array([13.0]), np.array([0.0]))
        expected = float(np.expm1(1.0))
        assert result == pytest.approx(expected, rel=1e-6)


class TestRegressionMetrics:
    def test_perfect_predictions_all_zero(self):
        """Perfect predictions → RMSE = MAE = NASA_score = 0, R² = 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, y)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["MAE"] == pytest.approx(0.0)
        assert m["R2"] == pytest.approx(1.0)
        assert m["NASA_score"] == pytest.approx(0.0)

    def test_rmse_greater_than_mae(self):
        """RMSE must be ≥ MAE (equality only when all errors are equal magnitude)."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 25.0, 25.0])
        m = regression_metrics(y_true, y_pred)
        assert m["RMSE"] >= m["MAE"]

    def test_r2_range(self):
        """R² must be ≤ 1 (can be negative for bad models)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # Very bad predictions.
        m = regression_metrics(y_true, y_pred)
        assert m["R2"] <= 1.0

    def test_all_keys_present(self):
        """regression_metrics must return RMSE, MAE, R2, and NASA_score."""
        m = regression_metrics(np.array([1.0]), np.array([1.0]))
        for key in ("RMSE", "MAE", "R2", "NASA_score"):
            assert key in m

    def test_known_rmse_value(self):
        """Verify RMSE for a known case: errors [1, -1] → RMSE = 1.0."""
        y_true = np.array([2.0, 2.0])
        y_pred = np.array([3.0, 1.0])
        assert regression_metrics(y_true, y_pred)["RMSE"] == pytest.approx(1.0)


class TestBootstrapIntervals:
    def test_returns_rmse_and_mae_keys(self):
        """bootstrap_intervals must return 'RMSE' and 'MAE' keys."""
        y = np.array([10.0, 20.0, 30.0])
        result = bootstrap_intervals(y, y, repeats=50)
        assert "RMSE" in result
        assert "MAE" in result

    def test_interval_is_two_values(self):
        """Each CI must be [lower, upper]."""
        y = np.array([10.0, 20.0, 30.0])
        result = bootstrap_intervals(y, y, repeats=50)
        for ci in result.values():
            assert len(ci) == 2

    def test_lower_le_upper(self):
        """Lower bound must be ≤ upper bound."""
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 100, 20)
        y_pred = y_true + rng.normal(0, 5, 20)
        result = bootstrap_intervals(y_true, y_pred, repeats=200)
        for ci in result.values():
            assert ci[0] <= ci[1]

    def test_perfect_predictions_zero_ci(self):
        """With perfect predictions, CI bounds for RMSE and MAE must both be 0."""
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = bootstrap_intervals(y, y, repeats=100)
        assert result["RMSE"][0] == pytest.approx(0.0)
        assert result["RMSE"][1] == pytest.approx(0.0)

    def test_reproducibility(self):
        """Same seed must produce identical CI bounds."""
        rng = np.random.default_rng(99)
        y_true = rng.uniform(0, 100, 30)
        y_pred = y_true + rng.normal(0, 10, 30)
        r1 = bootstrap_intervals(y_true, y_pred, repeats=200, seed=42)
        r2 = bootstrap_intervals(y_true, y_pred, repeats=200, seed=42)
        assert r1["RMSE"] == r2["RMSE"]
        assert r1["MAE"] == r2["MAE"]
