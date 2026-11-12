"""Tests for data loading, RUL calculation, and validation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import BASE_COLUMNS, add_rul, validate_trajectory


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_sample_frame(
    engine_ids: list[int] | None = None,
    cycles: list[int] | None = None,
) -> pd.DataFrame:
    """Return a minimal valid trajectory with two engines, two cycles each."""
    if engine_ids is None:
        engine_ids = [1, 1, 2, 2]
    if cycles is None:
        cycles = [1, 3, 1, 2]

    data: dict = {col: [1] * len(engine_ids) for col in BASE_COLUMNS}
    data["engine_id"] = engine_ids
    data["cycle"] = cycles
    # Set distinct sensor values per row so they're not all identical.
    for i in range(1, 22):
        data[f"sensor_{i}"] = [float(i + j) for j in range(len(engine_ids))]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# RUL calculation
# ---------------------------------------------------------------------------

class TestAddRul:
    def test_rul_is_engine_local(self):
        """RUL is computed per engine, not across the whole dataset."""
        frame = make_sample_frame(
            engine_ids=[1, 1, 2, 2],
            cycles=[1, 3, 1, 2],
        )
        result = add_rul(frame)
        # Engine 1: max=3 → RUL=[2, 0]; Engine 2: max=2 → RUL=[1, 0]
        assert result["RUL"].tolist() == [2, 0, 1, 0]

    def test_final_cycle_rul_is_zero(self):
        """Every engine's last cycle must have RUL=0."""
        frame = make_sample_frame()
        result = add_rul(frame)
        last_rows = result.groupby("engine_id")["cycle"].idxmax()
        assert (result.loc[last_rows, "RUL"] == 0).all()

    def test_rul_is_non_negative(self):
        """RUL values must never be negative."""
        frame = make_sample_frame()
        result = add_rul(frame)
        assert (result["RUL"] >= 0).all()

    def test_rul_col_added(self):
        """add_rul must append a 'RUL' column without removing existing columns."""
        frame = make_sample_frame()
        result = add_rul(frame)
        assert "RUL" in result.columns
        # Original columns must still be present.
        for col in BASE_COLUMNS:
            assert col in result.columns

    def test_single_cycle_engine_has_rul_zero(self):
        """An engine with only one observed cycle must have RUL=0."""
        data = {col: [1] for col in BASE_COLUMNS}
        data["engine_id"] = [99]
        data["cycle"] = [5]
        for i in range(1, 22):
            data[f"sensor_{i}"] = [float(i)]
        frame = pd.DataFrame(data)
        result = add_rul(frame)
        assert result["RUL"].iloc[0] == 0


# ---------------------------------------------------------------------------
# validate_trajectory
# ---------------------------------------------------------------------------

class TestValidateTrajectory:
    def test_valid_frame_passes(self):
        """A valid frame should raise nothing."""
        validate_trajectory(make_sample_frame())  # Must not raise.

    def test_missing_column_raises(self):
        """A frame missing a required column must raise ValueError."""
        frame = make_sample_frame().drop(columns=["sensor_1"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_trajectory(frame)

    def test_duplicate_engine_cycle_raises(self):
        """Duplicate (engine_id, cycle) pairs must be rejected."""
        frame = make_sample_frame(cycles=[1, 1, 1, 2])  # engine 1, cycle 1 twice
        with pytest.raises(ValueError, match="unique"):
            validate_trajectory(frame)

    def test_nonpositive_cycle_raises(self):
        """cycle = 0 must be rejected."""
        frame = make_sample_frame(cycles=[0, 1, 1, 2])
        with pytest.raises(ValueError, match="strictly positive"):
            validate_trajectory(frame)

    def test_nonpositive_engine_id_raises(self):
        """engine_id = 0 must be rejected."""
        frame = make_sample_frame(engine_ids=[0, 0, 1, 1])
        with pytest.raises(ValueError, match="strictly positive"):
            validate_trajectory(frame)

    def test_empty_frame_raises(self):
        """An empty DataFrame must raise ValueError."""
        frame = pd.DataFrame(columns=BASE_COLUMNS)
        with pytest.raises(ValueError, match="empty"):
            validate_trajectory(frame)
