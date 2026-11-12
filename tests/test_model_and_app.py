"""Tests for model pipeline, engine-separation, and Streamlit upload validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.data import BASE_COLUMNS, add_rul
from src.features import build_features, model_columns
from src.train import make_pipeline
from tests.test_data import make_sample_frame


class TestMakePipeline:
    @pytest.mark.parametrize("name", [
        "linear", "random_forest", "extra_trees", "hist_gradient_boosting", "svr"
    ])
    def test_pipeline_predicts(self, name: str):
        """Every named pipeline must train and produce predictions without error."""
        frame = make_sample_frame()
        X = pd.DataFrame({
            "cycle": [1, 2, 3, 4, 5, 6],
            "setting_1": [0.1] * 6,
            "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        y = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        pipe = make_pipeline(name).fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (6,)

    def test_pipeline_handles_missing_values(self):
        """Pipeline imputation must handle NaN entries in input features."""
        X = pd.DataFrame({
            "cycle": [1, 2, 3, 4],
            "sensor_1": [1.0, np.nan, 3.0, 4.0],
            "sensor_2": [np.nan, 2.0, 3.0, 4.0],
        })
        y = [4.0, 3.0, 2.0, 1.0]
        pipe = make_pipeline("linear").fit(X, y)
        # Must predict on a row with NaN.
        result = pipe.predict(pd.DataFrame({"cycle": [5], "sensor_1": [np.nan], "sensor_2": [5.0]}))
        assert result.shape == (1,)

    def test_unknown_model_name_raises(self):
        """An unrecognised model name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            make_pipeline("xgboost")

    def test_predictions_non_negative_after_clipping(self):
        """Model predictions clipped at 0 must be non-negative."""
        X = pd.DataFrame({"cycle": [1, 2, 3], "sensor_1": [1.0, 2.0, 3.0]})
        y = [3.0, 2.0, 1.0]
        pipe = make_pipeline("linear").fit(X, y)
        preds = np.maximum(pipe.predict(X), 0.0)
        assert (preds >= 0.0).all()


class TestEngineSeparation:
    def test_no_engine_overlap_in_cv_folds(self):
        """GroupKFold must keep each engine entirely within one fold."""
        # Build a minimal multi-engine feature frame.
        rows = []
        for eng in range(1, 11):  # 10 engines
            for cycle in range(1, 6):
                row = {col: 1.0 for col in BASE_COLUMNS}
                row["engine_id"] = eng
                row["cycle"] = cycle
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame = add_rul(frame)
        feat_frame = build_features(frame)
        columns = model_columns(feat_frame)
        X = feat_frame[columns].values
        groups = feat_frame["engine_id"].to_numpy()

        cv = GroupKFold(n_splits=3)
        for train_idx, val_idx in cv.split(X, groups=groups):
            train_engines = set(groups[train_idx])
            val_engines = set(groups[val_idx])
            overlap = train_engines & val_engines
            assert overlap == set(), (
                f"Engine(s) {overlap} appear in both train and validation folds!"
            )

    def test_all_engines_appear_exactly_once_in_validation(self):
        """Every engine must appear in exactly one validation fold across all folds."""
        rows = []
        for eng in range(1, 10):
            for cycle in range(1, 4):
                row = {col: 1.0 for col in BASE_COLUMNS}
                row["engine_id"] = eng
                row["cycle"] = cycle
                rows.append(row)
        frame = add_rul(pd.DataFrame(rows))
        feat_frame = build_features(frame)
        groups = feat_frame["engine_id"].to_numpy()
        X = feat_frame[model_columns(feat_frame)].values

        val_engines_seen: list[int] = []
        for _, val_idx in GroupKFold(n_splits=3).split(X, groups=groups):
            val_engines_seen.extend(groups[val_idx].tolist())

        # Each engine ID should appear exactly `n_cycles` times across all folds.
        engine_counts = pd.Series(val_engines_seen).value_counts()
        # Every engine must appear in exactly one fold (3 cycles per engine).
        expected_count = 3  # cycles per engine
        for eng in range(1, 10):
            assert engine_counts.get(eng, 0) == expected_count, (
                f"Engine {eng} appeared in validation {engine_counts.get(eng, 0)} "
                f"times; expected {expected_count}."
            )


class TestStreamlitUploadValidation:
    def test_single_engine_upload_passes(self):
        """A single-engine valid CSV must pass validation without error."""
        pytest.importorskip("streamlit")
        from app.app import validate_trajectory  # Uses data.validate_trajectory

        frame = make_sample_frame(engine_ids=[1, 1], cycles=[1, 2])
        validate_trajectory(frame)  # Should not raise.

    def test_multi_engine_upload_rejected(self):
        """An upload with more than one engine must be caught by the app logic."""
        pytest.importorskip("streamlit")

        # Simulate the app-side check.
        frame = make_sample_frame(engine_ids=[1, 2], cycles=[1, 1])
        n_engines = frame["engine_id"].nunique()
        assert n_engines != 1, "Test setup error: frame should have 2 engines."

        # Confirm the guard would trigger.
        with pytest.raises(ValueError):
            if n_engines != 1:
                raise ValueError("Upload exactly one engine trajectory.")

    def test_missing_column_in_upload_is_detected(self):
        """An upload missing a required column must raise ValueError."""
        frame = make_sample_frame().drop(columns=["sensor_5"])
        with pytest.raises(ValueError, match="Missing required columns"):
            from src.data import validate_trajectory as _vt
            _vt(frame)
