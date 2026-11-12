"""Train, select, evaluate, and persist FD001 RUL regressors.

Workflow
--------
1. Load FD001 training trajectories; compute rolling features.
2. Tune five regressors with 3-fold GroupKFold (grouped by engine_id).
3. Select the model with the lowest mean CV RMSE.
4. Evaluate exclusively on official test endpoints — the final observed cycle
   of each test engine matched to its official RUL label.
5. Run three ablation experiments (rolling+capped, raw+capped, rolling+uncapped).
6. Save model, metadata, diagnostics, and ablation results.

Usage
-----
    python -m src.train               # 5 randomised iterations per model
    python -m src.train --n-iter 20   # broader search
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.data import ROOT, load_test, load_train
from src.evaluate import bootstrap_intervals, regression_metrics
from src.features import build_features, model_columns

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

SEED = 42
RUL_CAP = 125       # Common benchmark choice; tested in ablation.
N_SPLITS = 3        # GroupKFold folds — keep all three engine groups intact.

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "metrics"
FIGURE_DIR = ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(name: str) -> Pipeline:
    """Return a named scikit-learn Pipeline with imputation and optional scaling.

    Linear regression and SVR require feature scaling; tree-based models do not.
    All pipelines include median imputation to handle early-cycle rolling-std NaNs.

    Parameters
    ----------
    name:
        One of ``'linear'``, ``'random_forest'``, ``'extra_trees'``,
        ``'hist_gradient_boosting'``, ``'svr'``.
    """
    imputer = ("imputer", SimpleImputer(strategy="median"))
    scaler = ("scaler", StandardScaler())
    models: dict[str, tuple] = {
        "linear": (
            imputer,
            scaler,
            ("model", LinearRegression()),
        ),
        "random_forest": (
            imputer,
            ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        ),
        "extra_trees": (
            imputer,
            ("model", ExtraTreesRegressor(random_state=SEED, n_jobs=-1)),
        ),
        "hist_gradient_boosting": (
            # HistGradientBoosting natively handles NaN values; we still use the
            # imputer for consistency so the pipeline API stays uniform.
            imputer,
            ("model", HistGradientBoostingRegressor(random_state=SEED)),
        ),
        "svr": (
            imputer,
            scaler,
            ("model", SVR(kernel="rbf")),
        ),
    }
    if name not in models:
        raise ValueError(
            f"Unknown model name '{name}'.  Choose from: {list(models)}."
        )
    return Pipeline(list(models[name]))


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, dict] = {
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.5, 0.7, 1.0],
    },
    "extra_trees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.5, 0.7, 1.0],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.15],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__max_iter": [100, 200, 300],
    },
    "svr": {
        "model__C": [1, 5, 10, 50],
        "model__gamma": ["scale", 0.005, 0.01, 0.05],
        "model__epsilon": [0.1, 0.5, 1.0],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_endpoints(
    test_features: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract the final observed cycle row for each test engine.

    Test RUL labels (one per engine) correspond to the last row seen for each
    engine — not a randomly drawn row.  Engine order follows ascending engine_id
    to match the official RUL file ordering.

    Returns
    -------
    X_test: feature matrix for each engine's last cycle.
    engine_ids: 1-D array of engine IDs in the same order.
    """
    idx = test_features.groupby("engine_id", sort=True)["cycle"].idxmax()
    subset = test_features.loc[idx].sort_values("engine_id")
    return subset[columns].reset_index(drop=True), subset["engine_id"].to_numpy()


def _plot_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path
) -> None:
    """Save a two-panel figure: true vs. predicted, and residuals vs. predicted."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    maximum = float(max(y_true.max(), y_pred.max()))

    axes[0].scatter(y_true, y_pred, alpha=0.7, edgecolors="none", s=35)
    axes[0].plot([0, maximum], [0, maximum], "k--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("True RUL (cycles)")
    axes[0].set_ylabel("Predicted RUL (cycles)")
    axes[0].set_title("FD001 test endpoints: true vs. predicted")
    axes[0].legend()

    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.7, edgecolors="none", s=35)
    axes[1].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Predicted RUL (cycles)")
    axes[1].set_ylabel("Residual (pred − true)")
    axes[1].set_title("Residuals vs. predicted RUL")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(n_iter: int = 5, bootstrap_repeats: int = 1000) -> dict:
    """Execute the full training, selection, and evaluation pipeline.

    Parameters
    ----------
    n_iter:
        Number of parameter settings tried per model in ``RandomizedSearchCV``.
        Increase for a wider search at the cost of runtime.
    bootstrap_repeats:
        Number of engine-resampling iterations for bootstrap CI computation.

    Returns
    -------
    Dictionary with keys ``'leaderboard'`` (DataFrame) and ``'metrics'`` (dict).
    """
    for directory in (MODEL_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # --- Data loading & feature engineering ---
    train_raw = load_train("FD001")
    test_raw, y_test = load_test("FD001")
    train_f = build_features(train_raw)
    test_f = build_features(test_raw)

    columns = model_columns(train_f)
    groups = train_f["engine_id"].to_numpy()
    X = train_f[columns]
    y_raw = train_f["RUL"].to_numpy(dtype=float)
    y_capped = np.minimum(y_raw, RUL_CAP)

    # Official test endpoints: one row per engine, in engine-ID order.
    X_test, engine_ids = _extract_endpoints(test_f, columns)
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Test endpoint count ({len(X_test)}) does not match official "
            f"RUL label count ({len(y_test)})."
        )

    cv = GroupKFold(n_splits=N_SPLITS)
    all_results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}

    # --- Linear regression baseline (no hyperparameter tuning) ---
    linear = make_pipeline("linear")
    scores = -cross_val_score(
        linear, X, y_capped,
        groups=groups, cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    all_results.append({
        "model": "linear",
        "cv_rmse_mean": float(scores.mean()),
        "cv_rmse_std": float(scores.std()),
        "best_params": {},
    })
    fitted_models["linear"] = linear.fit(X, y_capped)
    print(f"  linear  — CV RMSE {scores.mean():.3f} ± {scores.std():.3f}")

    # --- Tuned models ---
    for name, space in SEARCH_SPACE.items():
        search = RandomizedSearchCV(
            estimator=make_pipeline(name),
            param_distributions=space,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X, y_capped, groups=groups)
        best_idx = search.best_index_
        cv_rmse_mean = float(-search.best_score_)
        cv_rmse_std = float(search.cv_results_["std_test_score"][best_idx])
        all_results.append({
            "model": name,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "best_params": search.best_params_,
        })
        fitted_models[name] = search.best_estimator_
        print(
            f"  {name:<28} — CV RMSE {cv_rmse_mean:.3f} ± {cv_rmse_std:.3f} "
            f"| params: {search.best_params_}"
        )

    # --- Leaderboard ---
    all_results.sort(key=lambda row: row["cv_rmse_mean"])
    leaderboard = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "best_params"} for row in all_results]
    )
    leaderboard.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nLeaderboard (CV RMSE, ascending):")
    print(leaderboard.to_string(index=False))

    # --- Official test evaluation (selected model only) ---
    selected = all_results[0]
    model = fitted_models[selected["model"]]

    # Clip at zero; never silently cap test-set predictions at RUL_CAP.
    predictions = np.maximum(model.predict(X_test), 0.0)
    metrics = regression_metrics(y_test, predictions)
    ci = bootstrap_intervals(y_test, predictions, repeats=bootstrap_repeats)
    metrics.update({
        "selected_model": selected["model"],
        "bootstrap_95pct": ci,
    })

    pd.DataFrame({
        "engine_id": engine_ids,
        "true_rul": y_test,
        "predicted_rul": predictions,
        "error_pred_minus_true": predictions - y_test,
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    _plot_diagnostics(y_test, predictions, FIGURE_DIR / "test_diagnostics.png")

    # --- Ablation studies ---
    # Raw features = no rolling columns.
    raw_cols = [c for c in columns if "roll" not in c]
    ablations = [
        ("A_rolling_features_capped_target", X, y_capped),
        ("B_raw_features_capped_target", train_f[raw_cols], y_capped),
        ("C_rolling_features_uncapped_target", X, y_raw),
    ]
    ablation_rows = []
    for label, X_variant, y_variant in ablations:
        estimator = make_pipeline(selected["model"]).set_params(
            **selected["best_params"]
        )
        values = -cross_val_score(
            estimator, X_variant, y_variant,
            groups=groups, cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        ablation_rows.append({
            "setting": label,
            "cv_rmse_mean": float(values.mean()),
            "cv_rmse_std": float(values.std()),
        })
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(REPORT_DIR / "ablation.csv", index=False)
    print("\nAblation results:")
    print(abl_df.to_string(index=False))

    # --- Persist model and metadata ---
    metadata = {
        **metrics,
        "subset": "FD001",
        "rul_cap": RUL_CAP,
        "seed": SEED,
        "n_iter": n_iter,
        "cv": f"{N_SPLITS}-fold GroupKFold grouped by engine_id",
        "best_params": selected["best_params"],
        "feature_columns": columns,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    joblib.dump(model, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTest RMSE : {metrics['RMSE']:.4f}")
    print(f"Test MAE  : {metrics['MAE']:.4f}")
    print(f"Test R²   : {metrics['R2']:.4f}")
    print(f"NASA score: {metrics['NASA_score']:.2f}")
    print(f"Bootstrap 95% CI RMSE: {ci['RMSE']}")

    return {"leaderboard": leaderboard, "metrics": metrics}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate turbofan RUL regressors."
    )
    parser.add_argument(
        "--n-iter", type=int, default=5,
        help="Randomised search iterations per model (default: 5).",
    )
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=1000,
        help="Bootstrap resampling iterations for CIs (default: 1000).",
    )
    args = parser.parse_args()
    print("Starting training pipeline …")
    train(n_iter=args.n_iter, bootstrap_repeats=args.bootstrap_repeats)


if __name__ == "__main__":
    main()
