"""Post-hoc feature importance analysis for the saved turbofan RUL model.

Two complementary methods are used:

1. **Permutation importance** (model-agnostic, based on test-set RMSE increase).
2. **SHAP** (Shapley Additive exPlanations) on a small representative sample.

Both methods describe *association* within this dataset and model, not causal
relationships between sensor readings and engine failure.

Usage
-----
    python -m src.explain
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data import ROOT, load_test, load_train
from src.features import build_features, model_columns

MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"


def _load_artifacts() -> tuple:
    """Load the saved model and metadata; raise clearly if absent."""
    model_path = MODEL_DIR / "best_model.joblib"
    meta_path = MODEL_DIR / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found.  Run `python -m src.train` first."
        )
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return model, metadata


def _test_endpoints(
    test_features: pd.DataFrame, columns: list[str], y_test: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X_test, y_test) for each engine's final observed cycle."""
    idx = test_features.groupby("engine_id", sort=True)["cycle"].idxmax()
    subset = test_features.loc[idx].sort_values("engine_id")
    return subset[columns].reset_index(drop=True), y_test


def run_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    columns: list[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute and save permutation feature importance on the test endpoints.

    Returns a DataFrame sorted descending by mean importance.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "feature": columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(METRICS_DIR / "permutation_importance.csv", index=False)

    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="steelblue",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xlabel("Mean RMSE increase after permutation")
    ax.set_title(
        "Permutation feature importance\n"
        "(reflects model associations, not causal effects)"
    )
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)
    print("Permutation importance saved.")
    return importance


def run_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
    n_background: int = 50,
    n_sample: int = 30,
) -> None:
    """Compute SHAP values and save a beeswarm summary plot.

    Uses the permutation algorithm for compatibility with any scikit-learn
    pipeline.  Shapes are limited to keep computation tractable on a laptop.
    """
    try:
        import shap  # optional dependency
    except ImportError:
        print(
            "SHAP is not installed.  "
            "Install it with `pip install shap` and re-run."
        )
        return

    background = shap.sample(X_train, min(n_background, len(X_train)), random_state=42)
    sample = X_test.iloc[: min(n_sample, len(X_test))]

    explainer = shap.Explainer(
        model.predict,
        shap.maskers.Independent(background),
        algorithm="permutation",
    )
    explanation = explainer(sample, max_evals=2 * len(columns) + 1)

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP beeswarm (associations, not causal effects)", pad=12)
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("SHAP beeswarm plot saved.")

    # Bar chart of mean |SHAP|.
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Mean |SHAP| — top 15 features", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_bar.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("SHAP bar chart saved.")


def main() -> None:
    model, metadata = _load_artifacts()
    columns: list[str] = metadata["feature_columns"]

    train_raw = load_train("FD001")
    test_raw, y_test = load_test("FD001")
    train_f = build_features(train_raw)
    test_f = build_features(test_raw)

    X_train = train_f[columns]
    X_test, y_test = _test_endpoints(test_f, columns, y_test)

    print("Running permutation importance …")
    importance = run_permutation_importance(model, X_test, y_test, columns)
    print("\nTop 10 features by permutation importance:")
    print(importance.head(10).to_string(index=False))

    print("\nRunning SHAP analysis …")
    run_shap(model, X_train, X_test, columns)


if __name__ == "__main__":
    main()
