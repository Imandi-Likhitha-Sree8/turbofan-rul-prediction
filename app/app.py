"""Turbofan Engine Predictive Maintenance System — Streamlit Dashboard.

Sections
--------
1. Project Overview
2. Dataset Information
3. Exploratory Data Analysis
4. Model Comparison
5. Model Performance
6. RUL Prediction
7. Feature Explainability
8. Project Limitations

Usage
-----
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable when running via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import BASE_COLUMNS, validate_trajectory  # noqa: E402
from src.evaluate import nasa_score  # noqa: E402
from src.features import build_features  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURE_DIR = ROOT / "reports" / "figures"

DISCLAIMER = (
    "⚠️  **This is an educational research demonstration using simulated "
    "NASA C-MAPSS data. It is not validated for real-world aircraft "
    "maintenance or safety decisions.**"
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Turbofan Engine Predictive Maintenance System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/300px-NASA_logo.svg.png",
    width=100,
)
st.sidebar.title("Navigation")
SECTIONS = [
    "🏠 Project Overview",
    "📊 Dataset Information",
    "🔍 Exploratory Data Analysis",
    "🏆 Model Comparison",
    "📈 Model Performance",
    "🎯 RUL Prediction",
    "🧩 Feature Explainability",
    "⚠️ Project Limitations",
]
section = st.sidebar.radio("Go to", SECTIONS)
st.sidebar.markdown("---")
st.sidebar.caption("NASA C-MAPSS FD001 · scikit-learn · Streamlit")

# ---------------------------------------------------------------------------
# Header (always visible)
# ---------------------------------------------------------------------------
st.title("✈️ Turbofan Engine Predictive Maintenance System")
st.warning(DISCLAIMER)
st.markdown("---")


# ---------------------------------------------------------------------------
# Helper: load model artifacts (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_model():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def _model_available() -> bool:
    return MODEL_PATH.exists() and METADATA_PATH.exists()


def _figure(name: str) -> Path:
    return FIGURE_DIR / name


def _metric_csv(name: str) -> Path:
    return METRICS_DIR / name


# ---------------------------------------------------------------------------
# Section routing
# ---------------------------------------------------------------------------

if section == "🏠 Project Overview":
    st.header("Project Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### What is this project?
This project builds a **Remaining Useful Life (RUL) prediction** system for
simulated turbofan jet engines using the NASA C-MAPSS benchmark dataset (FD001).

### What is RUL?
**Remaining Useful Life** is the number of operating cycles a machine is
estimated to operate before it reaches a failure threshold. Accurate RUL
prediction enables:
- Proactive maintenance scheduling.
- Reduction of unplanned downtime.
- Lower operational costs.

### How does it work?
1. **Sensor data** from 100 simulated engines is collected until failure.
2. **Rolling features** (5-cycle mean and std per sensor) capture degradation trends.
3. **Five ML models** are trained and compared using engine-grouped cross-validation.
4. The **best model** is evaluated on the official NASA test set.
5. **SHAP values** explain which sensors drive predictions.

### Technology Stack
""")
        st.markdown("""
| Component | Technology |
|---|---|
| Data processing | Pandas, NumPy |
| ML models | scikit-learn |
| Explainability | SHAP, permutation importance |
| Dashboard | Streamlit |
| Model persistence | Joblib |
""")
    with col2:
        st.info("**Dataset:** NASA C-MAPSS FD001")
        st.info("**Task:** Regression — predict cycles to failure")
        st.info("**Engines:** 100 train, 100 test")
        st.info("**Features:** 21 sensors + 3 settings")
        st.info("**Models:** 5 compared")

elif section == "📊 Dataset Information":
    st.header("Dataset Information")
    st.markdown("""
### NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set

**Citation:**  
Saxena, A. & Goebel, K. (2008). *C-MAPSS Turbofan Engine Degradation Simulation
Data Set.* NASA Prognostics Center of Excellence Data Repository.  
[https://www.nasa.gov/pcoe](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

**FD001 Characteristics:**
""")
    info_df = pd.DataFrame({
        "Property": [
            "Training engines", "Test engines", "Operating conditions",
            "Fault modes", "Features per row", "Sensor columns",
            "Operating settings", "RUL cap (training)", "Target",
        ],
        "Value": [
            "100", "100", "1 (sea level)",
            "1 (HPC degradation)", "26",
            "21 (sensor_1 through sensor_21)",
            "3 (setting_1, setting_2, setting_3)",
            "125 cycles", "Remaining cycles to failure",
        ],
    })
    st.table(info_df)

    st.markdown("""
### Column Schema

| Column | Description |
|---|---|
| `engine_id` | Unique engine identifier (1–100) |
| `cycle` | Operating cycle number |
| `setting_1/2/3` | Operational setting values |
| `sensor_1` to `sensor_21` | Physical/process measurements |
| `RUL` *(training only)* | Computed remaining useful life |

### How Training RUL is Computed
```
RUL[engine, cycle] = max_cycle[engine] − current_cycle
```
The final row of each training engine has RUL = 0.  
The training target is **capped at 125** to reduce noise from early-life cycles.

### Official Test Labels
One RUL value per engine (the remaining cycles *after* the last observed cycle).
""")

elif section == "🔍 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.info(
        "EDA figures are generated during dataset download. "
        "Run `python -m src.data --download` and then execute the EDA notebook "
        "(`notebooks/exploratory_analysis.ipynb`) to populate these plots."
    )
    plots = {
        "Sensor Distributions": "sensor_distributions.png",
        "Sensor Correlation Heatmap": "sensor_correlation.png",
        "Engine Degradation Trajectories": "degradation_trajectories.png",
        "RUL Distribution": "rul_distribution.png",
    }
    cols = st.columns(2)
    for i, (title, fname) in enumerate(plots.items()):
        fig_path = _figure(fname)
        with cols[i % 2]:
            st.subheader(title)
            if fig_path.exists():
                st.image(str(fig_path))
            else:
                st.info(f"Not yet generated: `{fname}`")

elif section == "🏆 Model Comparison":
    st.header("Model Comparison")
    st.markdown("""
All models were compared using **3-fold GroupKFold cross-validation**,
grouped by `engine_id` to prevent data leakage across engines.
Model selection is based **exclusively** on validation RMSE — the official
test set was never used for selection.
""")
    comp_path = _metric_csv("model_comparison.csv")
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        # Highlight the best model row.
        best_idx = df["cv_rmse_mean"].idxmin()
        st.dataframe(
            df.style.format({"cv_rmse_mean": "{:.4f}", "cv_rmse_std": "{:.4f}"})
            .highlight_min(subset=["cv_rmse_mean"], color="#d4edda"),
            use_container_width=True,
        )
        st.success(
            f"✅ **Selected model:** {df.loc[best_idx, 'model']}  "
            f"(CV RMSE = {df.loc[best_idx, 'cv_rmse_mean']:.4f})"
        )
        abl_path = _metric_csv("ablation.csv")
        if abl_path.exists():
            st.subheader("Ablation Study")
            st.markdown("""
| Experiment | Description |
|---|---|
| **A** | Rolling features + capped RUL (125) — primary setup |
| **B** | Raw features + capped RUL — no rolling statistics |
| **C** | Rolling features + uncapped RUL — no label clipping |
""")
            abl_df = pd.read_csv(abl_path)
            st.dataframe(
                abl_df.style.format(
                    {"cv_rmse_mean": "{:.4f}", "cv_rmse_std": "{:.4f}"}
                ).highlight_min(subset=["cv_rmse_mean"], color="#d4edda"),
                use_container_width=True,
            )
    else:
        st.info("Run `python -m src.train` to generate model comparison results.")

elif section == "📈 Model Performance":
    st.header("Model Performance on Official Test Set")
    metrics_path = METRICS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{metrics.get('RMSE', 'N/A'):.4f}")
        c2.metric("MAE", f"{metrics.get('MAE', 'N/A'):.4f}")
        c3.metric("R²", f"{metrics.get('R2', 'N/A'):.4f}")
        c4.metric("NASA Score", f"{metrics.get('NASA_score', 'N/A'):.2f}")

        if "bootstrap_95pct" in metrics:
            st.markdown("### Bootstrap 95% Confidence Intervals (engine resampling)")
            ci = metrics["bootstrap_95pct"]
            ci_df = pd.DataFrame({
                "Metric": list(ci.keys()),
                "Lower 2.5%": [v[0] for v in ci.values()],
                "Upper 97.5%": [v[1] for v in ci.values()],
            })
            st.dataframe(ci_df.style.format({"Lower 2.5%": "{:.4f}", "Upper 97.5%": "{:.4f}"}))

        fig_path = _figure("test_diagnostics.png")
        if fig_path.exists():
            st.subheader("Diagnostic Plots")
            st.image(str(fig_path), caption="Left: True vs. Predicted RUL | Right: Residuals")

        pred_path = _metric_csv("test_predictions.csv")
        if pred_path.exists():
            st.subheader("Per-Engine Test Predictions")
            pred_df = pd.read_csv(pred_path)
            st.dataframe(pred_df.style.format({
                "true_rul": "{:.1f}",
                "predicted_rul": "{:.1f}",
                "error_pred_minus_true": "{:.1f}",
            }), use_container_width=True, height=300)
    else:
        st.info("Run `python -m src.train` to generate evaluation results.")

elif section == "🎯 RUL Prediction":
    st.header("RUL Prediction")
    st.markdown("""
Upload a **single-engine trajectory CSV** to predict its remaining useful life.

The file must contain the following columns:  
`engine_id`, `cycle`, `setting_1`, `setting_2`, `setting_3`, `sensor_1` … `sensor_21`

You can use a row from the official test set as an example.
""")
    if not _model_available():
        st.error(
            "Model files not found. Run `python -m src.train` first, "
            "then refresh this page."
        )
        st.stop()

    model, metadata = _load_model()
    col_meta = metadata["feature_columns"]

    upload = st.file_uploader(
        "Upload engine trajectory CSV",
        type=["csv"],
        help="Must have exactly one engine_id value and all 26 C-MAPSS columns.",
    )

    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            validate_trajectory(raw)

            n_engines = raw["engine_id"].nunique()
            if n_engines != 1:
                st.error(
                    f"Upload data for **exactly one** engine. "
                    f"Found {n_engines} distinct engine_ids."
                )
                st.stop()

            features = build_features(raw)
            missing_feats = sorted(set(col_meta) - set(features.columns))
            if missing_feats:
                st.error(f"Missing required feature columns: {missing_feats}")
                st.stop()

            # Use the last observed cycle row for prediction.
            last_idx = features["cycle"].idxmax()
            endpoint = features.loc[last_idx, col_meta].to_frame().T
            predicted_rul = max(0.0, float(model.predict(endpoint)[0]))

            st.markdown("### Prediction Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("🔢 Estimated RUL", f"{predicted_rul:.1f} cycles")
            r2.metric("📅 Last observed cycle", int(raw["cycle"].max()))
            r3.metric("🛠 Model used", metadata.get("selected_model", "N/A"))

            # Simple health indicator.
            if predicted_rul <= 30:
                st.error("🚨 **Critical:** Maintenance required very soon.")
            elif predicted_rul <= 75:
                st.warning("⚠️ **Caution:** Monitor closely and plan maintenance.")
            else:
                st.success("✅ **Healthy:** Sufficient life remaining.")

            st.caption(
                f"Model: {metadata.get('selected_model')} | "
                f"Subset: {metadata.get('subset')} | "
                f"Features: {len(col_meta)}"
            )

            # Show sensor trends for the uploaded engine.
            st.markdown("### Sensor Trends (selected sensors)")
            sensors_to_plot = [s for s in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"] if s in raw.columns]
            if sensors_to_plot:
                st.line_chart(raw.set_index("cycle")[sensors_to_plot])

            # Show raw data preview.
            with st.expander("📋 Uploaded data preview"):
                st.dataframe(raw.tail(10))

        except (ValueError, KeyError) as exc:
            st.error(f"Validation error: {exc}")

elif section == "🧩 Feature Explainability":
    st.header("Feature Explainability")
    st.info(
        "Run `python -m src.explain` to generate these plots. "
        "They describe model *associations*, not causal effects."
    )
    st.markdown("""
### Methods Used

| Method | Description |
|---|---|
| **Permutation Importance** | Measures RMSE increase when each feature is randomly shuffled |
| **SHAP (Shapley Values)** | Attributes each prediction to individual features |

> **Important:** These explain what the *model learned from this dataset*. They
> do not establish that any sensor *causes* engine failure. C-MAPSS is simulated
> data, so domain interpretations should be treated with caution.
""")

    perm_fig = _figure("permutation_importance.png")
    shap_fig = _figure("shap_summary.png")
    shap_bar = _figure("shap_bar.png")

    if perm_fig.exists():
        st.subheader("Permutation Feature Importance")
        st.image(str(perm_fig))
        perm_csv = _metric_csv("permutation_importance.csv")
        if perm_csv.exists():
            with st.expander("📋 Full importance table"):
                st.dataframe(pd.read_csv(perm_csv).head(20))
    else:
        st.info("Permutation importance not generated yet.")

    if shap_fig.exists():
        st.subheader("SHAP Beeswarm Plot")
        st.image(str(shap_fig))
    if shap_bar.exists():
        st.subheader("SHAP Mean |Value| Bar Chart")
        st.image(str(shap_bar))

elif section == "⚠️ Project Limitations":
    st.header("Project Limitations")
    st.error(DISCLAIMER)
    st.markdown("""
### Fundamental Limitations

**1. Simulated data**  
C-MAPSS is a physics simulation, not data from real aircraft engines. Sensor noise,
failure modes, and degradation profiles are idealised. Models trained on C-MAPSS
may not transfer to real hardware without extensive revalidation.

**2. Single operating condition and fault mode (FD001)**  
FD001 uses one operating point and one fault type. Real fleets operate under many
conditions and can exhibit multiple failure modes simultaneously.

**3. No uncertainty calibration**  
Point predictions are provided without calibrated uncertainty bounds.
Proper predictive maintenance requires knowing *how confident* the model is.

**4. No temporal or fleet-level effects**  
The model does not account for maintenance history, component swaps, or
fleet-wide degradation patterns.

**5. Engine-grouped CV, but test-set leakage risk in feature design**  
GroupKFold prevents split-row leakage, but rolling statistics computed on the
full training set before cross-validation can still cause optimistic bias if
done incorrectly. Here, features are computed within the pipeline fold.

**6. No deployment validation**  
Any real-world application requires safety review, regulatory approval,
domain-expert validation, and ongoing monitoring. This system provides **none**
of those.

### What This Project Is Good For
- Learning ML pipeline design and evaluation methodology.
- Benchmarking regression models on a standard dataset.
- Practicing SHAP-based explainability.
- Understanding data leakage risks in time-series ML.

### Citation
Saxena, A. & Goebel, K. (2008). C-MAPSS Turbofan Engine Degradation Simulation
Data Set. NASA Prognostics Center of Excellence.
""")
