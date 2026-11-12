# Predictive Maintenance for Turbofan Engines Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> ⚠️ **This is an educational research demonstration using simulated NASA C-MAPSS data. It is NOT validated for real-world aircraft maintenance or safety decisions.**

Predict the **Remaining Useful Life (RUL)** of simulated turbofan jet engines from sensor time-series data using the NASA C-MAPSS FD001 benchmark dataset. Five regression models are compared using **engine-grouped cross-validation** to prevent data leakage, and the best model is evaluated on the official held-out test set.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Technology Stack](#technology-stack)
5. [Project Architecture](#project-architecture)
6. [Installation (Windows PowerShell)](#installation-windows-powershell)
7. [Running the Project](#running-the-project)
8. [Model Descriptions](#model-descriptions)
9. [Evaluation Methodology](#evaluation-methodology)
10. [Experimental Results](#experimental-results)
11. [Project Limitations](#project-limitations)
12. [Future Improvements](#future-improvements)
13. [Publishing to GitHub](#publishing-to-github)

---

## Problem Statement

Unexpected engine failure causes costly unplanned downtime and poses safety risks. Predictive maintenance uses sensor data to estimate how many operating cycles remain before failure — the **Remaining Useful Life (RUL)**. Accurate RUL prediction allows maintenance teams to intervene before failure while avoiding unnecessary early replacements.

---

## Objectives

1. Train a linear regression baseline and four stronger regressors on NASA C-MAPSS FD001.
2. Tune models using **3-fold GroupKFold** cross-validation, keeping each engine entirely within one fold (no row-level leakage).
3. Select the best model on validation RMSE alone — never the test set.
4. Evaluate the selected model on the **official FD001 test set** (final observed cycle per engine, matched to official labels).
5. Report RMSE, MAE, R², NASA asymmetric score, and engine-resampled 95% bootstrap confidence intervals.
6. Run **ablation studies** comparing rolling features vs. raw features, and capped vs. uncapped training labels.
7. Explain model predictions via SHAP values and permutation feature importance.
8. Serve a Streamlit interactive dashboard for single-engine RUL prediction.

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set — FD001**

| Property | Value |
|---|---|
| Source | NASA Prognostics Center of Excellence |
| Download | [phm-datasets.s3.amazonaws.com](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) |
| Training engines | 100 (run-to-failure trajectories) |
| Test engines | 100 (truncated trajectories + official RUL labels) |
| Operating conditions | 1 (sea level) |
| Fault modes | 1 (HPC degradation) |
| Features per row | 26 (engine_id, cycle, 3 settings, 21 sensors) |

**Citation:**  
Saxena, A. & Goebel, K. (2008). *C-MAPSS Turbofan Engine Degradation Simulation Data Set.* NASA Prognostics Center of Excellence Data Repository. https://www.nasa.gov/pcoe

> ⚠️ **Data redistribution:** Check NASA's repository terms before redistributing raw data files. Raw data is excluded from this Git repository via `.gitignore`.

---

## Technology Stack

| Component | Library / Tool |
|---|---|
| Data processing | Pandas 2.2.3, NumPy 2.4.6 |
| Visualisation | Matplotlib 3.x |
| Machine Learning | scikit-learn 1.9.0 |
| Model persistence | Joblib |
| Explainability | SHAP, permutation importance |
| Dashboard | Streamlit |
| Testing | Pytest |
| Version control | Git |

---

## Project Architecture

```text
turbofan-rul-prediction/
│
├── app/
│   ├── __init__.py
│   └── app.py                   ← Streamlit dashboard (8 sections)
│
├── data/
│   ├── raw/                     ← Downloaded data (git-ignored)
│   │   └── CMAPSSData/          ← .txt files extracted here
│   └── processed/               ← Cleaned CSVs (git-ignored)
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py                  ← Download, extract, load, validate
│   ├── features.py              ← Rolling feature engineering
│   ├── train.py                 ← Model training, tuning, evaluation
│   ├── evaluate.py              ← Metrics: RMSE, MAE, R², NASA score, CI
│   └── explain.py               ← Permutation importance + SHAP
│
├── models/                      ← Saved model + metadata (git-ignored)
│   ├── best_model.joblib
│   └── metadata.json
│
├── reports/
│   ├── figures/                 ← Generated plots (git-ignored)
│   └── metrics/                 ← CSV/JSON reports (git-ignored)
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py             ← 11 tests: loading, RUL, validation
│   ├── test_features.py         ← 14 tests: rolling features, leakage
│   ├── test_metrics.py          ← 16 tests: NASA score, RMSE, CI
│   └── test_model_and_app.py    ← 12 tests: pipelines, engine separation
│
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation (Windows PowerShell)

```powershell
# 1. Clone or navigate to the project directory
cd "turbofan-rul-prediction"

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

**Python 3.10 or newer is required.** Install from [python.org](https://www.python.org/) if needed.

---

## Running the Project

Execute commands in this order from the project root with the virtual environment active:

```powershell
# Step 1 — Download and extract NASA C-MAPSS FD001 (~35 MB)
py -m src.data --download

# Step 2 — Train all 5 models, evaluate on official test set, run ablations
#           (5 iterations per model; use --n-iter 20 for a broader search)
py -m src.train --n-iter 8

# Step 3 — Generate SHAP and permutation importance plots
py -m src.explain

# Step 4 — Run the full test suite (should report 51 passed, 2 skipped)
py -m pytest tests/ -v

# Step 5 — Launch the Streamlit dashboard
streamlit run app/app.py

# Optional: Run EDA notebook (requires jupyter)
py -m pip install jupyter
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## Model Descriptions

| Model | Type | Notes |
|---|---|---|
| **Linear Regression** | Baseline | No hyperparameter tuning; establishes minimum benchmark |
| **Random Forest** | Ensemble (bagging) | Tuned: n_estimators, max_depth, min_samples_leaf, max_features |
| **Extra Trees** | Ensemble (extreme randomisation) | Same search space as RF; faster but higher variance |
| **HistGradientBoosting** | Ensemble (boosting) | Tuned: learning_rate, max_leaf_nodes, l2_regularization, max_iter |
| **SVR (RBF kernel)** | Kernel method | Tuned: C, gamma, epsilon; requires feature scaling |

All pipelines include **median imputation** (for early-cycle rolling std NaNs) and **StandardScaler** where required. Hyperparameters are searched with `RandomizedSearchCV` using the same `GroupKFold` as model comparison.

---

## Evaluation Methodology

### Training / Validation Split

- **3-fold GroupKFold** grouped by `engine_id` — each engine appears entirely in one fold.
- Prevents row-level data leakage between training and validation.
- Model selection uses **mean CV RMSE** only (test set never touched during selection).

### Test Evaluation

- One prediction per test engine: the **final observed cycle** of each truncated trajectory.
- Predictions matched to official RUL labels in ascending engine-ID order.
- Negative predictions clipped to 0 (RUL cannot be negative).

### Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | √(mean(error²)) | Penalises large errors; in cycle units |
| **MAE** | mean(\|error\|) | Average absolute error in cycles |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained |
| **NASA Score** | Σ exp(d/10)−1 if d≥0, else exp(−d/13)−1 | Asymmetric; overestimating RUL penalised ~3× more |

### Bootstrap Confidence Intervals

95% percentile intervals by resampling test engines (1000 iterations, seed=42).

---

## Experimental Results

> Results measured on the official NASA FD001 test set after training on `train_FD001.txt`.  
> Seed: 42 | 3-fold GroupKFold | n_iter=8 per model | Python 3.11.1 | scikit-learn 1.9.0

### Model Comparison (3-fold GroupKFold CV RMSE)

| Model | CV RMSE (mean) | CV RMSE (std) |
|---|---:|---:|
| **Extra Trees ✓** | **16.30** | **1.85** |
| Random Forest | 16.53 | 2.03 |
| HistGradientBoosting | 16.61 | 2.07 |
| SVR (RBF) | 17.43 | 1.69 |
| Linear Regression | 19.96 | 1.07 |

✓ Selected model: **Extra Trees Regressor**

Best hyperparameters: `n_estimators=100, max_depth=10, min_samples_leaf=2, max_features=0.5`

### Official Test Set Results (Extra Trees)

| Metric | Value |
|---|---:|
| **RMSE** | **18.19 cycles** |
| **MAE** | 12.99 cycles |
| **R²** | 0.808 |
| **NASA Score** | 619.93 |
| RMSE Bootstrap 95% CI | [15.33, 21.00] |
| MAE Bootstrap 95% CI | [10.49, 15.60] |

### Ablation Study (Extra Trees, same hyperparameters)

| Experiment | Setting | CV RMSE (mean) | CV RMSE (std) |
|---|---|---:|---:|
| **A** | Rolling features + capped target (125) | **16.30** | **1.85** |
| **B** | Raw features + capped target (125) | 16.62 | 1.41 |
| **C** | Rolling features + uncapped target | 37.08 | 1.20 |

**Findings:**
- **Rolling features** (5-cycle mean/std) improve performance over raw sensors (A < B).
- **RUL capping** at 125 cycles is critical — removing it (experiment C) increases CV RMSE by 127%, because early-life cycles with very high RUL add noise that the model cannot learn from.
- The small gap between A and B suggests rolling statistics provide modest but consistent benefit.

---

## Project Limitations

1. **Simulated data** — C-MAPSS is a physics simulation, not real sensor readings. Results may not transfer to real hardware.
2. **Single operating condition / fault mode** — FD001 has one regime; real fleets have dozens.
3. **No calibrated uncertainty** — Point predictions only; real maintenance systems need confidence bounds.
4. **No temporal drift** — The model does not account for fleet aging, maintenance events, or component swaps.
5. **Fixed feature window** — The 5-cycle rolling window is not tuned per engine type.
6. **Not deployment-ready** — No safety review, regulatory validation, or integration with maintenance systems.

---

## Future Improvements

1. **Cross-regime generalisation** — Train on FD001, test on FD002/FD004 (multi-condition). Measure transfer gap.
2. **Calibrated uncertainty** — Quantile regression or conformal prediction intervals with empirical coverage.
3. **Deep learning comparison** — 1D-CNN or LSTM for direct sequence modelling (requires care with leakage).
4. **Hyperparameter efficiency** — Bayesian optimisation (Optuna) instead of random search.
5. **Constant sensor pruning** — Automatically drop zero-variance sensors before training.
6. **Online adaptation** — Update the model as new engine observations arrive.

---

## Publishing to GitHub

After reviewing that raw data and model binaries are excluded from Git:

```powershell
# From the project root with the venv active:
git remote add origin https://github.com/Imandi-Likhitha-Sree8/turbofan-rul-prediction.git
git branch -M main
git push -u origin main
```
