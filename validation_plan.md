# Validation Plan for `ndimpute`

This document outlines the comprehensive validation strategy to ensure the `ndimpute` package delivers accurate, statistically valid imputations comparable to industry-standard R packages (`NADA`, `NADA2`, `survival`).

## 1. Objective

To verify that `ndimpute`:
1.  Replicates the results of **NADA/NADA2** (Robust ROS) for left-censored environmental data within acceptable numerical tolerance.
2.  Aligns with **survival/survreg** (Parametric) for right-censored reliability data.
3.  Behaves robustly for **Mixed Censoring** scenarios where standard packages often require manual workflows.
4.  Correctly handles edge cases, data quality issues, and varying censoring intensities.

## 2. Directory Structure

To ensure organization, **each validation scenario must be self-contained in its own subfolder within `validation/`**.

**Required Structure:**
```text
validation/
├── 01_textbook_tce/           # Tier 1: Textbook Case
│   ├── validation_report.md   # From template
│   ├── generate_data.R        # R script for truth
│   ├── validate.py            # Python script
│   └── plots/                 # Artifacts
├── 02_synthetic_lognormal/    # Tier 2: Synthetic
│   └── ...
├── 03_edge_high_censoring/    # Tier 3: Edge Case
│   └── ...
├── generate_r_benchmarks.R    # (Legacy/Shared script)
└── validate_against_r.py      # (Legacy/Shared script)
```

## 3. Reference Standards

| Censoring Type | Python Method (`ndimpute`) | Reference Standard (R) | Comparison Metric |
| :--- | :--- | :--- | :--- |
| **Left** | `method='ros'` | `NADA::ros` | Imputed Values, Slope, Intercept |
| **Left** | `method='substitution'` | Manual Calculation | Exact Match |
| **Right** | `method='parametric'` | `survival::survreg` + `predict(type='response')` | Conditional Mean $E[T\|T>C]$ |
| **Right** | `method='ros'` | `NADA::ros` (on inverted data) | Imputed Values |
| **Mixed** | `method='parametric'` | `survival::survreg` (Interval/Mixed support) | Fitted Parameters, Mean |

## 4. Validation Scenarios

We define four tiers of datasets for validation:

### Tier 1: Textbook Cases (The "Golden" Standard)
Datasets from *Nondetects and Data Analysis* (Helsel, 2005) or USEPA guidance.
*   **Case 1.1: TCE Concentrations.**
    *   **Features:** Single detection limit. Small N (~20).
    *   **Goal:** Exact replication of Helsel's ROS example.
*   **Case 1.2: Sulfate Data (Multiple Limits).**
    *   **Features:** Multiple detection limits ($<1, <2, <5$).
    *   **Goal:** Verify plotting position logic handles interspersing correctly (or documents the simplified approximation difference).

### Tier 2: Synthetic Distributions (Monte Carlo)
Controlled datasets generated from known distributions (LogNormal, Weibull) to test statistical recovery.
*   **Case 2.1: Lognormal Recovery (Left).**
    *   **Setup:** $N=50$, $\mu=2, \sigma=1$. Censoring at 30%, 50%, 80%.
    *   **Goal:** Imputed mean should be closer to True Mean than simple substitution (LOD/2).
*   **Case 2.2: Weibull Reliability (Right).**
    *   **Setup:** $N=100$, Shape=2, Scale=50. Random right censoring times.
    *   **Goal:** Verify Parametric conditional mean aligns with R `survival` predictions.
*   **Case 2.3: Mixed Censoring Stress.**
    *   **Setup:** $N=200$. 10% Left, 10% Right.
    *   **Goal:** Verify Parametric Mixed fit converges and produces logical imputations ($<L$ and $>R$).

### Tier 3: Edge Cases & Stability
Tests designed to break the implementation.
*   **Case 3.1: High Censoring (>90%).**
    *   **Setup:** $N=100$, 95 censored, 5 observed.
    *   **Expectation:** ROS should warn or run on minimal points. Parametric might fail convergence (should raise informative error).
*   **Case 3.2: Minimal N (Small Sample).**
    *   **Setup:** $N=3$ (1 censored, 2 observed).
    *   **Expectation:** ROS requires min 2 uncensored. Verify boundary condition.
*   **Case 3.3: 100% Censoring.**
    *   **Setup:** All values censored.
    *   **Expectation:** `ValueError` (Cannot fit model).
*   **Case 3.4: Invalid Data.**
    *   **Setup:** Negative values for Lognormal/Weibull fit.
    *   **Expectation:** `ValueError` with clear message.

### Tier 4: Comparative Performance (Methodology Check)
*   **Case 4.1: ROS vs Substitution.**
    *   **Setup:** Synthetic Lognormal.
    *   **Goal:** Demonstrate statistically that `method='ros'` produces significantly lower bias in Mean and Variance estimation than `method='substitution', strategy='half'`.
*   **Case 4.2: Parametric vs Reverse ROS (Right).**
    *   **Setup:** Weibull data.
    *   **Goal:** Compare the two right-censoring strategies. Parametric is expected to be superior if distribution assumption holds; ROS should be robust if distribution is misspecified.

## 5. Validation Workflow

Since `ndimpute` is a Python package and the standards are in R, the validation is a 2-step process:

1.  **Generate Benchmarks (R):**
    *   Run the specific `generate_data.R` inside the scenario folder.
    *   Exports `truth.csv` containing inputs and R-imputed values.

2.  **Verify Parity (Python):**
    *   Run the specific `validate.py` inside the scenario folder.
    *   Calculates Mean Absolute Error (MAE) between `ndimpute` and `R`.
    *   Generates a standardized report using `validation_template.md`.

## 6. Acceptance Criteria

*   **ROS (Left):** `ndimpute` values must be within 5% of `NADA::ros` modeled values (relaxed from 1% due to simplified plotting position logic in `_ros_left.py`).
*   **Parametric (Right):** `ndimpute` values must be within 0.1% of `survival` predicted conditional means.
*   **Substitution:** Must match exactly.
*   **Stability:** No unhandled exceptions for valid inputs. Informative errors for invalid inputs.
