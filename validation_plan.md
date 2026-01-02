# Validation Plan for `ndimpute`

This document outlines the comprehensive validation strategy to ensure the `ndimpute` package delivers accurate, statistically valid imputations comparable to industry-standard R packages (`NADA`, `NADA2`, `survival`).

## 1. Objective

To verify that `ndimpute`:
1.  Replicates the results of **NADA/NADA2** (Robust ROS) for left-censored environmental data within acceptable numerical tolerance.
2.  Aligns with **survival/survreg** (Parametric) for right-censored reliability data.
3.  Behaves robustly for **Mixed Censoring** scenarios where standard packages often require manual workflows.

## 2. Reference Standards

| Censoring Type | Python Method (`ndimpute`) | Reference Standard (R) | Comparison Metric |
| :--- | :--- | :--- | :--- |
| **Left** | `method='ros'` | `NADA::ros` | Imputed Values, Slope, Intercept |
| **Left** | `method='substitution'` | Manual Calculation | Exact Match |
| **Right** | `method='parametric'` | `survival::survreg` + `predict(type='response')` | Conditional Mean $E[T\|T>C]$ |
| **Right** | `method='ros'` | `NADA::ros` (on inverted data) | Imputed Values |
| **Mixed** | `method='parametric'` | `survival::survreg` (Interval/Mixed support) | Fitted Parameters, Mean |

## 3. Validation Datasets

We define three tiers of datasets for validation:

### Tier 1: Textbook Cases (The "Golden" Standard)
Datasets from *Nondetects and Data Analysis* (Helsel, 2005).
*   **Purpose:** Verify correctness against published, hand-calculated results.
*   **Example:** TCE concentrations with multiple detection limits.

### Tier 2: Synthetic Distributions (Monte Carlo)
Controlled datasets generated from known distributions (LogNormal, Weibull).
*   **Purpose:** Verify that imputation recovers the true distribution parameters better than substitution.
*   **Scenarios:**
    *   Lognormal ($n=50$, censoring=30%, 50%, 80%).
    *   Weibull ($n=100$, right-censoring=20%).
    *   Mixed ($n=100$, 10% left, 10% right).

### Tier 3: Edge Cases
*   **High Censoring:** >90% censored data.
*   **Low N:** $N < 5$ observations.
*   **Overlapping Intervals:** Multiple detection limits ($<1, <5$).

## 4. Validation Workflow

Since `ndimpute` is a Python package and the standards are in R, the validation is a 2-step process:

1.  **Generate Benchmarks (R):**
    *   Run `validation/generate_r_benchmarks.R`.
    *   This script generates synthetic data.
    *   Runs `NADA` and `survival` functions.
    *   Exports `truth_{scenario}.csv` containing inputs and R-imputed values.

2.  **Verify Parity (Python):**
    *   Run `validation/validate_against_r.py`.
    *   Loads `truth_{scenario}.csv`.
    *   Runs `ndimpute` on the inputs.
    *   Calculates Mean Absolute Error (MAE) between `ndimpute` and `R`.
    *   Passes if MAE < Tolerance (typically $10^{-5}$ for deterministic algos, loose for heuristics).

## 5. Acceptance Criteria

*   **ROS (Left):** `ndimpute` values must be within 1% of `NADA::ros` modeled values (allowing for minor differences in plotting position implementations).
*   **Parametric (Right):** `ndimpute` values must be within 0.1% of `survival` predicted conditional means.
*   **Substitution:** Must match exactly.

## 6. Implementation

The validation scripts are located in the `validation/` directory.
