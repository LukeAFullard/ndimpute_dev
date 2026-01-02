# ndimpute: Imputation for Arbitrarily Censored Data

**ndimpute** is a Python package designed for the robust imputation of censored data. It handles **Left Censoring** (common in environmental sciences, e.g., nondetects), **Right Censoring** (common in reliability and survival analysis), and **Mixed Censoring** (datasets containing both).

The package provides statistical best-practice methods (Regression on Order Statistics, Parametric Conditional Mean) alongside simpler substitution heuristics for comparison and flexibility.

## Features

*   **Left Censoring Imputation:**
    *   **Robust ROS (Regression on Order Statistics):** Imputes values based on a probability plot regression (lognormal distribution), preserving the statistical properties of the dataset (NADA parity).
    *   **Substitution:** Flexible strategies including LOD/2, Zero, LOD, or custom multipliers.
*   **Right Censoring Imputation:**
    *   **Parametric Conditional Mean:** Fits a Weibull distribution to the data (using `scipy.stats.CensoredData`) and imputes censored values with their expected residual life ($E[T | T > C]$).
    *   **Reverse ROS:** Adapts the ROS methodology for right-censored data.
    *   **Substitution:** Supported strategies are **Value** (Censoring Time) and **Custom Multipliers** (e.g., 1.1x). *Note: Zero and Half strategies are not supported for Right Censoring as they would imply values below the censoring limit.*
*   **Mixed Censoring Imputation:**
    *   **Parametric:** Handles simultaneous left and right censoring using generalized likelihood fitting.
    *   **Heuristic ROS:** Applies a sequential ROS approach (impute left, then impute right).
*   **Unified API:** A single function `impute()` handles all logic and returns a detailed DataFrame with imputation status.

## Installation

`ndimpute` requires Python 3.8+ and the following dependencies:
*   `numpy >= 1.24.0`
*   `pandas >= 2.0.0`
*   `scipy >= 1.13.0` (Required for `CensoredData` support)

## Usage

### 1. Left Censoring (Environmental Data)

Ideal for data with nondetects (values below a Limit of Detection, LOD).

```python
import numpy as np
from ndimpute import impute

# Data: 10, <2, <2, 5, 6
values = [10.0, 2.0, 2.0, 5.0, 6.0]
# Status: True if censored (< LOD)
status = [False, True, True, False, False]

# Method 1: Robust ROS (Recommended)
df_ros = impute(values, status, method='ros', censoring_type='left')
print(df_ros)

# Method 2: Substitution (LOD/2)
df_sub = impute(values, status, method='substitution', censoring_type='left', strategy='half')
```

### 2. Right Censoring (Reliability/Survival)

Ideal for time-to-event data where some events have not occurred by the end of the study.

```python
# Data: Observed failures at 100, 150; Censored at 200 (survived past 200)
values = [100.0, 150.0, 200.0]
# Status: True if censored (Event did NOT happen)
status = [False, False, True]

# Method 1: Parametric Conditional Mean (Weibull)
df_param = impute(values, status, method='parametric', censoring_type='right')

# Method 2: Reverse ROS
df_ros = impute(values, status, method='ros', censoring_type='right')

# Method 3: Substitution (Multiplier)
df_sub = impute(values, status, method='substitution', censoring_type='right',
                strategy='multiple', multiplier=1.1)
```

### 3. Mixed Censoring

For datasets containing both left-censored (e.g., lower limit) and right-censored (e.g., upper limit/saturation) values.

**Status Codes:**
*   `0`: Observed (Uncensored)
*   `-1`: Left Censored (Value < Limit)
*   `1`: Right Censored (Value > Limit)

```python
# Data: 10 (Obs), <5 (Left), >20 (Right)
values = [10.0, 5.0, 20.0]
status = [0, -1, 1]

# Method 1: Parametric (Weibull fit on mixed data)
df_mixed = impute(values, status, method='parametric', censoring_type='mixed')

# Method 2: Heuristic ROS
df_ros = impute(values, status, method='ros', censoring_type='mixed')

# Method 3: Mixed Substitution (Custom strategies)
df_sub = impute(values, status, method='substitution', censoring_type='mixed',
                left_strategy='half',               # <5 -> 2.5
                right_strategy='multiple', right_multiplier=1.1) # >20 -> 22.0
```

## API Reference

### `impute(values, status, method='ros', censoring_type='left', **kwargs)`

**Arguments:**

*   `values` (array-like): The observed values (or censoring limits).
*   `status` (array-like): Censoring indicator.
    *   Left/Right types: Boolean (`True` = Censored).
    *   Mixed type: Integer (`-1` = Left, `0` = Observed, `1` = Right).
*   `method` (str):
    *   `'ros'`: Regression on Order Statistics (Robust/Reverse/Heuristic).
    *   `'parametric'`: Conditional Mean Imputation (Weibull).
    *   `'substitution'`: Simple substitution.
*   `censoring_type` (str): `'left'`, `'right'`, or `'mixed'`.
*   `**kwargs`:
    *   `strategy` (str): For substitution (`'half'`, `'zero'`, `'value'`, `'multiple'`).
    *   `multiplier` (float): Factor for `'multiple'` strategy.
    *   `left_strategy`, `right_strategy`, etc.: For mixed substitution.

**Returns:**
A `pandas.DataFrame` containing:
*   `imputed_value`: The final imputed series.
*   `original_value`: Input values.
*   `censoring_status`: Input status.
*   `is_imputed`: Boolean flag indicating which values were modified.
