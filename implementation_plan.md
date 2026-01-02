# Implementation Plan for `ndimpute`

This document outlines the step-by-step implementation guide for the `ndimpute` package, focusing on the "Comprehensive Analysis of Statistical Methodologies..." document.

### Project Architecture: `ndimpute`

```text
ndimpute/
├── pyproject.toml
├── src/
│   └── ndimpute/
│       ├── __init__.py
│       ├── _ros_left.py     # Left Censoring (Environmental/Nondetects)
│       ├── _ros_right.py    # Right Censoring (Reverse ROS)
│       ├── _parametric.py   # Right Censoring (Conditional Mean)
│       └── api.py           # Unified wrapper
└── tests/
    ├── test_ros_left.py
    └── test_right_imputation.py
```

### Step 1: Dependencies & Setup

We rely on `scipy` (stats) and `numpy`.

**`pyproject.toml` dependencies:**

```toml
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.13.0",  # Required for CensoredData class
]
```

### Step 2: Left Censoring Imputation (Robust ROS)

**Objective:** Implement Robust Regression on Order Statistics for Nondetects.
**Validation Target:** R package `NADA`, function `ros`.

**Sub-steps:**

1. **Plotting Positions:** Implement the Hirsch-Stedinger method (or Kaplan-Meier adapted for left-censoring) to handle multiple detection limits.
2. **Regression:** Fit OLS line: $\log(Observed) = \beta_0 + \beta_1 \cdot Z$.
3. **Imputation:** Predict censored values using the regression line.

#### Code Snippet: `src/ndimpute/_ros_left.py`

```python
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress

def impute_ros_left(values, is_censored, dist='lognormal'):
    """
    Imputes left-censored data using Robust ROS.
    Matches logic of R's NADA package.

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
    """
    df = pd.DataFrame({'val': values, 'cens': is_censored})
    n = len(df)

    # 1. Sort data
    # Standard sorting places smaller values first.
    # For multiple LODs, logic requires careful handling of censored vs uncensored ranks.
    df = df.sort_values('val')

    # 2. Compute Plotting Positions (simplified Kaplan-Meier for Left Censoring)
    # We essentially flip the data to treat left-censoring as right-censoring
    # to use standard KM logic, then flip back.

    # (Simplified implementation for single detection limit for clarity)
    n_cens = df['cens'].sum()
    n_unc = n - n_cens
    prob_cens = n_cens / (n + 1) # Probability mass below LOD

    # Assign positions
    df['pp'] = 0.0
    # Censored points spread in [0, prob_cens]
    df.loc[df['cens'], 'pp'] = np.linspace(0.5/n, prob_cens, n_cens)
    # Uncensored points spread in [prob_cens, 1]
    df.loc[~df['cens'], 'pp'] = np.linspace(prob_cens + (0.5/n), 1 - (0.5/n), n_unc)

    # 3. Z-scores
    df['z'] = norm.ppf(df['pp'])

    # 4. Fit Regression on Uncensored Data
    # Model: log(val) = intercept + slope * Z
    y_obs = np.log(df.loc[~df['cens'], 'val']) if dist == 'lognormal' else df.loc[~df['cens'], 'val']
    x_obs = df.loc[~df['cens'], 'z']

    slope, intercept, _, _, _ = linregress(x_obs, y_obs)

    # 5. Impute
    z_cens = df.loc[df['cens'], 'z']
    predicted = intercept + slope * z_cens
    imputed_vals = np.exp(predicted) if dist == 'lognormal' else predicted

    # 6. Robustness: Only replace censored values
    result = df['val'].copy()
    result.loc[df['cens']] = imputed_vals

    return result.sort_index().values
```

**Validation Steps:**

1. **R Comparison:** Compare output against `NADA::ros(vals, cens)$modeled`.
2. **Edge Case:** If `n_unc < 2` (fewer than 2 observed points), raise `ValueError` as regression is impossible.

### Step 3: Right Censoring Imputation (Conditional Mean)

**Objective:** Parametric imputation for Survival/Reliability data.
**Validation Target:** R package `imputeCensoRd`, or manual `survival` integration.

**Sub-steps:**

1. **Fit:** Fit a parametric model (Weibull/Lognormal) using Maximum Likelihood.
2. **Integrate:** For each censored value $C$, calculate the expected residual life.

#### Code Snippet: `src/ndimpute/_parametric.py`

```python
import numpy as np
from scipy.stats import weibull_min, CensoredData
from scipy.integrate import quad

def impute_right_conditional(values, is_censored):
    """
    Imputes right-censored data using Conditional Mean Imputation.
    """
    data = np.array(values)
    cens = np.array(is_censored, dtype=bool)

    # 1. Fit Weibull using scipy's CensoredData
    # Note: Scipy's weibull_min uses (c, loc, scale)
    cd = CensoredData.right_censored(data, cens)
    shape, loc, scale = weibull_min.fit(cd, floc=0)

    # 2. Impute Loop
    imputed = data.copy()

    def survival_func(t):
        return weibull_min.sf(t, shape, scale=scale)

    for i in np.where(cens)[0]:
        C = data[i]
        S_C = survival_func(C)

        if S_C < 1e-9:
            continue # Probability is effectively zero, keep C

        # E[T | T > C] = C + (Integral_C^inf S(t) dt) / S(C)
        integral, _ = quad(survival_func, C, np.inf)
        imputed[i] = C + (integral / S_C)

    return imputed
```

**Validation Steps:**

1. **R Comparison:** Fit a Weibull in R (`survreg`). Predict `type="response"` (mean). *Note: R's `predict` often gives the unconditional mean; you may need to manually compute the conditional mean in R to validate exact parity.*
2. **Edge Case:** High censoring times where $S(C) \approx 0$. Ensure division by zero is handled.

### Step 4: Right Censoring Imputation (Reverse ROS)

**Objective:** Non-parametric alternative for Right Censoring.
**Validation Target:** Theoretical parity (no direct standard R package for "Reverse ROS", but conceptually mirrors NADA).

**Logic:**

1. Multiply values by -1 (or sort descending).
2. Treat Right Censored values as "Left Censored" in the reversed domain.
3. Apply the `impute_ros_left` logic.
4. Flip back.

#### Code Snippet: `src/ndimpute/_ros_right.py`

```python
from ._ros_left import impute_ros_left

def impute_ros_right(values, is_censored, dist='lognormal'):
    """
    Imputes right-censored data using Reverse ROS.
    """
    # 1. Reverse domain
    # For lognormal (dist>0), we can't just flip sign and log.
    # Instead, we invert: y' = 1/y.
    # Large values become small (Left Censored).

    if dist == 'lognormal':
        # Right censored at 100 -> Value is > 100
        # Inverted: Value is < 1/100 (Left Censored)
        inv_values = 1.0 / values

        # Call Left ROS
        imputed_inv = impute_ros_left(inv_values, is_censored, dist='lognormal')

        # Invert back
        return 1.0 / imputed_inv

    else:
        # Normal distribution -> flip sign
        flipped_values = -values
        imputed_flipped = impute_ros_left(flipped_values, is_censored, dist='normal')
        return -imputed_flipped
```

### Step 5: Unified API & Outputs

**Objective:** Provide a unified wrapper in `src/ndimpute/api.py`.

**Output Requirement:**
The package should return the imputed (complete) values, but must also return a column or array indicating the censoring type (e.g., whether a value was originally observed, left-censored, or right-censored).

#### Proposed API Signature

```python
def impute(values, status, method='ros', censoring_type='left'):
    """
    Unified imputation function.

    Args:
        values (array-like): The data values.
        status (array-like): Indicator.
            For left-censoring: True if < LOD.
            For right-censoring: True if event happened (uncensored) or False if censored?
            (Need to standardize input format).
        method (str): 'ros' or 'parametric'.
        censoring_type (str): 'left' or 'right'.

    Returns:
        pd.DataFrame: A dataframe containing:
            - 'imputed_value': The final value (observed or imputed).
            - 'original_value': The input value.
            - 'censoring_status': The original status input.
            - 'is_imputed': Boolean flag.
    """
    pass
```

### Validation Strategy & Packages

| Method | Python (`ndimpute`) | Validation Package (R) |
| --- | --- | --- |
| **ROS (Left)** | `_ros_left.py` | **NADA** (`ros`) |
| **Cond. Mean (Right)** | `_parametric.py` | **imputeCensoRd** (`condl_mean_impute`) |
| **Reverse ROS** | `_ros_right.py` | N/A (Validate via manual calc or simulation) |

**Edge Case Validations to Implement:**

1. **Multiple Detection Limits (Left):** Ensure plotting positions (Step 2 in `_ros_left`) handle datasets like `[<1, <5, 3, 4]` correctly without violating probability monotonicity.
2. **100% Censoring:** Ensure code raises `ValueError` immediately.
3. **Stability:** Test with very small datasets ($N<5$) and very large datasets ($N>10^5$).
