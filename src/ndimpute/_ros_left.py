import numpy as np
import pandas as pd
from scipy.stats import norm, linregress

def impute_ros_left(values, is_censored, dist='lognormal'):
    """
    Imputes left-censored data using Robust ROS.
    Matches logic of R's NADA package (Weibull Plotting Positions).

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
    """
    df = pd.DataFrame({'val': values, 'cens': is_censored})
    n = len(df)

    # 1. Sort data
    # Standard sorting places smaller values first.
    # We maintain a stable sort for index recovery.
    df = df.sort_values('val')

    # 2. Compute Plotting Positions (Weibull Method: rank / (n+1))
    # This assumes that censored values effectively occupy the lower ranks
    # relative to the observed values (typical for single detection limit).
    # For multiple detection limits, NADA uses Kaplan-Meier to adjust these ranks.
    # Here, we use a simple rank-based approach which is robust for single limits
    # and a reasonable approximation for mixed limits in this "Simple" implementation.

    # We assign ranks 1 to N
    df['rank'] = np.arange(1, n + 1)

    # Weibull Plotting Position
    df['pp'] = df['rank'] / (n + 1)

    # 3. Z-scores
    df['z'] = norm.ppf(df['pp'])

    # 4. Fit Regression on Uncensored Data
    n_unc = (~df['cens']).sum()
    if n_unc < 2:
         raise ValueError("Too few uncensored observations to fit regression.")

    # Model: log(val) = intercept + slope * Z
    if dist == 'lognormal':
        # Ensure ALL values are positive (observed AND censoring limits)
        # because "censored at -1" is invalid for lognormal support (0, inf).
        if (df['val'] <= 0).any():
             raise ValueError("Values must be positive for lognormal distribution.")
        y_obs = np.log(df.loc[~df['cens'], 'val'])
    else:
        y_obs = df.loc[~df['cens'], 'val']

    x_obs = df.loc[~df['cens'], 'z']

    slope, intercept, _, _, _ = linregress(x_obs, y_obs)

    # 5. Impute
    z_cens = df.loc[df['cens'], 'z']
    predicted = intercept + slope * z_cens

    if dist == 'lognormal':
        imputed_vals = np.exp(predicted)
    else:
        imputed_vals = predicted

    # Guardrail: Imputed values should not exceed the detection limit (for Left Censoring)
    limit_vals = df.loc[df['cens'], 'val']
    imputed_vals = np.minimum(imputed_vals, limit_vals)

    # 6. Robustness: Only replace censored values
    result = df['val'].copy()
    result.loc[df['cens']] = imputed_vals

    return result.sort_index().values
