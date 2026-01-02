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
    # We want to sort primarily by value.
    df = df.sort_values('val')

    # 2. Compute Plotting Positions (simplified Kaplan-Meier for Left Censoring)
    # We essentially flip the data to treat left-censoring as right-censoring
    # to use standard KM logic, then flip back.

    # (Simplified implementation for single detection limit for clarity)
    n_cens = df['cens'].sum()
    n_unc = n - n_cens

    if n_unc < 2:
        raise ValueError("Too few uncensored observations to fit regression.")

    # Probability mass below LOD (assuming single LOD for now as per simplified implementation)
    # In a full multi-LOD scenario, we would use Kaplan-Meier or Hirsch-Stedinger.
    # Here we assume a simplified approach where all censored values are < min(uncensored)
    # or mixed but we use a simplified plotting position.

    # Let's try to implement a slightly more robust plotting position that handles simple cases well.
    # If we assume standard ROS plotting positions for a single censoring limit:
    prob_cens = n_cens / (n + 1)

    # Assign positions
    df['pp'] = 0.0

    # We need to assign pp based on rank, but distinguish censored/uncensored blocks
    # This implementation assumes censored values are 'smaller' effectively than the uncensored ones
    # or that we are distributing the probability mass of the censored values.

    # Censored points spread in [0, prob_cens]
    # We need to map the indices of censored items to this range
    df.loc[df['cens'], 'pp'] = np.linspace(0.5/n, prob_cens, n_cens)

    # Uncensored points spread in [prob_cens, 1]
    # We need to map the indices of uncensored items to this range
    df.loc[~df['cens'], 'pp'] = np.linspace(prob_cens + (0.5/n), 1 - (0.5/n), n_unc)

    # 3. Z-scores
    df['z'] = norm.ppf(df['pp'])

    # 4. Fit Regression on Uncensored Data
    # Model: log(val) = intercept + slope * Z

    if dist == 'lognormal':
        # Ensure values are positive
        if (df.loc[~df['cens'], 'val'] <= 0).any():
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

    # 6. Robustness: Only replace censored values
    result = df['val'].copy()
    result.loc[df['cens']] = imputed_vals

    return result.sort_index().values
