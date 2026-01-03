import numpy as np
import pandas as pd
from scipy.stats import norm, linregress, ecdf, CensoredData

def impute_ros_left(values, is_censored, dist='lognormal', plotting_position='kaplan-meier'):
    """
    Imputes left-censored data using Robust ROS.

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
        dist (str): Distribution assumption ('lognormal' or 'normal').
        plotting_position (str): Method for calculating plotting positions.
            - 'kaplan-meier' (default): Uses Hirsch-Stedinger logic via scipy.stats.ecdf.
              Best for multiple detection limits.
            - 'simple' or 'weibull': Uses simple ranking (rank/(n+1)).
              Matches simple NADA approximations for single limits.
    """
    values = np.array(values)
    is_censored = np.array(is_censored, dtype=bool)
    n = len(values)

    # Common Setup: Log Transform if needed for regression Y
    unc_mask = ~is_censored
    y_unc = values[unc_mask]

    if len(y_unc) < 2:
         raise ValueError("Too few uncensored observations to fit regression.")

    if dist == 'lognormal':
        if (values <= 0).any():
             raise ValueError("Values must be positive for lognormal distribution.")
        y_reg = np.log(y_unc)
    elif dist == 'normal':
        y_reg = y_unc
    else:
        raise ValueError(f"Unknown distribution '{dist}'")

    # --- Branch 1: Kaplan-Meier (Hirsch-Stedinger) ---
    if plotting_position in ['kaplan-meier', 'ecdf', 'hirsch-stedinger']:
        neg_values = -values
        unc_data = neg_values[~is_censored]
        cens_data = neg_values[is_censored]

        cd = CensoredData(uncensored=unc_data, right=cens_data)
        res = ecdf(cd)

        # PPs for Uncensored
        pp_unc = res.sf.evaluate(-y_unc)

        # Scaling
        pp_unc = pp_unc * (n / (n + 1))
        pp_unc[pp_unc == 0] = 0.5 / (n + 1)
        pp_unc[pp_unc == 1] = 1.0 - (0.5 / (n + 1))

        z_unc = norm.ppf(pp_unc)

        # Fit
        slope, intercept, _, _, _ = linregress(z_unc, y_reg)

        # Impute
        y_cens = values[is_censored]
        pp_limits = res.sf.evaluate(-y_cens)
        pp_limits = pp_limits * (n / (n + 1))
        pp_limits[pp_limits == 0] = 0.5 / (n + 1)

        z_limits = norm.ppf(pp_limits)

        numerator = norm.pdf(z_limits)
        denominator = norm.cdf(z_limits)
        z_imputed = -numerator / denominator

        predicted = intercept + slope * z_imputed

    # --- Branch 2: Simple Ranking (Weibull) ---
    elif plotting_position in ['simple', 'weibull']:
        # Sort data to assign ranks
        df = pd.DataFrame({'val': values, 'cens': is_censored})
        df = df.sort_values('val')

        df['rank'] = np.arange(1, n + 1)
        df['pp'] = df['rank'] / (n + 1)
        df['z'] = norm.ppf(df['pp'])

        # Fit on Uncensored
        # We need to extract the Z-scores corresponding to the uncensored data *after* sorting
        # y_reg computed above matches the order of y_unc (values[unc_mask]).
        # But df is sorted. We must match them.

        # Let's re-extract y_reg from the sorted dataframe to ensure alignment
        y_reg_sorted = df.loc[~df['cens'], 'val']
        if dist == 'lognormal':
            y_reg_sorted = np.log(y_reg_sorted)

        x_obs = df.loc[~df['cens'], 'z']

        slope, intercept, _, _, _ = linregress(x_obs, y_reg_sorted)

        # Impute
        z_cens = df.loc[df['cens'], 'z']
        predicted_sorted = intercept + slope * z_cens

        # We need to put these back into the original order.
        # The 'predicted' array here corresponds to the sorted censored rows.
        # We can assign them into the sorted dataframe, then resort by index.
        predicted = predicted_sorted # Placeholder name for logic below

    else:
        raise ValueError(f"Unknown plotting_position '{plotting_position}'.")

    # --- Common Finalization ---
    if plotting_position in ['simple', 'weibull']:
        # Simple method predicts directly based on Z of that point
        if dist == 'lognormal':
            imputed_vals = np.exp(predicted)
        else:
            imputed_vals = predicted

        # Assign back to sorted df
        # Note: 'predicted' is a Series with index matching df[cens]
        df.loc[df['cens'], 'imputed'] = imputed_vals

        # Guardrail
        limit_vals = df.loc[df['cens'], 'val']
        df.loc[df['cens'], 'imputed'] = np.minimum(df.loc[df['cens'], 'imputed'], limit_vals)

        # Restore order
        result = df.sort_index()['val'].copy()
        result[is_censored] = df.sort_index().loc[is_censored, 'imputed']
        return result.values

    else:
        # Kaplan-Meier path (already computed 'predicted')
        if dist == 'lognormal':
            imputed_vals = np.exp(predicted)
        else:
            imputed_vals = predicted

        y_cens = values[is_censored]
        imputed_vals = np.minimum(imputed_vals, y_cens)

        result = values.copy()
        result[is_censored] = imputed_vals
        return result
