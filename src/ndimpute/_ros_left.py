import numpy as np
import pandas as pd
from scipy.stats import norm, linregress, ecdf, CensoredData

def impute_ros_left(values, is_censored, dist='lognormal'):
    """
    Imputes left-censored data using Robust ROS.
    Uses Kaplan-Meier (Hirsch-Stedinger) plotting positions via scipy.stats.ecdf
    to robustly handle multiple detection limits.

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
        dist (str): Distribution assumption ('lognormal' or 'normal').
    """
    values = np.array(values)
    is_censored = np.array(is_censored, dtype=bool)

    # 1. Calculate Plotting Positions using Kaplan-Meier (Hirsch-Stedinger)
    # Left Censoring (< L) is equivalent to Right Censoring (> -L) on negated data.
    # We transform Y -> -Y.
    neg_values = -values

    # Wrap in CensoredData
    # is_censored=True means Left Censored in original, so Right Censored in negated.
    # CensoredData.right_censored(observed, censored) takes arrays of values?
    # No, it takes `uncensored` and `right`.

    unc_data = neg_values[~is_censored]
    cens_data = neg_values[is_censored]

    cd = CensoredData(uncensored=unc_data, right=cens_data)

    # Fit ECDF
    res = ecdf(cd)

    # 2. Plotting Positions for Uncensored Data
    unc_mask = ~is_censored
    y_unc = values[unc_mask]

    # Evaluate SF of negated data at -y_unc
    # res.cdf.evaluate(x) gives P(X <= x)
    # res.sf.evaluate(x) gives P(X > x)
    # P(Y <= y) = P(-Y >= -y) = P(Z >= -y) approx P(Z > -y) for continuous
    pp_unc = res.cdf.evaluate(-y_unc)

    # Wait, ecdf cdf is P(X <= x).
    # If Z = -Y.
    # P(Y <= y) = P(-Z <= y) = P(Z >= -y) = S_Z(-y).
    # Correct. We want Survival Function of negated data.

    pp_unc = res.sf.evaluate(-y_unc)

    # Scaling to avoid 1.0/0.0
    n = len(values)
    pp_unc = pp_unc * (n / (n + 1))
    pp_unc[pp_unc == 0] = 0.5 / (n + 1)
    pp_unc[pp_unc == 1] = 1.0 - (0.5 / (n + 1))

    # 3. Fit Regression
    if len(y_unc) < 2:
         raise ValueError("Too few uncensored observations to fit regression.")

    z_unc = norm.ppf(pp_unc)

    if dist == 'lognormal':
        if (values <= 0).any():
             raise ValueError("Values must be positive for lognormal distribution.")
        y_reg = np.log(y_unc)
    elif dist == 'normal':
        y_reg = y_unc
    else:
        raise ValueError(f"Unknown distribution '{dist}'")

    slope, intercept, _, _, _ = linregress(z_unc, y_reg)

    # 4. Impute
    # For a point censored at L: T < L.
    # We know P(T < L) = P(Z > -L) = S_Z(-L).

    y_cens = values[is_censored]
    pp_limits = res.sf.evaluate(-y_cens)

    pp_limits = pp_limits * (n / (n + 1))
    pp_limits[pp_limits == 0] = 0.5 / (n + 1) # Guardrail

    z_limits = norm.ppf(pp_limits)

    # Calculate conditional mean Z for each censored point (Z < z_c)
    # E[Z | Z < z_c] = -pdf(z_c) / cdf(z_c)
    numerator = norm.pdf(z_limits)
    denominator = norm.cdf(z_limits) # This is pp_limits
    z_imputed = -numerator / denominator

    # Predict
    predicted = intercept + slope * z_imputed

    if dist == 'lognormal':
        imputed_vals = np.exp(predicted)
    else:
        imputed_vals = predicted

    # Guardrail
    imputed_vals = np.minimum(imputed_vals, y_cens)

    # 5. Construct Result
    result = values.copy()
    result[is_censored] = imputed_vals

    return result
