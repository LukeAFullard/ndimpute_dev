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
    uncensored_vals = data[~cens]
    censored_vals = data[cens]

    cd = CensoredData(uncensored=uncensored_vals, right=censored_vals)
    shape, loc, scale = weibull_min.fit(cd, floc=0)

    # 2. Impute Loop
    imputed = data.copy()

    def survival_func(t):
        return weibull_min.sf(t, shape, loc=loc, scale=scale)

    censored_indices = np.where(cens)[0]

    for i in censored_indices:
        C = data[i]
        S_C = survival_func(C)

        if S_C < 1e-9:
            continue # Probability is effectively zero, keep C

        # E[T | T > C] = C + (Integral_C^inf S(t) dt) / S(C)
        integral, _ = quad(survival_func, C, np.inf)
        imputed[i] = C + (integral / S_C)

    return imputed

def impute_mixed_parametric(values, status):
    """
    Imputes mixed-censored data (left and right) using Conditional Mean Imputation.

    Args:
        values (array): Data values.
        status (array): Status codes.
            0: Observed (Uncensored)
            -1: Left Censored (Value < L)
            1: Right Censored (Value > R)
    """
    data = np.array(values)
    status = np.array(status, dtype=int)

    # Masks
    mask_obs = (status == 0)
    mask_left = (status == -1)
    mask_right = (status == 1)

    # Construct CensoredData
    # Note: CensoredData takes arrays of values for each category
    cd = CensoredData(
        uncensored=data[mask_obs],
        left=data[mask_left],
        right=data[mask_right]
    )

    # Fit Weibull (assuming positive data)
    # If fitting fails or data is not positive, this will raise error from scipy
    # We fix location to 0 for standard 2-param Weibull
    shape, loc, scale = weibull_min.fit(cd, floc=0)

    dist = weibull_min(shape, loc=loc, scale=scale)

    imputed = data.copy()

    # Impute Left Censored: E[T | T < L]
    # = Integral_{-inf}^L x f(x) dx / F(L)
    # Since Weibull is defined on [0, inf), lower bound is 0.
    if np.any(mask_left):
        indices = np.where(mask_left)[0]
        for i in indices:
            L = data[i]
            cdf_L = dist.cdf(L)

            if cdf_L < 1e-9:
                # Probability mass below L is negligible, essentially 0?
                # For left censoring, this means the value is likely very small.
                # Let's default to a small epsilon or 0?
                # Or just keep L? Usually L/2 is better than L if model says prob is 0.
                # But if model says prob is 0, then L is an outlier for this model.
                imputed[i] = L / 2.0
            else:
                # expect calculates integral x*pdf(x)
                expected_val = dist.expect(lb=0, ub=L)
                imputed[i] = expected_val / cdf_L

    # Impute Right Censored: E[T | T > R]
    if np.any(mask_right):
        indices = np.where(mask_right)[0]
        for i in indices:
            R = data[i]
            sf_R = dist.sf(R)

            if sf_R < 1e-9:
                # Probability mass above R is negligible
                # Keep R as conservative estimate
                imputed[i] = R
            else:
                expected_val = dist.expect(lb=R, ub=np.inf)
                imputed[i] = expected_val / sf_R

    return imputed
