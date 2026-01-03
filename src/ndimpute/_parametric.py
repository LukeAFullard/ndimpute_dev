import numpy as np
from scipy.stats import weibull_min, CensoredData
from scipy.special import gamma, gammaincc, gammainc

def impute_right_conditional(values, is_censored):
    """
    Imputes right-censored data using Conditional Mean Imputation (Vectorized).
    """
    data = np.array(values)
    cens = np.array(is_censored, dtype=bool)

    if not np.any(cens):
        return data.copy()

    # 1. Fit Weibull using scipy's CensoredData
    uncensored_vals = data[~cens]
    censored_vals = data[cens]

    cd = CensoredData(uncensored=uncensored_vals, right=censored_vals)
    # weibull_min shape=k, scale=lambda
    shape, loc, scale = weibull_min.fit(cd, floc=0)

    # 2. Vectorized Imputation
    imputed = data.copy()

    # Extract censored values C
    C = data[cens]

    # E[T | T > C] = C + (Integral_C^inf S(t) dt) / S(C)
    # For Weibull: S(t) = exp(-(t/scale)^shape)

    # Calculate terms
    u_c = (C / scale) ** shape
    mean_unconditional = scale * gamma(1 + 1.0/shape)

    # Integral of t*f(t) from C to inf
    # Use gammaincc(1 + 1/k, u_c) for the upper integral
    integral_upper = mean_unconditional * gammaincc(1.0 + 1.0/shape, u_c)

    # S(C)
    S_C = np.exp(-u_c)

    valid_mask = S_C > 1e-15

    # E[T|T>C] = integral_upper / S_C
    expected_val = C.copy()
    expected_val[valid_mask] = integral_upper[valid_mask] / S_C[valid_mask]

    imputed[cens] = expected_val

    return imputed

def impute_mixed_parametric(values, status):
    """
    Imputes mixed-censored data (left and right) using Conditional Mean Imputation.
    """
    data = np.array(values)
    status = np.array(status, dtype=int)

    # Masks
    mask_obs = (status == 0)
    mask_left = (status == -1)
    mask_right = (status == 1)

    # Construct CensoredData
    cd = CensoredData(
        uncensored=data[mask_obs],
        left=data[mask_left],
        right=data[mask_right]
    )

    shape, loc, scale = weibull_min.fit(cd, floc=0)
    dist = weibull_min(shape, loc=loc, scale=scale)

    imputed = data.copy()
    mean_unconditional = scale * gamma(1 + 1.0/shape)

    # Impute Left Censored: E[T | T < L]
    if np.any(mask_left):
        L = data[mask_left]
        u_L = (L / scale) ** shape
        F_L = 1.0 - np.exp(-u_L) # CDF

        # Integral of t*f(t) from 0 to L
        # Use gammainc(1 + 1/k, u_L) for lower integral
        # Note: gammainc vs gammaincc. 'inc' is lower regularized.
        # Parameter 'a' must be 1 + 1/shape for the first moment.
        integral_lower = mean_unconditional * gammainc(1.0 + 1.0/shape, u_L)

        valid_mask = F_L > 1e-15
        vals = L.copy()
        vals[valid_mask] = integral_lower[valid_mask] / F_L[valid_mask]
        vals[~valid_mask] = L[~valid_mask] / 2.0 # Fallback for very small L

        imputed[mask_left] = vals

    # Impute Right Censored: E[T | T > R]
    if np.any(mask_right):
        R = data[mask_right]
        u_R = (R / scale) ** shape
        S_R = np.exp(-u_R)

        # Integral of t*f(t) from R to inf
        # Use gammaincc(1 + 1/k, u_R) for upper integral
        integral_upper = mean_unconditional * gammaincc(1.0 + 1.0/shape, u_R)

        valid_mask = S_R > 1e-15
        vals = R.copy()
        vals[valid_mask] = integral_upper[valid_mask] / S_R[valid_mask]

        imputed[mask_right] = vals

    return imputed
