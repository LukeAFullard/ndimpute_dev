import numpy as np
from scipy.stats import weibull_min, CensoredData
from scipy.special import gamma, gammaincc

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
    # Integral = scale/shape * Gamma(1/shape) * gammaincc(1/shape, (C/scale)^shape)
    # Actually, Integral_0^inf = Mean = scale * Gamma(1 + 1/shape)
    # Integral_C^inf = Mean * gammaincc(1 + 1/shape, (C/scale)^shape)?
    # Let's check math carefully.
    # Mean = scale * Gamma(1 + 1/shape)
    # Upper Incomplete Gamma function Gamma(s, x) = Integral_x^inf t^(s-1) e^-t dt
    # Our integral I = Integral_C^inf exp(-(t/scale)^shape) dt
    # Subst u = (t/scale)^shape => t = scale * u^(1/shape) => dt = scale/shape * u^(1/shape - 1) du
    # Limits: u_c = (C/scale)^shape to inf
    # I = Integral_{u_c}^inf e^-u * scale/shape * u^(1/shape - 1) du
    # I = (scale/shape) * Gamma(1/shape, u_c)
    # Using scipy.special.gammaincc (Regularized Upper Gamma Q(s,x) = Gamma(s,x)/Gamma(s)):
    # Gamma(1/shape, u_c) = Gamma(1/shape) * gammaincc(1/shape, u_c)
    # So I = (scale/shape) * Gamma(1/shape) * gammaincc(1/shape, u_c)
    # Note: (1/shape) * Gamma(1/shape) = Gamma(1 + 1/shape)
    # So I = scale * Gamma(1 + 1/shape) * gammaincc(1/shape, u_c)

    # Calculate terms
    u_c = (C / scale) ** shape
    mean_unconditional = scale * gamma(1 + 1.0/shape)

    # Note: gammaincc first arg is 'a', second is 'x'
    # Here a = 1/shape
    integral_val = scale * (1.0/shape) * gamma(1.0/shape) * gammaincc(1.0/shape, u_c)
    # Or simply:
    integral_val = mean_unconditional * gammaincc(1.0/shape, u_c)

    S_C = np.exp(-u_c)

    # Handle potentially small S_C (avoid division by zero)
    # If S_C is tiny, the conditional mean is essentially C (tail is 0)
    # or C + epsilon.
    valid_mask = S_C > 1e-15

    expected_val = C.copy()
    expected_val[valid_mask] = C[valid_mask] + (integral_val[valid_mask] / S_C[valid_mask])

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

    # Impute Left Censored: E[T | T < L]
    # This is harder to vectorize purely with gammaincc because it's the lower integral
    # But scipy.stats.dist.expect might be slow loop.
    # Can we vectorize?
    # E[T | T < L] = Integral_0^L t f(t) dt / F(L)
    # Integral_0^L = Total Mean - Integral_L^inf
    # Total Mean = scale * Gamma(1+1/k)
    # Integral_L^inf = scale * Gamma(1+1/k) * gammaincc(1/k, (L/scale)^k)
    # So Integral_0^L = Total Mean * (1 - gammaincc(...))
    # Which is Total Mean * gammainc(...) (Lower regularized gamma)

    if np.any(mask_left):
        L = data[mask_left]
        u_L = (L / scale) ** shape
        F_L = 1.0 - np.exp(-u_L) # CDF

        # Mean * gammainc (lower regularized)
        mean_unconditional = scale * gamma(1 + 1.0/shape)
        # Using gammainc (lower) which corresponds to integral from 0 to x
        # Note: gammainc vs gammaincc. 'inc' is lower, 'incc' is upper. Sum is 1.
        # However, the term inside is t*f(t).
        # We established I_upper = Mean * gammaincc(1/k, u)
        # So I_lower = Mean * gammainc(1/k, u)
        # Actually check parameter a=1/k again? Yes from deriv above.

        from scipy.special import gammainc
        integral_lower = mean_unconditional * gammainc(1.0/shape, u_L)

        valid_mask = F_L > 1e-15
        vals = L.copy()
        vals[valid_mask] = integral_lower[valid_mask] / F_L[valid_mask]
        vals[~valid_mask] = L[~valid_mask] / 2.0 # Fallback for very small L

        imputed[mask_left] = vals

    # Impute Right Censored: E[T | T > R]
    # Reuse vectorization logic from right_conditional
    if np.any(mask_right):
        R = data[mask_right]
        u_R = (R / scale) ** shape
        S_R = np.exp(-u_R)

        mean_unconditional = scale * gamma(1 + 1.0/shape)
        integral_upper = mean_unconditional * gammaincc(1.0/shape, u_R)

        valid_mask = S_R > 1e-15
        vals = R.copy()
        vals[valid_mask] = R[valid_mask] + (integral_upper[valid_mask] / S_R[valid_mask])

        imputed[mask_right] = vals

    return imputed
