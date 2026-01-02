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
    # CensoredData.right_censored takes (observed_data, censored_data)
    # OR we can use the constructor directly if we have indicators.
    # The signature is CensoredData(uncensored=..., left=..., right=..., interval=...)
    # Or helper class methods.

    # Let's check how to construct CensoredData correctly from (values, is_censored)
    # is_censored=True means RIGHT censored (value is a lower bound).

    uncensored_vals = data[~cens]
    censored_vals = data[cens]

    cd = CensoredData(uncensored=uncensored_vals, right=censored_vals)

    # fit returns (shape, loc, scale)
    # We fix loc=0 for standard 2-parameter Weibull (common in reliability)
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
