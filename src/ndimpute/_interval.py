import numpy as np
from scipy.stats import norm, linregress
from ._turnbull import turnbull_em, predict_turnbull

def impute_interval_ros(left, right, dist='lognormal'):
    """
    Imputes interval-censored data using ROS with plotting positions derived
    from the Turnbull Estimator.
    """
    left = np.array(left)
    right = np.array(right)

    # 1. Turnbull Estimator
    intervals, probs = turnbull_em(left, right)

    if len(probs) == 0:
        raise ValueError("Turnbull estimator failed to find valid intervals.")

    # 2. Calculate Plotting Positions
    # Fit regression to Turnbull CDF points
    mids = np.mean(intervals, axis=1)

    # Cumulative Probabilities
    cdf_vals = np.cumsum(probs)
    pp_turnbull = cdf_vals - (probs / 2.0)

    z_turnbull = norm.ppf(pp_turnbull)

    if dist == 'lognormal':
        # Handle non-positive midpoints if any (though Turnbull intervals should be within obs range)
        # If mid <= 0, we can't take log.
        valid_mids = mids > 0
        if not np.any(valid_mids):
             raise ValueError("Intervals must be positive for lognormal distribution.")

        y_fit = np.log(mids[valid_mids])
        z_fit = z_turnbull[valid_mids]
        weights = probs[valid_mids]
    else:
        y_fit = mids
        z_fit = z_turnbull
        weights = probs

    # Weighted OLS
    w_mean_x = np.average(z_fit, weights=weights)
    w_mean_y = np.average(y_fit, weights=weights)

    numerator = np.sum(weights * (z_fit - w_mean_x) * (y_fit - w_mean_y))
    denominator = np.sum(weights * (z_fit - w_mean_x)**2)

    slope = numerator / denominator
    intercept = w_mean_y - slope * w_mean_x

    # 3. Impute
    mu_model = intercept
    sigma_model = slope

    imputed = np.zeros_like(left, dtype=float)

    for i in range(len(left)):
        l_i, r_i = left[i], right[i]

        # Transform bounds to Z-space
        if dist == 'lognormal':
            with np.errstate(divide='ignore'):
                z_l = (np.log(l_i) - mu_model) / sigma_model if l_i > 0 else -np.inf
                z_r = (np.log(r_i) - mu_model) / sigma_model if not np.isinf(r_i) else np.inf
        else:
            z_l = (l_i - mu_model) / sigma_model
            z_r = (r_i - mu_model) / sigma_model if not np.isinf(r_i) else np.inf

        # Expected Z in truncated range
        phi_a = norm.pdf(z_l)
        phi_b = norm.pdf(z_r) if not np.isinf(z_r) else 0.0

        Phi_a = norm.cdf(z_l)
        Phi_b = norm.cdf(z_r) if not np.isinf(z_r) else 1.0

        denom = Phi_b - Phi_a
        if denom < 1e-9:
            # Interval is extremely far in tail or tiny. Use midpoint logic.
            # If lognormal, geometric mean? Or just transformed Z.
            e_z = (z_l + z_r) / 2 if not np.isinf(z_r) and not np.isinf(z_l) else z_l + 1
        else:
            e_z = (phi_a - phi_b) / denom

        # Back transform
        pred_val = mu_model + sigma_model * e_z

        if dist == 'lognormal':
            imputed[i] = np.exp(pred_val)
        else:
            imputed[i] = pred_val

    return imputed
