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

        # Ensure no zeros if lognormal
        if (values <= 0).any():
             raise ValueError("Values must be positive for lognormal distribution.")

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
