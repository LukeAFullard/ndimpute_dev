import pandas as pd
import numpy as np
from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right
from ._ros_mixed import impute_ros_mixed_heuristic
from ._parametric import impute_right_conditional, impute_mixed_parametric
from ._substitution import impute_sub_left, impute_sub_right, impute_sub_mixed
from ._interval import impute_interval_ros

def impute(values, status=None, method='ros', censoring_type='left', **kwargs):
    """
    Unified imputation function.

    Args:
        values (array-like): The data values.
            - Left/Right/Mixed: 1D array of values.
            - Interval: 2D array of shape (N, 2) representing (Left, Right) bounds.
        status (array-like, optional): Indicator. Required for non-interval types.
            - Left: True if < LOD.
            - Right: True if censored (> C).
            - Mixed: Integer array (-1: Left, 0: Observed, 1: Right).
        method (str): 'ros', 'parametric', or 'substitution'.
        censoring_type (str): 'left', 'right', 'mixed', or 'interval'.
        **kwargs: Additional arguments (dist, plotting_position, strategy, etc.)

    Returns:
        pd.DataFrame: A dataframe containing:
            - 'imputed_value': The final value (observed or imputed).
            - 'original_value': The input value (or string repr for intervals).
            - 'censoring_status': The original status input.
            - 'is_imputed': Boolean flag.
    """
    dist = kwargs.get('dist', 'lognormal')
    plotting_position = kwargs.get('plotting_position', 'kaplan-meier')

    if censoring_type == 'interval':
        # Values should be (N, 2)
        bounds = np.array(values)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("For censoring_type='interval', values must be (N, 2) array of bounds.")

        left, right = bounds[:, 0], bounds[:, 1]

        if method == 'ros':
            imputed_vals = impute_interval_ros(left, right, dist=dist)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented for interval censoring.")

        return pd.DataFrame({
            'imputed_value': imputed_vals,
            'original_left': left,
            'original_right': right,
            'censoring_status': 'interval',
            'is_imputed': True # All intervals are technically imputed/estimated
        })

    # ... Existing Logic for Left/Right/Mixed ...
    values = np.array(values)
    if status is None:
        raise ValueError("Status argument is required for left/right/mixed censoring.")

    # Status handling depends on type
    if censoring_type == 'mixed':
        status = np.array(status, dtype=int)
        is_imputed = (status != 0)
    else:
        status = np.array(status, dtype=bool)
        is_imputed = status

    if censoring_type == 'left':
        if method == 'ros':
            imputed_vals = impute_ros_left(values, status, dist=dist, plotting_position=plotting_position)
        elif method == 'substitution':
            strategy = kwargs.get('strategy', 'half')
            multiplier = kwargs.get('multiplier', None)
            imputed_vals = impute_sub_left(values, status, strategy=strategy, multiplier=multiplier)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented for left censoring.")

    elif censoring_type == 'right':
        if method == 'ros':
            imputed_vals = impute_ros_right(values, status, dist=dist, plotting_position=plotting_position)
        elif method == 'parametric':
            imputed_vals = impute_right_conditional(values, status)
        elif method == 'substitution':
            strategy = kwargs.get('strategy', 'value')
            multiplier = kwargs.get('multiplier', None)
            imputed_vals = impute_sub_right(values, status, strategy=strategy, multiplier=multiplier)
        else:
            raise ValueError(f"Unknown method '{method}' for right censoring.")

    elif censoring_type == 'mixed':
        if method == 'parametric':
            imputed_vals = impute_mixed_parametric(values, status)
        elif method == 'substitution':
            # Extract mixed kwargs
            left_kwargs = {
                'strategy': kwargs.get('left_strategy', 'half'),
                'multiplier': kwargs.get('left_multiplier', None)
            }
            right_kwargs = {
                'strategy': kwargs.get('right_strategy', 'value'),
                'multiplier': kwargs.get('right_multiplier', None)
            }
            imputed_vals = impute_sub_mixed(values, status, left_kwargs=left_kwargs, right_kwargs=right_kwargs)
        elif method == 'ros':
             imputed_vals = impute_ros_mixed_heuristic(values, status)
        else:
            raise ValueError(f"Unknown method '{method}' for mixed censoring.")

    else:
        raise ValueError("censoring_type must be 'left', 'right', 'mixed', or 'interval'")

    return pd.DataFrame({
        'imputed_value': imputed_vals,
        'original_value': values,
        'censoring_status': status,
        'is_imputed': is_imputed
    })
