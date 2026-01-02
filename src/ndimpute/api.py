import pandas as pd
import numpy as np
from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right
from ._ros_mixed import impute_ros_mixed_heuristic
from ._parametric import impute_right_conditional, impute_mixed_parametric
from ._substitution import impute_sub_left, impute_sub_right, impute_sub_mixed

def impute(values, status, method='ros', censoring_type='left', **kwargs):
    """
    Unified imputation function.

    Args:
        values (array-like): The data values.
        status (array-like): Indicator.
            For censoring_type='left': True if < LOD.
            For censoring_type='right': True if censored (> C).
            For censoring_type='mixed': Integer array (-1: Left, 0: Observed, 1: Right).
        method (str): 'ros', 'parametric', or 'substitution'.
        censoring_type (str): 'left', 'right', or 'mixed'.
        **kwargs: Additional arguments for specific methods.
            - Substitution: 'strategy', 'multiplier'.
            - Mixed Substitution: 'left_strategy', 'right_strategy', 'left_multiplier', 'right_multiplier'.

    Returns:
        pd.DataFrame: A dataframe containing:
            - 'imputed_value': The final value (observed or imputed).
            - 'original_value': The input value.
            - 'censoring_status': The original status input.
            - 'is_imputed': Boolean flag.
    """
    values = np.array(values)

    # Status handling depends on type
    if censoring_type == 'mixed':
        status = np.array(status, dtype=int)
        is_imputed = (status != 0)
    else:
        status = np.array(status, dtype=bool)
        is_imputed = status

    if censoring_type == 'left':
        if method == 'ros':
            imputed_vals = impute_ros_left(values, status)
        elif method == 'substitution':
            strategy = kwargs.get('strategy', 'half')
            multiplier = kwargs.get('multiplier', None)
            imputed_vals = impute_sub_left(values, status, strategy=strategy, multiplier=multiplier)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented for left censoring.")

    elif censoring_type == 'right':
        if method == 'ros':
            imputed_vals = impute_ros_right(values, status)
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
        raise ValueError("censoring_type must be 'left', 'right', or 'mixed'")

    return pd.DataFrame({
        'imputed_value': imputed_vals,
        'original_value': values,
        'censoring_status': status,
        'is_imputed': is_imputed
    })
