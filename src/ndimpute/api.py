import pandas as pd
import numpy as np
from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right
from ._parametric import impute_right_conditional
from ._substitution import impute_sub_left, impute_sub_right

def impute(values, status, method='ros', censoring_type='left', **kwargs):
    """
    Unified imputation function.

    Args:
        values (array-like): The data values.
        status (array-like): Indicator.
            For left-censoring: True if < LOD.
            For right-censoring: True if censored (event did not happen).
        method (str): 'ros', 'parametric', or 'substitution'.
        censoring_type (str): 'left' or 'right'.
        **kwargs: Additional arguments for specific methods (e.g., strategy for substitution).

    Returns:
        pd.DataFrame: A dataframe containing:
            - 'imputed_value': The final value (observed or imputed).
            - 'original_value': The input value.
            - 'censoring_status': The original status input.
            - 'is_imputed': Boolean flag.
    """
    values = np.array(values)
    status = np.array(status, dtype=bool)

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

    else:
        raise ValueError("censoring_type must be 'left' or 'right'")

    return pd.DataFrame({
        'imputed_value': imputed_vals,
        'original_value': values,
        'censoring_status': status,
        'is_imputed': status  # For simple censoring, status=True usually implies it needs imputation
    })
