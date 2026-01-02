import numpy as np
from ._ros_left import impute_ros_left
from ._ros_right import impute_ros_right

def impute_ros_mixed_heuristic(values, status):
    """
    Imputes mixed-censored data using a sequential heuristic ROS.

    Strategy:
    1. Treat Right Censored values as Observed (Status 0).
    2. Impute Left Censored values using standard ROS (impute_ros_left).
    3. Use the result from (2) where Left values are now imputed (Observed).
    4. Impute Right Censored values using Reverse ROS (impute_ros_right).

    Args:
        values (array): Data values.
        status (array): Status codes (-1: Left, 0: Obs, 1: Right).

    Returns:
        array: Imputed values.
    """
    values = np.array(values, dtype=float)
    status = np.array(status, dtype=int)

    # --- Pass 1: Impute Left ---
    # Treat Right Censored (1) as Observed (False in boolean mask for Left ROS)
    # Left Censored (-1) are True.
    mask_left = (status == -1)

    # We use the raw values for Right Censored data here.
    # This assumes R is a lower bound for the true value, so treating it as R
    # is a conservative estimate for the purpose of fitting the Left tail.
    # (Since Left tail < Right tail usually, the exact value of Right tail
    # matters less for the slope of the Left tail than the count N does).
    pass1_values = impute_ros_left(values, mask_left)

    # --- Pass 2: Impute Right ---
    # Now use the output of Pass 1.
    # Left Censored are now imputed values. Treat them as Observed (False).
    # Right Censored (1) are True.
    mask_right = (status == 1)

    final_values = impute_ros_right(pass1_values, mask_right)

    return final_values
