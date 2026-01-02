import numpy as np
import pandas as pd

def impute_sub_left(values, is_censored, strategy='half', multiplier=None):
    """
    Imputes left-censored data using simple substitution.

    Args:
        values (array): Observed values (LOD for censored).
        is_censored (bool array): True if value is censored (<).
        strategy (str):
            - 'half': Replace <LOD with LOD/2 (default).
            - 'zero': Replace <LOD with 0.
            - 'value' or 'lod': Replace <LOD with LOD.
            - 'multiple': Replace <LOD with LOD * multiplier.
        multiplier (float, optional): Factor to multiply by when strategy='multiple'.
    """
    values = np.array(values, dtype=float)
    is_censored = np.array(is_censored, dtype=bool)

    imputed = values.copy()
    cens_vals = values[is_censored]

    if strategy == 'half':
        imputed[is_censored] = cens_vals / 2.0
    elif strategy == 'zero':
        imputed[is_censored] = 0.0
    elif strategy in ['value', 'lod']:
        imputed[is_censored] = cens_vals
    elif strategy == 'multiple':
        if multiplier is None:
            raise ValueError("Must provide 'multiplier' argument when strategy='multiple'.")
        imputed[is_censored] = cens_vals * multiplier
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for left censoring substitution. Options: 'half', 'zero', 'value', 'multiple'.")

    return imputed

def impute_sub_right(values, is_censored, strategy='value', multiplier=None):
    """
    Imputes right-censored data using simple substitution.

    Args:
        values (array): Observed values (Censoring time C).
        is_censored (bool array): True if value is censored (>).
        strategy (str):
            - 'value' or 'c': Replace >C with C.
            - 'multiple': Replace >C with C * multiplier.
        multiplier (float, optional): Factor to multiply by when strategy='multiple'.
    """
    values = np.array(values, dtype=float)
    is_censored = np.array(is_censored, dtype=bool)

    imputed = values.copy()
    cens_vals = values[is_censored]

    if strategy in ['value', 'c']:
        imputed[is_censored] = cens_vals
    elif strategy == 'multiple':
        if multiplier is None:
            raise ValueError("Must provide 'multiplier' argument when strategy='multiple'.")
        imputed[is_censored] = cens_vals * multiplier
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for right censoring substitution. Options: 'value', 'multiple'.")

    return imputed
