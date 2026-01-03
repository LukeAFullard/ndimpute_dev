import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ndimpute import impute

def run_validation():
    print("Running Validation: 01 Textbook TCE")

    # 1. Load Data
    try:
        df_truth = pd.read_csv('truth.csv')
    except FileNotFoundError:
        print("Error: truth.csv not found. Run generate_data.R first.")
        return

    values = df_truth['original_value'].values
    # R exports TRUE/FALSE which pandas reads as bool.
    # ndimpute expects bool for Left Censoring (True=Censored).
    status = df_truth['censoring_status'].astype(bool).values

    # 2. Run ndimpute
    print("Running ndimpute(method='ros')...")
    df_result = impute(values, status, method='ros', censoring_type='left')

    # 3. Compare Results
    # Join with truth
    df_result['r_imputed'] = df_truth['r_imputed']

    # Filter for censored values (Observed should match exactly)
    cens_mask = df_result['censoring_status']

    py_imputed = df_result.loc[cens_mask, 'imputed_value']
    r_imputed = df_result.loc[cens_mask, 'r_imputed']

    mae = np.mean(np.abs(py_imputed - r_imputed))
    max_diff = np.max(np.abs(py_imputed - r_imputed))

    print("\n--- Comparison Results (Censored Values) ---")
    print(f"MAE: {mae:.6f}")
    print(f"Max Diff: {max_diff:.6f}")

    print("\n--- Detailed View ---")
    comparison = pd.DataFrame({
        'Original': df_result.loc[cens_mask, 'original_value'],
        'Python_ROS': py_imputed,
        'R_NADA_ROS': r_imputed,
        'Diff': py_imputed - r_imputed
    })
    print(comparison)

    # 4. Success Criteria Check
    # We relax exact matching because NADA might use different plotting position details
    # or R vs Python floating point logic.
    # 5% tolerance is specified in plan.
    # For small values < 1.0, 5% is 0.05.

    tolerance = 0.05
    if max_diff < tolerance:
        print("\n[PASS] Validation successful within tolerance.")
    else:
        print("\n[FAIL] Validation exceeded tolerance.")

    # 5. Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_result['imputed_value'], label='Python Imputed', marker='o', linestyle='None')
    plt.plot(df_result['r_imputed'], label='R Imputed', marker='x', linestyle='None')
    plt.title("Comparison of Imputed Values: Python vs R (NADA)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_plot.png")
    print("Saved comparison_plot.png")

if __name__ == "__main__":
    run_validation()
