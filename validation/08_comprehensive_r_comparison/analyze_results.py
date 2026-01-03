import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze():
    df = pd.read_csv("suite_results.csv")

    # 1. Summary Table
    summary = df.groupby(['Dist', 'Limit'])['MAE'].mean().reset_index()
    print("\n--- Mean MAE by Scenario ---")
    print(summary)

    # 2. Heatmap: MAE by N vs Censoring Level (for Lognormal Single)
    # Filter for Lognormal Single
    subset = df[(df['Dist'] == 'lognormal') & (df['Limit'] == 'single')].copy()

    # Extract params from Scenario filename if needed, but we didn't save them explicit columns
    # We need to parse filename again or rely on order?
    # Better to parse from filename column

    def extract_params(row):
        # benchmark_lognormal_n200_cens0.2_single.csv
        parts = row['Scenario'].split('_')
        n = int(parts[2].replace('n', ''))
        cens = float(parts[3].replace('cens', ''))
        return pd.Series({'N': n, 'Censoring': cens})

    df[['N', 'Censoring']] = df.apply(extract_params, axis=1)

    # Heatmap Data
    heatmap_data = df[df['Dist'] == 'lognormal'].pivot_table(
        index='Censoring', columns='N', values='MAE', aggfunc='mean'
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".4f")
    plt.title("Mean MAE: Lognormal Distribution")
    plt.ylabel("Censoring Level")
    plt.xlabel("Sample Size (N)")
    plt.savefig("heatmap_mae.png")
    print("Saved heatmap_mae.png")

    # 3. Boxplot: Bias by Limit Type
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Limit', y='MAE', hue='Dist')
    plt.title("Distribution of Errors by Limit Type")
    plt.savefig("boxplot_mae.png")
    print("Saved boxplot_mae.png")

if __name__ == "__main__":
    analyze()
