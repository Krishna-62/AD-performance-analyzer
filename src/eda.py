# src/eda.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(data):
    """
    Plots distributions of numerical features in the dataset.
    """
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_cols].hist(bins=30, figsize=(12, 10))
    plt.tight_layout()
    plt.show()

def plot_correlations(data):
    """
    Plots a heatmap of the correlation matrix.
    """
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
