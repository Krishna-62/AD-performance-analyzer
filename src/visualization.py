# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_distribution(df, column, title="Distribution"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column].dropna(), kde=True, ax=ax)  # Drop missing values for better visualization
    ax.set_title(title)
    return fig

def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("📊 Correlation Matrix")
    return fig

# ✅ Add this helper function to show visuals
def show_visuals(df):
    st.subheader("📊 Feature Distributions")
    
    # Feature distributions for selected columns
    for col in ["Impressions", "Clicks", "Cost", "Conversions"]:
        st.pyplot(plot_distribution(df, col, f"Distribution of {col}"))
        st.markdown("<br>", unsafe_allow_html=True)  # Add some space between charts

    st.subheader("📈 Correlation Heatmap")
    st.pyplot(plot_correlation_matrix(df))
