import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# ✅ Fix path to access src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Import project modules
from src.data_loader import load_data, clean_data, classify_ctr
from src.kpi_calculator import calculate_kpis
from src.model import (
    train_ctr_classifier,
    predict_ctr,
    train_conversion_model,
    predict_conversion,
    plot_feature_importance,
    export_feature_importance_csv,
    export_feature_importance_image
)
from src.recommendation_engine import generate_recommendations
from src.visualization import show_visuals

# ✅ Streamlit Page Config
st.set_page_config(page_title="Ad Performance Analyzer", layout="wide")
st.title("📊 Ad Performance Analyzer Dashboard")

# ✅ Theme Toggle
theme = st.radio("Select Theme", ["Light", "Dark"], horizontal=True, key="theme_toggle")
if theme == "Dark":
    st.markdown('<style>body { background-color: #2e2e2e; color: white; }</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body { background-color: white; color: black; }</style>', unsafe_allow_html=True)

# ✅ Upload CSV
uploaded_file = st.file_uploader("Upload your ad data CSV", type=["csv"])

if uploaded_file:
    # ✅ Load & clean
    df = load_data(uploaded_file)
    df = clean_data(df)

    # ✅ KPIs & CTR classification
    df = calculate_kpis(df)
    df = classify_ctr(df)

    # ✅ Train & Predict CTR
    ctr_model, ctr_report, y_true, y_pred = train_ctr_classifier(df)
    df["Predicted_CTR_High"] = df.apply(
        lambda row: predict_ctr(ctr_model, [row["Impressions"], row["Clicks"], row["Conversions"], row["Cost"]]), axis=1
    )

    # ✅ Train & Predict Conversions
    conv_model, mse = train_conversion_model(df)
    df["Predicted_Conversions"] = df.apply(
        lambda row: predict_conversion(conv_model, [row["Impressions"], row["Clicks"], row["Cost"]]), axis=1
    )

    # ✅ Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📁 Data", 
        "📈 KPIs", 
        "🧠 Model Insights", 
        "📊 Feature Distributions", 
        "📌 Campaign Recommendations", 
        "🔮 Real-Time Predictions",
        "📊 Model Performance Metrics"
    ])

    with tab1:
        st.subheader("Preview Data")
        st.dataframe(df.head(10))

    with tab2:
        st.subheader("Key Performance Indicators")
        kpi_cols = ["CTR", "CPC", "CPA", "ConversionRate"]
        st.dataframe(df[kpi_cols].describe())

    with tab3:
        st.subheader("CTR Classification Report")
        st.text(ctr_report)

        st.subheader("Conversion Model MSE")
        st.metric("Mean Squared Error", f"{mse:.5f}")

        # ------------------ Feature Importance: CTR ------------------
        st.subheader("🔍 Feature Importance for CTR Model")
        ctr_feature_names = ["Impressions", "Clicks", "Conversions", "Cost"]

        ctr_chart_type = st.radio("Choose chart type for CTR", ["Bar Chart", "Pie Chart"], horizontal=True, key="ctr_chart")
        fig_ctr = plot_feature_importance(
            ctr_model,
            ctr_feature_names,
            return_fig=True,
            as_pie=(ctr_chart_type == "Pie Chart")
        )
        st.pyplot(fig_ctr)

        ctr_csv = export_feature_importance_csv(ctr_model, ctr_feature_names)
        st.download_button("📥 Download CTR Importance as CSV", data=ctr_csv, file_name="ctr_feature_importance.csv", mime="text/csv")

        ctr_img = export_feature_importance_image(ctr_model, ctr_feature_names, as_pie=(ctr_chart_type == "Pie Chart"))
        st.download_button("🖼️ Download CTR Importance as Image", data=ctr_img, file_name="ctr_importance.png", mime="image/png")

        st.markdown("---")

        # ------------------ Feature Importance: Conversion ------------------
        st.subheader("🔍 Feature Importance for Conversion Model")
        conv_feature_names = ["Impressions", "Clicks", "Cost"]

        conv_chart_type = st.radio("Choose chart type for Conversion", ["Bar Chart", "Pie Chart"], horizontal=True, key="conv_chart")
        fig_conv = plot_feature_importance(
            conv_model,
            conv_feature_names,
            return_fig=True,
            as_pie=(conv_chart_type == "Pie Chart")
        )
        st.pyplot(fig_conv)

        conv_csv = export_feature_importance_csv(conv_model, conv_feature_names)
        st.download_button("📥 Download Conversion Importance as CSV", data=conv_csv, file_name="conversion_feature_importance.csv", mime="text/csv")

        conv_img = export_feature_importance_image(conv_model, conv_feature_names, as_pie=(conv_chart_type == "Pie Chart"))
        st.download_button("🖼️ Download Conversion Importance as Image", data=conv_img, file_name="conversion_importance.png", mime="image/png")

        # ------------------ Confusion Matrix for CTR ------------------
        st.subheader("📊 CTR Classification Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot(cmap='Blues')
        st.pyplot(cm_display.figure_)

        st.markdown("---")

        # ------------------ Predicted vs Actual Conversion ------------------
        st.subheader("📊 Predicted vs Actual Conversion")
        fig_conv_scatter = plt.figure(figsize=(6, 4))
        plt.scatter(df["Predicted_Conversions"], df["Conversions"], alpha=0.6, color='blue')
        plt.plot(
            [df["Predicted_Conversions"].min(), df["Predicted_Conversions"].max()],
            [df["Predicted_Conversions"].min(), df["Predicted_Conversions"].max()],
            color="red", linestyle="--"
        )
        plt.title("Predicted vs Actual Conversions")
        plt.xlabel("Predicted Conversions")
        plt.ylabel("Actual Conversions")
        st.pyplot(fig_conv_scatter)

    with tab4:
        show_visuals(df)

    with tab5:
        st.subheader("📌 Campaign Recommendations")
        recs = generate_recommendations(df)
        st.table(recs)

    with tab6:
        st.subheader("🔮 Real-Time Predictions")

        st.markdown("Make real-time predictions below by entering new values:")

        impressions_input = st.number_input("Impressions", min_value=0, value=1000)
        clicks_input = st.number_input("Clicks", min_value=0, value=100)
        conversions_input = st.number_input("Conversions", min_value=0, value=10)
        cost_input = st.number_input("Cost", min_value=0.0, value=100.0, step=1.0)

        if st.button("Predict CTR & Conversions"):
            ctr_pred = predict_ctr(ctr_model, [impressions_input, clicks_input, conversions_input, cost_input])
            conv_pred = predict_conversion(conv_model, [impressions_input, clicks_input, cost_input])

            st.success(f"📈 Predicted CTR High: {'Yes' if ctr_pred == 1 else 'No'}")
            st.success(f"🔢 Predicted Conversions: {conv_pred:.2f}")

        st.markdown("---")
        st.subheader("📁 Download Full Processed Dataset")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="ad_analysis_output.csv", mime='text/csv')

    with tab7:
        st.subheader("📊 Model Performance Metrics")

        # Metrics for CTR Model
        st.subheader("CTR Model Performance")
        ctr_accuracy = np.mean(ctr_model.predict(df[["Impressions", "Clicks", "Conversions", "Cost"]]) == (df["Predicted_CTR_High"] == 1))
        st.write("Accuracy: {:.2f}%".format(ctr_accuracy * 100))

        # Metrics for Conversion Model
        st.subheader("Conversion Model Performance")
        mse_conversion = mse  # Already calculated during model training
        st.write("Mean Squared Error (MSE): {:.2f}".format(mse_conversion))

