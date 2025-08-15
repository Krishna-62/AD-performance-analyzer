# src/recommendation_engine.py

import pandas as pd

def generate_recommendations(df):
    recommendations = []

    for idx, row in df.iterrows():
        suggestion = []

        if row["CTR"] < 0.02:
            suggestion.append("📉 Improve Ad Creative")
        if row["CPC"] > 2.0:
            suggestion.append("💸 Lower Bid or Targeting")
        if row["CPA"] > 10.0:
            suggestion.append("📊 Optimize Landing Page")
        if row["ConversionRate"] < 0.05:
            suggestion.append("🚀 A/B Test CTA")

        if not suggestion:
            suggestion.append("✅ Performing Well")

        recommendations.append(", ".join(suggestion))

    return pd.DataFrame({
        "Campaign/Ad": df.get("Campaign", df.index),
        "Suggestions": recommendations
    })
