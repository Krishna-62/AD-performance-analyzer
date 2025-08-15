def calculate_kpis(df):
    df = df.copy()
    df["CTR"] = df["Clicks"] / df["Impressions"]
    df["CPC"] = df["Cost"] / df["Clicks"]
    df["CPA"] = df["Cost"] / df["Conversions"]
    df["ConversionRate"] = df["Conversions"] / df["Clicks"]
    return df
