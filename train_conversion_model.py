import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_conversion_model(data_path="data/ad_data_1.csv", model_path="models/conversion_model.pkl"):
    df = pd.read_csv(data_path)

    # Feature Engineering
    df["CTR"] = df["Clicks"] / df["Impressions"]
    df["CPC"] = df["Cost"] / df["Clicks"].replace(0, pd.NA)
    df["CPA"] = df["Cost"] / df["Conversions"].replace(0, pd.NA)
    df["ConversionRate"] = df["Conversions"] / df["Clicks"].replace(0, pd.NA)

    # Clean data
    df = df.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    df.dropna(inplace=True)

    # Define features & target
    X = df[["CTR", "CPC", "CPA"]]
    y = df["ConversionRate"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate performance
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Save metrics
    with open("models/conversion_metrics.txt", "w") as f:
        f.write(f"MSE: {mse:.5f}")

    print(f"✅ Conversion model saved. MSE: {mse:.5f}")

if __name__ == "__main__":
    train_conversion_model()
