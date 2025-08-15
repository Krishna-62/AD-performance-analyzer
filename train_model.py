import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_ctr_classifier(data_path="data/ad_data_1.csv", model_path="models/ctr_classifier.pkl"):
    # Load dataset
    data = pd.read_csv(data_path)

    # Safe Feature Engineering
    data["CTR"] = data["Clicks"] / data["Impressions"]
    data["CPC"] = data["Cost"] / data["Clicks"].replace(0, pd.NA)
    data["CPA"] = data["Cost"] / data["Conversions"].replace(0, pd.NA)
    data["ConversionRate"] = data["Conversions"] / data["Clicks"].replace(0, pd.NA)

    # Clean problematic values
    data.replace([pd.NA, float("inf"), float("-inf")], pd.NA, inplace=True)
    data.dropna(inplace=True)

    # Create target column: High CTR (1 if above median, 0 otherwise)
    threshold = data["CTR"].median()
    data["High_CTR"] = (data["CTR"] >= threshold).astype(int)

    # Features and labels
    X = data[["CTR", "CPC", "CPA", "ConversionRate"]]
    y = data["High_CTR"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and save metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open("models/ctr_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(classification_report(y_test, model.predict(X_test)))

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to: {model_path}")

if __name__ == "__main__":
    train_and_save_ctr_classifier()
