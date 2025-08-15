import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def clean_data(data):
    # Drop missing values
    data = data.dropna()

    # Compute CTR
    data["CTR"] = data["Clicks"] / data["Impressions"]

    # Add binary classification target
    threshold = 0.05
    data["High_CTR"] = (data["CTR"] >= threshold).astype(int)

    return data


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def classify_ctr(data, threshold=0.05):
    data["High_CTR"] = (data["CTR"] >= threshold).astype(int)

    features = ["Impressions", "Clicks", "Conversions", "Cost"]
    target = "High_CTR"

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return data
