# src/model.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
import pandas as pd
import numpy as np
import io

# -------------------- CTR Classifier --------------------

def train_ctr_classifier(df):
    features = ["Impressions", "Clicks", "Conversions", "Cost"]
    X = df[features]
    y = df["High_CTR"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    print("📈 Best Parameters for CTR Classifier:", grid.best_params_)
    return best_model, report, y_test, y_pred

# -------------------- Conversion Regressor --------------------

def train_conversion_model(df):
    features = ["Impressions", "Clicks", "Cost"]
    X = df[features]
    y = df["Conversions"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10, 20]
    }

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"✅ Conversion model saved. MSE: {mse:.5f}")
    return best_model, mse

# -------------------- Save & Load --------------------

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

# -------------------- Prediction Functions --------------------

def predict_ctr(model, input_data):
    # Ensure input_data is an array or series
    if isinstance(input_data, dict):
        input_data = np.array(list(input_data.values())).reshape(1, -1)
    elif isinstance(input_data, pd.Series):
        input_data = input_data.values.reshape(1, -1)
    elif isinstance(input_data, pd.DataFrame):
        input_data = input_data.values  # Already 2D
    else:
        input_data = np.array(input_data).reshape(1, -1)

    return model.predict(input_data)[0]

def predict_conversion(model, input_data):
    # Ensure input_data is an array or series
    if isinstance(input_data, dict):
        input_data = np.array(list(input_data.values())).reshape(1, -1)
    elif isinstance(input_data, pd.Series):
        input_data = input_data.values.reshape(1, -1)
    elif isinstance(input_data, pd.DataFrame):
        input_data = input_data.values  # Already 2D
    else:
        input_data = np.array(input_data).reshape(1, -1)

    return model.predict(input_data)[0]

# -------------------- Feature Importance --------------------

def get_feature_importance_df(model, feature_names):
    importances = model.feature_importances_
    return pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

def plot_feature_importance(model, feature_names, title="Feature Importance", return_fig=False, as_pie=False):
    df = get_feature_importance_df(model, feature_names)

    fig, ax = plt.subplots(figsize=(8, 5))

    if as_pie:
        ax.pie(df["Importance"], labels=df["Feature"], autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        plt.title(f"{title} (Pie Chart)")
    else:
        sns.barplot(x="Importance", y="Feature", data=df, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()

# -------------------- Export Feature Importance --------------------

def export_feature_importance_csv(model, feature_names):
    df = get_feature_importance_df(model, feature_names)
    return df.to_csv(index=False).encode('utf-8')

def export_feature_importance_image(model, feature_names, as_pie=False):
    fig = plot_feature_importance(model, feature_names, return_fig=True, as_pie=as_pie)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# -------------------- Confusion Matrix --------------------

def plot_confusion_matrix(y_true, y_pred, class_labels=None):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Display confusion matrix with labels if provided
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return fig
