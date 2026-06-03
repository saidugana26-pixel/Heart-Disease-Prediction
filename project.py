import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

import joblib


# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_data():

    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(
        "heart.csv",
        header=None,
        names=columns,
        na_values="?"
    )

    # Remove missing values
    df.dropna(inplace=True)

    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)

    print("\nDataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)

    return df


# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess_data(df):

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    print("\nScaler saved successfully")

    return X_scaled, y


# -----------------------------
# 3. Build Model
# -----------------------------
def build_model():

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    return model


# -----------------------------
# 4. Train Model
# -----------------------------
def train_model(X_scaled, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model()

    model.fit(X_train, y_train)

    joblib.dump(model, "heart_model.pkl")

    print("\nModel saved successfully")

    return model, X_test, y_test


# -----------------------------
# 5. Evaluate Model
# -----------------------------
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


# -----------------------------
# 6. Feature Importance
# -----------------------------
def explain_features(model, X_test, y_test):

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42
    )

    importance = result.importances_mean

    print("\nFeature Importance:")
    print(importance)

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(len(importance)),
        importance
    )

    plt.title("Permutation Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")

    plt.show()


# -----------------------------
# 7. Prediction
# -----------------------------
def predict_sample(sample):

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("heart_model.pkl")

    sample = np.array(sample).reshape(1, -1)

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    return prediction[0]


# -----------------------------
# 8. Main Function
# -----------------------------
def main():

    df = load_data()

    X_scaled, y = preprocess_data(df)

    model, X_test, y_test = train_model(
        X_scaled,
        y
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    explain_features(
        model,
        X_test,
        y_test
    )

    sample = [
        63, 1, 3, 145, 233,
        1, 0, 150, 0, 2.3,
        0, 0, 1
    ]

    pred = predict_sample(sample)

    if pred == 1:
        print("\nPrediction: Heart Disease Detected")
    else:
        print("\nPrediction: No Heart Disease Detected")


if __name__ == "__main__":
    main()