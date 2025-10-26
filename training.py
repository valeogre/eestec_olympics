import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from math import radians, sin, cos, sqrt, atan2
import joblib
import os

# ================================
# Helper Functions
# ================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance (in km) between two coordinates."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def preprocess(df, label_encoders=None, training=True):
    """Preprocess the dataset: compute features and encode categorical variables."""
    print("Preparing dataframe...")

    # Combine date + time if available
    df["trans_datetime"] = pd.to_datetime(
        df["trans_date"] + " " + df["trans_time"], errors="coerce"
    )
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Compute age safely
    df["age"] = ((df["trans_datetime"] - df["dob"]).dt.days / 365.25).fillna(0)

    # Extract transaction hour
    df["hour"] = df["trans_datetime"].dt.hour.fillna(0)

    # Compute geographical distance
    df["distance_km"] = df.apply(
        lambda x: haversine(x["lat"], x["long"], x["merch_lat"], x["merch_long"]), axis=1
    )

    # Encode categorical columns
    cat_cols = ["category", "merchant", "state", "gender", "job"]
    if training:
        print("Fitting label encoders...")
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            le = label_encoders[col]
            df[col] = df[col].map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )

    # Replace NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, label_encoders


# ================================
# Main Training Script
# ================================

def main():
    csv_path = "hackathon-labeled-train.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep="|")

    # Preprocess
    df, label_encoders = preprocess(df, training=True)

    # Feature selection
    feature_cols = [
        "amt", "age", "hour", "distance_km",
        "category", "merchant", "state", "gender", "job"
    ]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["is_fraud"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/test split: 80.00%/20.00%")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Model parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(50)
        ],
    )

    # Evaluate model
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\nAUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))

    # Save model and encoders
    print("\nSaving model and encoders...")
    joblib.dump(model, "fraud_model.pkl")
    for col, le in label_encoders.items():
        joblib.dump(le, f"le_{col}.pkl")

    print("✅ Model training complete. Files saved: fraud_model.pkl + le_*.pkl")


if __name__ == "__main__":
    main()
