import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data, scale_features
import joblib
import logging

logging.basicConfig(filename="../logs/project.log", level=logging.INFO)


def train_model():
    logging.info("Loading data...")
    train = pd.read_csv("../data/train.csv", parse_dates=["Date"])
    train = preprocess_data(train)

    X = train[
        [
            "Store",
            "Promo",
            "Year",
            "Month",
            "WeekOfYear",
            "DayOfWeek",
            "CompetitionDistance",
        ]
    ]
    y = train["Sales"]

    X_scaled, scaler = scale_features(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    logging.info("Training model...")
    pipeline = Pipeline(
        [("model", RandomForestRegressor(n_estimators=100, random_state=42))]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    logging.info(f"Validation RMSE: {rmse}")

    joblib.dump(pipeline, "../models/rf_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    logging.info("Model saved.")


if __name__ == "__main__":
    train_model()
