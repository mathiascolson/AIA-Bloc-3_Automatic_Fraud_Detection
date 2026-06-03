from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd

from src.preprocessing import prepare_features_for_inference


def load_sklearn_model(
    tracking_uri: str,
    model_name: str,
    model_alias: str,
):
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}@{model_alias}"

    print(f"[MLFLOW] Chargement du modèle : {model_uri}")

    return mlflow.sklearn.load_model(model_uri)


def predict_fraud(
    df: pd.DataFrame,
    tracking_uri: str,
    model_name: str,
    model_alias: str,
    fraud_alert_threshold: float,
) -> pd.DataFrame:
    features = prepare_features_for_inference(df)

    model = load_sklearn_model(
        tracking_uri=tracking_uri,
        model_name=model_name,
        model_alias=model_alias,
    )

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Le modèle chargé ne fournit pas de méthode predict_proba()."
        )

    fraud_probability = model.predict_proba(features)[:, 1]
    fraud_prediction = (fraud_probability >= fraud_alert_threshold).astype(int)

    predictions = df.copy()

    predictions["fraud_probability"] = fraud_probability
    predictions["is_fraud_predicted"] = fraud_prediction
    predictions["fraud_alert_threshold"] = fraud_alert_threshold
    predictions["model_name"] = model_name
    predictions["model_alias"] = model_alias

    return predictions