import hashlib
import json
from io import StringIO
from typing import Any

import pandas as pd
import requests
from sqlalchemy import create_engine, text

from src.config import settings
from src.notification import send_fraud_alert_notification


def get_database_engine():
    if not settings.fraud_database_url:
        raise ValueError("FRAUD_DATABASE_URL is missing in .env")

    return create_engine(settings.fraud_database_url)


def build_payment_api_url() -> str:
    return (
        settings.payment_api_url.rstrip("/")
        + settings.payment_api_current_transactions_endpoint
    )


def build_prediction_api_url() -> str:
    return (
        settings.prediction_api_url.rstrip("/")
        + settings.prediction_api_predict_endpoint
    )


def hash_identifier(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    import math

    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c

def read_pandas_split_json(response_json: str) -> pd.DataFrame:
    return pd.read_json(StringIO(response_json), orient="split")


def fetch_current_transaction() -> pd.DataFrame:
    url = build_payment_api_url()

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return read_pandas_split_json(response.json())


def normalize_realtime_transaction(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The real-time API returned an empty DataFrame.")

    row = df.iloc[0].copy()
    
    distance_customer_merchant_km = haversine_distance_km(
        lat1=float(row["lat"]),
        lon1=float(row["long"]),
        lat2=float(row["merch_lat"]),
        lon2=float(row["merch_long"]),
    )

    required_columns = [
        "merchant",
        "category",
        "amt",
        "gender",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "trans_num",
        "merch_lat",
        "merch_long",
        "current_time",
    ]

    missing_columns = [col for col in required_columns if col not in row.index]

    if missing_columns:
        raise ValueError(f"Missing columns from real-time API: {missing_columns}")

    current_time = pd.to_datetime(row["current_time"])

    transaction_id = str(row["trans_num"])

    prediction_payload = {
        "trans_date_trans_time": str(current_time),
        "merchant": str(row["merchant"]),
        "category": str(row["category"]),
        "amt": float(row["amt"]),
        "gender": str(row["gender"]),
        "city": str(row["city"]),
        "state": str(row["state"]),
        "zip": str(row["zip"]),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "city_pop": int(row["city_pop"]),
        "job": str(row["job"]),
        "dob": str(row["dob"]),
        "unix_time": int(current_time.timestamp()),
        "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
        "trans_num": transaction_id,
        "distance_customer_merchant_km": distance_customer_merchant_km,
    }

    db_transaction = {
        "transaction_id": transaction_id,
        "transaction_datetime": current_time.to_pydatetime(),
        "amount": float(row["amt"]),
        "currency": "USD",
        "merchant_id": str(row["merchant"]),
        "merchant_category": str(row["category"]),
        "customer_id": hash_identifier(row.get("cc_num")),
        "country": "US",
        "payment_method": "card",
        "raw_payload": {
            "transaction_id": transaction_id,
            "merchant": str(row["merchant"]),
            "category": str(row["category"]),
            "amount": float(row["amt"]),
            "gender": str(row["gender"]),
            "city": str(row["city"]),
            "state": str(row["state"]),
            "zip": str(row["zip"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "city_pop": int(row["city_pop"]),
            "job": str(row["job"]),
            "dob": str(row["dob"]),
            "merch_lat": float(row["merch_lat"]),
            "merch_long": float(row["merch_long"]),
            "distance_customer_merchant_km": distance_customer_merchant_km,
            "current_time": str(current_time),
            "cc_num_hash": hash_identifier(row.get("cc_num")),
            "source_label_is_fraud": int(row["is_fraud"]) if "is_fraud" in row else None,
        },
    }

    return {
        "transaction_id": transaction_id,
        "prediction_payload": prediction_payload,
        "db_transaction": db_transaction,
    }


def call_prediction_api(prediction_payload: dict[str, Any]) -> dict[str, Any]:
    url = build_prediction_api_url()

    response = requests.post(
        url,
        json=prediction_payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def insert_transaction(db_transaction: dict[str, Any]) -> None:
    engine = get_database_engine()

    query = text(
        """
        INSERT INTO transactions (
            transaction_id,
            transaction_datetime,
            amount,
            currency,
            merchant_id,
            merchant_category,
            customer_id,
            country,
            payment_method,
            raw_payload
        )
        VALUES (
            :transaction_id,
            :transaction_datetime,
            :amount,
            :currency,
            :merchant_id,
            :merchant_category,
            :customer_id,
            :country,
            :payment_method,
            CAST(:raw_payload AS JSONB)
        )
        ON CONFLICT (transaction_id) DO NOTHING;
        """
    )

    params = {
        **db_transaction,
        "raw_payload": json.dumps(db_transaction["raw_payload"]),
    }

    with engine.begin() as connection:
        connection.execute(query, params)


def insert_prediction(
    transaction_id: str,
    prediction_response: dict[str, Any],
) -> None:
    engine = get_database_engine()

    query = text(
        """
        INSERT INTO predictions (
            transaction_id,
            prediction,
            fraud_probability,
            model_name,
            model_version,
            mlflow_run_id
        )
        VALUES (
            :transaction_id,
            :prediction,
            :fraud_probability,
            :model_name,
            :model_version,
            :mlflow_run_id
        )
        ON CONFLICT (transaction_id)
        DO NOTHING;
        """
    )

    params = {
        "transaction_id": transaction_id,
        "prediction": int(prediction_response["is_fraud_predicted"]),
        "fraud_probability": float(prediction_response["fraud_probability"]),
        "model_name": prediction_response.get("model_name"),
        "model_version": prediction_response.get("model_key"),
        "mlflow_run_id": prediction_response.get("run_id"),
    }

    with engine.begin() as connection:
        connection.execute(query, params)


def insert_alert_if_needed(
    transaction_id: str,
    prediction_response: dict[str, Any],
    prediction_payload: dict[str, Any],
    db_transaction: dict[str, Any],
) -> bool:
    is_fraud_predicted = int(prediction_response["is_fraud_predicted"])

    if is_fraud_predicted != 1:
        return False

    fraud_probability = float(prediction_response["fraud_probability"])
    alert_threshold = float(prediction_response["fraud_alert_threshold"])

    engine = get_database_engine()

    insert_query = text(
        """
        INSERT INTO alerts (
            transaction_id,
            fraud_probability,
            alert_threshold,
            notification_channel,
            notification_status
        )
        VALUES (
            :transaction_id,
            :fraud_probability,
            :alert_threshold,
            :notification_channel,
            :notification_status
        )
        ON CONFLICT (transaction_id)
        DO NOTHING
        RETURNING alert_id;
        """
    )

    params = {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "alert_threshold": alert_threshold,
        "notification_channel": settings.notification_channel,
        "notification_status": "created",
    }

    with engine.begin() as connection:
        inserted_alert = connection.execute(
            insert_query,
            params,
        ).mappings().first()

    if inserted_alert is None:
        return False

    notification_result = send_fraud_alert_notification(
        {
            "transaction_id": transaction_id,
            "fraud_probability": fraud_probability,
            "fraud_alert_threshold": alert_threshold,

            "amount": prediction_payload.get("amt"),
            "currency": db_transaction.get("currency"),
            "merchant": prediction_payload.get("merchant"),
            "category": prediction_payload.get("category"),
            "customer_id": db_transaction.get("customer_id"),

            "city": prediction_payload.get("city"),
            "state": prediction_payload.get("state"),
            "customer_lat": prediction_payload.get("lat"),
            "customer_long": prediction_payload.get("long"),
            "merchant_lat": prediction_payload.get("merch_lat"),
            "merchant_long": prediction_payload.get("merch_long"),
            "distance_customer_merchant_km": prediction_payload.get(
                "distance_customer_merchant_km"
            ),

            "transaction_datetime": prediction_payload.get(
                "trans_date_trans_time"
            ),
        }
    )

    notification_status = notification_result.get(
        "notification_status",
        "unknown",
    )

    update_query = text(
        """
        UPDATE alerts
        SET notification_status = :notification_status
        WHERE alert_id = :alert_id;
        """
    )

    with engine.begin() as connection:
        connection.execute(
            update_query,
            {
                "alert_id": inserted_alert["alert_id"],
                "notification_status": notification_status,
            },
        )

    return True


def run_realtime_pipeline_once() -> dict[str, Any]:
    df = fetch_current_transaction()

    normalized = normalize_realtime_transaction(df)

    transaction_id = normalized["transaction_id"]
    prediction_payload = normalized["prediction_payload"]
    db_transaction = normalized["db_transaction"]

    insert_transaction(db_transaction)

    prediction_response = call_prediction_api(prediction_payload)

    insert_prediction(
        transaction_id=transaction_id,
        prediction_response=prediction_response,
    )

    alert_created = insert_alert_if_needed(
        transaction_id=transaction_id,
        prediction_response=prediction_response,
        prediction_payload=prediction_payload,
        db_transaction=db_transaction,
    )

    return {
        "transaction_id": transaction_id,
        "fraud_probability": prediction_response["fraud_probability"],
        "fraud_alert_threshold": prediction_response["fraud_alert_threshold"],
        "is_fraud_predicted": prediction_response["is_fraud_predicted"],
        "model_name": prediction_response.get("model_name"),
        "run_id": prediction_response.get("run_id"),
        "alert_created": alert_created,
    }


if __name__ == "__main__":
    result = run_realtime_pipeline_once()
    print(json.dumps(result, indent=4))