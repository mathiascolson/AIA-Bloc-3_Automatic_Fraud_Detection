import json
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text

from src.config import settings
from src.realtime_pipeline import insert_alert_if_needed


def get_database_engine():
    if not settings.fraud_database_url:
        raise ValueError("FRAUD_DATABASE_URL is missing in .env")

    return create_engine(settings.fraud_database_url)


def get_latest_transaction() -> dict[str, Any]:
    engine = get_database_engine()

    query = text(
        """
        SELECT
            transaction_id,
            transaction_datetime,
            amount,
            currency,
            merchant_id,
            merchant_category,
            customer_id,
            raw_payload
        FROM transactions
        ORDER BY inserted_at DESC
        LIMIT 1;
        """
    )

    with engine.begin() as connection:
        row = connection.execute(query).mappings().first()

    if row is None:
        raise ValueError(
            "No transaction found in NeonDB. Run one realtime pipeline test first."
        )

    return dict(row)


def parse_raw_payload(raw_payload: Any) -> dict[str, Any]:
    if raw_payload is None:
        return {}

    if isinstance(raw_payload, dict):
        return raw_payload

    if isinstance(raw_payload, str):
        return json.loads(raw_payload)

    return dict(raw_payload)


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


def get_distance_customer_merchant_km(raw_payload: dict[str, Any]) -> float | None:
    existing_distance = raw_payload.get("distance_customer_merchant_km")

    if existing_distance is not None:
        return float(existing_distance)

    required_fields = ["lat", "long", "merch_lat", "merch_long"]

    missing_fields = [
        field
        for field in required_fields
        if raw_payload.get(field) is None
    ]

    if missing_fields:
        print(
            "Distance cannot be computed. Missing fields:",
            ", ".join(missing_fields),
        )
        return None

    return haversine_distance_km(
        lat1=float(raw_payload["lat"]),
        lon1=float(raw_payload["long"]),
        lat2=float(raw_payload["merch_lat"]),
        lon2=float(raw_payload["merch_long"]),
    )


def main() -> None:
    transaction = get_latest_transaction()
    raw_payload = parse_raw_payload(transaction.get("raw_payload"))

    transaction_datetime = transaction["transaction_datetime"]

    if isinstance(transaction_datetime, datetime):
        transaction_datetime_value = transaction_datetime.isoformat()
    else:
        transaction_datetime_value = str(transaction_datetime)

    distance_customer_merchant_km = get_distance_customer_merchant_km(raw_payload)

    prediction_payload = {
        "trans_date_trans_time": transaction_datetime_value,
        "merchant": transaction["merchant_id"],
        "category": transaction["merchant_category"],
        "amt": float(transaction["amount"]),
        "city": raw_payload.get("city", "unknown"),
        "state": raw_payload.get("state", "unknown"),
        "lat": raw_payload.get("lat", "unknown"),
        "long": raw_payload.get("long", "unknown"),
        "merch_lat": raw_payload.get("merch_lat", "unknown"),
        "merch_long": raw_payload.get("merch_long", "unknown"),
        "distance_customer_merchant_km": distance_customer_merchant_km,
    }

    db_transaction = {
        "currency": transaction["currency"],
        "customer_id": transaction["customer_id"],
    }

    forced_prediction_response = {
        "fraud_probability": 0.9876,
        "fraud_alert_threshold": 0.90,
        "is_fraud_predicted": 1,
    }

    alert_created = insert_alert_if_needed(
        transaction_id=transaction["transaction_id"],
        prediction_response=forced_prediction_response,
        prediction_payload=prediction_payload,
        db_transaction=db_transaction,
    )

    result = {
        "alert_created": alert_created,
        "transaction_id": transaction["transaction_id"],
        "fraud_probability": forced_prediction_response["fraud_probability"],
        "fraud_alert_threshold": forced_prediction_response[
            "fraud_alert_threshold"
        ],
        "distance_customer_merchant_km": distance_customer_merchant_km,
    }

    print("Forced alert integration test result:")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()