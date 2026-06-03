from io import StringIO

import pandas as pd
import requests

from src.config import settings


def read_pandas_split_json(response_json) -> pd.DataFrame:
    if isinstance(response_json, str):
        return pd.read_json(StringIO(response_json), orient="split")

    raise TypeError(
        f"Expected response_json to be a string, got {type(response_json)}"
    )


def normalize_realtime_transaction(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("The real-time API returned an empty DataFrame.")

    row = df.iloc[0].copy()

    if "current_time" not in row:
        raise ValueError("Missing expected column from real-time API: current_time")

    current_time = pd.to_datetime(row["current_time"])

    payload = {
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
    }

    if "trans_num" in row:
        payload["trans_num"] = str(row["trans_num"])

    return payload


def main() -> None:
    payment_url = (
        settings.payment_api_url.rstrip("/")
        + settings.payment_api_current_transactions_endpoint
    )

    prediction_url = (
        settings.prediction_api_url.rstrip("/")
        + settings.prediction_api_predict_endpoint
    )

    print("Calling payment API:", payment_url)

    payment_response = requests.get(payment_url, timeout=30)
    print("Payment API status:", payment_response.status_code)
    payment_response.raise_for_status()

    df = read_pandas_split_json(payment_response.json())

    print("\nRaw real-time transaction:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    payload = normalize_realtime_transaction(df)

    print("\nPayload sent to prediction API:")
    print(payload)

    print("\nCalling prediction API:", prediction_url)

    prediction_response = requests.post(
        prediction_url,
        json=payload,
        timeout=30,
    )

    print("Prediction API status:", prediction_response.status_code)
    print("Prediction response:")
    print(prediction_response.json())

    prediction_response.raise_for_status()


if __name__ == "__main__":
    main()