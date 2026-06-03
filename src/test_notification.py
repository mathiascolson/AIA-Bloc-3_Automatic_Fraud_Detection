import json

from src.notification import send_fraud_alert_notification


def main() -> None:
    fake_alert = {
        "transaction_id": "test_transaction_001",
        "fraud_probability": 0.9876,
        "fraud_alert_threshold": 0.90,

        "amount": 412.75,
        "currency": "USD",
        "merchant": "fraud_Kilback LLC",
        "category": "shopping_net",
        "customer_id": "customer_hash_abc123",

        "city": "Phoenix",
        "state": "AZ",
        "customer_lat": 33.4484,
        "customer_long": -112.0740,
        "merchant_lat": 36.1699,
        "merchant_long": -115.1398,
        "distance_customer_merchant_km": 411.52,

        "transaction_datetime": "2026-06-03 08:12:45",
    }

    result = send_fraud_alert_notification(fake_alert)

    print("Notification test result:")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()