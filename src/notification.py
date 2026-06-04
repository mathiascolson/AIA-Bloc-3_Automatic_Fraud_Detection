from typing import Any
import requests
from src.config import get_settings

def build_fraud_alert_message(alert_data: dict[str, Any]) -> str:
    transaction_id = alert_data.get("transaction_id", "unknown")
    fraud_probability = float(alert_data.get("fraud_probability", 0.0))
    alert_threshold = float(alert_data.get("fraud_alert_threshold", 0.0))

    amount = alert_data.get("amount", "unknown")
    currency = alert_data.get("currency", "USD")
    merchant = alert_data.get("merchant", "unknown")
    category = alert_data.get("category", "unknown")
    customer_id = alert_data.get("customer_id", "unknown")

    city = alert_data.get("city", "unknown")
    state = alert_data.get("state", "unknown")
    customer_lat = alert_data.get("customer_lat", "unknown")
    customer_long = alert_data.get("customer_long", "unknown")
    merchant_lat = alert_data.get("merchant_lat", "unknown")
    merchant_long = alert_data.get("merchant_long", "unknown")
    distance_km = alert_data.get("distance_customer_merchant_km")

    transaction_datetime = alert_data.get("transaction_datetime", "unknown")

    distance_text = (
        f"{float(distance_km):.2f} km"
        if distance_km is not None
        else "unknown"
    )

    return (
        "🚨 **Fraud alert detected**\n\n"
        f"**Transaction ID:** `{transaction_id}`\n"
        f"**Fraud probability:** `{fraud_probability:.4f}`\n"
        f"**Alert threshold:** `{alert_threshold:.2f}`\n\n"
        f"**Amount:** `{amount} {currency}`\n"
        f"**Merchant:** `{merchant}`\n"
        f"**Category:** `{category}`\n"
        f"**Customer ID:** `{customer_id}`\n\n"
        f"**Customer location:** `{city}, {state}`\n"
        f"**Customer coordinates:** `{customer_lat}, {customer_long}`\n"
        f"**Merchant coordinates:** `{merchant_lat}, {merchant_long}`\n"
        f"**Customer ↔ merchant distance:** `{distance_text}`\n\n"
        f"**Transaction datetime:** `{transaction_datetime}`"
    )


def send_discord_notification(message: str) -> dict[str, Any]:
    settings = get_settings()
    webhook_url = settings.discord_webhook_url

    if not webhook_url:
        return {
            "notification_sent": False,
            "notification_channel": "discord",
            "notification_status": "skipped",
            "reason": "DISCORD_WEBHOOK_URL is missing",
        }

    response = requests.post(
        webhook_url,
        json={"content": message},
        timeout=10,
    )

    if response.status_code not in (200, 204):
        return {
            "notification_sent": False,
            "notification_channel": "discord",
            "notification_status": "failed",
            "status_code": response.status_code,
            "response_text": response.text,
        }

    return {
        "notification_sent": True,
        "notification_channel": "discord",
        "notification_status": "sent",
        "status_code": response.status_code,
    }


def send_fraud_alert_notification(alert_data: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    channel = settings.notification_channel.lower()

    if channel != "discord":
        return {
            "notification_sent": False,
            "notification_channel": channel,
            "notification_status": "skipped",
            "reason": f"Unsupported notification channel: {channel}",
        }

    message = build_fraud_alert_message(alert_data)
    return send_discord_notification(message)