from __future__ import annotations

import requests
import pandas as pd


def build_discord_fraud_alert_message(alert: dict) -> str:
    """
    Construit un message Discord pour une alerte fraude.

    Le message évite les données personnelles directes.
    """

    trans_num = alert.get("trans_num")
    amount = alert.get("amt")
    merchant = alert.get("merchant")
    category = alert.get("category")
    city = alert.get("city")
    state = alert.get("state")
    transaction_time = alert.get("transaction_time")
    fraud_probability = alert.get("fraud_probability")
    fraud_alert_threshold = alert.get("fraud_alert_threshold")
    model_name = alert.get("model_name")
    model_alias = alert.get("model_alias")

    return (
        "🚨 **Fraud alert detected**\n"
        f"- Transaction: `{trans_num}`\n"
        f"- Fraud probability: `{fraud_probability:.4f}`\n"
        f"- Alert threshold: `{fraud_alert_threshold:.4f}`\n"
        f"- Amount: `{amount}`\n"
        f"- Merchant: `{merchant}`\n"
        f"- Category: `{category}`\n"
        f"- Location: `{city}, {state}`\n"
        f"- Transaction time: `{transaction_time}`\n"
        f"- Model: `{model_name}`\n"
        f"- Alias: `{model_alias}`"
    )


def send_discord_message(
    webhook_url: str,
    message: str,
) -> None:
    """
    Envoie un message Discord via webhook.
    """

    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is missing.")

    response = requests.post(
        webhook_url,
        json={"content": message},
        timeout=30,
    )

    response.raise_for_status()


def send_discord_fraud_alerts(
    alerts: pd.DataFrame,
    webhook_url: str,
) -> int:
    """
    Envoie une notification Discord pour chaque nouvelle alerte fraude.

    Retourne le nombre de notifications envoyées.
    """

    if alerts.empty:
        print("[DISCORD] Aucune alerte à notifier.")
        return 0

    sent_count = 0

    for alert in alerts.to_dict(orient="records"):
        message = build_discord_fraud_alert_message(alert)

        send_discord_message(
            webhook_url=webhook_url,
            message=message,
        )

        sent_count += 1

    print(f"[DISCORD] Notifications envoyées : {sent_count}")

    return sent_count