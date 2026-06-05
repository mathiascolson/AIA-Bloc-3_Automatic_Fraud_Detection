from __future__ import annotations

import requests
import pandas as pd


def _get_value(alert: dict, *keys: str, default=None):
    """
    Récupère la première valeur disponible parmi plusieurs clés possibles.
    Utile pour rester compatible avec différents noms de colonnes.
    """

    for key in keys:
        value = alert.get(key)

        if value is not None and not pd.isna(value):
            return value

    return default


def _format_probability(value) -> str:
    """
    Formate une probabilité de fraude sur 4 décimales.
    """

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_threshold(value) -> str:
    """
    Formate le seuil d'alerte sur 2 décimales.
    """

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_amount(value) -> str:
    """
    Formate le montant en USD.
    """

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2f} USD"
    except (TypeError, ValueError):
        return "N/A"


def _format_coordinates(lat, lon, decimals: int = 6) -> str:
    """
    Formate une paire de coordonnées géographiques.
    """

    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return "N/A"

    try:
        return f"{float(lat):.{decimals}f}, {float(lon):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_distance(value) -> str:
    """
    Formate la distance client / marchand en kilomètres.
    """

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2f} km"
    except (TypeError, ValueError):
        return "N/A"


def _format_datetime(value) -> str:
    """
    Formate le datetime de transaction sans perdre d'information.
    """

    if value is None or pd.isna(value):
        return "N/A"

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def build_discord_fraud_alert_message(alert: dict) -> str:
    """
    Construit le message Discord détaillé pour une alerte fraude.

    Le format reprend les informations métier utiles :
    - transaction
    - score de fraude
    - seuil d'alerte
    - montant
    - marchand
    - catégorie
    - client
    - localisation client
    - coordonnées client
    - coordonnées marchand
    - distance client / marchand
    - datetime transaction
    """

    trans_num = _get_value(alert, "trans_num", "transaction_id", default="N/A")

    fraud_probability = _get_value(
        alert,
        "fraud_probability",
        "fraud_proba",
        "prediction_probability",
        default=None,
    )

    fraud_alert_threshold = _get_value(
        alert,
        "fraud_alert_threshold",
        "alert_threshold",
        "threshold",
        default=None,
    )

    amount = _get_value(alert, "amt", "amount", default=None)
    merchant = _get_value(alert, "merchant", default="N/A")
    category = _get_value(alert, "category", default="N/A")

    customer_id = _get_value(
        alert,
        "cc_num",
        "customer_id",
        default="N/A",
    )

    city = _get_value(alert, "city", default=None)
    state = _get_value(alert, "state", default=None)

    customer_lat = _get_value(alert, "lat", "customer_lat", default=None)
    customer_lon = _get_value(alert, "long", "lon", "customer_long", "customer_lon", default=None)

    merchant_lat = _get_value(alert, "merch_lat", "merchant_lat", default=None)
    merchant_lon = _get_value(alert, "merch_long", "merchant_long", "merchant_lon", default=None)

    distance_km = _get_value(
        alert,
        "distance_km",
        "customer_merchant_distance_km",
        "distance_customer_merchant_km",
        default=None,
    )

    transaction_datetime = _get_value(
        alert,
        "trans_date_trans_time",
        "transaction_datetime",
        "transaction_time",
        default=None,
    )

    if city and state:
        customer_location = f"{city}, {state}"
    elif city:
        customer_location = str(city)
    elif state:
        customer_location = str(state)
    else:
        customer_location = "N/A"

    return (
        "🚨 Fraud alert detected\n\n"
        f"Transaction ID: {trans_num}\n"
        f"Fraud probability: {_format_probability(fraud_probability)}\n"
        f"Alert threshold: {_format_threshold(fraud_alert_threshold)}\n\n"
        f"Amount: {_format_amount(amount)}\n"
        f"Merchant: {merchant}\n"
        f"Category: {category}\n"
        f"Customer ID: {customer_id}\n\n"
        f"Customer location: {customer_location}\n"
        f"Customer coordinates: {_format_coordinates(customer_lat, customer_lon, decimals=4)}\n"
        f"Merchant coordinates: {_format_coordinates(merchant_lat, merchant_lon, decimals=6)}\n"
        f"Customer ↔ merchant distance: {_format_distance(distance_km)}\n\n"
        f"Transaction datetime: {_format_datetime(transaction_datetime)}"
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