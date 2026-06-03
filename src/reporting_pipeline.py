import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings
from src.s3_utils import upload_file_to_s3


def get_database_engine():
    if not settings.fraud_database_url:
        raise ValueError("FRAUD_DATABASE_URL is missing in .env")

    return create_engine(settings.fraud_database_url)


def get_default_report_date() -> date:
    return date.today() - timedelta(days=1)


def extract_daily_data(report_date: date) -> pd.DataFrame:
    engine = get_database_engine()

    query = text(
        """
        SELECT
            t.transaction_id,
            t.transaction_datetime,
            t.amount,
            t.currency,
            t.merchant_id,
            t.merchant_category,
            t.customer_id,
            t.country,
            t.payment_method,
            t.raw_payload,
            t.inserted_at,

            p.prediction_id,
            p.prediction,
            p.fraud_probability,
            p.model_name,
            p.model_version,
            p.mlflow_run_id,
            p.prediction_datetime,

            a.alert_id,
            a.notification_channel,
            a.notification_status,
            a.notification_datetime

        FROM transactions t
        LEFT JOIN predictions p
            ON t.transaction_id = p.transaction_id
        LEFT JOIN alerts a
            ON t.transaction_id = a.transaction_id
        WHERE DATE(t.transaction_datetime) = :report_date
        ORDER BY t.transaction_datetime ASC;
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {"report_date": report_date},
        )

        rows = result.fetchall()
        columns = result.keys()

    return pd.DataFrame(rows, columns=columns)


def compute_daily_metrics(df: pd.DataFrame, report_date: date) -> dict[str, Any]:
    if df.empty:
        return {
            "report_date": str(report_date),
            "total_transactions": 0,
            "detected_frauds": 0,
            "fraud_rate": 0.0,
            "total_amount": 0.0,
            "fraud_amount": 0.0,
            "average_fraud_probability": 0.0,
            "max_fraud_probability": 0.0,
            "alert_count": 0,
        }

    unique_transactions = df.drop_duplicates(subset=["transaction_id"])

    total_transactions = int(unique_transactions["transaction_id"].nunique())

    detected_frauds = int(
        unique_transactions.loc[
            unique_transactions["prediction"] == 1,
            "transaction_id",
        ].nunique()
    )

    fraud_rate = (
        detected_frauds / total_transactions
        if total_transactions > 0
        else 0.0
    )

    total_amount = float(unique_transactions["amount"].fillna(0).sum())

    fraud_amount = float(
        unique_transactions.loc[
            unique_transactions["prediction"] == 1,
            "amount",
        ].fillna(0).sum()
    )

    average_fraud_probability = float(
        unique_transactions["fraud_probability"].fillna(0).mean()
    )

    max_fraud_probability = float(
        unique_transactions["fraud_probability"].fillna(0).max()
    )

    alert_count = int(df["alert_id"].dropna().nunique())

    return {
        "report_date": str(report_date),
        "total_transactions": total_transactions,
        "detected_frauds": detected_frauds,
        "fraud_rate": fraud_rate,
        "total_amount": total_amount,
        "fraud_amount": fraud_amount,
        "average_fraud_probability": average_fraud_probability,
        "max_fraud_probability": max_fraud_probability,
        "alert_count": alert_count,
    }


def build_report_s3_key(report_date: date) -> str:
    prefix = settings.s3_reports_prefix.rstrip("/")
    return f"{prefix}/fraud_daily_report_{report_date}.csv"


def export_report_to_s3(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    report_date: date,
) -> str:
    report_s3_key = build_report_s3_key(report_date)

    if df.empty:
        export_df = pd.DataFrame([metrics])
    else:
        export_df = df.copy()
        export_df["report_date"] = str(report_date)
        export_df["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
        newline="",
    ) as tmp_file:
        temp_path = tmp_file.name
        export_df.to_csv(tmp_file, index=False)

    try:
        upload_file_to_s3(
            local_path=temp_path,
            s3_key=report_s3_key,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return report_s3_key


def upsert_daily_report(metrics: dict[str, Any], report_s3_key: str) -> None:
    engine = get_database_engine()

    query = text(
        """
        INSERT INTO daily_reports (
            report_date,
            total_transactions,
            detected_frauds,
            fraud_rate,
            total_amount,
            fraud_amount,
            report_s3_key
        )
        VALUES (
            :report_date,
            :total_transactions,
            :detected_frauds,
            :fraud_rate,
            :total_amount,
            :fraud_amount,
            :report_s3_key
        )
        ON CONFLICT (report_date)
        DO UPDATE SET
            total_transactions = EXCLUDED.total_transactions,
            detected_frauds = EXCLUDED.detected_frauds,
            fraud_rate = EXCLUDED.fraud_rate,
            total_amount = EXCLUDED.total_amount,
            fraud_amount = EXCLUDED.fraud_amount,
            report_s3_key = EXCLUDED.report_s3_key,
            generated_at = CURRENT_TIMESTAMP;
        """
    )

    params = {
        "report_date": metrics["report_date"],
        "total_transactions": metrics["total_transactions"],
        "detected_frauds": metrics["detected_frauds"],
        "fraud_rate": metrics["fraud_rate"],
        "total_amount": metrics["total_amount"],
        "fraud_amount": metrics["fraud_amount"],
        "report_s3_key": report_s3_key,
    }

    with engine.begin() as connection:
        connection.execute(query, params)


def run_daily_report_pipeline(target_date: date | None = None) -> dict[str, Any]:
    report_date = target_date or get_default_report_date()

    df = extract_daily_data(report_date)
    metrics = compute_daily_metrics(df, report_date)
    report_s3_key = export_report_to_s3(df, metrics, report_date)
    upsert_daily_report(metrics, report_s3_key)

    return {
        **metrics,
        "report_s3_key": report_s3_key,
    }


if __name__ == "__main__":
    result = run_daily_report_pipeline()
    print(json.dumps(result, indent=4))