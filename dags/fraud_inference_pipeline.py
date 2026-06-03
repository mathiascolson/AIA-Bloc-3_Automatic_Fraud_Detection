from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

from airflow.decorators import dag, task
from airflow.models import Variable


# Permet à Airflow de retrouver le package src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.api_client import fetch_current_transactions
from src.db import (
    create_predictions_table_if_not_exists,
    get_connection,
    insert_predictions_batch,
)
from src.inference import predict_fraud
from src.validation import validate_transactions


def get_required_config(name: str) -> str:
    """
    Récupère une configuration depuis :
    1. les variables d'environnement ;
    2. les Airflow Variables.

    Lève une erreur explicite si la valeur est absente.
    """

    value = os.getenv(name)

    if value:
        return value

    value = Variable.get(name, default_var=None)

    if value:
        return value

    raise ValueError(f"Configuration manquante : {name}")


def get_optional_config(name: str) -> str | None:
    """
    Récupère une configuration optionnelle depuis :
    1. les variables d'environnement ;
    2. les Airflow Variables.
    """

    value = os.getenv(name)

    if value:
        return value

    return Variable.get(name, default_var=None)


def get_payment_api_url() -> str:
    """
    Construit l'URL complète de l'API de transactions.

    Priorité :
    1. FRAUD_API_URL si défini ;
    2. PAYMENT_API_URL + PAYMENT_API_CURRENT_TRANSACTIONS_ENDPOINT.
    """

    fraud_api_url = get_optional_config("FRAUD_API_URL")

    if fraud_api_url:
        return fraud_api_url

    payment_api_url = get_required_config("PAYMENT_API_URL")
    endpoint = get_required_config("PAYMENT_API_CURRENT_TRANSACTIONS_ENDPOINT")

    return payment_api_url.rstrip("/") + "/" + endpoint.lstrip("/")


def dataframe_to_xcom_json(df: pd.DataFrame) -> str:
    """
    Convertit un DataFrame en JSON compatible XCom.
    """

    return df.to_json(
        orient="split",
        date_format="iso",
        index=True,
    )


def xcom_json_to_dataframe(raw_json: str) -> pd.DataFrame:
    """
    Reconstruit un DataFrame depuis un JSON XCom.
    """

    return pd.read_json(
        StringIO(raw_json),
        orient="split",
    )


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="fraud_inference_pipeline",
    description="Pipeline d'inférence fraude : API Jedha -> MLflow champion -> NeonDB",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(hours=2),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "inference", "mlflow", "neondb"],
)
def fraud_inference_pipeline():
    @task(
        task_id="fetch_transactions",
        execution_timeout=timedelta(minutes=2),
    )
    def fetch_transactions() -> str:
        api_url = get_payment_api_url()

        df = fetch_current_transactions(api_url)

        if df.empty:
            raise ValueError("Aucune transaction récupérée depuis l'API.")

        print(f"[DAG] Transactions récupérées : {len(df)}")

        return dataframe_to_xcom_json(df)

    @task(
        task_id="validate_transactions",
        execution_timeout=timedelta(minutes=2),
    )
    def validate_transactions_task(raw_transactions_json: str) -> str:
        df = xcom_json_to_dataframe(raw_transactions_json)

        validate_transactions(df)

        print(f"[DAG] Transactions validées : {len(df)}")

        return dataframe_to_xcom_json(df)

    @task(
        task_id="predict_transactions",
        execution_timeout=timedelta(minutes=5),
    )
    def predict_transactions(validated_transactions_json: str) -> str:
        df = xcom_json_to_dataframe(validated_transactions_json)

        tracking_uri = get_required_config("MLFLOW_TRACKING_URI")
        model_name = get_required_config("MLFLOW_MODEL_NAME")
        fraud_alert_threshold = float(get_required_config("FRAUD_ALERT_THRESHOLD"))

        model_alias = get_optional_config("MLFLOW_MODEL_ALIAS") or "champion"

        predictions = predict_fraud(
            df=df,
            tracking_uri=tracking_uri,
            model_name=model_name,
            model_alias=model_alias,
            fraud_alert_threshold=fraud_alert_threshold,
        )

        print(f"[DAG] Prédictions générées : {len(predictions)}")
        print(
            "[DAG] Fraudes prédites : "
            f"{int(predictions['is_fraud_predicted'].sum())}"
        )

        return dataframe_to_xcom_json(predictions)

    @task(
        task_id="store_predictions",
        execution_timeout=timedelta(minutes=3),
    )
    def store_predictions(predictions_json: str) -> int:
        predictions = xcom_json_to_dataframe(predictions_json)

        database_url = get_required_config("FRAUD_DATABASE_URL")

        conn = get_connection(database_url)

        try:
            create_predictions_table_if_not_exists(conn)
            insert_predictions_batch(conn, predictions)
        finally:
            conn.close()

        inserted_rows = len(predictions)

        print(f"[DAG] Prédictions insérées dans NeonDB : {inserted_rows}")

        return inserted_rows

    raw_transactions = fetch_transactions()
    validated_transactions = validate_transactions_task(raw_transactions)
    predictions = predict_transactions(validated_transactions)
    store_predictions(predictions)


fraud_inference_pipeline()