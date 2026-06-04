from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


# Permet à Airflow de retrouver le package src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.api_client import fetch_current_transactions
from src.data_quality import validate_incoming_transactions
from src.db import (
    count_unintegrated_labeled_transactions,
    create_labeled_transactions_table_if_not_exists,
    create_predictions_table_if_not_exists,
    get_connection,
    insert_labeled_transactions_batch,
    insert_predictions_batch,
)
from src.inference import predict_fraud
from src.validation import validate_transactions
from src.xcom_utils import dataframe_to_xcom_json, xcom_json_to_dataframe


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


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="fraud_inference_pipeline",
    description=(
        "Pipeline d'inférence fraude : API Jedha -> MLflow champion -> "
        "NeonDB -> trigger conditionnel du retraining/CD"
    ),
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
        task_id="validate_transactions_with_gx",
        execution_timeout=timedelta(minutes=3),
    )
    def validate_transactions_with_gx(validated_transactions_json: str) -> str:
        df = xcom_json_to_dataframe(validated_transactions_json)

        validate_incoming_transactions(df)

        print(
            "[DAG] Transactions validées par Great Expectations : "
            f"{len(df)}"
        )

        return dataframe_to_xcom_json(df)

    @task(
        task_id="predict_transactions",
        execution_timeout=timedelta(minutes=5),
    )
    def predict_transactions(validated_transactions_json: str) -> str:
        df = xcom_json_to_dataframe(validated_transactions_json)

        tracking_uri = get_required_config("MLFLOW_TRACKING_URI")
        model_name = get_required_config("MLFLOW_MODEL_NAME")
        fraud_alert_threshold = float(
            get_required_config("FRAUD_ALERT_THRESHOLD")
        )

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

    @task(
        task_id="store_labeled_transactions",
        execution_timeout=timedelta(minutes=3),
    )
    def store_labeled_transactions(validated_transactions_json: str) -> int:
        transactions = xcom_json_to_dataframe(validated_transactions_json)

        database_url = get_required_config("FRAUD_DATABASE_URL")

        conn = get_connection(database_url)

        try:
            create_labeled_transactions_table_if_not_exists(conn)
            inserted_rows = insert_labeled_transactions_batch(
                conn=conn,
                df=transactions,
            )
        finally:
            conn.close()

        print(
            "[DAG] Transactions labellisées stockées pour retraining : "
            f"{inserted_rows}"
        )

        return inserted_rows

    @task(
        task_id="check_retraining_trigger_condition",
        execution_timeout=timedelta(minutes=2),
    )
    def check_retraining_trigger_condition(
        stored_labeled_transactions_count: int,
        stored_predictions_count: int,
    ) -> int:
        """
        Vérifie si le DAG de retraining/CD doit être déclenché.

        Le seuil MIN_NEW_TRANSACTIONS_FOR_RETRAINING ne déclenche rien seul.
        Cette tâche rend le déclenchement événementiel :
        si le nombre de transactions labellisées non intégrées atteint le seuil,
        le DAG fraud_retraining_cd_pipeline est déclenché.
        """

        print(
            "[DAG] Transactions labellisées insérées pendant ce run : "
            f"{stored_labeled_transactions_count}"
        )

        print(
            "[DAG] Prédictions stockées pendant ce run : "
            f"{stored_predictions_count}"
        )

        min_new_transactions = int(
            get_optional_config("MIN_NEW_TRANSACTIONS_FOR_RETRAINING") or "100"
        )

        database_url = get_required_config("FRAUD_DATABASE_URL")

        conn = get_connection(database_url)

        try:
            unintegrated_count = count_unintegrated_labeled_transactions(conn)
        finally:
            conn.close()

        print(
            "[DAG] Transactions labellisées non intégrées : "
            f"{unintegrated_count}"
        )

        print(
            "[DAG] Seuil minimal de réentraînement : "
            f"{min_new_transactions}"
        )

        if unintegrated_count < min_new_transactions:
            raise AirflowSkipException(
                "Trigger retraining/CD annulé : seuil non atteint. "
                f"{unintegrated_count} transaction(s) disponible(s), "
                f"seuil={min_new_transactions}."
            )

        print(
            "[DAG] Seuil atteint : déclenchement du DAG "
            "fraud_retraining_cd_pipeline."
        )

        return unintegrated_count

    trigger_retraining_cd = TriggerDagRunOperator(
        task_id="trigger_retraining_cd_pipeline",
        trigger_dag_id="fraud_retraining_cd_pipeline",
        wait_for_completion=False,
        reset_dag_run=False,
        conf={
            "triggered_by": "fraud_inference_pipeline",
            "unintegrated_transactions": (
                "{{ ti.xcom_pull("
                "task_ids='check_retraining_trigger_condition'"
                ") }}"
            ),
            "source_dag_id": "{{ dag.dag_id }}",
            "source_run_id": "{{ run_id }}",
            "source_execution_date": "{{ ts }}",
        },
    )

    raw_transactions = fetch_transactions()

    validated_transactions = validate_transactions_task(raw_transactions)

    gx_validated_transactions = validate_transactions_with_gx(
        validated_transactions
    )

    predictions = predict_transactions(gx_validated_transactions)

    stored_predictions = store_predictions(predictions)

    stored_labeled_transactions = store_labeled_transactions(
        gx_validated_transactions
    )

    retraining_condition = check_retraining_trigger_condition(
        stored_labeled_transactions,
        stored_predictions,
    )

    retraining_condition >> trigger_retraining_cd


fraud_inference_pipeline()