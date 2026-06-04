import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


def get_required_env_var(name: str) -> str:
    """
    Return a required environment variable.

    Raises:
        ValueError: if the variable is missing or empty.
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")

    return value


def get_optional_env_var(name: str) -> str | None:
    """
    Return an optional environment variable.

    Returns None if the variable is missing or empty.
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return None

    return value


@dataclass(frozen=True)
class Settings:
    # ============================================================
    # PROJECT
    # ============================================================

    project_name: str
    env: str

    # ============================================================
    # AWS / S3
    # ============================================================

    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_default_region: str

    s3_bucket_name: str
    s3_raw_data_key: str
    s3_processed_data_key: str
    s3_reports_prefix: str
    s3_production_model_key: str
    s3_production_model_metadata_key: str

    # ============================================================
    # MLFLOW
    # ============================================================

    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_artifact_root: str
    mlflow_backend_store_uri: str
    mlflow_model_name: str

    # ============================================================
    # APPLICATION DATABASE - NEONDB
    # ============================================================

    fraud_database_url: str

    # ============================================================
    # REAL-TIME PAYMENT API
    # ============================================================

    payment_api_url: str
    payment_api_current_transactions_endpoint: str

    # ============================================================
    # PREDICTION API
    # ============================================================

    prediction_api_url: str
    prediction_api_health_endpoint: str
    prediction_api_predict_endpoint: str

    # ============================================================
    # FRAUD ALERTS
    # ============================================================

    fraud_alert_threshold: float
    notification_channel: str
    discord_webhook_url: str | None

    # ============================================================
    # AIRFLOW
    # ============================================================

    airflow_ingestion_schedule: str
    airflow_daily_report_schedule: str
    airflow_training_schedule: str | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        # ============================================================
        # PROJECT
        # ============================================================

        project_name=get_required_env_var("PROJECT_NAME"),
        env=get_required_env_var("ENV"),

        # ============================================================
        # AWS / S3
        # ============================================================

        aws_access_key_id=get_optional_env_var("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_optional_env_var("AWS_SECRET_ACCESS_KEY"),
        aws_default_region=get_required_env_var("AWS_DEFAULT_REGION"),

        s3_bucket_name=get_required_env_var("S3_BUCKET_NAME"),
        s3_raw_data_key=get_required_env_var("S3_RAW_DATA_KEY"),
        s3_processed_data_key=get_required_env_var("S3_PROCESSED_DATA_KEY"),
        s3_reports_prefix=get_required_env_var("S3_REPORTS_PREFIX"),
        s3_production_model_key=get_required_env_var("S3_PRODUCTION_MODEL_KEY"),
        s3_production_model_metadata_key=get_required_env_var(
            "S3_PRODUCTION_MODEL_METADATA_KEY"
        ),

        # ============================================================
        # MLFLOW
        # ============================================================

        mlflow_tracking_uri=get_required_env_var("MLFLOW_TRACKING_URI"),
        mlflow_experiment_name=get_required_env_var("MLFLOW_EXPERIMENT_NAME"),
        mlflow_artifact_root=get_required_env_var("MLFLOW_ARTIFACT_ROOT"),
        mlflow_backend_store_uri=get_required_env_var("MLFLOW_BACKEND_STORE_URI"),
        mlflow_model_name=get_required_env_var("MLFLOW_MODEL_NAME"),

        # ============================================================
        # APPLICATION DATABASE - NEONDB
        # ============================================================

        fraud_database_url=get_required_env_var("FRAUD_DATABASE_URL"),

        # ============================================================
        # REAL-TIME PAYMENT API
        # ============================================================

        payment_api_url=get_required_env_var("PAYMENT_API_URL"),
        payment_api_current_transactions_endpoint=get_required_env_var(
            "PAYMENT_API_CURRENT_TRANSACTIONS_ENDPOINT"
        ),

        # ============================================================
        # PREDICTION API
        # ============================================================

        prediction_api_url=get_required_env_var("PREDICTION_API_URL"),
        prediction_api_health_endpoint=get_required_env_var(
            "PREDICTION_API_HEALTH_ENDPOINT"
        ),
        prediction_api_predict_endpoint=get_required_env_var(
            "PREDICTION_API_PREDICT_ENDPOINT"
        ),

        # ============================================================
        # FRAUD ALERTS
        # ============================================================

        fraud_alert_threshold=float(os.getenv("FRAUD_ALERT_THRESHOLD", "0.90")),
        notification_channel=get_required_env_var("NOTIFICATION_CHANNEL"),
        discord_webhook_url=get_optional_env_var("DISCORD_WEBHOOK_URL"),

        # ============================================================
        # AIRFLOW
        # ============================================================

        airflow_ingestion_schedule=get_required_env_var(
            "AIRFLOW_INGESTION_SCHEDULE"
        ),
        airflow_daily_report_schedule=get_required_env_var(
            "AIRFLOW_DAILY_REPORT_SCHEDULE"
        ),
        airflow_training_schedule=get_optional_env_var(
            "AIRFLOW_TRAINING_SCHEDULE"
        ),
    )