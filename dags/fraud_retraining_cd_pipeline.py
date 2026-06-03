from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models import Variable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_required_config(name: str) -> str:
    value = os.getenv(name)

    if value:
        return value

    value = Variable.get(name, default_var=None)

    if value:
        return value

    raise ValueError(f"Configuration manquante : {name}")


def get_optional_config(name: str) -> str | None:
    value = os.getenv(name)

    if value:
        return value

    return Variable.get(name, default_var=None)


default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="fraud_retraining_cd_pipeline",
    description=(
        "Pipeline CD modèle : vérification CI GitHub, entraînement challenger, "
        "comparaison avec champion, promotion MLflow et historisation NeonDB"
    ),
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "training", "cd", "mlflow", "github"],
)
def fraud_retraining_cd_pipeline():
    @task(
        task_id="check_github_ci",
        execution_timeout=timedelta(minutes=2),
    )
    def check_github_ci() -> dict:
        from src.github_ci import is_latest_workflow_successful

        owner = get_required_config("GITHUB_OWNER")
        repo = get_required_config("GITHUB_REPO")
        workflow_name = get_required_config("GITHUB_WORKFLOW_NAME")
        branch = get_optional_config("GITHUB_BRANCH") or "main"
        github_token = get_optional_config("GITHUB_TOKEN")

        ci_ok, latest_run = is_latest_workflow_successful(
            owner=owner,
            repo=repo,
            workflow_name=workflow_name,
            branch=branch,
            github_token=github_token,
        )

        status = latest_run.get("status")
        conclusion = latest_run.get("conclusion")
        html_url = latest_run.get("html_url")

        print(f"[GITHUB] Dernier workflow : {html_url}")
        print(f"[GITHUB] status={status} | conclusion={conclusion}")

        if not ci_ok:
            raise ValueError(
                "CD arrêtée : la dernière CI GitHub n'est pas en succès. "
                f"status={status} | conclusion={conclusion} | url={html_url}"
            )

        return {
            "ci_ok": ci_ok,
            "status": status,
            "conclusion": conclusion,
            "html_url": html_url,
        }

    @task(
        task_id="train_challenger",
        execution_timeout=timedelta(minutes=45),
    )
    def train_challenger(ci_status: dict) -> str:
        if not ci_status.get("ci_ok"):
            raise ValueError("Entraînement annulé : CI GitHub non validée.")

        from src.train_model_candidates import main as train_candidates

        print("[TRAINING] Démarrage entraînement candidats.")
        train_candidates()
        print("[TRAINING] Entraînement terminé. Le meilleur modèle est enregistré en challenger.")

        return "challenger_trained"

    @task(
        task_id="compare_and_promote",
        execution_timeout=timedelta(minutes=5),
    )
    def compare_and_promote(_: str) -> dict:
        from src.mlflow_utils import (
            get_model_alias_info,
            promote_model_alias,
        )
        from src.model_promotion import (
            build_cd_decision,
            is_candidate_better,
        )

        tracking_uri = get_required_config("MLFLOW_TRACKING_URI")
        model_name = get_required_config("MLFLOW_MODEL_NAME")

        champion_info = get_model_alias_info(
            tracking_uri=tracking_uri,
            model_name=model_name,
            model_alias="champion",
        )

        challenger_info = get_model_alias_info(
            tracking_uri=tracking_uri,
            model_name=model_name,
            model_alias="challenger",
        )

        promoted, reason = is_candidate_better(
            champion_metrics=champion_info["metrics"],
            challenger_metrics=challenger_info["metrics"],
        )
        
        def export_promoted_model_to_s3_production(
            tracking_uri: str,
            model_name: str,
            model_version: str,
            challenger_info: dict,
        ) -> None:
            import json
            import tempfile
            from datetime import datetime, timezone
            from pathlib import Path

            import joblib
            import mlflow
            import boto3

            production_model_key = get_required_config("S3_PRODUCTION_MODEL_KEY")
            production_metadata_key = get_required_config("S3_PRODUCTION_MODEL_METADATA_KEY")
            bucket_name = get_required_config("S3_BUCKET_NAME")

            model_uri = f"models:/{model_name}/{model_version}"

            mlflow.set_tracking_uri(tracking_uri)
            promoted_model = mlflow.sklearn.load_model(model_uri)

            metadata = {
                "artifact_role": "production",
                "model_name": model_name,
                "model_version": model_version,
                "run_id": challenger_info.get("run_id"),
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_alias": "challenger",
                "target_alias": "champion",
                "metrics": challenger_info.get("metrics", {}),
                "s3_production_model_key": production_model_key,
                "s3_production_model_metadata_key": production_metadata_key,
            }

            s3_client = boto3.client("s3")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)

                model_path = tmp_dir_path / "fraud_pipeline.joblib"
                metadata_path = tmp_dir_path / "model_metadata.json"

                joblib.dump(promoted_model, model_path)

                metadata_path.write_text(
                    json.dumps(metadata, indent=4),
                    encoding="utf-8",
                )

                s3_client.upload_file(
                    Filename=str(model_path),
                    Bucket=bucket_name,
                    Key=production_model_key,
                )

                s3_client.upload_file(
                    Filename=str(metadata_path),
                    Bucket=bucket_name,
                    Key=production_metadata_key,
                )

            print(
                "[S3] Modèle promu exporté vers "
                f"s3://{bucket_name}/{production_model_key}"
            )
            print(
                "[S3] Métadonnées production exportées vers "
                f"s3://{bucket_name}/{production_metadata_key}"
            )

        if promoted:
            promote_model_alias(
                tracking_uri=tracking_uri,
                model_name=model_name,
                model_alias="champion",
                model_version=str(challenger_info["version"]),
            )

            export_promoted_model_to_s3_production(
                tracking_uri=tracking_uri,
                model_name=model_name,
                model_version=str(challenger_info["version"]),
                challenger_info=challenger_info,
            )

        decision = build_cd_decision(
            model_name=model_name,
            champion_info=champion_info,
            challenger_info=challenger_info,
            promoted=promoted,
            decision_reason=reason,
        )

        print(f"[CD] champion version={champion_info['version']}")
        print(f"[CD] challenger version={challenger_info['version']}")
        print(f"[CD] promoted={promoted}")
        print(f"[CD] reason={reason}")

        return decision

    @task(
        task_id="store_cd_decision",
        execution_timeout=timedelta(minutes=3),
    )
    def store_cd_decision(decision: dict) -> int:
        from src.db import (
            create_model_cd_decisions_table_if_not_exists,
            get_connection,
            insert_model_cd_decision,
        )

        database_url = get_required_config("FRAUD_DATABASE_URL")

        conn = get_connection(database_url)

        try:
            create_model_cd_decisions_table_if_not_exists(conn)
            insert_model_cd_decision(conn, decision)
        finally:
            conn.close()

        print("[CD] Décision CD historisée dans NeonDB.")

        return 1

    ci_status = check_github_ci()
    challenger_training = train_challenger(ci_status)
    decision = compare_and_promote(challenger_training)
    store_cd_decision(decision)


fraud_retraining_cd_pipeline()