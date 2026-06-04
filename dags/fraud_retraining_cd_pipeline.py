from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
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
        "Pipeline CD modèle : vérification CI GitHub, mise à jour du dataset "
        "d'entraînement, entraînement challenger, comparaison avec champion, "
        "promotion MLflow et historisation NeonDB"
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
        task_id="update_training_dataset_from_labeled_transactions",
        execution_timeout=timedelta(minutes=10),
    )
    def update_training_dataset_from_labeled_transactions(
        ci_status: dict,
    ) -> int:
        from src.db import (
            count_unintegrated_labeled_transactions,
            fetch_unintegrated_labeled_transactions,
            get_connection,
            mark_labeled_transactions_as_integrated,
        )
        from src.training_dataset_store import (
            append_labeled_transactions_to_training_dataset,
        )

        if not ci_status.get("ci_ok"):
            raise ValueError(
                "Mise à jour du dataset annulée : CI GitHub non validée."
            )

        database_url = get_required_config("FRAUD_DATABASE_URL")
        bucket_name = get_required_config("S3_BUCKET_NAME")
        raw_data_key = get_required_config("S3_RAW_DATA_KEY")
        processed_data_key = get_required_config("S3_PROCESSED_DATA_KEY")

        min_new_transactions = int(
            get_optional_config("MIN_NEW_TRANSACTIONS_FOR_RETRAINING") or "100"
        )

        conn = get_connection(database_url)

        try:
            available_transactions = (
                count_unintegrated_labeled_transactions(conn)
            )

            print(
                "[DAG] Transactions labellisées non intégrées : "
                f"{available_transactions}"
            )

            print(
                "[DAG] Seuil minimal de réentraînement : "
                f"{min_new_transactions}"
            )

            if available_transactions < min_new_transactions:
                raise AirflowSkipException(
                    "Volume insuffisant pour réentraînement : "
                    f"{available_transactions} transaction(s) disponible(s), "
                    f"seuil={min_new_transactions}."
                )

            labeled_transactions = fetch_unintegrated_labeled_transactions(conn)

            integrated_trans_nums = (
                append_labeled_transactions_to_training_dataset(
                    bucket_name=bucket_name,
                    raw_data_key=raw_data_key,
                    processed_data_key=processed_data_key,
                    labeled_transactions_df=labeled_transactions,
                )
            )

            if not integrated_trans_nums:
                already_present_trans_nums = (
                    labeled_transactions["trans_num"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                reconciled_count = mark_labeled_transactions_as_integrated(
                    conn=conn,
                    trans_nums=already_present_trans_nums,
                )

                print(
                    "[DAG] Transactions déjà présentes dans le dataset "
                    "et marquées comme intégrées : "
                    f"{reconciled_count}"
                )

                raise AirflowSkipException(
                    "Aucune nouvelle transaction à ajouter au dataset. "
                    "Les transactions déjà présentes ont été marquées "
                    "comme intégrées."
                )

            updated_count = mark_labeled_transactions_as_integrated(
                conn=conn,
                trans_nums=integrated_trans_nums,
            )

        finally:
            conn.close()

        print(
            "[DAG] Transactions intégrées au dataset d'entraînement : "
            f"{updated_count}"
        )

        return updated_count

    @task(
        task_id="train_challenger",
        execution_timeout=timedelta(minutes=45),
    )
    def train_challenger(integrated_transactions_count: int) -> str:
        if integrated_transactions_count <= 0:
            raise AirflowSkipException(
                "Entraînement annulé : aucune transaction intégrée au dataset."
            )

        from src.train_model_candidates import main as train_candidates

        print(
            "[TRAINING] Démarrage entraînement candidats après intégration de "
            f"{integrated_transactions_count} transaction(s)."
        )

        train_candidates()

        print(
            "[TRAINING] Entraînement terminé. "
            "Le meilleur modèle est enregistré en challenger."
        )

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

            import boto3
            import joblib
            import mlflow

            production_model_key = get_required_config(
                "S3_PRODUCTION_MODEL_KEY"
            )
            production_metadata_key = get_required_config(
                "S3_PRODUCTION_MODEL_METADATA_KEY"
            )
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

    integrated_transactions_count = (
        update_training_dataset_from_labeled_transactions(ci_status)
    )

    challenger_training = train_challenger(integrated_transactions_count)

    decision = compare_and_promote(challenger_training)

    store_cd_decision(decision)


fraud_retraining_cd_pipeline()