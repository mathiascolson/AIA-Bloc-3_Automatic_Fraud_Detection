from __future__ import annotations

from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

load_dotenv()


def get_model_uri(model_name: str, model_alias: str) -> str:
    """
    Construit l'URI MLflow d'un modèle à partir de son alias.
    """

    return f"models:/{model_name}@{model_alias}"


def load_pyfunc_model(
    tracking_uri: str,
    model_name: str,
    model_alias: str,
):
    """
    Charge un modèle MLflow PyFunc à partir d'un alias.
    """

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = get_model_uri(
        model_name=model_name,
        model_alias=model_alias,
    )

    print(f"[MLFLOW] Chargement du modèle : {model_uri}")

    return mlflow.pyfunc.load_model(model_uri)


def get_mlflow_client(tracking_uri: str) -> MlflowClient:
    """
    Retourne un client MLflow configuré.
    """

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)

def get_model_version_by_alias(
    tracking_uri: str,
    model_name: str,
    model_alias: str,
):
    """
    Récupère une version de modèle MLflow à partir d'un alias.
    Exemple : fraud_proj@champion ou fraud_proj@challenger.
    """

    client = get_mlflow_client(tracking_uri)

    return client.get_model_version_by_alias(
        name=model_name,
        alias=model_alias,
    )


def get_run_metrics(
    tracking_uri: str,
    run_id: str,
) -> dict:
    """
    Récupère les métriques d'un run MLflow.
    """

    client = get_mlflow_client(tracking_uri)
    run = client.get_run(run_id)

    return dict(run.data.metrics)


def get_model_alias_info(
    tracking_uri: str,
    model_name: str,
    model_alias: str,
) -> dict:
    """
    Récupère les informations principales d'un modèle via son alias :
    version, run_id, source et métriques du run associé.
    """

    model_version = get_model_version_by_alias(
        tracking_uri=tracking_uri,
        model_name=model_name,
        model_alias=model_alias,
    )

    metrics = get_run_metrics(
        tracking_uri=tracking_uri,
        run_id=model_version.run_id,
    )

    return {
        "model_name": model_name,
        "alias": model_alias,
        "version": model_version.version,
        "run_id": model_version.run_id,
        "source": model_version.source,
        "metrics": metrics,
    }


def promote_model_alias(
    tracking_uri: str,
    model_name: str,
    model_alias: str,
    model_version: str,
) -> None:
    """
    Affecte un alias MLflow à une version de modèle.
    Utilisé pour promouvoir challenger vers champion.
    """

    client = get_mlflow_client(tracking_uri)

    client.set_registered_model_alias(
        name=model_name,
        alias=model_alias,
        version=model_version,
    )

    print(
        f"[MLFLOW] Alias '{model_alias}' affecté à "
        f"{model_name} version {model_version}"
    )