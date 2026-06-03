import mlflow

from src.config import settings


def main() -> None:
    if not settings.mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing in .env")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="test_mlflow_connection") as run:
        mlflow.log_param("connection_test", True)
        mlflow.log_metric("test_metric", 1.0)

        print("MLflow connection successful.")
        print("Tracking URI:", settings.mlflow_tracking_uri)
        print("Experiment:", settings.mlflow_experiment_name)
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()