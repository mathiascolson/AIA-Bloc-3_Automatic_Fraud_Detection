import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import settings
from src.preprocessing import prepare_features_and_target, build_preprocessor
from src.s3_utils import read_csv_from_s3, upload_file_to_s3


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


PRODUCTION_THRESHOLD = settings.fraud_alert_threshold
THRESHOLDS = sorted(set([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, PRODUCTION_THRESHOLD]))


def compute_class_imbalance_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)

    if positive_count == 0:
        raise ValueError("No positive class found in target.")

    return negative_count / positive_count


def build_model_candidates(scale_pos_weight: float) -> dict:
    candidates = {
        "logistic_regression_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="saga",
            random_state=42,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["xgboost_weighted"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    return candidates


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def evaluate_thresholds(y_true, y_proba, thresholds: list[float]) -> list[dict]:
    results = []

    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_threshold).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred_threshold, zero_division=0),
                "recall": recall_score(y_true, y_pred_threshold, zero_division=0),
                "f1_score": f1_score(y_true, y_pred_threshold, zero_division=0),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )

    return results


def log_threshold_metrics(threshold_results: list[dict]) -> None:
    for row in threshold_results:
        threshold_label = str(row["threshold"]).replace(".", "_")

        mlflow.log_metric(f"precision_threshold_{threshold_label}", row["precision"])
        mlflow.log_metric(f"recall_threshold_{threshold_label}", row["recall"])
        mlflow.log_metric(f"f1_threshold_{threshold_label}", row["f1_score"])
        mlflow.log_metric(f"false_positives_threshold_{threshold_label}", row["false_positives"])
        mlflow.log_metric(f"false_negatives_threshold_{threshold_label}", row["false_negatives"])
        mlflow.log_metric(f"true_positives_threshold_{threshold_label}", row["true_positives"])


def wait_for_model_version_ready(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    timeout_seconds: int = 120,
    poll_interval_seconds: int = 5,
) -> None:
    """
    Attend que la version enregistrée dans le Model Registry soit prête.
    """

    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        version = client.get_model_version(
            name=model_name,
            version=model_version,
        )

        status = version.status

        if status == "READY":
            return

        if status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"Échec d'enregistrement MLflow pour "
                f"{model_name} version {model_version}."
            )

        print(
            f"[MLFLOW] Version {model_name} v{model_version} "
            f"en statut {status}. Attente..."
        )
        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timeout : {model_name} version {model_version} "
        f"n'est pas passée en READY après {timeout_seconds} secondes."
    )


def register_best_model_as_challenger(best_result: dict) -> dict:
    """
    Enregistre le meilleur candidat dans le Model Registry MLflow
    et lui attribue l'alias challenger.
    """

    model_registry_name = settings.mlflow_model_name
    model_uri = best_result["model_uri"]

    print(
        f"\nRegistering best candidate in MLflow Model Registry: "
        f"{model_registry_name}"
    )
    print(f"Model URI: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_registry_name,
    )

    client = MlflowClient()

    wait_for_model_version_ready(
        client=client,
        model_name=model_registry_name,
        model_version=registered_model.version,
    )

    client.set_registered_model_alias(
        name=model_registry_name,
        alias="challenger",
        version=registered_model.version,
    )

    print(
        f"[MLFLOW] Alias challenger assigned to "
        f"{model_registry_name} version {registered_model.version}"
    )

    return {
        "registered_model_name": model_registry_name,
        "registered_model_version": registered_model.version,
        "registered_model_alias": "challenger",
        "model_uri": model_uri,
    }


def train_one_candidate(
    model_name: str,
    classifier,
    X_train,
    X_test,
    y_train,
    y_test,
    class_imbalance_ratio: float,
) -> dict:
    model_pipeline = build_pipeline(classifier)

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("classifier", classifier.__class__.__name__)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("class_imbalance_ratio", class_imbalance_ratio)
        mlflow.log_param("production_threshold", PRODUCTION_THRESHOLD)

        print(f"\nTraining candidate: {model_name}")
        model_pipeline.fit(X_train, y_train)

        print(f"Evaluating candidate: {model_name}")
        y_pred_default = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        default_metrics = compute_metrics(y_test, y_pred_default, y_proba)
        threshold_results = evaluate_thresholds(y_test, y_proba, THRESHOLDS)

        for metric_name, metric_value in default_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        log_threshold_metrics(threshold_results)

        production_row = next(
            row for row in threshold_results
            if row["threshold"] == PRODUCTION_THRESHOLD
        )

        mlflow.log_metric("production_precision", production_row["precision"])
        mlflow.log_metric("production_recall", production_row["recall"])
        mlflow.log_metric("production_f1_score", production_row["f1_score"])
        mlflow.log_metric("production_false_positives", production_row["false_positives"])
        mlflow.log_metric("production_false_negatives", production_row["false_negatives"])
        mlflow.log_metric("production_true_positives", production_row["true_positives"])

        classification_report_dict = classification_report(
            y_test,
            y_pred_default,
            output_dict=True,
            zero_division=0,
        )

        confusion_matrix_default = confusion_matrix(y_test, y_pred_default)

        metadata = {
            "model_name": model_name,
            "classifier": classifier.__class__.__name__,
            "run_id": run_id,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "s3_raw_data_key": settings.s3_raw_data_key,
            "production_threshold": PRODUCTION_THRESHOLD,
            "default_metrics": default_metrics,
            "threshold_analysis": threshold_results,
            "classification_report_default_threshold": classification_report_dict,
            "confusion_matrix_default_threshold": confusion_matrix_default.tolist(),
            "threshold_source": "env:FRAUD_ALERT_THRESHOLD",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            metadata_path = tmp_dir_path / f"{model_name}_metadata.json"
            report_path = tmp_dir_path / f"{model_name}_classification_report.json"
            threshold_path = tmp_dir_path / f"{model_name}_threshold_analysis.json"

            metadata_path.write_text(
                json.dumps(metadata, indent=4),
                encoding="utf-8",
            )

            report_path.write_text(
                json.dumps(classification_report_dict, indent=4),
                encoding="utf-8",
            )

            threshold_path.write_text(
                json.dumps(threshold_results, indent=4),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(threshold_path), artifact_path="evaluation")

            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model",
            )

        model_uri = f"runs:/{run_id}/model"

        result = {
            "model_name": model_name,
            "run_id": run_id,
            "model_uri": model_uri,
            "pipeline": model_pipeline,
            "default_metrics": default_metrics,
            "threshold_results": threshold_results,
            "production_metrics": production_row,
            "selection_score": default_metrics["average_precision"],
        }

        print(f"\n=== {model_name} ===")
        print(f"average_precision: {default_metrics['average_precision']:.4f}")
        print(f"roc_auc: {default_metrics['roc_auc']:.4f}")
        print(
            f"threshold={PRODUCTION_THRESHOLD:.2f} | "
            f"precision={production_row['precision']:.4f} | "
            f"recall={production_row['recall']:.4f} | "
            f"f1={production_row['f1_score']:.4f} | "
            f"FP={production_row['false_positives']} | "
            f"FN={production_row['false_negatives']} | "
            f"TP={production_row['true_positives']}"
        )

        return result


def export_best_model_to_s3(best_result: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        model_path = tmp_dir_path / "fraud_pipeline.joblib"
        metadata_path = tmp_dir_path / "model_metadata.json"

        metadata = {
            "model_name": best_result["model_name"],
            "run_id": best_result["run_id"],
            "selection_score": best_result["selection_score"],
            "selection_metric": "average_precision",
            "production_threshold": PRODUCTION_THRESHOLD,
            "production_metrics": best_result["production_metrics"],
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "s3_production_model_key": settings.s3_production_model_key,
            "s3_production_model_metadata_key": settings.s3_production_model_metadata_key,
            "threshold_source": "env:FRAUD_ALERT_THRESHOLD",
        }

        joblib.dump(best_result["pipeline"], model_path)

        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        upload_file_to_s3(
            local_path=model_path,
            s3_key=settings.s3_production_model_key,
        )

        upload_file_to_s3(
            local_path=metadata_path,
            s3_key=settings.s3_production_model_metadata_key,
        )


def main() -> None:
    if not settings.mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing in .env")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    print("Loading training dataset from S3...")
    df = read_csv_from_s3(settings.s3_raw_data_key)

    print("Preparing features and target...")
    X, y = prepare_features_and_target(df)

    print("Dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts(normalize=True))

    class_imbalance_ratio = compute_class_imbalance_ratio(y)
    print("Class imbalance ratio:", class_imbalance_ratio)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    candidates = build_model_candidates(class_imbalance_ratio)

    results = []

    for model_name, classifier in candidates.items():
        result = train_one_candidate(
            model_name=model_name,
            classifier=classifier,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_imbalance_ratio=class_imbalance_ratio,
        )

        results.append(result)

    best_result = max(results, key=lambda item: item["selection_score"])

    print("\n=== BEST MODEL ===")
    print("Model:", best_result["model_name"])
    print("Run ID:", best_result["run_id"])
    print("Selection metric: average_precision")
    print("Selection score:", best_result["selection_score"])
    print("Production threshold:", PRODUCTION_THRESHOLD)
    print("Production metrics:", best_result["production_metrics"])

    print("\nExporting best model to S3 production path...")
    export_best_model_to_s3(best_result)

    registry_info = register_best_model_as_challenger(best_result)

    print("\n=== MODEL REGISTRY ===")
    print("Registered model name:", registry_info["registered_model_name"])
    print("Registered model version:", registry_info["registered_model_version"])
    print("Registered model alias:", registry_info["registered_model_alias"])
    print("Registered model URI:", registry_info["model_uri"])

    print("Training candidates completed successfully.")


if __name__ == "__main__":
    main()