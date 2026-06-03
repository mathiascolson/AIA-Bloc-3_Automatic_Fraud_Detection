import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

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


def build_model_pipeline() -> Pipeline:
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="saga",
        random_state=42,
    )

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


def main() -> None:
    print("Loading training dataset from S3...")
    df = read_csv_from_s3(settings.s3_raw_data_key)

    print("Preparing features and target...")
    X, y = prepare_features_and_target(df)

    print("Dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts(normalize=True))

    print("Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    model_pipeline = build_model_pipeline()

    if not settings.mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing in .env")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    print("Training model...")

    with mlflow.start_run(run_name="baseline_logistic_regression_balanced") as run:
        run_id = run.info.run_id

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("solver", "saga")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_features_before_encoding", X.shape[1])
        mlflow.log_param("target_column", "is_fraud")
        mlflow.log_param("s3_raw_data_key", settings.s3_raw_data_key)
        mlflow.log_param("production_threshold", settings.fraud_alert_threshold)

        model_pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        threshold_results = evaluate_thresholds(y_test, y_proba, thresholds)

        for row in threshold_results:
            mlflow.log_metric(
                f"precision_threshold_{row['threshold']}",
                row["precision"],
            )
            mlflow.log_metric(
                f"recall_threshold_{row['threshold']}",
                row["recall"],
            )
            mlflow.log_metric(
                f"f1_threshold_{row['threshold']}",
                row["f1_score"],
            )
            mlflow.log_metric(
                f"false_positives_threshold_{row['threshold']}",
                row["false_positives"],
            )
            mlflow.log_metric(
                f"false_negatives_threshold_{row['threshold']}",
                row["false_negatives"],
            )

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred)

        metadata = {
            "model_name": settings.mlflow_model_name,
            "model_type": "LogisticRegression",
            "run_id": run_id,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "s3_raw_data_key": settings.s3_raw_data_key,
            "s3_production_model_key": settings.s3_production_model_key,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "threshold_analysis": threshold_results,
            "production_threshold": settings.fraud_alert_threshold,
            "threshold_source": "env:FRAUD_ALERT_THRESHOLD",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            model_path = tmp_dir_path / "fraud_pipeline.joblib"
            metadata_path = tmp_dir_path / "model_metadata.json"
            report_path = tmp_dir_path / "classification_report.json"
            cm_path = tmp_dir_path / "confusion_matrix.json"

            joblib.dump(model_pipeline, model_path)

            metadata_path.write_text(
                json.dumps(metadata, indent=4),
                encoding="utf-8",
            )

            report_path.write_text(
                json.dumps(report, indent=4),
                encoding="utf-8",
            )

            cm_path.write_text(
                json.dumps(cm.tolist(), indent=4),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model",
                registered_model_name=settings.mlflow_model_name,
            )

            print("Uploading production model to S3...")
            upload_file_to_s3(
                local_path=model_path,
                s3_key=settings.s3_production_model_key,
            )

            upload_file_to_s3(
                local_path=metadata_path,
                s3_key=settings.s3_production_model_metadata_key,
            )

        print("\n=== METRICS ===")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print("\n=== CONFUSION MATRIX ===")
        print(cm)

        print("\nMLflow run_id:", run_id)
        print("Training completed successfully.")
        
        print("\n=== THRESHOLD ANALYSIS ===")
        
        for row in threshold_results:
            print(
                f"threshold={row['threshold']:.2f} | "
                f"precision={row['precision']:.4f} | "
                f"recall={row['recall']:.4f} | "
                f"f1={row['f1_score']:.4f} | "
                f"FP={row['false_positives']} | "
                f"FN={row['false_negatives']} | "
                f"TP={row['true_positives']}"
            )


if __name__ == "__main__":
    main()