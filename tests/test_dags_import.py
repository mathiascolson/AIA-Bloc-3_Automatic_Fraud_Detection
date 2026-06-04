from pathlib import Path

from airflow.models import DagBag


def load_dag_bag() -> DagBag:
    project_root = Path(__file__).resolve().parents[1]
    dags_path = project_root / "dags"

    return DagBag(
        dag_folder=str(dags_path),
        include_examples=False,
        read_dags_from_db=False,
    )


def test_airflow_dags_import_without_errors():
    dag_bag = load_dag_bag()

    assert dag_bag.import_errors == {}
    assert "fraud_inference_pipeline" in dag_bag.dags
    assert "fraud_retraining_cd_pipeline" in dag_bag.dags


def test_fraud_inference_pipeline_contains_retraining_trigger():
    dag_bag = load_dag_bag()

    dag = dag_bag.dags["fraud_inference_pipeline"]

    task_ids = {task.task_id for task in dag.tasks}

    assert "validate_transactions_with_gx" in task_ids
    assert "check_retraining_trigger_condition" in task_ids
    assert "trigger_retraining_cd_pipeline" in task_ids


def test_fraud_retraining_cd_pipeline_has_no_schedule():
    dag_bag = load_dag_bag()

    dag = dag_bag.dags["fraud_retraining_cd_pipeline"]

    assert dag.schedule_interval is None