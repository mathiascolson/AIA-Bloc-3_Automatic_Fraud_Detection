import pandas as pd
from datetime import date

from src.training_dataset_store import (
    merge_training_dataset_with_new_transactions,normalize_training_dataset_types,
)

def test_normalize_training_dataset_types_converts_dob_to_datetime():
    df = pd.DataFrame(
        {
            "trans_num": ["tx_001", "tx_002"],
            "dob": ["1980-01-01", date(1990, 2, 3)],
            "current_time": ["2026-06-04T08:00:00", "2026-06-04T08:01:00"],
            "amt": ["10.5", 20.0],
            "is_fraud": [0, 1],
        }
    )

    normalized_df = normalize_training_dataset_types(df)

    assert pd.api.types.is_datetime64_any_dtype(normalized_df["dob"])
    assert pd.api.types.is_datetime64_any_dtype(normalized_df["current_time"])
    assert pd.api.types.is_numeric_dtype(normalized_df["amt"])
    assert pd.api.types.is_integer_dtype(normalized_df["is_fraud"])


def test_merge_training_dataset_adds_only_new_transactions():
    training_df = pd.DataFrame(
        {
            "trans_num": ["tx_001", "tx_002"],
            "amt": [10.0, 20.0],
            "is_fraud": [0, 1],
        }
    )

    new_transactions_df = pd.DataFrame(
        {
            "trans_num": ["tx_002", "tx_003"],
            "amt": [20.0, 30.0],
            "is_fraud": [1, 0],
        }
    )

    updated_df, integrated_trans_nums, already_present_trans_nums = (
        merge_training_dataset_with_new_transactions(
            training_df=training_df,
            new_transactions_df=new_transactions_df,
        )
    )

    assert len(updated_df) == 3
    assert integrated_trans_nums == ["tx_003"]
    assert already_present_trans_nums == ["tx_002"]
    assert "tx_003" in updated_df["trans_num"].tolist()


def test_merge_training_dataset_returns_no_rows_when_all_duplicates():
    training_df = pd.DataFrame(
        {
            "trans_num": ["tx_001"],
            "amt": [10.0],
            "is_fraud": [0],
        }
    )

    new_transactions_df = pd.DataFrame(
        {
            "trans_num": ["tx_001"],
            "amt": [10.0],
            "is_fraud": [0],
        }
    )

    updated_df, integrated_trans_nums, already_present_trans_nums = (
        merge_training_dataset_with_new_transactions(
            training_df=training_df,
            new_transactions_df=new_transactions_df,
        )
    )

    assert len(updated_df) == 1
    assert integrated_trans_nums == []
    assert already_present_trans_nums == ["tx_001"]


def test_merge_training_dataset_identifies_already_present_transactions_for_reconciliation():
    training_df = pd.DataFrame(
        {
            "trans_num": ["tx_001", "tx_002"],
            "amt": [10.0, 20.0],
            "is_fraud": [0, 1],
        }
    )

    labeled_transactions_df = pd.DataFrame(
        {
            "trans_num": ["tx_001", "tx_003"],
            "amt": [10.0, 30.0],
            "is_fraud": [0, 0],
        }
    )

    updated_df, integrated_trans_nums, already_present_trans_nums = (
        merge_training_dataset_with_new_transactions(
            training_df=training_df,
            new_transactions_df=labeled_transactions_df,
        )
    )

    assert len(updated_df) == 3
    assert integrated_trans_nums == ["tx_003"]
    assert already_present_trans_nums == ["tx_001"]

