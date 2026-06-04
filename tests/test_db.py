import pandas as pd
import pytest

from src.db import insert_labeled_transactions_batch, insert_predictions_batch


def test_insert_predictions_batch_raises_when_required_columns_missing():
    df = pd.DataFrame(
        {
            "trans_num": ["abc"],
            "cc_num": [123],
        }
    )

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        insert_predictions_batch(conn=None, df=df)


def test_insert_labeled_transactions_batch_raises_when_required_columns_missing():
    df = pd.DataFrame(
        {
            "trans_num": ["abc"],
            "cc_num": [123],
            "is_fraud": [0],
        }
    )

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        insert_labeled_transactions_batch(conn=None, df=df)