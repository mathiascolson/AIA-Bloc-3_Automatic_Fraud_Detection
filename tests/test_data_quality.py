import pandas as pd
import pytest

from src.data_quality import DataQualityError, validate_incoming_transactions


def make_valid_transactions_df():
    return pd.DataFrame(
        {
            "cc_num": [123456789],
            "merchant": ["fraud_Test Merchant"],
            "category": ["shopping_pos"],
            "amt": [10.0],
            "first": ["John"],
            "last": ["Doe"],
            "gender": ["M"],
            "street": ["1 Test Street"],
            "city": ["Paris"],
            "state": ["FR"],
            "zip": [75000],
            "lat": [48.8566],
            "long": [2.3522],
            "city_pop": [2000000],
            "job": ["Engineer"],
            "dob": ["1990-01-01"],
            "trans_num": ["tx_001"],
            "merch_lat": [48.85],
            "merch_long": [2.35],
            "is_fraud": [0],
            "current_time": ["2026-06-04T10:00:00"],
        }
    )


def test_validate_incoming_transactions_passes_on_valid_data():
    df = make_valid_transactions_df()

    validate_incoming_transactions(df)


def test_validate_incoming_transactions_fails_on_negative_amount():
    df = make_valid_transactions_df()
    df.loc[0, "amt"] = -10.0

    with pytest.raises(DataQualityError):
        validate_incoming_transactions(df)


def test_validate_incoming_transactions_fails_on_invalid_label():
    df = make_valid_transactions_df()
    df.loc[0, "is_fraud"] = 3

    with pytest.raises(DataQualityError):
        validate_incoming_transactions(df)


def test_validate_incoming_transactions_fails_on_duplicate_trans_num():
    df = pd.concat(
        [
            make_valid_transactions_df(),
            make_valid_transactions_df(),
        ],
        ignore_index=True,
    )

    with pytest.raises(DataQualityError):
        validate_incoming_transactions(df)