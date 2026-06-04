import pandas as pd
from scipy import sparse

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def _numeric_values_for_feature(feature: str) -> list[float]:
    examples = {
        "amt": [10.0, 20.0],
        "zip": [10001, 90001],
        "lat": [40.0, 41.0],
        "long": [-70.0, -71.0],
        "city_pop": [1000, 2000],
        "merch_lat": [40.5, 41.5],
        "merch_long": [-70.5, -71.5],
        "customer_age": [35, 42],
        "transaction_hour": [10, 22],
        "transaction_dayofweek": [1, 5],
        "transaction_day_of_week": [1, 5],
        "transaction_month": [1, 12],
    }

    return examples.get(feature, [1.0, 2.0])


def _categorical_values_for_feature(feature: str) -> list[str]:
    examples = {
        "category": ["shopping_pos", "gas_transport"],
        "gender": ["M", "F"],
        "state": ["NY", "CA"],
        "job": ["Engineer", "Teacher"],
    }

    return examples.get(feature, ["A", "B"])


def test_preprocessor_outputs_sparse_matrix():
    data = {}

    for feature in NUMERIC_FEATURES:
        data[feature] = _numeric_values_for_feature(feature)

    for feature in CATEGORICAL_FEATURES:
        data[feature] = _categorical_values_for_feature(feature)

    df = pd.DataFrame(data)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert sparse.issparse(transformed)