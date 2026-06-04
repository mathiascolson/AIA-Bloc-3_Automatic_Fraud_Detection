import pandas as pd

from src.xcom_utils import (
    dataframe_to_xcom_json,
    xcom_json_to_dataframe,
)


def test_dataframe_xcom_roundtrip():
    df = pd.DataFrame(
        {
            "trans_num": ["abc"],
            "amt": [12.5],
            "is_fraud": [0],
        }
    )

    raw_json = dataframe_to_xcom_json(df)
    restored = xcom_json_to_dataframe(raw_json)

    assert restored.shape == df.shape
    assert restored.loc[0, "trans_num"] == "abc"
    assert restored.loc[0, "amt"] == 12.5
    assert restored.loc[0, "is_fraud"] == 0