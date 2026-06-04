from __future__ import annotations

from io import StringIO

import pandas as pd


def dataframe_to_xcom_json(df: pd.DataFrame) -> str:
    """
    Convertit un DataFrame en JSON compatible XCom Airflow.
    """

    return df.to_json(
        orient="split",
        date_format="iso",
        index=True,
    )


def xcom_json_to_dataframe(raw_json: str) -> pd.DataFrame:
    """
    Reconstruit un DataFrame depuis un JSON XCom Airflow.
    """

    return pd.read_json(
        StringIO(raw_json),
        orient="split",
    )