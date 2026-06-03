from __future__ import annotations

from io import StringIO

import pandas as pd

from src.model_schema import EXPECTED_COLUMNS, CRITICAL_COLUMNS


def validate_transactions(df: pd.DataFrame) -> None:
    """
    Valide les transactions avant preprocessing et inférence.

    Contrôles :
    - présence des colonnes attendues ;
    - absence de valeurs nulles sur les colonnes critiques ;
    - cohérence des montants ;
    - validité des coordonnées géographiques ;
    - convertibilité de current_time en datetime.
    """

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    null_columns = [
        col for col in CRITICAL_COLUMNS
        if df[col].isnull().any()
    ]

    if null_columns:
        raise ValueError(f"Valeurs nulles détectées dans : {null_columns}")

    if (df["amt"] < 0).any():
        raise ValueError("Montants négatifs détectés dans la colonne amt.")

    if not df["lat"].between(-90, 90).all():
        raise ValueError("Latitude client invalide détectée dans la colonne lat.")

    if not df["long"].between(-180, 180).all():
        raise ValueError("Longitude client invalide détectée dans la colonne long.")

    if not df["merch_lat"].between(-90, 90).all():
        raise ValueError("Latitude marchand invalide détectée dans la colonne merch_lat.")

    if not df["merch_long"].between(-180, 180).all():
        raise ValueError("Longitude marchand invalide détectée dans la colonne merch_long.")

    try:
        pd.to_datetime(df["current_time"])
    except Exception as exc:
        raise ValueError(
            "La colonne current_time ne peut pas être convertie en datetime."
        ) from exc


def dataframe_to_xcom_json(df: pd.DataFrame) -> str:
    """
    Convertit un DataFrame en JSON compatible avec XCom.
    """

    return df.to_json(orient="split", date_format="iso")


def xcom_json_to_dataframe(raw_json: str) -> pd.DataFrame:
    """
    Reconstruit un DataFrame depuis un JSON XCom.
    """

    return pd.read_json(StringIO(raw_json), orient="split")