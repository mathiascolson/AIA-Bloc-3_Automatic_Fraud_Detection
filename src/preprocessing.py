from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_schema import MODEL_FEATURE_COLUMNS


NUMERIC_FEATURES = [
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "customer_age",
    "transaction_hour",
    "transaction_dayofweek",
    "transaction_month",
    "is_weekend",
    "distance_customer_merchant_km",
]

CATEGORICAL_FEATURES = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job",
]


def compute_customer_age(
    current_time: pd.Series,
    dob: pd.Series,
) -> pd.Series:
    current_time_dt = pd.to_datetime(current_time, errors="coerce")
    dob_dt = pd.to_datetime(dob, errors="coerce")

    age = (current_time_dt - dob_dt).dt.days / 365.25

    return age


def compute_haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    earth_radius_km = 6371.0

    lat1_rad = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1_rad = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2_rad = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2_rad = np.radians(pd.to_numeric(lon2, errors="coerce"))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "current_time" not in data.columns:
        if "trans_date_trans_time" in data.columns:
            data["current_time"] = data["trans_date_trans_time"]
        else:
            raise ValueError(
                "Colonne temporelle manquante : aucune colonne "
                "'current_time' ou 'trans_date_trans_time' trouvée."
            )

    data["current_time"] = pd.to_datetime(
        data["current_time"],
        errors="coerce",
    )

    data["dob"] = pd.to_datetime(
        data["dob"],
        errors="coerce",
    )

    data["customer_age"] = compute_customer_age(
        current_time=data["current_time"],
        dob=data["dob"],
    )

    data["transaction_hour"] = data["current_time"].dt.hour
    data["transaction_dayofweek"] = data["current_time"].dt.dayofweek
    data["transaction_month"] = data["current_time"].dt.month
    data["is_weekend"] = data["transaction_dayofweek"].isin([5, 6]).astype(int)

    data["distance_customer_merchant_km"] = compute_haversine_distance_km(
        lat1=data["lat"],
        lon1=data["long"],
        lat2=data["merch_lat"],
        lon2=data["merch_long"],
    )

    missing_features = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Colonnes de features manquantes après preprocessing : "
            f"{missing_features}"
        )

    return data[MODEL_FEATURE_COLUMNS].copy()


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "is_fraud" not in df.columns:
        raise ValueError("La colonne cible 'is_fraud' est absente du dataset.")

    X = prepare_features_for_inference(df)
    y = df["is_fraud"].astype(int)

    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
        remainder="drop",
    )

    return preprocessor