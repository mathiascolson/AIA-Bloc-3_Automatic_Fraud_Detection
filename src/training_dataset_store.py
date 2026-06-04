from __future__ import annotations

from io import BytesIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError


def get_s3_client():
    """
    Crée un client S3 à partir des variables d'environnement AWS.
    """

    return boto3.client("s3")


def s3_object_exists(
    bucket_name: str,
    object_key: str,
) -> bool:
    """
    Vérifie si un objet existe dans S3.
    """

    s3_client = get_s3_client()

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key=object_key,
        )
        return True

    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")

        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False

        raise


def read_csv_from_s3(
    bucket_name: str,
    object_key: str,
) -> pd.DataFrame:
    """
    Lit un fichier CSV depuis S3.
    """

    s3_client = get_s3_client()

    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=object_key,
    )

    return pd.read_csv(response["Body"])


def read_parquet_from_s3(
    bucket_name: str,
    object_key: str,
) -> pd.DataFrame:
    """
    Lit un fichier Parquet depuis S3.
    """

    s3_client = get_s3_client()

    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=object_key,
    )

    body = response["Body"].read()

    return pd.read_parquet(BytesIO(body))


def write_parquet_to_s3(
    df: pd.DataFrame,
    bucket_name: str,
    object_key: str,
) -> None:
    """
    Écrit un DataFrame en Parquet dans S3.
    """

    s3_client = get_s3_client()

    normalized_df = normalize_training_dataset_types(df)

    buffer = BytesIO()

    normalized_df.to_parquet(
        buffer,
        index=False,
    )

def load_or_initialize_training_dataset(
    bucket_name: str,
    raw_data_key: str,
    processed_data_key: str,
) -> pd.DataFrame:
    """
    Charge le dataset processed s'il existe.

    Sinon, initialise le dataset processed depuis le dataset raw historique.
    """

    if s3_object_exists(
        bucket_name=bucket_name,
        object_key=processed_data_key,
    ):
        print(
            "[S3] Chargement du dataset processed : "
            f"s3://{bucket_name}/{processed_data_key}"
        )

        return read_parquet_from_s3(
            bucket_name=bucket_name,
            object_key=processed_data_key,
        )

    print(
        "[S3] Dataset processed absent. Initialisation depuis raw : "
        f"s3://{bucket_name}/{raw_data_key}"
    )

    training_df = read_csv_from_s3(
        bucket_name=bucket_name,
        object_key=raw_data_key,
    )

    write_parquet_to_s3(
        df=training_df,
        bucket_name=bucket_name,
        object_key=processed_data_key,
    )

    return training_df


def merge_training_dataset_with_new_transactions(
    training_df: pd.DataFrame,
    new_transactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Ajoute les nouvelles transactions labellisées au dataset d'entraînement.

    Retourne :
    - le dataset mis à jour ;
    - les trans_num effectivement ajoutés ;
    - les trans_num déjà présents dans le dataset.
    """

    if new_transactions_df.empty:
        return training_df.copy(), [], []

    if "trans_num" not in training_df.columns:
        raise ValueError(
            "La colonne trans_num est absente du dataset d'entraînement."
        )

    if "trans_num" not in new_transactions_df.columns:
        raise ValueError(
            "La colonne trans_num est absente des nouvelles transactions."
        )

    existing_trans_nums = set(
        training_df["trans_num"]
        .dropna()
        .astype(str)
        .tolist()
    )

    candidate_transactions = new_transactions_df.copy()

    candidate_transactions["_trans_num_str"] = (
        candidate_transactions["trans_num"]
        .astype(str)
    )

    already_present_transactions = candidate_transactions[
        candidate_transactions["_trans_num_str"].isin(existing_trans_nums)
    ].copy()

    new_transactions = candidate_transactions[
        ~candidate_transactions["_trans_num_str"].isin(existing_trans_nums)
    ].copy()

    already_present_trans_nums = (
        already_present_transactions["_trans_num_str"]
        .dropna()
        .astype(str)
        .tolist()
    )

    new_transactions = new_transactions.drop(
        columns=["_trans_num_str"]
    )

    if new_transactions.empty:
        return (
            training_df.copy(),
            [],
            already_present_trans_nums,
        )

    merged_columns = list(training_df.columns)

    extra_columns = [
        column
        for column in new_transactions.columns
        if column not in merged_columns
    ]

    merged_columns = merged_columns + extra_columns

    aligned_training_df = training_df.reindex(
        columns=merged_columns
    )

    aligned_new_transactions = new_transactions.reindex(
        columns=merged_columns
    )

    updated_training_df = pd.concat(
        [
            aligned_training_df,
            aligned_new_transactions,
        ],
        ignore_index=True,
    )

    integrated_trans_nums = (
        new_transactions["trans_num"]
        .astype(str)
        .tolist()
    )

    return (
        updated_training_df,
        integrated_trans_nums,
        already_present_trans_nums,
    )


def append_labeled_transactions_to_training_dataset(
    bucket_name: str,
    raw_data_key: str,
    processed_data_key: str,
    labeled_transactions_df: pd.DataFrame,
) -> list[str]:
    """
    Ajoute les transactions labellisées non intégrées au dataset processed S3.

    Retourne les trans_num effectivement ajoutés.
    """

    training_df = load_or_initialize_training_dataset(
        bucket_name=bucket_name,
        raw_data_key=raw_data_key,
        processed_data_key=processed_data_key,
    )

    updated_training_df, integrated_trans_nums, already_present_trans_nums = (
        merge_training_dataset_with_new_transactions(
            training_df=training_df,
            new_transactions_df=labeled_transactions_df,
        )
    )

    if not integrated_trans_nums:
        print(
            "[S3] Aucune nouvelle transaction à ajouter au dataset "
            "d'entraînement."
        )
        print(
            "[S3] Transactions déjà présentes dans le dataset : "
            f"{len(already_present_trans_nums)}"
        )
        return []

    write_parquet_to_s3(
        df=updated_training_df,
        bucket_name=bucket_name,
        object_key=processed_data_key,
    )

    print(
        "[S3] Transactions ajoutées au dataset d'entraînement : "
        f"{len(integrated_trans_nums)}"
    )

    return integrated_trans_nums

def normalize_training_dataset_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les types avant écriture Parquet.

    Objectif :
    - éviter les colonnes object hétérogènes ;
    - garantir une écriture stable avec pyarrow ;
    - conserver les noms de colonnes attendus par le pipeline d'entraînement.
    """

    normalized_df = df.copy()

    string_columns = [
        "trans_num",
        "merchant",
        "category",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "job",
    ]

    numeric_columns = [
        "cc_num",
        "amt",
        "zip",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "is_fraud",
    ]

    datetime_columns = [
        "dob",
        "current_time",
    ]

    for column in string_columns:
        if column in normalized_df.columns:
            normalized_df[column] = normalized_df[column].astype("string")

    for column in numeric_columns:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_numeric(
                normalized_df[column],
                errors="coerce",
            )

    for column in datetime_columns:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_datetime(
                normalized_df[column],
                errors="coerce",
            )

    if "is_fraud" in normalized_df.columns:
        normalized_df["is_fraud"] = (
            normalized_df["is_fraud"]
            .fillna(0)
            .astype(int)
        )

    return normalized_df