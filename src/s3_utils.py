import io
from pathlib import Path

import boto3
import pandas as pd

from src.config import get_settings


def get_s3_client():
    """
    Create and return a boto3 S3 client using environment variables.
    """
    settings = get_settings()

    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_default_region,
    )


def upload_file_to_s3(local_path: str | Path, s3_key: str) -> None:
    """
    Upload a local file to S3.
    """
    settings = get_settings()
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if not settings.s3_bucket_name:
        raise ValueError("S3_BUCKET_NAME is missing from environment variables.")

    s3 = get_s3_client()

    s3.upload_file(
        Filename=str(local_path),
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
    )

    print(f"Uploaded to s3://{settings.s3_bucket_name}/{s3_key}")


def read_csv_from_s3(s3_key: str) -> pd.DataFrame:
    """
    Read a CSV file from S3 into a pandas DataFrame.
    """
    settings = get_settings()

    if not settings.s3_bucket_name:
        raise ValueError("S3_BUCKET_NAME is missing from environment variables.")

    s3 = get_s3_client()

    response = s3.get_object(
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
    )

    content = response["Body"].read()

    return pd.read_csv(io.BytesIO(content))


def write_dataframe_to_s3_csv(df: pd.DataFrame, s3_key: str) -> None:
    """
    Write a pandas DataFrame to S3 as CSV.
    """
    settings = get_settings()

    if not settings.s3_bucket_name:
        raise ValueError("S3_BUCKET_NAME is missing from environment variables.")

    s3 = get_s3_client()

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    s3.put_object(
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv; charset=utf-8",
    )

    print(f"DataFrame written to s3://{settings.s3_bucket_name}/{s3_key}")


def write_dataframe_to_s3_parquet(df: pd.DataFrame, s3_key: str) -> None:
    """
    Write a pandas DataFrame to S3 as Parquet.
    """
    settings = get_settings()

    if not settings.s3_bucket_name:
        raise ValueError("S3_BUCKET_NAME is missing from environment variables.")

    s3 = get_s3_client()

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)

    s3.put_object(
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    print(f"DataFrame written to s3://{settings.s3_bucket_name}/{s3_key}")