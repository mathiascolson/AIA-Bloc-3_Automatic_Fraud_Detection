import pandas as pd

from src.config import settings
from src.s3_utils import read_csv_from_s3


def main() -> None:
    df = read_csv_from_s3(settings.s3_raw_data_key)

    print("\n=== BASIC INFO ===")
    print("Shape:", df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    print("\n=== TARGET DISTRIBUTION ===")
    target_counts = df["is_fraud"].value_counts(dropna=False)
    target_rates = df["is_fraud"].value_counts(normalize=True, dropna=False)

    print(target_counts)
    print("\nRates:")
    print(target_rates)

    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe())

    print("\n=== CATEGORICAL CARDINALITY ===")
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")

    print("\n=== SAMPLE FRAUDS ===")
    print(df[df["is_fraud"] == 1].head())

    print("\n=== SAMPLE NON-FRAUDS ===")
    print(df[df["is_fraud"] == 0].head())


if __name__ == "__main__":
    main()