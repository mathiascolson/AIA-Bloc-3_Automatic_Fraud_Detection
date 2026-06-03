from src.config import settings
from src.s3_utils import read_csv_from_s3


def main() -> None:
    print("Bucket:", settings.s3_bucket_name)
    print("Raw data key:", settings.s3_raw_data_key)

    df = read_csv_from_s3(settings.s3_raw_data_key)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print(df.head())


if __name__ == "__main__":
    main()