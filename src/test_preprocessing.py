from src.config import settings
from src.s3_utils import read_csv_from_s3
from src.preprocessing import prepare_features_and_target, build_preprocessor


def main() -> None:
    df = read_csv_from_s3(settings.s3_raw_data_key)

    X, y = prepare_features_and_target(df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nX columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    print("\nTransformed X shape:", X_transformed.shape)
    print("Preprocessing test successful.")


if __name__ == "__main__":
    main()