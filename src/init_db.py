from pathlib import Path

from sqlalchemy import create_engine, text

from src.config import settings


def init_database() -> None:
    if not settings.fraud_database_url:
        raise ValueError("FRAUD_DATABASE_URL is missing from environment variables.")

    sql_path = Path("sql/init_fraud_app_db.sql")

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_script = sql_path.read_text(encoding="utf-8")

    engine = create_engine(settings.fraud_database_url)

    with engine.begin() as connection:
        for statement in sql_script.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(text(statement))

    print("Fraud application database initialized successfully.")


if __name__ == "__main__":
    init_database()