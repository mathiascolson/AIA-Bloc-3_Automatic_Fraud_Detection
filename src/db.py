from __future__ import annotations

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


def get_connection(database_url: str):
    """
    Ouvre une connexion PostgreSQL / NeonDB.
    """

    if not database_url:
        raise ValueError("URL de connexion PostgreSQL / NeonDB manquante.")

    return psycopg2.connect(database_url)


def create_predictions_table_if_not_exists(conn) -> None:
    """
    Crée la table d'historisation des prédictions si elle n'existe pas.
    """

    query = """
    CREATE TABLE IF NOT EXISTS fraud_predictions (
        id SERIAL PRIMARY KEY,
        trans_num TEXT,
        cc_num BIGINT,
        amt DOUBLE PRECISION,
        category TEXT,
        transaction_time TIMESTAMP,
        is_fraud_predicted INTEGER,
        fraud_probability DOUBLE PRECISION,
        fraud_alert_threshold DOUBLE PRECISION,
        model_name TEXT,
        model_alias TEXT,
        predicted_at TIMESTAMP DEFAULT NOW()
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(query)

    conn.commit()


def insert_predictions_batch(conn, df) -> None:
    """
    Insère les prédictions en batch avec execute_values.
    """

    cols = [
        "trans_num",
        "cc_num",
        "amt",
        "category",
        "current_time",
        "is_fraud_predicted",
        "fraud_probability",
        "fraud_alert_threshold",
        "model_name",
        "model_alias",
    ]

    missing_columns = [col for col in cols if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes pour l'insertion NeonDB : {missing_columns}"
        )

    if df.empty:
        print("[DB] Aucune prédiction à insérer.")
        return

    data = df[cols].copy()

    data["current_time"] = pd.to_datetime(
        data["current_time"],
        errors="coerce",
    )

    records = data.values.tolist()

    query = """
    INSERT INTO fraud_predictions (
        trans_num,
        cc_num,
        amt,
        category,
        transaction_time,
        is_fraud_predicted,
        fraud_probability,
        fraud_alert_threshold,
        model_name,
        model_alias
    )
    VALUES %s;
    """

    with conn.cursor() as cursor:
        execute_values(cursor, query, records)

    conn.commit()

    print(f"[DB] Prédictions insérées dans NeonDB : {len(records)}")


def create_model_cd_decisions_table_if_not_exists(conn) -> None:
    """
    Crée la table d'historisation des décisions de CD modèle.
    Sera surtout utilisée dans la phase réentraînement/CD.
    """

    query = """
    CREATE TABLE IF NOT EXISTS model_cd_decisions (
        id SERIAL PRIMARY KEY,
        model_name TEXT,
        champion_version TEXT,
        challenger_version TEXT,
        champion_average_precision DOUBLE PRECISION,
        challenger_average_precision DOUBLE PRECISION,
        champion_recall_fraud DOUBLE PRECISION,
        challenger_recall_fraud DOUBLE PRECISION,
        challenger_precision_fraud DOUBLE PRECISION,
        promoted BOOLEAN,
        decision_reason TEXT,
        decided_at TIMESTAMP DEFAULT NOW()
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(query)

    conn.commit()
    
def insert_model_cd_decision(conn, decision: dict) -> None:
    """
    Insère une décision de CD modèle dans NeonDB.
    """

    required_keys = [
        "model_name",
        "champion_version",
        "challenger_version",
        "champion_average_precision",
        "challenger_average_precision",
        "champion_recall_fraud",
        "challenger_recall_fraud",
        "challenger_precision_fraud",
        "promoted",
        "decision_reason",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in decision
    ]

    if missing_keys:
        raise ValueError(
            f"Clés manquantes pour l'insertion CD NeonDB : {missing_keys}"
        )

    query = """
    INSERT INTO model_cd_decisions (
        model_name,
        champion_version,
        challenger_version,
        champion_average_precision,
        challenger_average_precision,
        champion_recall_fraud,
        challenger_recall_fraud,
        challenger_precision_fraud,
        promoted,
        decision_reason
    )
    VALUES (
        %(model_name)s,
        %(champion_version)s,
        %(challenger_version)s,
        %(champion_average_precision)s,
        %(challenger_average_precision)s,
        %(champion_recall_fraud)s,
        %(challenger_recall_fraud)s,
        %(challenger_precision_fraud)s,
        %(promoted)s,
        %(decision_reason)s
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(query, decision)

    conn.commit()

    print(
        "[DB] Décision CD insérée dans NeonDB : "
        f"model={decision['model_name']} | "
        f"champion={decision['champion_version']} | "
        f"challenger={decision['challenger_version']} | "
        f"promoted={decision['promoted']}"
    )