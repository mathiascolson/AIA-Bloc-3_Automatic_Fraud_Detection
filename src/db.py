from __future__ import annotations

import math

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


def _replace_nan_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les NaN / NaT pandas par None pour insertion PostgreSQL.
    """

    return df.where(pd.notnull(df), None)


def _haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float | None:
    """
    Calcule la distance en kilomètres entre deux points GPS.

    Retourne None si une coordonnée est manquante ou invalide.
    """

    values = [lat1, lon1, lat2, lon2]

    if any(pd.isna(value) for value in values):
        return None

    try:
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))
    except (TypeError, ValueError):
        return None

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    earth_radius_km = 6371.0

    return round(earth_radius_km * c, 2)


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


def insert_predictions_batch(conn, df: pd.DataFrame) -> None:
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

    data = _replace_nan_with_none(data)

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


def create_fraud_alerts_table_if_not_exists(conn) -> None:
    """
    Crée la table d'alertes fraude si elle n'existe pas.

    Une alerte est créée uniquement pour les transactions prédites comme fraude.
    La contrainte UNIQUE sur trans_num évite les doublons lors des retries Airflow.
    """

    query = """
    CREATE TABLE IF NOT EXISTS fraud_alerts (
        id SERIAL PRIMARY KEY,
        trans_num TEXT UNIQUE NOT NULL,
        cc_num BIGINT,
        amt DOUBLE PRECISION,
        merchant TEXT,
        category TEXT,
        city TEXT,
        state TEXT,
        lat DOUBLE PRECISION,
        long DOUBLE PRECISION,
        merch_lat DOUBLE PRECISION,
        merch_long DOUBLE PRECISION,
        customer_merchant_distance_km DOUBLE PRECISION,
        transaction_time TIMESTAMP,
        fraud_probability DOUBLE PRECISION NOT NULL,
        fraud_alert_threshold DOUBLE PRECISION NOT NULL,
        model_name TEXT,
        model_alias TEXT,
        notification_channel TEXT DEFAULT 'discord',
        notification_status TEXT DEFAULT 'created',
        created_at TIMESTAMP DEFAULT NOW(),
        notified_at TIMESTAMP
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(query)

    conn.commit()


def insert_fraud_alerts_batch(
    conn,
    df: pd.DataFrame,
    notification_channel: str = "discord",
) -> pd.DataFrame:
    """
    Insère les nouvelles alertes fraude dans NeonDB.

    Seules les lignes avec is_fraud_predicted = 1 sont insérées.
    Les doublons sont ignorés via ON CONFLICT (trans_num) DO NOTHING.

    Retourne uniquement les alertes réellement créées pendant cet appel,
    afin d'éviter d'envoyer plusieurs notifications Discord pour la même fraude.
    """

    required_columns = [
        "trans_num",
        "amt",
        "category",
        "current_time",
        "is_fraud_predicted",
        "fraud_probability",
        "fraud_alert_threshold",
        "model_name",
        "model_alias",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes pour la création d'alertes : {missing_columns}"
        )

    fraud_alerts = df[df["is_fraud_predicted"] == 1].copy()

    if fraud_alerts.empty:
        print("[DB] Aucune fraude prédite, aucune alerte créée.")
        return fraud_alerts

    optional_columns = [
        "cc_num",
        "merchant",
        "city",
        "state",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "customer_merchant_distance_km",
    ]

    for column in optional_columns:
        if column not in fraud_alerts.columns:
            fraud_alerts[column] = None

    if fraud_alerts["customer_merchant_distance_km"].isna().all():
        fraud_alerts["customer_merchant_distance_km"] = fraud_alerts.apply(
            lambda row: _haversine_distance_km(
                row.get("lat"),
                row.get("long"),
                row.get("merch_lat"),
                row.get("merch_long"),
            ),
            axis=1,
        )

    cols = [
        "trans_num",
        "cc_num",
        "amt",
        "merchant",
        "category",
        "city",
        "state",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "customer_merchant_distance_km",
        "current_time",
        "fraud_probability",
        "fraud_alert_threshold",
        "model_name",
        "model_alias",
    ]

    data = fraud_alerts[cols].copy()

    data["current_time"] = pd.to_datetime(
        data["current_time"],
        errors="coerce",
    )

    data["notification_channel"] = notification_channel
    data["notification_status"] = "created"

    data = _replace_nan_with_none(data)

    records = data[
        [
            "trans_num",
            "cc_num",
            "amt",
            "merchant",
            "category",
            "city",
            "state",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "customer_merchant_distance_km",
            "current_time",
            "fraud_probability",
            "fraud_alert_threshold",
            "model_name",
            "model_alias",
            "notification_channel",
            "notification_status",
        ]
    ].values.tolist()

    query = """
    INSERT INTO fraud_alerts (
        trans_num,
        cc_num,
        amt,
        merchant,
        category,
        city,
        state,
        lat,
        long,
        merch_lat,
        merch_long,
        customer_merchant_distance_km,
        transaction_time,
        fraud_probability,
        fraud_alert_threshold,
        model_name,
        model_alias,
        notification_channel,
        notification_status
    )
    VALUES %s
    ON CONFLICT (trans_num) DO NOTHING
    RETURNING
        trans_num,
        cc_num,
        amt,
        merchant,
        category,
        city,
        state,
        lat,
        long,
        merch_lat,
        merch_long,
        customer_merchant_distance_km,
        transaction_time,
        fraud_probability,
        fraud_alert_threshold,
        model_name,
        model_alias,
        notification_channel,
        notification_status,
        created_at;
    """

    with conn.cursor() as cursor:
        inserted_rows = execute_values(
            cursor,
            query,
            records,
            fetch=True,
        )

    conn.commit()

    inserted_alerts = pd.DataFrame(
        inserted_rows,
        columns=[
            "trans_num",
            "cc_num",
            "amt",
            "merchant",
            "category",
            "city",
            "state",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "customer_merchant_distance_km",
            "transaction_time",
            "fraud_probability",
            "fraud_alert_threshold",
            "model_name",
            "model_alias",
            "notification_channel",
            "notification_status",
            "created_at",
        ],
    )

    print(
        "[DB] Nouvelles alertes fraude créées : "
        f"{len(inserted_alerts)}"
    )

    return inserted_alerts


def mark_fraud_alerts_as_notified(
    conn,
    trans_nums: list[str],
    notification_status: str = "sent",
) -> int:
    """
    Marque les alertes comme notifiées après envoi Discord.
    """

    if not trans_nums:
        print("[DB] Aucune alerte à marquer comme notifiée.")
        return 0

    query = """
    UPDATE fraud_alerts
    SET
        notification_status = %s,
        notified_at = NOW()
    WHERE trans_num = ANY(%s);
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                notification_status,
                trans_nums,
            ),
        )
        updated_count = cursor.rowcount

    conn.commit()

    print(
        "[DB] Alertes marquées comme notifiées : "
        f"{updated_count}"
    )

    return updated_count


def create_model_cd_decisions_table_if_not_exists(conn) -> None:
    """
    Crée la table d'historisation des décisions de CD modèle.
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


def create_labeled_transactions_table_if_not_exists(conn) -> None:
    """
    Crée la table des transactions complètes labellisées.

    Cette table sert au réentraînement incrémental :
    - chaque transaction validée est stockée avec son label is_fraud ;
    - les lignes non encore intégrées au dataset d'entraînement ont
      integrated_in_training = FALSE ;
    - après intégration dans data/processed/training_dataset.parquet,
      elles sont marquées comme intégrées.
    """

    query = """
    CREATE TABLE IF NOT EXISTS labeled_transactions (
        id SERIAL PRIMARY KEY,
        trans_num TEXT UNIQUE NOT NULL,
        cc_num BIGINT,
        merchant TEXT,
        category TEXT,
        amt DOUBLE PRECISION,
        first_name TEXT,
        last_name TEXT,
        gender TEXT,
        street TEXT,
        city TEXT,
        state TEXT,
        zip INTEGER,
        lat DOUBLE PRECISION,
        long DOUBLE PRECISION,
        city_pop INTEGER,
        job TEXT,
        dob DATE,
        merch_lat DOUBLE PRECISION,
        merch_long DOUBLE PRECISION,
        is_fraud INTEGER NOT NULL,
        transaction_time TIMESTAMP,
        received_at TIMESTAMP DEFAULT NOW(),
        integrated_in_training BOOLEAN DEFAULT FALSE,
        integrated_at TIMESTAMP
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(query)

    conn.commit()


def insert_labeled_transactions_batch(conn, df: pd.DataFrame) -> int:
    """
    Insère les transactions complètes labellisées dans NeonDB.

    Les doublons sont ignorés via ON CONFLICT (trans_num) DO NOTHING.
    """

    cols = [
        "trans_num",
        "cc_num",
        "merchant",
        "category",
        "amt",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "merch_lat",
        "merch_long",
        "is_fraud",
        "current_time",
    ]

    missing_columns = [col for col in cols if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes pour l'insertion des transactions "
            f"labellisées : {missing_columns}"
        )

    if df.empty:
        print("[DB] Aucune transaction labellisée à insérer.")
        return 0

    data = df[cols].copy()

    data["dob"] = pd.to_datetime(
        data["dob"],
        errors="coerce",
    ).dt.date

    data["current_time"] = pd.to_datetime(
        data["current_time"],
        errors="coerce",
    )

    data = _replace_nan_with_none(data)

    records = data.values.tolist()

    query = """
    INSERT INTO labeled_transactions (
        trans_num,
        cc_num,
        merchant,
        category,
        amt,
        first_name,
        last_name,
        gender,
        street,
        city,
        state,
        zip,
        lat,
        long,
        city_pop,
        job,
        dob,
        merch_lat,
        merch_long,
        is_fraud,
        transaction_time
    )
    VALUES %s
    ON CONFLICT (trans_num) DO NOTHING;
    """

    with conn.cursor() as cursor:
        execute_values(cursor, query, records)
        inserted_count = cursor.rowcount

    conn.commit()

    print(
        "[DB] Transactions labellisées insérées dans NeonDB : "
        f"{inserted_count}"
    )

    return inserted_count


def count_unintegrated_labeled_transactions(conn) -> int:
    """
    Compte les transactions labellisées non encore intégrées
    au dataset d'entraînement.
    """

    query = """
    SELECT COUNT(*)
    FROM labeled_transactions
    WHERE integrated_in_training = FALSE;
    """

    with conn.cursor() as cursor:
        cursor.execute(query)
        count = cursor.fetchone()[0]

    return int(count)


def fetch_unintegrated_labeled_transactions(conn) -> pd.DataFrame:
    """
    Récupère les transactions labellisées non encore intégrées
    au dataset d'entraînement.

    Les colonnes retournées reprennent les noms attendus par le pipeline
    d'entraînement, notamment first et last.
    """

    query = """
    SELECT
        trans_num,
        cc_num,
        merchant,
        category,
        amt,
        first_name AS first,
        last_name AS last,
        gender,
        street,
        city,
        state,
        zip,
        lat,
        long,
        city_pop,
        job,
        dob,
        merch_lat,
        merch_long,
        is_fraud,
        transaction_time AS current_time
    FROM labeled_transactions
    WHERE integrated_in_training = FALSE
    ORDER BY received_at ASC;
    """

    df = pd.read_sql_query(query, conn)

    return df


def mark_labeled_transactions_as_integrated(
    conn,
    trans_nums: list[str],
) -> int:
    """
    Marque comme intégrées les transactions ajoutées au dataset
    d'entraînement processed.
    """

    if not trans_nums:
        print("[DB] Aucune transaction labellisée à marquer comme intégrée.")
        return 0

    query = """
    UPDATE labeled_transactions
    SET
        integrated_in_training = TRUE,
        integrated_at = NOW()
    WHERE trans_num = ANY(%s)
      AND integrated_in_training = FALSE;
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (trans_nums,))
        updated_count = cursor.rowcount

    conn.commit()

    print(
        "[DB] Transactions labellisées marquées comme intégrées : "
        f"{updated_count}"
    )

    return updated_count