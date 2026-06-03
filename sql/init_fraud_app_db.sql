-- ============================================================
-- Fraud Detection Application Database Initialization
-- ============================================================

-- Drop existing tables if needed
-- Uncomment only if you want to fully reset the application database.
--
-- DROP TABLE IF EXISTS daily_reports CASCADE;
-- DROP TABLE IF EXISTS alerts CASCADE;
-- DROP TABLE IF EXISTS predictions CASCADE;
-- DROP TABLE IF EXISTS transactions CASCADE;


-- ============================================================
-- Transactions table
-- Stores raw and normalized payment transactions.
-- ============================================================

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    transaction_datetime TIMESTAMP,
    amount NUMERIC,
    currency TEXT,
    merchant_id TEXT,
    merchant_category TEXT,
    customer_id TEXT,
    country TEXT,
    payment_method TEXT,
    raw_payload JSONB,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- Predictions table
-- Stores fraud prediction outputs for each transaction.
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    transaction_id TEXT REFERENCES transactions(transaction_id),
    prediction INTEGER,
    fraud_probability NUMERIC,
    model_name TEXT,
    model_version TEXT,
    mlflow_run_id TEXT,
    prediction_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- Alerts table
-- Stores fraud alerts generated when the fraud probability
-- exceeds the configured production threshold.
-- ============================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    transaction_id TEXT REFERENCES transactions(transaction_id),
    fraud_probability NUMERIC,
    alert_threshold NUMERIC,
    notification_channel TEXT,
    notification_status TEXT,
    notification_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- Daily reports table
-- Stores daily monitoring aggregates.
-- ============================================================

CREATE TABLE IF NOT EXISTS daily_reports (
    report_id SERIAL PRIMARY KEY,
    report_date DATE UNIQUE,
    total_transactions INTEGER,
    detected_frauds INTEGER,
    fraud_rate NUMERIC,
    total_amount NUMERIC,
    fraud_amount NUMERIC,
    report_s3_key TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- Standard indexes
-- Improve query performance for monitoring and reporting.
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_transactions_transaction_datetime
ON transactions(transaction_datetime);

CREATE INDEX IF NOT EXISTS idx_transactions_inserted_at
ON transactions(inserted_at);

CREATE INDEX IF NOT EXISTS idx_transactions_merchant_category
ON transactions(merchant_category);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id
ON predictions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_predictions_prediction_datetime
ON predictions(prediction_datetime);

CREATE INDEX IF NOT EXISTS idx_predictions_prediction
ON predictions(prediction);

CREATE INDEX IF NOT EXISTS idx_alerts_transaction_id
ON alerts(transaction_id);

CREATE INDEX IF NOT EXISTS idx_alerts_notification_datetime
ON alerts(notification_datetime);

CREATE INDEX IF NOT EXISTS idx_daily_reports_report_date
ON daily_reports(report_date);


-- ============================================================
-- Idempotency indexes
-- Prevent duplicate predictions and duplicate alerts when
-- an Airflow task is retried, replayed, or manually tested.
-- ============================================================

CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_transaction_id_unique
ON predictions(transaction_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_transaction_id_unique
ON alerts(transaction_id);