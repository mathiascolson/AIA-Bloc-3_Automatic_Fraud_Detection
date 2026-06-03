from __future__ import annotations


RAW_INPUT_SCHEMA = {
    "cc_num": "int64",
    "merchant": "object",
    "category": "object",
    "amt": "float64",
    "gender": "object",
    "city": "object",
    "state": "object",
    "zip": "int64",
    "lat": "float64",
    "long": "float64",
    "city_pop": "int64",
    "job": "object",
    "dob": "object",
    "trans_num": "object",
    "merch_lat": "float64",
    "merch_long": "float64",
    "current_time": "datetime64[ns]",
}

RAW_INPUT_COLUMNS = list(RAW_INPUT_SCHEMA.keys())

CRITICAL_COLUMNS = [
    "cc_num",
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "merch_lat",
    "merch_long",
    "current_time",
]

MODEL_FEATURE_COLUMNS = [
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "merch_lat",
    "merch_long",
    "customer_age",
    "transaction_hour",
    "transaction_dayofweek",
    "transaction_month",
    "is_weekend",
    "distance_customer_merchant_km",
]

# Compatibilité avec le code existant.
MODEL_SCHEMA = RAW_INPUT_SCHEMA
EXPECTED_COLUMNS = RAW_INPUT_COLUMNS