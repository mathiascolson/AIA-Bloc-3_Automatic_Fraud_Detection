import os
import pandas as pd
from dotenv import load_dotenv

from src.notification import send_discord_fraud_alerts


load_dotenv()

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

alerts = pd.DataFrame(
    [
        {
            "trans_num": "test_discord_alert_001",
            "fraud_probability": 0.9876,
            "fraud_alert_threshold": 0.90,
            "amt": 15.86,
            "merchant": "fraud_Streich Ltd",
            "category": "home",
            "cc_num": "5ff16478f9ba6d3b6f0d6de8c8812876c161a0d3bcf47c89ac10c11858af445f",
            "city": "San Antonio",
            "state": "TX",
            "lat": 29.3641,
            "long": -98.4924,
            "merch_lat": 29.885253,
            "merch_long": -98.311015,
            "distance_km": 60.54,
            "trans_date_trans_time": "2026-06-03T07:30:48.646000",
        }
    ]
)

sent = send_discord_fraud_alerts(
    alerts=alerts,
    webhook_url=webhook_url,
)

print(f"Notifications envoyées : {sent}")