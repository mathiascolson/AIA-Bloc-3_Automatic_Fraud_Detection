import json
from datetime import date

from src.reporting_pipeline import run_daily_report_pipeline


def main() -> None:
    result = run_daily_report_pipeline(
        target_date=date(2026, 6, 2)
    )

    print("Daily report pipeline executed successfully.")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()