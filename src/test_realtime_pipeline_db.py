import json

from src.realtime_pipeline import run_realtime_pipeline_once


def main() -> None:
    result = run_realtime_pipeline_once()

    print("Realtime pipeline executed successfully.")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()