from pathlib import Path
import sys

import pandas as pd
import requests


def collect_holidays():
    project_dir = Path(__file__).resolve().parent.parent.parent
    raw_path = (
        project_dir / "data" / "external_data" / "raw" / "holidays"
        / "thailand_public_holidays_2026.json"
    )
    if raw_path.exists():
        raise FileExistsError(f"File already exists: {raw_path}")

    url = (
        "https://raw.githubusercontent.com/ppraserts/thailand-open-data/"
        "main/data/thai-public-holidays/2026.json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as file:
        file.write(response.content)

    holidays = pd.DataFrame(data["holidays"])
    print(f"HTTP status: {response.status_code}")
    print(f"Saved raw JSON: {raw_path}")
    print(f"Holiday records: {len(holidays)}")
    print("Available columns:", holidays.columns.tolist())
    print("\nFirst 5 records:")
    print(holidays.head(5).to_string(index=False))
    return holidays


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        holidays = collect_holidays()
    except (requests.RequestException, ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Holiday collection failed: {error}")
