from pathlib import Path

import pandas as pd
import requests


def collect_weather(latitude, longitude, start_date, end_date):
    project_dir = Path(__file__).resolve().parent.parent.parent
    external_data_dir = project_dir / "data" / "external_data"
    filename = f"weather_{latitude}_{longitude}_{start_date}_{end_date}"
    raw_path = external_data_dir / "raw" / "weather" / f"{filename}.json"
    csv_path = external_data_dir / "processed" / "weather" / f"{filename}.csv"

    for path in (raw_path, csv_path):
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "cloud_cover,wind_speed_10m"
        ),
        "timezone": "Asia/Bangkok",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data.get("reason", "Open-Meteo returned an error"))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as file:
        file.write(response.content)

    weather = pd.DataFrame(data["hourly"])
    if weather.empty:
        raise ValueError("Open-Meteo returned no hourly records")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(csv_path, index=False, encoding="utf-8", mode="x")

    print(f"Saved raw response: {raw_path}")
    print(f"Saved hourly CSV: {csv_path}")
    return weather


if __name__ == "__main__":
    try:
        weather = collect_weather(13.7563, 100.5018, "2026-08-25", "2026-08-31")
        print(f"\nHourly records: {len(weather)}")
        print(weather.head().to_string(index=False))
    except (requests.RequestException, ValueError, OSError) as error:
        raise SystemExit(f"Weather collection failed: {error}")
