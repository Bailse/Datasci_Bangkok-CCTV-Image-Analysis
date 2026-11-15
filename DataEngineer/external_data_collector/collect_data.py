import argparse
from pathlib import Path
import sys
import time

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


def collect_era5_grids():
    project_dir = Path(__file__).resolve().parent.parent.parent
    external_dir = project_dir / "data" / "external_data"
    raw_dir = external_dir / "raw" / "weather"
    mapping_path = external_dir / "processed" / "weather" / "camera_weather_grid_mapping.csv"
    grids = [
        ("A", 915, 13.72639, 100.49213, 13.75, 100.50),
        ("B", 1804, 13.71021, 100.62955, 13.75, 100.75),
    ]
    raw_paths = [raw_dir / f"weather_era5_grid_{grid[0]}_2026-08-25_2026-08-31.json" for grid in grids]
    for path in [*raw_paths, mapping_path]:
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")

    cameras = pd.read_csv(project_dir / "data" / "cctv_info.csv")
    grid_a_ids = [
        11, 18, 26, 32, 36, 68, 70, 75, 82, 88,
        112, 118, 119, 120, 123, 124, 125, 126, 127, 135,
        136, 137, 139, 141, 150, 158, 169, 170, 176, 177,
        178, 179, 186, 188, 194, 196, 206, 211, 212, 215,
        219, 224, 225, 228, 230, 233, 239, 250, 251, 256,
        277, 301, 332, 363, 370, 371, 395, 411, 412, 413,
        414, 432, 500, 529, 542, 546, 556, 569, 572, 576,
        595, 599, 603, 604, 619, 630, 913, 915, 938, 939,
        1028, 1039, 1060, 1063, 1070, 1071, 1076, 1078,
        1103, 1112, 1132, 1188, 1195, 1361, 1362, 1368,
        1408, 1589, 1815,
    ]
    if len(cameras) != 100 or not cameras["camera_id"].is_unique:
        raise ValueError("Expected 100 unique camera IDs")
    if set(cameras["camera_id"]) != set(grid_a_ids + [1804]):
        raise ValueError("Camera IDs differ from the verified spatial audit")
    if not cameras["latitude"].between(-90, 90).all() or not cameras["longitude"].between(-180, 180).all():
        raise ValueError("Missing or invalid camera coordinates")
    mapping = cameras[["camera_id", "latitude", "longitude"]].rename(columns={
        "latitude": "camera_latitude", "longitude": "camera_longitude",
    })
    mapping["grid_latitude"] = 13.75
    mapping["grid_longitude"] = mapping["camera_id"].map({
        **{camera_id: 100.50 for camera_id in grid_a_ids}, 1804: 100.75,
    })
    expected_units = {
        "time": "iso8601", "temperature_2m": "°C",
        "relative_humidity_2m": "%", "precipitation": "mm",
        "cloud_cover": "%", "wind_speed_10m": "km/h",
    }
    expected_times = pd.date_range("2026-08-25", "2026-08-31 23:00", freq="h")
    for grid, raw_path in zip(grids, raw_paths):
        label, camera_id, latitude, longitude, grid_latitude, grid_longitude = grid
        camera = cameras.loc[cameras["camera_id"].eq(camera_id)].iloc[0]
        if camera["latitude"] != latitude or camera["longitude"] != longitude:
            raise ValueError(f"Representative camera {camera_id} coordinates changed")
        params = {
            "latitude": latitude, "longitude": longitude,
            "models": "era5", "timezone": "Asia/Bangkok",
            "cell_selection": "land", "elevation": "nan",
            "start_date": "2026-08-25", "end_date": "2026-08-31",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m",
            "temperature_unit": "celsius", "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
        print(f"\nGrid {label}: HTTP {response.status_code}", flush=True)
        print("Request URL:", response.url, flush=True)
        response.raise_for_status()
        data = response.json()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            file.write(response.content)
        if data.get("error"):
            raise ValueError(data.get("reason", "Open-Meteo returned an error"))
        if (data["latitude"], data["longitude"]) != (grid_latitude, grid_longitude):
            raise ValueError(f"Grid {label}: returned coordinates differ from the expected grid")
        if data["timezone"] != "Asia/Bangkok" or data["utc_offset_seconds"] != 25200:
            raise ValueError(f"Grid {label}: unexpected timezone")
        if data["hourly_units"] != expected_units:
            raise ValueError(f"Grid {label}: unexpected hourly units")
        weather = pd.DataFrame(data["hourly"])
        times = pd.DatetimeIndex(pd.to_datetime(weather["time"], errors="raise"))
        if not times.equals(expected_times):
            raise ValueError(f"Grid {label}: expected 168 consecutive local hours")
        print("Returned grid:", data["latitude"], data["longitude"])
        print("Returned grid elevation:", data["elevation"])
        print("Hourly records:", len(weather))
        print("Missing values:", weather.isna().sum().to_dict())
        print(weather.head(5).to_string(index=False))
        print("Saved unchanged JSON:", raw_path.relative_to(project_dir), flush=True)
        if label == "A":
            time.sleep(1)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False, encoding="utf-8", mode="x")
    print(f"\nCamera mapping records: {len(mapping)}")
    print(mapping.groupby(["grid_latitude", "grid_longitude"]).size().to_string())
    print(mapping.loc[mapping["camera_id"].isin([915, 1804])].to_string(index=False))
    print("Saved mapping:", mapping_path.relative_to(project_dir))
    return mapping


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Collect Bangkok external weather data")
    parser.add_argument("--era5-grids", action="store_true", help="Collect the two approved ERA5 grids and camera mapping")
    args = parser.parse_args()
    try:
        if args.era5_grids:
            collect_era5_grids()
        else:
            weather = collect_weather(13.7563, 100.5018, "2026-08-25", "2026-08-31")
            print(f"\nHourly records: {len(weather)}")
            print(weather.head().to_string(index=False))
    except (requests.RequestException, ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Weather collection failed: {error}")
