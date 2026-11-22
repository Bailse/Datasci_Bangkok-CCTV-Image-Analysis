"""Collect a current OSM snapshot around each camera, not the August 2026 network."""

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PILOT_ID = 915
CAMERA_COLUMNS = ["camera_id", "latitude", "longitude"]
ROAD_COLUMNS = [
    "camera_id", "osm_id", "road_name", "highway", "lanes", "oneway",
    "maxspeed", "surface", "geometry",
]


def read_cameras(csv_path):
    cameras = pd.read_csv(csv_path, usecols=CAMERA_COLUMNS)
    missing_ids = cameras["camera_id"].isna().sum()
    duplicate_ids = cameras["camera_id"].duplicated().sum()
    missing_coords = cameras[["latitude", "longitude"]].isna().any(axis=1).sum()
    print(f"Metadata: {len(cameras)} rows, {cameras['camera_id'].nunique()} unique IDs, "
          f"{missing_ids} missing IDs, {duplicate_ids} duplicate IDs, "
          f"{missing_coords} rows with missing coordinates")
    if missing_ids or duplicate_ids or missing_coords:
        raise ValueError("Camera IDs and coordinates must be complete, with unique IDs")
    if not (cameras["latitude"].between(-90, 90) & cameras["longitude"].between(-180, 180)).all():
        raise ValueError("Camera coordinates are outside latitude or longitude bounds")
    return cameras


def collect_camera(session, camera, output_dir, endpoint):
    path = output_dir / f"camera_{camera.camera_id}.json"
    existing = path.exists()
    if existing:
        content = path.read_bytes()
        status = "existing file reused"
    else:
        query = (f'[out:json][timeout:25];\n'
                 f'way(around:300,{camera.latitude},{camera.longitude})["highway"];\n'
                 'out geom;')
        time.sleep(2)
        response = session.get(endpoint, params={"data": query}, timeout=(10, 50))
        response.raise_for_status()
        content = response.content
        status = f"HTTP {response.status_code}"

    data = json.loads(content)
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        raise ValueError("Expected an Overpass JSON response with an elements list")
    if data.get("remark"):
        raise ValueError(f"Overpass returned a remark: {data['remark']}")
    ways = data["elements"]
    if any(
        not isinstance(way, dict) or way.get("type") != "way"
        or not isinstance(way.get("id"), int) or not isinstance(way.get("tags"), dict)
        or "highway" not in way["tags"]
        for way in ways
    ):
        raise ValueError("Expected highway ways with OSM IDs")
    if not existing:
        output_dir.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(content)
    print(f"Camera {camera.camera_id}: {status}, {len(ways)} ways")
    return ways


def road_records(camera_id, ways):
    return [{
        "camera_id": camera_id,
        "osm_id": way["id"],
        "road_name": way["tags"].get("name"),
        "highway": way["tags"].get("highway"),
        "lanes": way["tags"].get("lanes"),
        "oneway": way["tags"].get("oneway"),
        "maxspeed": way["tags"].get("maxspeed"),
        "surface": way["tags"].get("surface"),
        "geometry": way.get("geometry"),
    } for way in ways]


def report(records):
    roads = pd.DataFrame(records, columns=ROAD_COLUMNS)
    print(f"OSM ways across camera responses: {len(roads):,}")
    print(f"Unique OSM way IDs: {roads['osm_id'].nunique():,}")
    print("Missing values per field:")
    print(roads.isna().sum().to_string())
    print("First five records (geometry shown as point count):")
    sample = roads.head(5).copy()
    sample["geometry"] = sample["geometry"].map(lambda points: len(points) if points is not None else None)
    print(sample.to_string(index=False))


def collect_osm(pilot_only=False, endpoint=OVERPASS_URL):
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "data" / "external_data" / "raw" / "osm"
    cameras = read_cameras(project_dir / "data" / "cctv_info.csv")
    pilot = cameras[cameras["camera_id"].eq(PILOT_ID)]
    if pilot.empty:
        raise ValueError(f"Pilot camera {PILOT_ID} is missing from the metadata")
    others = cameras[~cameras["camera_id"].eq(PILOT_ID)]
    records = []
    failed = []
    print(f"Overpass endpoint for new requests: {endpoint}")
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Bangkok-CCTV-Image-Analysis/1.0 (university road network data collection)",
            "Accept": "application/json",
        })
        camera = next(pilot.itertuples(index=False))
        ways = collect_camera(session, camera, output_dir, endpoint)
        tags = sorted({tag for way in ways for tag in way["tags"]})
        print(f"Pilot {camera.camera_id} at {camera.latitude}, {camera.longitude}. Tags: {', '.join(tags)}")
        records.extend(road_records(camera.camera_id, ways))
        if not pilot_only:
            for camera in others.itertuples(index=False):
                try:
                    ways = collect_camera(session, camera, output_dir, endpoint)
                    records.extend(road_records(camera.camera_id, ways))
                except (requests.RequestException, ValueError, OSError) as error:
                    failed.append(camera.camera_id)
                    print(f"Camera {camera.camera_id} failed: {error}")
                    if (isinstance(error, requests.HTTPError)
                            and error.response.status_code in (406, 429, 502, 503, 504)):
                        retry_after = error.response.headers.get("Retry-After", "")
                        time.sleep(max(30, int(retry_after)) if retry_after.isdigit() else 30)

    processed = 1 if pilot_only else len(cameras)
    print(f"Cameras processed: {processed}. Successful: {processed - len(failed)}. Failed: {len(failed)}.")
    print(f"Failed camera IDs: {failed}")
    print(f"Raw OSM responses: {output_dir}")
    print("Current OSM snapshot. Ways within 300 m are candidates for inspection only.")
    report(records)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Collect and inspect only camera 915")
    parser.add_argument("--endpoint", default=OVERPASS_URL, help="Overpass interpreter URL")
    args = parser.parse_args()
    try:
        collect_osm(args.pilot, args.endpoint)
    except (requests.RequestException, ValueError, OSError) as error:
        raise SystemExit(f"OSM collection failed: {error}")
