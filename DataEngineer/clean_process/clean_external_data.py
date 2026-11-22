import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def clean_weather(raw_path, requested_coordinates, expected_grid):
    """Validate one documented ERA5 response, preserving its original values."""
    data = json.loads(raw_path.read_bytes())
    expected_units = {
        "time": "iso8601",
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "cloud_cover": "%",
        "wind_speed_10m": "km/h",
    }
    for column, unit in expected_units.items():
        if data["hourly_units"].get(column) != unit:
            raise ValueError(f"Unexpected weather unit for {column}")
    if data["timezone"] != "Asia/Bangkok" or data["utc_offset_seconds"] != 25200:
        raise ValueError("Expected Asia/Bangkok weather with UTC offset +07:00")

    if (data["latitude"], data["longitude"]) != expected_grid:
        raise ValueError(f"Unexpected returned weather grid in {raw_path.name}")
    if not isinstance(data.get("elevation"), (int, float)):
        raise ValueError("Missing numeric grid elevation in weather metadata")
    latitude, longitude = requested_coordinates
    start_date, end_date = "2026-08-25", "2026-08-31"
    coordinates = {
        "requested_latitude": float(latitude),
        "requested_longitude": float(longitude),
        "grid_latitude": float(data["latitude"]),
        "grid_longitude": float(data["longitude"]),
    }
    for column, value in coordinates.items():
        limit = 90 if "latitude" in column else 180
        if not -limit <= value <= limit:
            raise ValueError(f"Invalid coordinate: {column}")

    weather = pd.DataFrame(data["hourly"])
    if weather["time"].duplicated().any():
        raise ValueError("Duplicate weather timestamps; inspect the raw records")
    timestamps = pd.to_datetime(weather["time"], format="%Y-%m-%dT%H:%M", errors="raise")
    if timestamps.isna().any() or timestamps.dt.tz is not None:
        raise ValueError("Expected nonmissing local weather timestamps without offsets")
    weather["time"] = timestamps.dt.tz_localize("Asia/Bangkok")
    weather = weather.sort_values("time").reset_index(drop=True)
    expected_hours = pd.date_range(
        start_date,
        pd.Timestamp(end_date) + pd.Timedelta(days=1),
        freq="h", inclusive="left", tz="Asia/Bangkok",
    )
    if not pd.DatetimeIndex(weather["time"]).equals(expected_hours):
        raise ValueError("Weather timestamps do not cover every requested hour")

    for column in expected_units:
        if column != "time":
            weather[column] = pd.to_numeric(weather[column], errors="raise")
    weather = weather.rename(columns={
        "temperature_2m": "temperature_2m_c",
        "relative_humidity_2m": "relative_humidity_2m_percent",
        "precipitation": "precipitation_mm",
        "cloud_cover": "cloud_cover_percent",
        "wind_speed_10m": "wind_speed_10m_kmh",
    })
    for column, value in coordinates.items():
        weather[column] = value
    weather["timezone"] = "Asia/Bangkok"
    # The raw response does not echo these request options. Their provenance is
    # the explicit ERA5 collection configuration recorded in README_external_data.
    weather["model"] = "era5"
    weather["cell_selection"] = "land"
    # A bare "nan" would be read as a missing value by default CSV readers.
    weather["elevation_setting"] = "nan (no downscaling)"
    weather["grid_elevation_m"] = data["elevation"]
    return weather


def clean_camera_mapping(camera_path, mapping_path, weather):
    """Rebuild camera columns using the existing, verified spatial audit map."""
    cameras = pd.read_csv(camera_path)
    audited = pd.read_csv(mapping_path)
    columns = ["camera_id", "camera_latitude", "camera_longitude", "grid_latitude", "grid_longitude"]
    if audited.columns.tolist() != columns:
        raise ValueError("Unexpected camera weather mapping columns")
    if len(cameras) != 100 or len(audited) != 100:
        raise ValueError("Expected exactly 100 camera metadata and mapping records")
    if not cameras["camera_id"].is_unique or not audited["camera_id"].is_unique:
        raise ValueError("Duplicate camera_id in metadata or mapping")
    original = cameras[["camera_id", "latitude", "longitude"]].rename(columns={
        "latitude": "camera_latitude", "longitude": "camera_longitude",
    })
    if set(original["camera_id"]) != set(audited["camera_id"]):
        raise ValueError("Camera IDs differ from the spatial audit; rerun the audit")
    # A mapping is valid only for the exact coordinates that were audited.
    original_sorted = original.sort_values("camera_id").reset_index(drop=True)
    audited_sorted = audited[original.columns].sort_values("camera_id").reset_index(drop=True)
    if not original_sorted.equals(audited_sorted):
        raise ValueError("Camera coordinates differ from the spatial audit; rerun the audit")
    mapping = original.merge(audited[["camera_id", "grid_latitude", "grid_longitude"]],
                             on="camera_id", how="left", validate="one_to_one")
    if mapping.isna().any().any():
        raise ValueError("Missing camera coordinates or grid assignments")
    if not mapping["camera_latitude"].between(-90, 90).all() or not mapping["camera_longitude"].between(-180, 180).all():
        raise ValueError("Camera coordinates outside valid ranges")
    if not mapping["grid_latitude"].eq(13.75).all():
        raise ValueError("Unexpected camera grid latitude")
    if mapping.loc[mapping["camera_id"].eq(1804), "grid_longitude"].tolist() != [100.75]:
        raise ValueError("Camera 1804 must map to Grid B")
    if not mapping.loc[mapping["camera_id"].ne(1804), "grid_longitude"].eq(100.50).all():
        raise ValueError("The other 99 audited cameras must map to Grid A")
    available = weather[["grid_latitude", "grid_longitude"]].drop_duplicates()
    covered = mapping.merge(available, on=["grid_latitude", "grid_longitude"], how="left", indicator=True)
    if not covered["_merge"].eq("both").all():
        raise ValueError("A camera maps to a grid without cleaned hourly weather")
    return mapping


def clean_holidays(raw_path):
    data = json.loads(raw_path.read_bytes())
    if data["year"] != 2026 or data["year_buddhist_era"] != 2569:
        raise ValueError("Expected the 2026 Thailand holiday source")
    holidays = pd.DataFrame(data["holidays"])
    if len(holidays) != data["total_holidays"]:
        raise ValueError("Holiday count differs from the source metadata")
    holidays["date"] = pd.to_datetime(holidays["date"], format="%Y-%m-%d", errors="raise")
    if holidays["date"].isna().any() or not holidays["date"].dt.year.eq(2026).all():
        raise ValueError("Missing holiday dates or dates outside 2026")
    if holidays["date"].duplicated().any():
        raise ValueError("Duplicate holiday dates; inspect rather than discard information")
    for column in ["name_th", "name_en", "type", "note"]:
        holidays[column] = holidays[column].astype("string")
    holidays["is_substitution"] = holidays["is_substitution"].astype("boolean")

    calendar = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-12-31")})
    calendar = calendar.merge(holidays, on="date", how="left", validate="one_to_one", indicator=True)
    calendar["is_holiday"] = calendar.pop("_merge").eq("both")
    calendar["is_weekend"] = calendar["date"].dt.dayofweek.ge(5)
    calendar["day_of_week"] = calendar["date"].dt.day_name()
    calendar["source"] = data["source"]
    calendar["source_year_be"] = data["year_buddhist_era"]
    calendar["official_verification_status"] = "incomplete"
    return calendar


def traffic_numbers(values):
    values = values.str.replace(",", "", regex=False).replace("", pd.NA)
    return pd.to_numeric(values, errors="raise").astype("Float64")


def clean_bem(csv_path, html_path):
    html = html_path.read_text(encoding="utf-8")
    if "หน่วย: พันเที่ยว" not in html or "ปริมาณจราจรเฉลี่ยต่อวัน" not in html:
        raise ValueError("Cannot confirm BEM's original measure and unit")
    month_numbers = {
        "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4,
        "พ.ค.": 5, "มิ.ย.": 6, "ก.ค.": 7, "ส.ค.": 8,
        "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
    }
    wide = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    if wide.columns.tolist() != ["Unnamed: 0", *month_numbers, "เฉลี่ย"]:
        raise ValueError("Unexpected BEM table columns")
    wide = wide.rename(columns={"Unnamed: 0": "year_be"})
    wide["year_be"] = pd.to_numeric(wide["year_be"], errors="raise").astype("Int64")
    if wide["year_be"].isna().any() or wide["year_be"].duplicated().any():
        raise ValueError("Missing or duplicate BEM years")

    monthly = wide.melt(
        id_vars="year_be", value_vars=list(month_numbers),
        var_name="month_th", value_name="traffic_volume",
    )
    monthly["year"] = monthly["year_be"] - 543
    monthly["month"] = monthly["month_th"].map(month_numbers).astype("Int64")
    monthly["traffic_volume"] = traffic_numbers(monthly["traffic_volume"])
    monthly = monthly[["year_be", "year", "month", "month_th", "traffic_volume"]]
    monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)

    annual = wide[["year_be", "เฉลี่ย"]].copy()
    annual["year"] = annual["year_be"] - 543
    annual["published_average_traffic_volume"] = traffic_numbers(annual.pop("เฉลี่ย"))
    months_with_data = monthly.groupby("year_be")["traffic_volume"].count()
    annual["months_with_data"] = annual["year_be"].map(months_with_data).astype("Int64")
    annual["is_complete_year"] = annual["months_with_data"].eq(12)
    annual = annual.sort_values("year").reset_index(drop=True)
    for table in [monthly, annual]:
        table["measure"] = "Average Traffic Volume per Day"
        table["unit"] = "Thousand Trips"
        table["unit_th"] = "พันเที่ยว"
    return monthly, annual


def clean_accidents(raw_path):
    columns = {
        "ปีที่เกิดเหตุ": "accident_year",
        "วันที่เกิดเหตุ": "accident_date_raw",
        "เวลา": "accident_time_raw",
        "วันที่รายงาน": "report_date_raw",
        "เวลาที่รายงาน": "report_time_raw",
        "หน่วยงาน": "agency",
        "สายทางหน่วยงาน": "road_network",
        "รหัสสายทาง": "route_code",
        "สายทาง": "road_name",
        "ก.ม.": "kilometer_marker",
        "จังหวัด": "province",
        "รถคันที่1": "first_vehicle_type",
        "บริเวณที่เกิดเหตุ/ลักษณะทาง": "road_layout",
        "บริเวณที่เกิดเหตุอื่น": "other_location_detail",
        "มูลเหตุสันนิษฐาน": "suspected_cause",
        "ลักษณะการเกิดอุบัติเหตุ": "accident_type",
        "สภาพอากาศ": "weather",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "จำนวนรถที่เกิดเหตุ(รวมคันที่1)": "vehicles_total",
        "จำนวนที่เกิดเหตุทั้งหมด(รวมคนเดินเท้า)": "participants_total",
        "รถจักรยานยนต์": "motorcycles",
        "รถสามล้อเครื่อง": "motorized_three_wheelers",
        "รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ": "passenger_cars",
        "รถตู้": "vans",
        "รถปิคอัพโดยสาร": "passenger_pickups",
        "รถโดยสารมากกว่า4ล้อ": "buses_over_four_wheels",
        "รถปิคอัพบรรทุก4ล้อ": "four_wheel_cargo_pickups",
        "รถบรรทุก6ล้อ": "six_wheel_trucks",
        "รถบรรทุกมากกว่า6ล้อไม่เกิน10ล้อ": "trucks_seven_to_ten_wheels",
        "รถบรรทุกมากกว่า10ล้อ(รถพ่วง)": "trucks_over_ten_wheels",
        "รถอีแต๋น": "e_tan_farm_vehicles",
        "อื่นๆ": "other_vehicles",
        "คนเดินเท้า": "pedestrians",
        "จำนวนผู้เสียชีวิต": "fatalities",
        "จำนวนผู้บาดเจ็บสาหัส": "serious_injuries",
        "จำนวนผู้บาดเจ็บเล็กน้อย": "minor_injuries",
        "รวมจำนวนผู้บาดเจ็บ": "injuries_total",
    }
    raw = pd.read_csv(raw_path, dtype={"รหัสสายทาง": "string"},
                      keep_default_na=False, na_values=[""], float_precision="round_trip")
    accidents = raw[list(columns)].rename(columns=columns)
    accidents = accidents.loc[
        accidents["province"].eq("กรุงเทพมหานคร") & accidents["accident_year"].eq(2026)
    ].convert_dtypes()
    if accidents.empty:
        raise ValueError("No 2026 Bangkok accident records in the MOT CSV")
    accidents.insert(0, "source_row", accidents.index + 1)
    dates = pd.to_datetime(accidents["accident_date_raw"], format="%d/%m/%Y")
    timestamps = pd.to_datetime(
        accidents["accident_date_raw"] + " " + accidents["accident_time_raw"],
        format="%d/%m/%Y %H:%M",
    )
    if not dates.dt.year.eq(2026).all() or timestamps.isna().any():
        raise ValueError("Missing event timestamps or Bangkok accident dates outside 2026")
    accidents.insert(4, "accident_date", dates)
    accidents.insert(5, "accident_at", timestamps.dt.tz_localize("Asia/Bangkok"))
    for column, limit in [("latitude", 90), ("longitude", 180)]:
        values = pd.to_numeric(accidents[column]).astype("Float64")
        if not values.dropna().between(-limit, limit).all():
            raise ValueError(f"MOT {column} outside valid bounds")
        accidents[column] = values
    return accidents.reset_index(drop=True)


def clean_osm(raw_dir, camera_path):
    cameras = pd.read_csv(camera_path, usecols=["camera_id"])
    paths = sorted(raw_dir.glob("camera_*.json"))
    if not paths or [p.name for p in paths] != sorted(f"camera_{i}.json" for i in cameras["camera_id"]):
        raise ValueError("OSM response files do not match the camera metadata IDs")

    motor_roads = {
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
        "residential", "service", "living_street", "busway",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    }
    paths_and_walkways = {"footway", "path", "steps", "pedestrian", "cycleway", "corridor", "bridleway"}
    tag_columns = {"name": "road_name", "highway": "highway", "lanes": "lanes",
                   "oneway": "oneway", "maxspeed": "maxspeed", "surface": "surface"}
    ways_by_id = {}
    snapshots = set()
    records = []
    for path in paths:
        camera_id = int(path.stem.split("_")[1])
        response = json.loads(path.read_bytes())
        if response.get("remark"):
            raise ValueError(f"Incomplete Overpass response: {path.name}")
        snapshots.add(response["osm3s"]["timestamp_osm_base"])
        for way in response["elements"]:
            osm_id, tags = way["id"], way["tags"]
            if way["type"] != "way" or ways_by_id.setdefault(osm_id, way) != way:
                raise ValueError(f"Conflicting OSM way or invalid element: {osm_id}")
            highway = tags["highway"]
            group = "other"
            if highway in motor_roads:
                group = "motor_road"
            elif highway in paths_and_walkways:
                group = "path"
            record = {"camera_id": camera_id, "osm_id": osm_id}
            record.update({column: tags.get(tag) for tag, column in tag_columns.items()})
            record["highway_group"] = group
            records.append(record)

    table = pd.DataFrame(records).astype({c: "string" for c in [*tag_columns.values(), "highway_group"]})
    if table.duplicated(["camera_id", "osm_id"]).any():
        raise ValueError("Repeated camera–way pairs in the OSM responses")
    features = []
    for osm_id, way in sorted(ways_by_id.items()):
        coordinates = [[point["lon"], point["lat"]] for point in way["geometry"]]
        if len(coordinates) < 2 or any(not (-180 <= lon <= 180 and -90 <= lat <= 90)
                                       for lon, lat in coordinates):
            raise ValueError(f"Invalid geometry for OSM way {osm_id}")
        features.append({
            "type": "Feature",
            "properties": {"osm_id": osm_id},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        })
    print(f"OSM source snapshot(s) in UTC: {', '.join(sorted(snapshots))}")
    return table, {"type": "FeatureCollection", "features": features}


def show_checks(name, table, key_columns):
    print(f"\n{name}: {len(table)} rows")
    print("First five records:")
    print(table.head(5).to_string(index=False, max_colwidth=75))
    print("Missing values per column:")
    print(table.isna().sum().to_string())
    print("Duplicate rows:", int(table.duplicated().sum()))
    print(f"Duplicate keys {key_columns}:", int(table.duplicated(key_columns).sum()))
    print("Data types before CSV export (CSV does not store types):")
    print(table.dtypes.to_string())


def export_and_verify(table, csv_path, date_format=None, replace=False):
    parquet_path = csv_path.with_suffix(".parquet")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if replace or not csv_path.exists():
        table.to_csv(
            csv_path, index=False, encoding="utf-8", mode="w" if replace else "x",
            date_format=date_format,
        )
    if replace or not parquet_path.exists():
        with parquet_path.open("wb" if replace else "xb") as file:
            table.to_parquet(file, engine="pyarrow", index=False)

    restored = pd.read_parquet(parquet_path, engine="pyarrow")
    pd.testing.assert_frame_equal(restored, table)
    csv_table = pd.read_csv(csv_path)
    if len(csv_table) != len(restored) or csv_table.columns.tolist() != restored.columns.tolist():
        raise ValueError(f"CSV and Parquet row counts or columns differ: {csv_path}")
    print(f"Verified CSV + Parquet: {csv_path.name}, {parquet_path.name}")
    print(f"  {len(restored)} rows; matching columns; Parquet values and dtypes preserved")


def verify_csv_values(path, table):
    dtypes = table.dtypes.to_dict()
    # CSV stores timestamps as text. Restore each timestamp type for comparison.
    for column in ("accident_date", "accident_at"):
        dtypes.pop(column, None)
    restored = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""],
                           float_precision="round_trip")
    if "accident_at" in table:
        restored["accident_date"] = pd.to_datetime(restored["accident_date"], format="%Y-%m-%d")
        restored["accident_at"] = pd.to_datetime(restored["accident_at"], utc=True).dt.tz_convert("Asia/Bangkok")
    pd.testing.assert_frame_equal(restored, table, check_exact=True)


def clean_accidents_and_osm():
    project_dir = Path(__file__).resolve().parents[2]
    external_dir = project_dir / "data" / "external_data"
    raw, processed = external_dir / "raw", external_dir / "processed"
    accidents = clean_accidents(raw / "accidents" / "accident2026.csv")
    roads, geojson = clean_osm(raw / "osm", project_dir / "data" / "cctv_info.csv")
    outputs = [
        (accidents, processed / "accidents" / "bangkok_accidents_2026.csv"),
        (roads, processed / "osm" / "camera_osm_ways.csv"),
    ]
    geo_path = processed / "osm" / "osm_ways.geojson"

    # Check all existing outputs before creating any missing files.
    for table, csv_path in outputs:
        parquet_path = csv_path.with_suffix(".parquet")
        if csv_path.exists():
            verify_csv_values(csv_path, table)
        if parquet_path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(parquet_path, engine="pyarrow"),
                                          table, check_exact=True)
    if geo_path.exists() and json.loads(geo_path.read_bytes()) != geojson:
        raise FileExistsError(f"Existing GeoJSON differs; refusing to overwrite: {geo_path}")

    for table, csv_path in outputs:
        export_and_verify(table, csv_path)
        parquet = pd.read_parquet(csv_path.with_suffix(".parquet"), engine="pyarrow")
        pd.testing.assert_frame_equal(parquet, table, check_exact=True)
        verify_csv_values(csv_path, parquet)
        print(f"  Verified exact CSV and Parquet values for {csv_path.stem}")
    if not geo_path.exists():
        with geo_path.open("x", encoding="utf-8") as file:
            json.dump(geojson, file, separators=(",", ":"), allow_nan=False)
    if json.loads(geo_path.read_bytes()) != geojson:
        raise ValueError(f"GeoJSON verification failed: {geo_path}")

    show_checks("Bangkok MOT accidents", accidents, ["source_row"])
    print("Repeated source rows excluding source_row:", accidents.drop(columns="source_row").duplicated().sum())
    location_key = ["accident_at", "latitude", "longitude"]
    located = accidents.dropna(subset=location_key)
    print("Repeated timestamp-and-coordinate combinations:", located.duplicated(location_key).sum())
    dates = accidents["accident_date"]
    print(f"Bangkok date coverage: {dates.min().date()} to {dates.max().date()}")
    print("August 25–31, 2026 records:", dates.between("2026-08-25", "2026-08-31").sum())
    print("Bangkok coordinates (decimal degrees):")
    print(accidents[["latitude", "longitude"]].agg(["min", "max"]).to_string())

    show_checks("Camera–OSM way pairs", roads, ["camera_id", "osm_id"])
    print(f"Cameras represented: {roads['camera_id'].nunique()}")
    features = geojson["features"]
    longitudes, latitudes = zip(*(point for feature in features for point in feature["geometry"]["coordinates"]))
    print(f"Verified {len(features)} GeoJSON LineStrings with unique integer osm_id values")
    print(f"OSM geometry bounds (west, south, east, north): "
          f"{min(longitudes)}, {min(latitudes)}, {max(longitudes)}, {max(latitudes)}")
    print("First five geometries (osm_id, first [longitude, latitude], point count):")
    for feature in features[:5]:
        coordinates = feature["geometry"]["coordinates"]
        print(feature["properties"]["osm_id"], coordinates[0], len(coordinates))


def main(replace_weather=False):
    project_dir = Path(__file__).resolve().parent.parent.parent
    external_dir = project_dir / "data" / "external_data"
    raw_dir = external_dir / "raw"
    processed_dir = external_dir / "processed"
    output_paths = [
        processed_dir / "weather" / "weather_clean.csv",
        processed_dir / "holidays" / "thailand_calendar_2026.csv",
        processed_dir / "bem" / "bem_monthly_traffic.csv",
        processed_dir / "bem" / "bem_annual_averages.csv",
    ]
    grids = [
        ("A", (13.72639, 100.49213), (13.75, 100.50)),
        ("B", (13.71021, 100.62955), (13.75, 100.75)),
    ]
    weather_parts = []
    for label, requested_coordinates, expected_grid in grids:
        raw_path = raw_dir / "weather" / f"weather_era5_grid_{label}_2026-08-25_2026-08-31.json"
        weather_parts.append(clean_weather(raw_path, requested_coordinates, expected_grid))
    weather = pd.concat(weather_parts, ignore_index=True)
    weather_keys = ["grid_latitude", "grid_longitude", "time"]
    if len(weather) != 336 or weather[weather_keys[:2]].drop_duplicates().shape[0] != 2:
        raise ValueError("Expected 336 weather rows covering exactly two ERA5 grids")
    mapping_path = processed_dir / "weather" / "camera_weather_grid_mapping.csv"
    mapping = clean_camera_mapping(project_dir / "data" / "cctv_info.csv", mapping_path, weather)
    holidays = clean_holidays(raw_dir / "holidays" / "thailand_public_holidays_2026.json")
    monthly, annual = clean_bem(
        raw_dir / "bem" / "traffic_volume_raw.csv",
        raw_dir / "bem" / "traffic_volume.html",
    )
    outputs = [
        ("Weather", weather, weather_keys),
        ("2026 holiday calendar", holidays, ["date"]),
        ("BEM monthly", monthly, ["year", "month"]),
        ("BEM published annual averages", annual, ["year"]),
    ]
    for name, table, keys in outputs:
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate keys in {name}")
    for path, (name, table, keys) in zip(output_paths, outputs):
        date_format = "%Y-%m-%d" if name == "2026 holiday calendar" else None
        if path.exists():
            expected_csv = table.to_csv(index=False, date_format=date_format).encode("utf-8")
            if path.read_bytes() != expected_csv:
                if name != "Weather" or not replace_weather:
                    raise FileExistsError(f"Existing CSV differs; refusing to overwrite: {path}")
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path, engine="pyarrow")
            try:
                pd.testing.assert_frame_equal(existing, table)
            except AssertionError as error:
                if name != "Weather" or not replace_weather:
                    raise FileExistsError(
                        f"Existing Parquet differs; refusing to overwrite: {parquet_path}"
                    ) from error

    expected_mapping = mapping.to_csv(index=False).encode("utf-8")
    if mapping_path.exists() and mapping_path.read_bytes() != expected_mapping:
        raise FileExistsError(f"Existing mapping differs; refusing to overwrite: {mapping_path}")

    for path, (name, table, keys) in zip(output_paths, outputs):
        date_format = "%Y-%m-%d" if name == "2026 holiday calendar" else None
        export_and_verify(table, path, date_format, replace=name == "Weather" and replace_weather)
        show_checks(name, table, keys)
    if not mapping_path.exists():
        mapping.to_csv(mapping_path, index=False, encoding="utf-8", mode="x")
    pd.testing.assert_frame_equal(pd.read_csv(mapping_path), mapping)
    show_checks("Camera weather grid mapping", mapping, ["camera_id"])
    print("\nHoliday official government verification: incomplete.")
    print("BEM 2026 coverage:")
    print(annual.loc[annual["year"].eq(2026)].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Clean external datasets without changing raw files")
    parser.add_argument("--replace-weather", action="store_true",
                        help="Explicitly allow replacement of weather_clean.csv and weather_clean.parquet only")
    parser.add_argument("--accidents-osm", action="store_true",
                        help="Clean and verify only MOT accidents and OpenStreetMap")
    args = parser.parse_args()
    try:
        if args.accidents_osm:
            clean_accidents_and_osm()
        else:
            main(replace_weather=args.replace_weather)
    except (ValueError, KeyError, OSError, ImportError, AssertionError) as error:
        raise SystemExit(f"External data cleaning failed: {error}")
