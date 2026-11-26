"""Join CCTV image counts to Bangkok weather, calendar, road and traffic data.

From the repository root (requires pandas, numpy, pyarrow and XlsxWriter):
    python DataSci/analysis/analyze_cctv.py --counts "path/to/image_counts (1).parquet"
"""

import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
EXTERNAL = DATA / "external_data" / "processed"
DEFAULT_OUT = HERE / "outputs" / "cctv_aug2026"
CAMERA = "camera_id"
SOURCE_COLUMNS = [CAMERA, "ts_filename", "filename", "car", "motor", "day_night"]
IMAGE_KEY = [CAMERA, "filename", "ts_filename"]
GRID = ["grid_latitude", "grid_longitude"]
TIERS = ["motorway", "trunk", "primary", "secondary", "tertiary", "other_motor_road", "no_motor_road"]
INSTANT_WEATHER = ["temperature_2m_c", "relative_humidity_2m_percent", "cloud_cover_percent", "wind_speed_10m_kmh"]
RADIUS_M = 300
ONE_HOUR = pd.Timedelta(hours=1)


def attach(images, lookup, keys, source):
    """Require one matching lookup record per image, including zero-valued observations."""
    out = images.merge(lookup, how="left", on=keys, validate="many_to_one", indicator=True)
    missing = out["_merge"].ne("both")
    if missing.any():
        examples = out.loc[missing, keys].drop_duplicates().head().to_dict("records")
        raise ValueError(f"{source}: {int(missing.sum())} unmatched image rows, e.g. {examples}")
    return out.drop(columns="_merge")


def road_context():
    ways = pd.read_parquet(EXTERNAL / "osm/camera_osm_ways.parquet")
    ways[CAMERA] = ways[CAMERA].astype(str)
    if ways.duplicated([CAMERA, "osm_id"]).any():
        raise ValueError("Duplicate camera/OSM way pair")
    ways["motor_road"] = ways["highway_group"].eq("motor_road")
    rank = {tier: pos for pos, tier in enumerate(TIERS)}
    base = ways["highway"].str.replace(r"_link$", "", regex=True)
    other = np.where(ways["motor_road"], rank["other_motor_road"], rank["no_motor_road"])
    ways["rank"] = base.map(rank).fillna(pd.Series(other, index=ways.index)).astype(int)
    nearby = ways.groupby(CAMERA, as_index=False).agg(
        osm_ways_300m=("osm_id", "size"), osm_motor_ways_300m=("motor_road", "sum"),
        rank=("rank", "min"),
    )
    nearby["road_class_300m"] = nearby.pop("rank").map(dict(enumerate(TIERS)))
    return nearby


def accident_context(images, cameras):
    """MOT records at city/day grain and located reports within 300 m by camera/clock-hour."""
    events = pd.read_parquet(EXTERNAL / "accidents/bangkok_accidents_2026.parquet",
                             columns=["source_row", "accident_date", "accident_at", "latitude", "longitude"])
    dates = pd.DatetimeIndex(images["date_local"].unique()).sort_values()
    if dates.min() < events.accident_date.min() or dates.max() > events.accident_date.max():
        raise ValueError("Image dates outside the observed MOT reporting period")
    events = events.loc[events.accident_date.isin(dates)].copy()
    if not events.source_row.is_unique:
        raise ValueError("Duplicate MOT source record")
    located = events.dropna(subset=["latitude", "longitude"])
    lat, lon = np.radians(located[["latitude", "longitude"]].to_numpy(dtype=float)).T
    cam_lat, cam_lon = np.radians(cameras[["latitude", "longitude"]].to_numpy(dtype=float)).T
    # Haversine distance between each reported point and each camera in this analysis.
    half_chord = (np.sin((lat[:, None] - cam_lat) / 2) ** 2 + np.cos(lat[:, None])
                  * np.cos(cam_lat) * np.sin((lon[:, None] - cam_lon) / 2) ** 2)
    distance_m = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(half_chord, 0, 1)))
    event_idx, cam_idx = np.nonzero(distance_m <= RADIUS_M)
    nearby = located.iloc[event_idx][["source_row", "accident_date", "accident_at"]].copy()
    nearby[CAMERA] = cameras.iloc[cam_idx][CAMERA].to_numpy()
    nearby["hour_start"] = nearby.accident_at.dt.floor("h")
    per_hour = nearby.groupby([CAMERA, "hour_start"]).source_row.nunique().rename("mot_reports_300m_hour")
    images = images.merge(per_hour.reset_index(), on=[CAMERA, "hour_start"], how="left", validate="many_to_one")
    images["mot_reports_300m_hour"] = images.mot_reports_300m_hour.fillna(0).astype(int)
    daily = pd.DataFrame({
        "mot_reports_bkk": events.groupby("accident_date").source_row.nunique(),
        "mot_geocoded": located.groupby("accident_date").source_row.nunique(),
        "mot_within_300m": nearby.groupby("accident_date").source_row.nunique(),
    }).reindex(dates).fillna(0).astype(int).rename_axis("date_local").reset_index()
    return images, daily


def enrich(counts):
    if counts.empty or counts[SOURCE_COLUMNS].isna().any().any() or counts.duplicated(IMAGE_KEY).any():
        raise ValueError("Expected non-empty image counts with complete source fields and unique image keys")
    amounts = counts[["car", "motor"]]
    if (amounts < 0).any().any() or not all(pd.api.types.is_integer_dtype(v) for v in amounts.dtypes):
        raise ValueError("Car and motorcycle detections must be nonnegative integer counts")
    images = counts.copy()
    local = images.ts_filename.dt.tz_convert("Asia/Bangkok")
    images["vehicles_total"] = images.car + images.motor
    images["motor_share"] = images.motor.div(images.vehicles_total.where(images.vehicles_total.ne(0)))
    images["date_local"] = local.dt.tz_localize(None).dt.normalize()
    images["hour_local"] = local.dt.hour
    images["hour_start"] = local.dt.floor("h")
    images["rain_interval_end"] = images.hour_start + ONE_HOUR

    calendar = pd.read_parquet(EXTERNAL / "holidays/thailand_calendar_2026.parquet",
                               columns=["date", "day_of_week", "is_weekend", "is_holiday"])
    images = attach(images, calendar.rename(columns={"date": "date_local"}), ["date_local"], "Thai calendar")
    images["day_type"] = np.select([images.is_holiday, images.is_weekend], ["public_holiday", "weekend"], default="weekday")
    cameras = pd.read_csv(DATA / "cctv_info.csv", dtype={CAMERA: str},
                          usecols=[CAMERA, "location", "location_detail", "latitude", "longitude"])
    cameras = cameras.loc[cameras[CAMERA].isin(images[CAMERA])].reset_index(drop=True)
    images = attach(images, cameras, [CAMERA], "camera metadata")
    images = attach(images, road_context(), [CAMERA], "OSM")

    mapping = pd.read_csv(EXTERNAL / "weather/camera_weather_grid_mapping.csv", dtype={CAMERA: str})
    mapping = mapping.rename(columns={"camera_latitude": "latitude", "camera_longitude": "longitude"})
    images = attach(images, mapping, [CAMERA, "latitude", "longitude"], "ERA5 grid mapping")
    weather = pd.read_parquet(EXTERNAL / "weather/weather_clean.parquet")
    instant = weather[[*GRID, "time", *INSTANT_WEATHER]].rename(columns={"time": "hour_start"})
    images = attach(images, instant, [*GRID, "hour_start"], "hourly weather")
    rain = weather[[*GRID, "time", "precipitation_mm"]].rename(columns={"time": "rain_interval_end", "precipitation_mm": "rain_mm_hour"})
    images = attach(images, rain, [*GRID, "rain_interval_end"], "hourly precipitation")
    if images[["rain_mm_hour", *INSTANT_WEATHER]].isna().any().any() or images.rain_mm_hour.lt(0).any():
        raise ValueError("Missing or invalid weather values")
    images["rain_group"] = np.where(images.rain_mm_hour.gt(0), "wet", "dry")
    images, daily_mot = accident_context(images, cameras)
    pd.testing.assert_frame_equal(images[counts.columns].reset_index(drop=True), counts.reset_index(drop=True), check_exact=True)
    return images, daily_mot


def aggregate(images, keys):
    """Pooled rates use the sum of detections divided by the actual number of frames."""
    result = images.groupby(keys, as_index=False, observed=True, dropna=False).agg(
        images=("filename", "size"), cameras=(CAMERA, "nunique"), dates=("date_local", "nunique"),
        cars=("car", "sum"), motors=("motor", "sum"), detections=("vehicles_total", "sum"),
    )
    for label, numerator in [("cars_per_image", "cars"), ("motors_per_image", "motors"), ("vehicles_per_image", "detections")]:
        result[label] = result[numerator] / result.images
    result["motor_share"] = result.motors.div(result.detections.where(result.detections.ne(0)))
    return result


def matched_rain(images):
    # Two means per stratum, across the dates with dry and wet observations respectively.
    keys = [CAMERA, "hour_local", "day_type"]
    parts = images.groupby([*keys, "rain_group"]).vehicles_total.agg(["size", "mean"])
    paired = parts.unstack("rain_group").reindex(columns=pd.MultiIndex.from_product([["size", "mean"], ["dry", "wet"]])).dropna()
    paired.columns = ["dry_images", "wet_images", "dry_vehicles_per_image", "wet_vehicles_per_image"]
    paired = paired.astype({"dry_images": int, "wet_images": int}).reset_index()
    paired["wet_minus_dry"] = paired.wet_vehicles_per_image - paired.dry_vehicles_per_image
    return paired


def bem_months(images):
    monthly = pd.read_parquet(EXTERNAL / "bem/bem_monthly_traffic.parquet")
    monthly["month_start"] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    values = monthly.set_index("month_start").traffic_volume
    if not values.index.is_unique:
        raise ValueError("Duplicate BEM month")
    months = images.date_local.dt.to_period("M").drop_duplicates().sort_values().dt.to_timestamp()
    bem = pd.DataFrame({"month_start": months.to_numpy()})
    bem["bem_thousand_trips_per_day"] = bem.month_start.map(values)
    bem["previous_month"] = (bem.month_start - pd.DateOffset(months=1)).map(values)
    bem["previous_year"] = (bem.month_start - pd.DateOffset(years=1)).map(values)
    for col, ref in [("mom_change", "previous_month"), ("yoy_change", "previous_year")]:
        bem[col] = bem.bem_thousand_trips_per_day.div(bem[ref].where(bem[ref].ne(0))) - 1
    return bem


def build_tables(images, daily_mot):
    tables = {
        "day_type": aggregate(images, ["day_type"]),
        "daily": aggregate(images, ["date_local", "day_type"]).merge(daily_mot, on="date_local", validate="one_to_one"),
        "hourly": aggregate(images, ["hour_local", "day_type"]),
        "rain": aggregate(images, ["rain_group"]),
        "roads": aggregate(images, ["road_class_300m"]),
        "cameras": aggregate(images, [CAMERA, "location", "road_class_300m"]).sort_values("vehicles_per_image", ascending=False),
    }
    controls = pd.Series([len(images), images.car.sum(), images.motor.sum()], index=["images", "cars", "motors"])
    for name, table in tables.items():
        if not table[controls.index].sum().eq(controls).all():
            raise ValueError(f"Aggregate totals for {name} do not reconcile with the image counts")
    exposure = images[[*GRID, "hour_start", "rain_group"]].drop_duplicates()
    grid_hours = exposure.groupby("rain_group").size().rename("grid_hours")
    tables["rain"] = tables["rain"].merge(grid_hours, on="rain_group", validate="one_to_one")
    tables["roads"] = tables["roads"].sort_values("road_class_300m", key=lambda s: s.map(TIERS.index))
    tables["pairs"] = matched_rain(images)
    tables["bem"] = bem_months(images)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path, help="Supplied Gold counts parquet, one row per image")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    from cctv_workbook import write_workbook

    args.out.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.counts)
    source_hash = hashlib.sha256(args.counts.read_bytes()).hexdigest()
    manifest_path = DATA / "bronze_data/manifest.parquet"
    manifest_matches = None
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path, columns=IMAGE_KEY)
        matched = source[IMAGE_KEY].merge(manifest, how="left", on=IMAGE_KEY, indicator=True, validate="one_to_one")
        manifest_matches = int(matched["_merge"].eq("both").sum())
        if manifest_matches != len(source):
            raise ValueError(f"Only {manifest_matches}/{len(source)} image keys match the Bronze manifest")
    images, daily_mot = enrich(source)
    tables = build_tables(images, daily_mot)
    # Sibling files inherit the output folder's access permissions on Windows.
    # Publish only after both exports are complete, using a distinct name for each run.
    with ExitStack() as cleanup:
        token = uuid4().hex
        parquet = args.out / f".{token}.parquet"
        workbook = args.out / f".{token}.xlsx"
        for staging_file in (parquet, workbook):
            cleanup.callback(staging_file.unlink, missing_ok=True)
        arrow = pa.Table.from_pandas(images, preserve_index=False)
        metadata = {**(arrow.schema.metadata or {}), b"counts_file": args.counts.name.encode(),
                    b"counts_sha256": source_hash.encode(), b"grain": b"one source CCTV image"}
        pq.write_table(arrow.replace_schema_metadata(metadata), parquet, compression="zstd")
        pd.testing.assert_frame_equal(pd.read_parquet(parquet), images, check_exact=True)
        write_workbook(workbook, images, tables, args.counts.name, source_hash, manifest_matches)
        parquet.replace(args.out / "image_counts_enriched.parquet")
        workbook.replace(args.out / "traffic_insights.xlsx")
    print(f"{len(images):,} images, {images[CAMERA].nunique()} cameras, {len(images.columns)} columns -> {args.out}")


if __name__ == "__main__":
    main()
