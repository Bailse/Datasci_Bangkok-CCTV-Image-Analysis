from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENRICHED_PATH = (
    PROJECT_ROOT
    / "DataSci"
    / "analysis"
    / "outputs"
    / "cctv_aug2026"
    / "image_counts_enriched.parquet"
)
CCTV_INFO_PATH = PROJECT_ROOT / "data" / "cctv_info.csv"
OSM_WAYS_PATH = PROJECT_ROOT / "data" / "external_data" / "processed" / "osm" / "camera_osm_ways.parquet"
ACCIDENTS_PATH = PROJECT_ROOT / "data" / "external_data" / "processed" / "accidents" / "bangkok_accidents_2026.parquet"
BEM_MONTHLY_PATH = PROJECT_ROOT / "data" / "external_data" / "processed" / "bem" / "bem_monthly_traffic.parquet"
MODEL_METRICS_PATH = PROJECT_ROOT / "DataSci" / "model_eval" / "comparison" / "metrics.csv"
SPEED_BENCHMARK_PATH = PROJECT_ROOT / "DataSci" / "model_eval" / "speed_benchmark.csv"
MODEL_CHARTS_DIR = PROJECT_ROOT / "DataSci" / "model_eval" / "comparison" / "charts"


def _find_data_file() -> Path | None:
    if ENRICHED_PATH.exists():
        return ENRICHED_PATH
    for p in PROJECT_ROOT.rglob("*.parquet"):
        if "bronze_data" not in p.parts and "external_data" not in p.parts and "Users" not in p.parts:
            return p
    return None


@st.cache_data
def load_enriched_data() -> tuple[pd.DataFrame, str]:
    data_path = _find_data_file()
    if data_path is None:
        rng = np.random.default_rng(42)
        rows = []
        for i in range(1, 21):
            cam_id = f"CAM-{i:03d}"
            for h in range(24):
                peak = 25 if h in (8, 9, 14, 15) else 5
                rows.append({
                    "camera_id": cam_id,
                    "location": f"Camera Location {i}",
                    "location_detail": "Sample Data",
                    "latitude": 13.7563 + rng.normal(0, 0.04),
                    "longitude": 100.5018 + rng.normal(0, 0.04),
                    "hour_local": h,
                    "hour": h,
                    "vehicles_total": int(max(1, rng.normal(20 + peak, 5))),
                    "car": int(max(0, rng.normal(12 + peak * 0.6, 3))),
                    "motor": int(max(0, rng.normal(8 + peak * 0.4, 3))),
                    "day_type": "weekday",
                    "rain_group": "dry",
                    "day_night": "day" if 6 <= h <= 18 else "night",
                })
        return pd.DataFrame(rows), "Fallback mode: sample dataset"

    df = pd.read_parquet(data_path).copy()
    df["camera_id"] = df["camera_id"].astype(str)

    if "hour_local" in df.columns:
        df["hour"] = pd.to_numeric(df["hour_local"], errors="coerce")
    elif "ts_filename" in df.columns:
        df["hour"] = pd.to_datetime(df["ts_filename"], errors="coerce").dt.hour
    else:
        df["hour"] = 0
    df["hour"] = df["hour"].fillna(0).astype(int)

    if "location_detail" in df.columns and "location" in df.columns:
        df["camera_label"] = df["location_detail"].fillna(df["location"]).fillna(df["camera_id"])
    elif "location" in df.columns:
        df["camera_label"] = df["location"].fillna(df["camera_id"])
    else:
        df["camera_label"] = df["camera_id"]

    if CCTV_INFO_PATH.exists() and {"latitude", "longitude"}.issubset(df.columns):
        info = pd.read_csv(CCTV_INFO_PATH)
        if "camera_id" in info.columns and {"latitude", "longitude"}.issubset(info.columns):
            info["camera_id"] = info["camera_id"].astype(str)
            info_map = info.drop_duplicates("camera_id").set_index("camera_id")[["latitude", "longitude"]]
            df["latitude"] = df["latitude"].fillna(df["camera_id"].map(info_map["latitude"]))
            df["longitude"] = df["longitude"].fillna(df["camera_id"].map(info_map["longitude"]))

    return df, f"Loaded dataset: {data_path.relative_to(PROJECT_ROOT)} ({len(df):,} rows)"


@st.cache_data
def load_supplementary_data() -> dict:
    extras = {}

    if OSM_WAYS_PATH.exists():
        osm_df = pd.read_parquet(OSM_WAYS_PATH).copy()
        if "camera_id" in osm_df.columns:
            osm_df["camera_id"] = osm_df["camera_id"].astype(str)
        if "lanes" in osm_df.columns:
            osm_df["lanes_num"] = pd.to_numeric(osm_df["lanes"], errors="coerce")
        extras["osm"] = osm_df

    if ACCIDENTS_PATH.exists():
        acc_df = pd.read_parquet(ACCIDENTS_PATH).copy()
        for col in ("latitude", "longitude"):
            if col in acc_df.columns:
                acc_df[col] = pd.to_numeric(acc_df[col], errors="coerce")
        extras["accidents"] = acc_df

    if BEM_MONTHLY_PATH.exists():
        extras["bem"] = pd.read_parquet(BEM_MONTHLY_PATH).copy()

    if MODEL_METRICS_PATH.exists():
        extras["model_metrics"] = pd.read_csv(MODEL_METRICS_PATH)

    if SPEED_BENCHMARK_PATH.exists():
        extras["speed_benchmark"] = pd.read_csv(SPEED_BENCHMARK_PATH)

    if MODEL_CHARTS_DIR.exists():
        charts = sorted(MODEL_CHARTS_DIR.glob("*.png"))
        if charts:
            extras["model_charts"] = [str(p) for p in charts]

    return extras