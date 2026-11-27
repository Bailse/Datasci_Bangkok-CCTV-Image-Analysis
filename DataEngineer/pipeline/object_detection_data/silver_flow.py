import argparse
from pathlib import Path

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE

import config as cfg
from detector import detect_camera
from quality import flag_bad_frames


def camera_out_path(camera_id: str) -> Path:
    return cfg.SILVER_DIR / f"camera_{camera_id}.parquet"


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_parquet(cfg.BRONZE_MANIFEST)
    manifest["camera_id"] = manifest.camera_id.astype(str)
    manifest["ts_filename"] = pd.to_datetime(                   # bronze keeps raw string; parse here
        manifest.ts_filename_raw, format="%Y%m%d%H%M%S"
    ).dt.tz_localize("Asia/Bangkok")
    return manifest


def camera_files(camera_id: str, manifest: pd.DataFrame, limit: int | None = None):
    sub = manifest[manifest.camera_id == camera_id].sort_values("filename")
    if limit:
        sub = sub.head(limit)
    return list(zip(sub.filename, sub.rel_path, sub.ts_filename))


def write_atomic(df: pd.DataFrame, out: Path):
    tmp = out.with_suffix(".tmp")                     # atomic write: no half files
    df.to_parquet(tmp, index=False)
    tmp.replace(out)


@task(retries=1, retry_delay_seconds=10, cache_policy=NO_CACHE)
def process_camera(camera_id: str, manifest: pd.DataFrame, limit: int | None):
    log = get_run_logger()
    out = camera_out_path(camera_id)
    if out.exists():                                  # RESUME: skip finished cameras
        log.info(f"skip {camera_id} (done)")
        return str(out)

    files = camera_files(camera_id, manifest, limit)
    df = detect_camera(camera_id, files)
    df = flag_bad_frames(df, files)                   # status no_signal / frozen -> left out of gold

    write_atomic(df, out)
    log.info(f"{camera_id}: {len(df)} images")
    return str(out)


def reflag_silver() -> pd.Series:
    """Re-apply quality.py to the existing silver files (no YOLO), e.g. after changing its thresholds."""
    manifest = load_manifest()
    parts = sorted(cfg.SILVER_DIR.glob("camera_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No silver camera files found in {cfg.SILVER_DIR}")
    statuses = []
    for p in parts:
        df = pd.read_parquet(p)
        camera_id = str(df.camera_id.iloc[0])
        df = flag_bad_frames(df, camera_files(camera_id, manifest))
        write_atomic(df, p)
        statuses.append(df.status)
    return pd.concat(statuses).value_counts()


@task(cache_policy=NO_CACHE)
def build_gold() -> int:
    parts = sorted(cfg.SILVER_DIR.glob("camera_*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No silver camera files found in {cfg.SILVER_DIR}; run silver detection first."
        )
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    ok = df[df.status == "ok"].copy()

    for out_col, src_cols in cfg.GOLD_GROUPS.items():
        ok[out_col] = ok[src_cols].sum(axis=1).astype(int)
    hour = pd.to_datetime(ok.ts_filename).dt.tz_convert("Asia/Bangkok").dt.hour
    is_night = (hour >= cfg.NIGHT_START_HOUR) | (hour < cfg.NIGHT_END_HOUR)
    ok["day_night"] = is_night.map({True: "night", False: "day"})

    gold = ok[["camera_id", "ts_filename", "filename", *cfg.GOLD_GROUPS, "day_night"]]
    cfg.GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(cfg.GOLD_PATH, index=False)
    return len(gold)


@flow(name="silver-yolo-detection")
def silver_flow(shard: int = 0, n_shards: int = 1,
                cameras: list[str] | None = None, limit: int | None = None):
    log = get_run_logger()
    cfg.SILVER_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    all_cams = sorted(manifest.camera_id.unique())
    if cameras:
        requested_cameras = {str(camera) for camera in cameras}
        missing_cameras = sorted(requested_cameras.difference(all_cams))
        if missing_cameras:
            available = ", ".join(all_cams[:10])
            raise ValueError(
                f"Camera IDs not found in bronze manifest: {', '.join(missing_cameras)}. "
                f"Available IDs include: {available}"
            )
        all_cams = [c for c in all_cams if c in requested_cameras]
    my_cams = all_cams[shard::n_shards]               # split work across teammates
    log.info(f"{len(my_cams)}/{len(all_cams)} cameras in this shard")

    for cam in my_cams:                               # sequential: YOLO already uses all CPU cores
        process_camera(cam, manifest, limit)

    if n_shards == 1:                                 # gold only when one machine has everything
        log.info(f"gold rows: {build_gold()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--cameras", nargs="*")
    ap.add_argument("--limit", type=int, help="images per camera, for smoke tests")
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--reflag", action="store_true",
                    help="re-mark no_signal/frozen frames in existing silver files (no YOLO), then rebuild gold")
    a = ap.parse_args()
    if a.reflag:
        print(reflag_silver().to_string())
        print("gold rows:", build_gold.fn())
    elif a.gold_only:
        print(build_gold.fn())                        # re-apply gold rules, no YOLO
    else:
        silver_flow(a.shard, a.n_shards, a.cameras, a.limit)