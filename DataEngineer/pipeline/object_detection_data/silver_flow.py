import argparse
from pathlib import Path

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE

import config as cfg
from detector import detect_camera


def camera_out_path(camera_id: str) -> Path:
    return cfg.SILVER_DIR / f"camera_{camera_id}.parquet"


@task(retries=1, retry_delay_seconds=10, cache_policy=NO_CACHE)
def process_camera(camera_id: str, manifest: pd.DataFrame, limit: int | None):
    log = get_run_logger()
    out = camera_out_path(camera_id)
    if out.exists():                                  # RESUME: skip finished cameras
        log.info(f"skip {camera_id} (done)")
        return str(out)

    sub = manifest[manifest.camera_id == camera_id].sort_values("filename")
    if limit:
        sub = sub.head(limit)
    files = list(zip(sub.filename, sub.rel_path, sub.ts_filename))

    df = detect_camera(camera_id, files)

    tmp = out.with_suffix(".tmp")                     # atomic write: no half files
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    log.info(f"{camera_id}: {len(df)} images")
    return str(out)


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
    ok["day_night"] = (ok.brightness >= cfg.NIGHT_BRIGHTNESS_THRESHOLD
                       ).map({True: "day", False: "night"})

    gold = ok[["camera_id", "ts_filename", "filename", *cfg.GOLD_GROUPS, "day_night"]]
    cfg.GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(cfg.GOLD_PATH, index=False)
    return len(gold)


@flow(name="silver-yolo-detection")
def silver_flow(shard: int = 0, n_shards: int = 1,
                cameras: list[str] | None = None, limit: int | None = None):
    log = get_run_logger()
    cfg.SILVER_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(cfg.BRONZE_MANIFEST)
    manifest["camera_id"] = manifest.camera_id.astype(str)
    manifest["ts_filename"] = pd.to_datetime(                   # bronze keeps raw string; parse here
        manifest.ts_filename_raw, format="%Y%m%d%H%M%S"
    ).dt.tz_localize("Asia/Bangkok")

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
    a = ap.parse_args()
    if a.gold_only:
        print(build_gold.fn())                        # re-apply gold rules, no YOLO
    else:
        silver_flow(a.shard, a.n_shards, a.cameras, a.limit)