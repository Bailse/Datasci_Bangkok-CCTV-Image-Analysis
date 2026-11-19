"""Step 2: build a small ground-truth set by counting vehicles by hand (no box drawing).

Run from the repo root:
    python DataSci/model_eval/02_make_count_sheet.py

Output:
    DataSci/model_eval/count_set/images/*.jpg   100 clean images (1 per camera, ~50% night)
    DataSci/model_eval/count_set/counts.csv     fill in car / motorcycle / heavy for each row
    DataSci/model_eval/count_set/RULES.txt      counting rules (everyone must follow the same ones)

Re-running will NOT overwrite counts.csv if it already exists (protects your work).
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = REPO_ROOT / "raw_data/cctv_images"
MANIFEST = REPO_ROOT / "data/bronze_data/manifest.parquet"
OUT_DIR = Path(__file__).resolve().parent / "count_set"

MIN_FILE_BYTES = 2000                         # skip blank "no signal" frames
NIGHT_HOURS = set(range(19, 24)) | set(range(0, 6))

RULES = """\
COUNTING RULES (follow exactly, so everyone counts the same way)

Columns to fill: car, motorcycle, heavy   (whole numbers, 0 if none)

car        = sedan, SUV, van, pickup, taxi, songthaew (pickup with seats)
motorcycle = motorcycle / scooter. Rider + bike = 1 motorcycle. Tuk-tuk = motorcycle.
heavy      = bus, minibus, 6-wheel+ truck, trailer

Count a vehicle when:
  - it is on the road or road shoulder, moving OR parked
  - at least half of it is visible (cut at the image edge -> count only if >= half)
  - you can tell it is a vehicle, even if far away
Do NOT count:
  - vehicles too small/blurry to tell they are vehicles (just a light or a few pixels)
  - reflections, vehicles on billboards/screens
  - bicycles, people

If an image is unusable (black, frozen, camera covered), put 1 in "skip" and leave counts empty.
Unsure about something? Write it in "note" instead of guessing silently.
Zoom in: images are only 352x288.
"""


def select_images(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One image per camera, alternating night/day across cameras (~50% night), random date/time."""
    df = manifest[manifest.file_size_bytes >= MIN_FILE_BYTES].copy()
    ts = pd.to_datetime(df.ts_filename_raw, format="%Y%m%d%H%M%S")
    df["is_night"] = ts.dt.hour.isin(NIGHT_HOURS)

    picks = []
    for i, (cam, g) in enumerate(sorted(df.groupby("camera_id"), key=lambda x: x[0])):
        want_night = i % 2 == 0
        pool = g[g.is_night == want_night]
        if pool.empty:                        # camera has no image of that kind -> take any
            pool = g
        picks.append(pool.sample(n=1, random_state=seed + i))
    return pd.concat(picks, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    manifest = pd.read_parquet(MANIFEST)
    manifest["camera_id"] = manifest.camera_id.astype(str)
    sel = select_images(manifest, a.seed)
    # ASCII names: <camera>_<YYYYMMDD>_<HHMMSS>.jpg
    sel["image"] = (sel.camera_id + "_" + sel.ts_filename_raw.str[:8] + "_"
                    + sel.ts_filename_raw.str[8:] + ".jpg")
    sel["time"] = pd.to_datetime(sel.ts_filename_raw, format="%Y%m%d%H%M%S").dt.strftime("%Y-%m-%d %H:%M")

    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for _, s in sel.iterrows():
        shutil.copy2(IMAGE_ROOT / s.rel_path, img_dir / s.image)

    (OUT_DIR / "RULES.txt").write_text(RULES, encoding="utf-8")

    sheet = sel[["image", "camera_id", "time", "is_night"]].copy()
    sheet["day_night"] = sheet.pop("is_night").map({True: "night", False: "day"})
    for col in ("car", "motorcycle", "heavy", "skip"):
        sheet[col] = ""
    sheet["note"] = ""
    sheet["counted_by"] = ""
    sheet["rel_path"] = sel.rel_path              # link back to the original file (for step 3)

    csv_path = OUT_DIR / "counts.csv"
    if csv_path.exists():
        print(f"{csv_path} already exists -> not overwritten")
    else:
        sheet.to_csv(csv_path, index=False, encoding="utf-8-sig")   # utf-8-sig: opens fine in Excel
        print(f"wrote {csv_path}")
    print(f"{len(sel)} images from {sel.camera_id.nunique()} cameras, "
          f"night {(sheet.day_night == 'night').mean():.0%} -> {img_dir}")


if __name__ == "__main__":
    main()
