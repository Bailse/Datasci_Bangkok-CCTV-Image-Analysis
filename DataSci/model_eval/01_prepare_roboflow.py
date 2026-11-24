"""Fine-tuning step 1: pick images to label and pre-label them with yolo11x, ready to upload to Roboflow.

Pre-labels = YOLO's boxes already drawn, so in Roboflow you only fix them (delete wrong boxes,
add missed ones, fix classes) instead of drawing every box from zero.

Run from the repo root:
    python DataSci/model_eval/01_prepare_roboflow.py                  # 6 images x 100 cameras = 600
    python DataSci/model_eval/01_prepare_roboflow.py --per-cam 4      # smaller set

Output (DataSci/model_eval/finetune/):
    roboflow_upload/data.yaml          class names (YOLO/COCO names, see CLASSES) - Roboflow reads this
    roboflow_upload/train/images/      the picked images, ASCII names <camera>_<YYYYMMDD>_<HHMMSS>.jpg
    roboflow_upload/train/labels/      one .txt per image with the pre-label boxes (YOLO format)
    roboflow_upload.zip                same folder zipped: drag this into Roboflow "Upload Data"
    selection.csv                      which images were picked and why

How images are picked (per camera):
  - never the 100 count-set images (or anything within 30 min of them): those stay the test set
  - candidates spread over different days/hours (no near-identical consecutive frames)
  - ~50% night, and half of each group are the busiest scenes (most vehicles), because night
    traffic jams are where the off-the-shelf model is weakest
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IMAGE_ROOT = REPO_ROOT / "raw_data/cctv_images"
MANIFEST = REPO_ROOT / "data/bronze_data/manifest.parquet"
COUNTS_CSV = HERE / "count_set/counts.csv"
OUT_DIR = HERE / "finetune"
UPLOAD_DIR = OUT_DIR / "roboflow_upload"

MIN_FILE_BYTES = 2000                         # skip blank "no signal" frames
NIGHT_HOURS = set(range(19, 24)) | set(range(0, 6))   # same definition as 100Imgs_Testing/make_count_sheet.py
TEST_GAP = pd.Timedelta(minutes=30)          # keep away from count-set frames (they look the same)

# Keep YOLO's own (COCO) classes = the ones silver stores (TRACKED in config.py). Grouping into
# analysis columns (e.g. car+truck) happens later in gold, so the classes don't need to be decided now.
# Index in the upload = position in this list; 02_train_finetune.py maps the names back to COCO ids.
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
COCO_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
COCO_TO_UPLOAD = {COCO_IDS[n]: i for i, n in enumerate(CLASSES)}
VEHICLES = {COCO_IDS[n] for n in ("car", "motorcycle", "bus", "truck")}   # for "busiest scene"


def load_candidates(cand_per_cam: int, seed: int, cameras: int | None) -> pd.DataFrame:
    m = pd.read_parquet(MANIFEST)
    m["camera_id"] = m.camera_id.astype(str)
    m = m[m.file_size_bytes >= MIN_FILE_BYTES].copy()
    m["ts"] = pd.to_datetime(m.ts_filename_raw, format="%Y%m%d%H%M%S")
    m["is_night"] = m.ts.dt.hour.isin(NIGHT_HOURS)

    # drop the test images and their neighbours in time (same camera)
    test = pd.read_csv(COUNTS_CSV, encoding="utf-8-sig", dtype={"camera_id": str})
    near_test = pd.Series(False, index=m.index)
    for rel in test.rel_path:
        hit = m[m.rel_path == rel]
        if hit.empty:
            continue
        cam, t = hit.camera_id.iloc[0], hit.ts.iloc[0]
        near_test |= (m.camera_id == cam) & ((m.ts - t).abs() <= TEST_GAP)
    m = m[~near_test]

    cams = sorted(m.camera_id.unique())
    if cameras:
        cams = cams[:cameras]

    picks = []
    for i, cam in enumerate(cams):
        g = m[m.camera_id == cam]
        # one frame per (day, hour) slot -> spread over time, no near-duplicates
        slots = g.groupby([g.ts.dt.date, g.ts.dt.hour], group_keys=False).sample(n=1, random_state=seed + i)
        half = cand_per_cam // 2
        night, day = slots[slots.is_night], slots[~slots.is_night]
        n_night = min(half, len(night))
        picks.append(night.sample(n=n_night, random_state=seed + i))
        picks.append(day.sample(n=min(cand_per_cam - n_night, len(day)), random_state=seed + i))
    return pd.concat(picks, ignore_index=True)


def prelabel(cands: pd.DataFrame, weights: str, conf: float, iou: float, batch: int):
    """Run YOLO on every candidate. Returns (boxes per row, vehicle count per row)."""
    from ultralytics import YOLO
    import cv2
    import numpy as np

    model = YOLO(weights)
    boxes, counts = [], []
    rows = list(cands.itertuples())
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        imgs = [cv2.imdecode(np.fromfile(str(IMAGE_ROOT / r.rel_path), np.uint8), cv2.IMREAD_COLOR)
                for r in chunk]                       # imdecode: cv2.imread fails on Thai paths
        results = model.predict(imgs, conf=conf, iou=iou, imgsz=640,
                                classes=list(COCO_TO_UPLOAD), verbose=False)
        for res in results:
            cls = [int(c) for c in res.boxes.cls.tolist()]
            boxes.append([(COCO_TO_UPLOAD[c], *xywh) for c, xywh in zip(cls, res.boxes.xywhn.tolist())])
            counts.append(sum(c in VEHICLES for c in cls))
        print(f"  pre-labelled {min(start + batch, len(rows))}/{len(rows)}", end="\r")
    print()
    return boxes, counts


def choose(cands: pd.DataFrame, per_cam: int, busy_frac: float, seed: int) -> pd.DataFrame:
    """Per camera and per day/night group: busiest `busy_frac` + random rest."""
    out = []
    for (cam, night), g in cands.groupby(["camera_id", "is_night"]):
        n = per_cam // 2 if night else per_cam - per_cam // 2
        n = min(n, len(g))
        n_busy = round(n * busy_frac)
        busy = g.nlargest(n_busy, "pred_count")
        rest = g.drop(busy.index)
        rand = rest.sample(n=min(n - n_busy, len(rest)), random_state=seed)
        out += [busy.assign(reason="busy"), rand.assign(reason="random")]
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cam", type=int, default=6, help="images to label per camera")
    ap.add_argument("--candidates", type=int, default=20, help="candidates per camera to choose from")
    ap.add_argument("--busy-frac", type=float, default=0.5, help="share of picks that are the busiest scenes")
    ap.add_argument("--weights", default="yolo11x.pt", help="model used for pre-labels")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="pre-label confidence. Lower = more boxes to delete, higher = more to draw")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cameras", type=int, default=None, help="only the first N cameras (quick test)")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    img_dir, lbl_dir = UPLOAD_DIR / "train/images", UPLOAD_DIR / "train/labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    cands = load_candidates(a.candidates, a.seed, a.cameras)
    print(f"{len(cands)} candidates from {cands.camera_id.nunique()} cameras -> pre-labelling with {a.weights}")
    boxes, counts = prelabel(cands, a.weights, a.conf, a.iou, a.batch)
    cands["pred_count"] = counts
    cands["boxes"] = boxes

    sel = choose(cands, a.per_cam, a.busy_frac, a.seed)
    sel["image"] = sel.camera_id + "_" + sel.ts_filename_raw.str[:8] + "_" + sel.ts_filename_raw.str[8:] + ".jpg"
    for s in sel.itertuples():
        shutil.copy2(IMAGE_ROOT / s.rel_path, img_dir / s.image)        # original bytes, no re-encode
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in s.boxes]
        (lbl_dir / s.image).with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")

    (UPLOAD_DIR / "data.yaml").write_text(
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\ntrain: train/images\nval: train/images\n", encoding="utf-8")
    zip_path = shutil.make_archive(str(OUT_DIR / "roboflow_upload"), "zip", UPLOAD_DIR)

    sel["day_night"] = sel.is_night.map({True: "night", False: "day"})
    sel = sel.rename(columns={"pred_count": "pred_vehicles"})
    sel[["image", "camera_id", "day_night", "reason", "pred_vehicles", "rel_path"]].to_csv(
        OUT_DIR / "selection.csv", index=False, encoding="utf-8-sig")

    print(f"{len(sel)} images from {sel.camera_id.nunique()} cameras, night {sel.is_night.mean():.0%}, "
          f"{sum(len(b) for b in sel.boxes)} pre-label boxes")
    print(f"upload this to Roboflow -> {zip_path}")


if __name__ == "__main__":
    main()
