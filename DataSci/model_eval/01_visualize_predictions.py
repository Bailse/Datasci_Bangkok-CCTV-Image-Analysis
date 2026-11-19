"""Step 1: draw YOLO predictions (e.g. "car 0.90") on sample images, one folder per camera.

Run from the repo root:
    python DataSci/model_eval/01_visualize_predictions.py
    python DataSci/model_eval/01_visualize_predictions.py --model yolo11m.pt --cameras 1028 1039

Output:
    DataSci/model_eval/predictions/<model_name>/<camera_id>/<image>.jpg
    DataSci/model_eval/predictions/<model_name>/detections.csv   (one row per box)
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from viz import draw_result

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = REPO_ROOT / "raw_data/cctv_images"
MANIFEST = REPO_ROOT / "data/bronze_data/manifest.parquet"
DEFAULT_MODEL = REPO_ROOT / "DataEngineer/pipeline/object_detection_data/yolo11s.pt"
OUT_ROOT = Path(__file__).resolve().parent / "predictions"

# same classes the silver pipeline tracks (COCO ids)
TRACKED = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
MIN_FILE_BYTES = 2000          # ~1.4 KB files are blank "no signal" frames
SAMPLE_HOURS = (8, 13, 20)     # morning rush, midday, night


def pick_samples(manifest: pd.DataFrame, hours) -> pd.DataFrame:
    """For each camera and each target hour, take the image closest to that hour on the first day."""
    df = manifest[manifest.file_size_bytes >= MIN_FILE_BYTES].copy()
    ts = pd.to_datetime(df.ts_filename_raw, format="%Y%m%d%H%M%S")
    df["date"] = ts.dt.date
    df["minutes"] = ts.dt.hour * 60 + ts.dt.minute
    first_day = df.groupby("camera_id").date.transform("min")
    df = df[df.date == first_day]

    picks = []
    for h in hours:
        d = df.assign(gap=(df.minutes - h * 60).abs())
        picks.append(d.loc[d.groupby("camera_id").gap.idxmin()])
    return pd.concat(picks).drop_duplicates("filename")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="weights file, e.g. yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cameras", nargs="*", help="only these camera ids")
    ap.add_argument("--hours", nargs="*", type=int, default=list(SAMPLE_HOURS))
    ap.add_argument("--scale", type=int, default=3,
                    help="upscale factor before drawing (bigger = sharper labels, bigger files)")
    a = ap.parse_args()

    manifest = pd.read_parquet(MANIFEST)
    manifest["camera_id"] = manifest.camera_id.astype(str)
    if a.cameras:
        manifest = manifest[manifest.camera_id.isin(a.cameras)]
    samples = pick_samples(manifest, a.hours)
    print(f"{len(samples)} images from {samples.camera_id.nunique()} cameras")

    model = YOLO(a.model)
    out_dir = OUT_ROOT / Path(a.model).stem
    rows = []
    for _, s in samples.iterrows():
        res = model.predict(str(IMAGE_ROOT / s.rel_path), conf=a.conf, imgsz=a.imgsz,
                            classes=list(TRACKED.values()), verbose=False)[0]
        cam_dir = out_dir / s.camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        # imencode + write_bytes handles Thai file names on Windows (cv2.imwrite does not)
        ok, buf = cv2.imencode(".jpg", draw_result(res, a.scale))
        (cam_dir / s.filename).write_bytes(buf.tobytes())

        for cls, conf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
            rows.append({"camera_id": s.camera_id, "filename": s.filename,
                         "class": res.names[int(cls)], "conf": round(conf, 3),
                         "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})

    pd.DataFrame(rows).to_csv(out_dir / "detections.csv", index=False, encoding="utf-8-sig")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
