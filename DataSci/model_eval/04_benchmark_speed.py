"""Step 4: measure how fast each YOLO size runs on THIS machine, and estimate the full silver run.

Run from the repo root:
    python DataSci/model_eval/04_benchmark_speed.py
    python DataSci/model_eval/04_benchmark_speed.py --models yolo11s.pt yolo11x.pt --n-images 300

It does not write anything into data/ and does not touch the silver pipeline.
Timing mimics silver: read image from disk -> predict in batches of 16.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = REPO_ROOT / "raw_data/cctv_images"
MANIFEST = REPO_ROOT / "data/bronze_data/manifest.parquet"
SILVER_WEIGHTS = REPO_ROOT / "DataEngineer/pipeline/object_detection_data/yolo11s.pt"
TOTAL_IMAGES = 61_319
CLASSES = [0, 1, 2, 3, 5, 7]            # same classes silver tracks


def imread(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)   # Thai paths ok


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.075)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    gpu = torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0) if gpu else 'CPU (no CUDA!)'}   torch {torch.__version__}")
    if not gpu:
        print("WARNING: running on CPU. Install CUDA PyTorch first if you have an NVIDIA GPU.")

    m = pd.read_parquet(MANIFEST)
    m = m[m.file_size_bytes >= 2000]                       # skip blank frames
    paths = [IMAGE_ROOT / p for p in m.sample(n=a.n_images, random_state=0).rel_path]

    rows = []
    for name in a.models:
        weights = str(SILVER_WEIGHTS) if name == "yolo11s.pt" and SILVER_WEIGHTS.exists() else name
        model = YOLO(weights)
        kw = dict(conf=a.conf, iou=a.iou, imgsz=a.imgsz, classes=CLASSES, verbose=False)
        model.predict([imread(p) for p in paths[: a.batch]], **kw)          # warm-up (not timed)
        sync()

        t_read = t_pred = 0.0
        for i in range(0, len(paths), a.batch):
            t0 = time.perf_counter()
            imgs = [imread(p) for p in paths[i:i + a.batch]]
            t1 = time.perf_counter()
            model.predict(imgs, **kw)
            sync()
            t2 = time.perf_counter()
            t_read += t1 - t0
            t_pred += t2 - t1

        n = len(paths)
        ms = (t_read + t_pred) / n * 1000
        full_h = ms * TOTAL_IMAGES / 1000 / 3600
        vram = torch.cuda.max_memory_allocated() / 1e9 if gpu else 0
        rows.append({"model": Path(name).stem, "ms_per_img_total": round(ms, 1),
                     "ms_read": round(t_read / n * 1000, 1), "ms_predict": round(t_pred / n * 1000, 1),
                     "img_per_sec": round(1000 / ms, 1), "full_run_hours": round(full_h, 2),
                     "full_run_minutes": round(full_h * 60), "peak_vram_gb": round(vram, 2)})
        print(rows[-1])
        del model
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = Path(__file__).resolve().parent / "speed_benchmark.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print(f"estimate = ms_per_img_total x {TOTAL_IMAGES:,} images (silver adds a little overhead on top)")


if __name__ == "__main__":
    main()
