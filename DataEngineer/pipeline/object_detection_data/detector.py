from collections import Counter
from functools import lru_cache

import cv2
import pandas as pd

import config as cfg


@lru_cache(maxsize=1)
def get_model():
    from ultralytics import YOLO
    return YOLO(cfg.MODEL_WEIGHTS)          # downloads weights on first use


def read_image(path):
    """Return BGR array or None if the file is missing/corrupt."""
    return cv2.imread(str(path))            # cv2 returns None instead of raising


def mean_brightness(img) -> float:
    small = cv2.resize(img, (160, 90))
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def detect_camera(camera_id: str, files: list[tuple[str, str, object]]) -> pd.DataFrame:
    """files = [(filename, rel_path, ts_filename), ...] for ONE camera."""
    model = get_model()
    class_ids = list(cfg.TRACKED.values())
    id_to_name = {v: k for k, v in cfg.TRACKED.items()}
    rows = []

    for start in range(0, len(files), cfg.BATCH):
        chunk = files[start:start + cfg.BATCH]
        imgs, meta = [], []
        for fname, rel_path, ts_filename in chunk:
            img = read_image(cfg.IMAGE_ROOT / rel_path)
            if img is None:                              # keep the failure visible
                rows.append({"camera_id": camera_id, "ts_filename": ts_filename, "filename": fname,
                             "status": "read_error", "brightness": None,
                             **{n: None for n in cfg.TRACKED}})
                continue
            imgs.append(img)
            meta.append((fname, ts_filename, mean_brightness(img)))

        if not imgs:
            continue
        results = model.predict(imgs, conf=cfg.CONF, imgsz=cfg.IMGSZ,
                                classes=class_ids, verbose=False)
        for (fname, ts_filename, bright), res in zip(meta, results):
            counts = Counter(id_to_name[int(c)] for c in res.boxes.cls)
            rows.append({"camera_id": camera_id, "ts_filename": ts_filename, "filename": fname,
                         "status": "ok", "brightness": bright,
                         **{n: counts.get(n, 0) for n in cfg.TRACKED}})
    return pd.DataFrame(rows)