"""Run several YOLO models / parameter sets on the SAME 100 hand-counted images and compare them.

Uses the images listed in count_set/counts.csv (made by make_count_sheet.py).

Run from the repo root:
    python DataSci/model_eval/100Imgs_Testing/compare_configs.py                       # all configs below
    python DataSci/model_eval/100Imgs_Testing/compare_configs.py --configs yolo11x yolo11m_finetune
    python DataSci/model_eval/100Imgs_Testing/compare_configs.py --configs yolo11x --extra-model DataSci/model_eval/finetune/runs/yolo11m_ft/weights/best.pt
                                     # try a new fine-tuned model (02_train_finetune.py) at several conf values

Output (DataSci/model_eval/comparison/):
    <config>/<camera_id>/<image>.jpg   boxes drawn, one folder per camera (1 image each)
    side_by_side/<image>.jpg           all configs next to each other, with counts on top
    predicted_counts.csv               car / motorcycle / heavy per image per config
    metrics.csv                        error vs. your hand counts (only rows you filled in)
                                       + gpu_ms_per_img (speed on this machine, batch 1)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_EVAL = Path(__file__).resolve().parents[1]      # DataSci/model_eval (this file is in 100Imgs_Testing/)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODEL_EVAL))                   # viz.py lives in model_eval/
from viz import draw_result, header  # noqa: E402

IMAGE_ROOT = REPO_ROOT / "raw_data/cctv_images"
COUNTS_CSV = MODEL_EVAL / "count_set/counts.csv"
OUT_DIR = MODEL_EVAL / "comparison"
FINETUNED = REPO_ROOT / "DataEngineer/pipeline/object_detection_data/models/yolo11m_finetune.pt"

# ---- parameter sets to compare: edit / add freely --------------------------------------
# model: weights file (yolo11n/s/m/l/x.pt download automatically on first use)
# iou: NMS overlap threshold (optional, default 0.7 = YOLO default)
# The 3 pretrained sizes share ONE setting (the best found for yolo11x in round-2 tuning,
# tuning_history/round2/summary_round2.csv), so only model size changes between them.
# The fine-tuned model uses its own best conf from 0.1-0.4.
CONFIGS = {
    "yolo11m": dict(model="yolo11m.pt", conf=0.075, imgsz=640, agnostic=False, iou=0.5),
    "yolo11l": dict(model="yolo11l.pt", conf=0.075, imgsz=640, agnostic=False, iou=0.5),
    "yolo11x": dict(model="yolo11x.pt", conf=0.075, imgsz=640, agnostic=False, iou=0.5),   # silver before fine-tuning
    "yolo11m_finetune": dict(model=str(FINETUNED), conf=0.1, imgsz=640, agnostic=False, iou=0.5),  # current silver
}
# ------------------------------------------------------------------------------------------

CLASSES = ["car", "motorcycle", "heavy"]
COCO_TO_OURS = {2: "car", 3: "motorcycle", 5: "heavy", 7: "heavy"}   # bus/truck -> heavy


def class_map(model) -> dict:
    """Model class id -> our class. Fine-tuned models already use our names; COCO models need mapping."""
    names = model.names
    if set(CLASSES) <= set(names.values()):
        return {i: n for i, n in names.items() if n in CLASSES}
    return COCO_TO_OURS


def imread_unicode(path: Path):
    """cv2.imread fails on Thai paths on Windows; decode from bytes instead."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".jpg", img)[1].tobytes())


def count_str(c):
    return f"car {c['car']}  moto {c['motorcycle']}  heavy {c['heavy']}"


def evaluate(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error and bias (pred - truth) per config, overall and day/night."""
    gt = gt[gt.skip.fillna(0).astype(str).str.strip().isin(["", "0", "0.0"])]
    gt = gt.dropna(subset=CLASSES)
    if gt.empty:
        return pd.DataFrame()
    m = pred.merge(gt[["image", *CLASSES]], on="image", suffixes=("_pred", "_gt"))
    rows = []
    for (cfg, dn), g in pd.concat([m, m.assign(day_night="all")]).groupby(["config", "day_night"]):
        r = {"config": cfg, "day_night": dn, "n_images": len(g)}
        tot_pred = sum(g[f"{c}_pred"] for c in CLASSES)
        tot_gt = sum(g[f"{c}_gt"] for c in CLASSES)
        for c in CLASSES:
            err = g[f"{c}_pred"] - g[f"{c}_gt"]
            r[f"{c}_mae"] = round(err.abs().mean(), 2)
            r[f"{c}_bias"] = round(err.mean(), 2)
        r["total_mae"] = round((tot_pred - tot_gt).abs().mean(), 2)
        r["total_bias"] = round((tot_pred - tot_gt).mean(), 2)
        r["total_count_ratio"] = round(tot_pred.sum() / max(tot_gt.sum(), 1), 2)   # 1.0 = perfect
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["day_night", "total_mae"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), help="names from CONFIGS")
    ap.add_argument("--scale", type=int, default=3, help="upscale before drawing (sharper text)")
    ap.add_argument("--extra-model", type=Path, help="also test these weights (e.g. a fine-tuned best.pt)")
    ap.add_argument("--extra-confs", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4],
                    help="conf values tried for --extra-model (fine-tuned models usually want higher conf)")
    ap.add_argument("--extra-iou", type=float, default=0.5)
    a = ap.parse_args()

    if a.extra_model:
        tag = a.extra_model.parent.parent.name if a.extra_model.parent.name == "weights" else a.extra_model.stem
        for c in a.extra_confs:
            name = f"{tag}_c{c}"
            CONFIGS[name] = dict(model=str(a.extra_model), conf=c, imgsz=640, agnostic=False, iou=a.extra_iou)
            a.configs.append(name)

    sheet = pd.read_csv(COUNTS_CSV, encoding="utf-8-sig", dtype={"camera_id": str})
    for c in CLASSES:
        sheet[c] = pd.to_numeric(sheet[c], errors="coerce")

    from ultralytics import YOLO

    pred_rows = []
    speed = {}                                         # config -> GPU/CPU ms per image
    drawn = {}                                         # (config, image) -> annotated image
    for name in a.configs:
        cfg = CONFIGS[name]
        print(f"[{name}] {cfg}")
        model = YOLO(cfg["model"])
        cmap = class_map(model)
        model.predict(np.zeros((288, 352, 3), np.uint8), imgsz=cfg["imgsz"], verbose=False)   # warm-up
        infer_ms = []
        for _, s in sheet.iterrows():
            img = imread_unicode(IMAGE_ROOT / s.rel_path)
            res = model.predict(img, conf=cfg["conf"], iou=cfg.get("iou", 0.7), imgsz=cfg["imgsz"],
                                classes=list(cmap), agnostic_nms=cfg["agnostic"],
                                verbose=False)[0]
            infer_ms.append(sum(res.speed.values()))    # preprocess + inference + NMS
            counts = {c: 0 for c in CLASSES}
            for cls in res.boxes.cls.tolist():
                counts[cmap[int(cls)]] += 1
            pred_rows.append({"config": name, "image": s.image, "camera_id": s.camera_id,
                              "day_night": s.day_night, **counts})

            plot = draw_result(res, a.scale)
            imwrite_unicode(OUT_DIR / name / s.camera_id / s.image, plot)
            drawn[(name, s.image)] = header(plot, scale=a.scale, lines=[f"{name}: {Path(cfg['model']).stem} conf={cfg['conf']}"
                                                   f" agn={cfg['agnostic']}", count_str(counts)])
        speed[name] = round(float(np.median(infer_ms)), 1)

    # side by side: every config for the same image in one file
    for _, s in sheet.iterrows():
        panels = [drawn[(n, s.image)] for n in a.configs]
        h = max(p.shape[0] for p in panels)
        panels = [np.vstack([p, np.zeros((h - p.shape[0], p.shape[1], 3), np.uint8)]) for p in panels]
        row = np.hstack(panels)
        gt_line = f"{s.image}  ({s.day_night})"
        if not s[CLASSES].isna().any():
            gt_line += "   HAND COUNT: " + count_str({c: int(s[c]) for c in CLASSES})
        imwrite_unicode(OUT_DIR / "side_by_side" / s.image, header(row, [gt_line], scale=a.scale))

    pred = pd.DataFrame(pred_rows)
    pred = pred.sort_values(["image", "config"]).reset_index(drop=True)   # A/B rows of same image together
    pred.to_csv(OUT_DIR / "predicted_counts.csv", index=False, encoding="utf-8-sig")
    metrics = evaluate(pred, sheet)
    if metrics.empty:
        print("no hand counts filled in yet -> images saved, metrics skipped")
    else:
        metrics["gpu_ms_per_img"] = metrics.config.map(speed)
        metrics.to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
        print(metrics.to_string(index=False))
    print(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
