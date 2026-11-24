"""Fine-tuning step 2: train YOLO on the boxes you fixed in Roboflow.

In Roboflow: Generate a version WITHOUT preprocessing resize and WITHOUT augmentations
(YOLO does its own augmentation while training), then Export -> format "YOLOv11" -> download zip.

Run from the repo root:
    python DataSci/model_eval/02_train_finetune.py --export path/to/roboflow_export.zip
    python DataSci/model_eval/02_train_finetune.py --export path/to/unzipped_folder --model yolo11l.pt

What it does:
  1. reads all images/labels from the export (train/valid/test folders, whatever Roboflow made)
  2. re-splits them BY CAMERA (val = ~15% of cameras). Frames from one camera look almost the same,
     so a random split would let the model "see" the val images and the val score would lie.
     Labels are written with YOLO's own COCO class ids (car=2, motorcycle=3, ...), so the fine-tuned
     model is a drop-in for silver: only MODEL_WEIGHTS / CONF in config.py change.
  3. trains, and prints where best.pt is. Then evaluate it on the 100 hand-counted images:
     python DataSci/model_eval/100Imgs_Testing/compare_configs.py --configs yolo11x yolo11m_finetune --extra-model <best.pt>

Output (DataSci/model_eval/finetune/):
    dataset/                 images + labels split into train/val, data.yaml
    runs/<name>/             training curves, val results, weights/best.pt
"""
import argparse
import random
import re
import shutil
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "finetune"
DATASET_DIR = OUT_DIR / "dataset"
# classes we label (must match 01_prepare_roboflow.py); everything else in COCO just gets no boxes
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def unpack(export: Path) -> Path:
    if export.suffix.lower() != ".zip":
        return export
    dest = OUT_DIR / "roboflow_export"
    if dest.exists():
        shutil.rmtree(dest)
    with zipfile.ZipFile(export) as z:
        z.extractall(dest)
    return dest


def read_names(root: Path) -> list[str]:
    ymls = list(root.rglob("data.yaml"))
    if not ymls:
        raise SystemExit(f"no data.yaml in {root} - export as 'YOLOv11' format from Roboflow")
    names = yaml.safe_load(ymls[0].read_text(encoding="utf-8"))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def camera_of(img: Path) -> str:
    """Roboflow renames '1028_20260826_191055.jpg' to '1028_20260826_191055_jpg.rf.<hash>.jpg'."""
    m = re.match(r"(\d+)_", img.name)
    if not m:
        raise SystemExit(f"can't read camera id from {img.name}")
    return m.group(1)


def build_dataset(root: Path, coco_names: dict, val_frac: float, seed: int) -> Path:
    names = read_names(root)
    unknown = set(names) - set(CLASSES)
    if unknown:
        raise SystemExit(f"unexpected classes in Roboflow export: {unknown} (rename them in Roboflow)")
    coco_id = {n: i for i, n in coco_names.items()}
    remap = {i: coco_id[n] for i, n in enumerate(names)}    # Roboflow sorts names alphabetically
    print(f"export classes -> COCO ids: { {n: coco_id[n] for n in names} }")

    images = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
              and p.parent.name == "images"]
    cams = sorted({camera_of(p) for p in images})
    rng = random.Random(seed)
    val_cams = set(rng.sample(cams, max(1, round(len(cams) * val_frac))))

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    n = {"train": 0, "val": 0}
    n_boxes = {c: 0 for c in names}
    for img in images:
        split = "val" if camera_of(img) in val_cams else "train"
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        lines = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:        # skip polygons/empty lines; we only use boxes
                    continue
                cls = remap[int(parts[0])]
                n_boxes[coco_names[cls]] += 1
                lines.append(" ".join([str(cls), *parts[1:]]))
        for sub in ("images", "labels"):
            (DATASET_DIR / sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, DATASET_DIR / "images" / split / img.name)
        (DATASET_DIR / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n[split] += 1

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(DATASET_DIR), "train": "images/train",
                                         "val": "images/val", "names": coco_names}),
                         encoding="utf-8")
    print(f"{len(images)} images: train {n['train']}, val {n['val']} "
          f"(val cameras: {len(val_cams)}/{len(cams)}), boxes {n_boxes}")
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path, help="Roboflow YOLOv11 export (.zip or folder)")
    ap.add_argument("--model", default="yolo11m.pt", help="start weights: yolo11s/m/l/x.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower it if you get CUDA out of memory")
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (try 10 if < 300 images)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of cameras used for validation")
    ap.add_argument("--name", default=None, help="run name (default: <model>_ft)")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader processes. 0 = load in the main process (safe on Windows: each worker "
                         "re-loads torch/CUDA and can exhaust RAM -> MemoryError). Try 2 if you have RAM to spare")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    data_yaml = build_dataset(unpack(a.export), dict(model.names), a.val_frac, a.seed)

    name = a.name or f"{Path(a.model).stem}_ft"
    model.train(
        data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, freeze=a.freeze,
        patience=20,          # stop if val mAP hasn't improved for 20 epochs
        hsv_v=0.5,            # stronger brightness changes: day / night / headlights
        fliplr=0.5, flipud=0.0,   # CCTV is never upside down
        project=str(OUT_DIR / "runs"), name=name, exist_ok=True, seed=a.seed,
        workers=a.workers,
    )
    best = OUT_DIR / "runs" / name / "weights" / "best.pt"
    print(f"\nbest weights -> {best}")
    print("evaluate on the 100 hand-counted images:")
    print(f'  python DataSci/model_eval/100Imgs_Testing/compare_configs.py --configs yolo11x yolo11m_finetune --extra-model "{best}"')


if __name__ == "__main__":
    main()
