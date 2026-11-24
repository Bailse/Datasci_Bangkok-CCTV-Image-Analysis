from pathlib import Path

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parents[2]

IMAGE_ROOT      = REPO_ROOT / "raw_data/cctv_images"          # <camera_id>/<filename>
BRONZE_MANIFEST = REPO_ROOT / "data/bronze_data/manifest.parquet"
SILVER_DIR      = REPO_ROOT / "data/silver_data/detections"
GOLD_PATH       = REPO_ROOT / "data/gold/image_counts.parquet"


# --- YOLO (silver) ---
# fine-tuned on 597 labelled Bangkok CCTV images (DataSci/model_eval/06_train_finetune.py, best epoch 14).
# On the 100 hand-counted images: total MAE 2.21 vs 2.82 for yolo11x @ conf 0.075.
# *.pt is gitignored: share this file with the team separately (or copy finetune/runs/yolo11m_ft/weights/best.pt here).
MODEL_WEIGHTS = str(ROOT / "models/yolo11m_finetune.pt")   # was yolo11x.pt (before that yolo11s.pt)
CONF = 0.1                     # best of 0.1-0.4 for the fine-tuned model (was 0.075 for yolo11x)
IMGSZ = 640                    
BATCH = 16
IOU = 0.5                      # tuned: NMS overlap threshold (YOLO default 0.7)

# COCO class ids we store in silver (can adjust))
TRACKED = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}

# --- Gold rules (cheap to change, no YOLO rerun) ---
GOLD_GROUPS = {
    "car":   ["car"],            
    "motor": ["motorcycle"],
}
NIGHT_BRIGHTNESS_THRESHOLD = 70  # mean gray (0-255); tune by looking at samples