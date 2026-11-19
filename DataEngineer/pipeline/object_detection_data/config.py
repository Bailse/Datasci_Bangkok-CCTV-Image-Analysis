from pathlib import Path

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parents[2]

IMAGE_ROOT      = REPO_ROOT / "raw_data/cctv_images"          # <camera_id>/<filename>
BRONZE_MANIFEST = REPO_ROOT / "data/bronze_data/manifest.parquet"
SILVER_DIR      = REPO_ROOT / "data/silver_data/detections"
GOLD_PATH       = REPO_ROOT / "data/gold/image_counts.parquet"


# --- YOLO (silver) ---
MODEL_WEIGHTS = "yolo11x.pt"   # tuned (DataSci/model_eval): was yolo11s.pt
CONF = 0.075                   # tuned: was 0.25
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