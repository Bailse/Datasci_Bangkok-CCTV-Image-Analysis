# Model evaluation: choosing YOLO parameters for the silver pipeline

Goal: find the YOLO settings that count vehicles in our CCTV images most accurately,
so the silver/gold counts are trustworthy. No fine-tuning here, only off-the-shelf models + settings.

## Result

| Setting | MAE (vehicles off per image) | Day | Night | Counted vs. truth | Full run on RTX 4060 |
|---|---|---|---|---|---|
| Current silver: yolo11s, conf 0.25, iou 0.7 | 6.60 | 5.86 | 7.34 | 44% | ~8 min |
| yolo11s, conf 0.075, iou 0.5 | 4.04 | 3.10 | 4.98 | 85% | ~8 min |
| yolo11l, conf 0.075, iou 0.5 | 3.44 | 2.98 | 3.90 | 88% | ~16 min |
| **yolo11x, conf 0.075, iou 0.5 (recommended)** | **2.97** | **2.32** | **3.62** | **87%** | **~37 min** |

MAE = average absolute difference between YOLO's count and the hand count, per image (lower is better).
Measured on 100 images (1 per camera, 50 day / 50 night). The winner was also checked on two random
halves of the 100 images (2-fold cross-validation MAE 3.14), so it is not tuned to this one sample.

Things that did NOT help: imgsz 960/1280 (worse: images are only 352x288), agnostic_nms, test-time
augmentation, separate day/night conf (tiny gain, not worth it).

Still weak (needs fine-tuning, not parameters): motorcycles at night in traffic jams (~2.3 off per image
for every model), and counts are still ~13% low overall. Treat dashboard numbers as trends, not exact counts.

Ground truth caveat: `count_set/counts.csv` was hand-counted by Claude (AI) following `count_set/RULES.txt`.
Dense night scenes are approximate (see the `note` column). A human spot-check is recommended before
quoting these numbers in the presentation.

## Changes for the silver pipeline (DataEngineer/pipeline/object_detection_data)

`config.py`
```python
MODEL_WEIGHTS = "yolo11x.pt"   # downloads automatically (~115 MB) - do NOT commit it, GitHub limit is 100 MB
CONF  = 0.075
IOU   = 0.5                    # new
IMGSZ = 640                    # unchanged
```

`detector.py`, in `model.predict(...)` add:
```python
iou=cfg.IOU,
```

Before re-running: delete the old `data/silver_data/detections/camera_*.parquet` files made with the old
settings (the flow skips cameras that already have a file), then rebuild gold.

Optional (`GOLD_GROUPS`): YOLO's "truck" is mostly pickups. If the team wants a heavy-vehicle column:
`"car": ["car", "truck"], "motor": ["motorcycle"], "heavy": ["bus"]` (no YOLO re-run needed, `--gold-only`).

## What each setting in `config.py` means

### Paths
| Setting | Meaning |
|---|---|
| `IMAGE_ROOT` | Folder with the raw CCTV images: `raw_data/cctv_images/<camera_id>/<file>.jpg` |
| `BRONZE_MANIFEST` | Bronze table: one row per image (camera, timestamp, path, file size) |
| `SILVER_DIR` | Silver output: one parquet per camera with YOLO counts per image |
| `GOLD_PATH` | Gold output: all cameras combined, grouped into the columns used for analysis |

### YOLO (silver) - changing these requires re-running silver
| Setting | Value | What it does | What we found |
|---|---|---|---|
| `MODEL_WEIGHTS` | `yolo11x.pt` | Which YOLO model. Sizes n < s < m < l < x: bigger sees more detail but is slower | Accuracy improves with size; x is best (MAE 2.97 vs 4.04 for s) and still only ~37 min for all images on an RTX 4060 |
| `CONF` | `0.075` | Minimum confidence to keep a box. Lower keeps more boxes (finds more vehicles, but also more false boxes); higher drops uncertain vehicles | The biggest lever. At 0.25 YOLO threw away about half of the real vehicles, especially at night. 0.075 was best; 0.05 starts to over-count |
| `IOU` | `0.5` | Duplicate-box filter (NMS). If two boxes of the same class overlap more than this, only the more confident one is kept. Lower = removes duplicates more aggressively | 0.5 reduced double-counting of the same car in daytime vs. the default 0.7 |
| `IMGSZ` | `640` | Size YOLO resizes each image to before predicting. Affects predictions | Our images are only 352x288; 960/1280 made results worse (upscaling adds no detail). 480 is about as accurate and faster |
| `BATCH` | `16` | How many images go to the GPU at once. Speed/memory only, does not change results | Uses ~2.7 GB VRAM with yolo11x |
| `TRACKED` | 6 classes | Which COCO classes are detected and stored in silver: person, bicycle, car, motorcycle, bus, truck | Keeping all 6 in silver lets gold be regrouped later without re-running YOLO |

Note: `CONF` and `IOU` were chosen by comparing vehicle *counts* with hand counts, not by checking every
box. Some misses and false boxes can cancel out in the count. Per-class errors and spot checks of the
images suggest the extra night detections are mostly real vehicles, while daytime still has some duplicate/false boxes.

### Gold - changing these does NOT require re-running YOLO (`python silver_flow.py --gold-only`)
| Setting | What it does |
|---|---|
| `GOLD_GROUPS` | Maps silver classes to gold columns. Now: `car` = car, `motor` = motorcycle. bus/truck/person/bicycle are in silver but not used in gold yet. YOLO's "truck" is mostly pickups, so if a heavy-vehicle column is wanted: `"car": ["car", "truck"], "heavy": ["bus"]` |
| `NIGHT_BRIGHTNESS_THRESHOLD` | Images with mean brightness (0-255) below this are labeled `night` in gold's `day_night` column. 70 was set by eye and has not been validated |

## Files

| File | What it does |
|---|---|
| `01_visualize_predictions.py` | Draws YOLO boxes ("car 0.90") on 3 images per camera (08:00/13:00/20:00), one folder per camera |
| `02_make_count_sheet.py` | Picks 100 images (1 per camera, half night) and creates the hand-count sheet |
| `03_compare_configs.py` | Runs several parameter sets on the same 100 images, saves side-by-side images and error metrics |
| `04_benchmark_speed.py` | Measures speed of yolo11 s/m/l/x on your machine and estimates the full silver run time |
| `viz.py` | Helper: upscales images before drawing so labels are sharp (does not affect predictions) |
| `count_set/counts.csv` | Hand counts (ground truth) for the 100 images |
| `count_set/RULES.txt` | Counting rules used for the hand counts |
| `comparison/round2/summary_round2.csv` | Final comparison of all tested settings |
| `speed_benchmark.csv` | Speed results on RTX 4060 |

All scripts run from the repo root, e.g. `python DataSci/model_eval/03_compare_configs.py`.
Images (`predictions/`, `comparison/**/*.jpg`, `count_set/images/`) are not in git: they can be
regenerated with the scripts, or see the team Google Drive.
