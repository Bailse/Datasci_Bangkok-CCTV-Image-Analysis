# Model evaluation: choosing the YOLO model for the silver pipeline

Goal: count vehicles in our CCTV images as accurately as possible, so the silver/gold counts are trustworthy.
We did it in two stages:

1. **Tune settings** (conf / iou / imgsz) of off-the-shelf YOLO models: best was yolo11x, conf 0.075, iou 0.5.
2. **Fine-tune** yolo11m on 597 of our own CCTV images labelled in Roboflow. This beat every off-the-shelf
   model, so **silver now uses the fine-tuned model**.

## Result (current)

Measured on 100 hand-counted images (1 per camera, 50 day / 50 night). The three off-the-shelf sizes use the
same setting (conf 0.075, iou 0.5, imgsz 640), so only model size changes between them.

| Model | Total MAE | Day | Night | Car | Motorcycle | Heavy | Counted vs. truth |
|---|---|---|---|---|---|---|---|
| yolo11m | 3.71 | 3.22 | 4.20 | 2.37 | 2.46 | 1.32 | 82% |
| yolo11l | 3.16 | 2.92 | 3.40 | 2.19 | 2.47 | 1.22 | 89% |
| yolo11x | 2.82 | 2.30 | 3.34 | 2.06 | 2.34 | 1.18 | 87% |
| **yolo11m fine-tuned, conf 0.1 (silver)** | **2.21** | **1.98** | **2.44** | **1.56** | **1.11** | **0.54** | **89%** |

MAE = average absolute difference between the model's count and the hand count, per image (lower is better).

- Fine-tuning cut yolo11m's error by 40% (3.71 -> 2.21) and beats yolo11x, a model ~3x bigger (57M vs 20M parameters).
- Biggest gain is motorcycles (2.46 -> 1.11): off-the-shelf models miss many Bangkok motorcycles.
- Still weak: night. Every model under-counts at night (fine-tuned counts ~78% of night vehicles).
  Treat dashboard numbers as trends, not exact counts.
- Heavy caveat: `LABELING_RULES.md` labels pickups as `truck`, while the hand counts put pickups under `car`.
  The compare script maps truck -> heavy, so the car/heavy split is not perfectly comparable. Use total and
  motorcycle MAE for conclusions.

Slide charts: `comparison/charts/` (made by `100Imgs_Testing/make_charts.py`).
Full numbers: `comparison/metrics.csv`.

Ground truth caveat: `count_set/counts.csv` was hand-counted by Claude (AI) following `count_set/RULES.txt`.
Dense night scenes are approximate (see the `note` column). A human spot-check is recommended before
quoting these numbers in the presentation.

### Stage 1 history: tuning off-the-shelf models

Full table: `tuning_history/round2/summary_round2.csv` (15 settings). Highlights:

| Setting | MAE | Day | Night |
|---|---|---|---|
| Old silver: yolo11s, conf 0.25, iou 0.7 | 6.60 | 5.86 | 7.34 |
| yolo11s, conf 0.075, iou 0.5 | 4.04 | 3.10 | 4.98 |
| yolo11l, conf 0.075, iou 0.5 | 3.44 | 2.98 | 3.90 |
| yolo11x, conf 0.075, iou 0.5 (best, 2-fold cross-validation MAE 3.14) | 2.97 | 2.32 | 3.62 |

(Stage 1 numbers were measured before small fixes to `counts.csv`, so they differ slightly from the
current table above. Compare models within one table only.)

Things that did NOT help: imgsz 960/1280 (worse: images are only 352x288), agnostic_nms, test-time
augmentation, separate day/night conf (tiny gain, not worth it).

## Silver pipeline settings (DataEngineer/pipeline/object_detection_data/config.py)

```python
MODEL_WEIGHTS = str(ROOT / "models/yolo11m_finetune.pt")   # fine-tuned, ~40 MB, committed to git
CONF  = 0.1
IOU   = 0.5
IMGSZ = 640
```

`models/*.pt` is the only `.pt` allowed in git (see `.gitignore`); the off-the-shelf `yolo11*.pt` files are
downloaded automatically by ultralytics when a script needs them.

Before re-running silver with a new model: delete the old `data/silver_data/detections/camera_*.parquet`
files (the flow skips cameras that already have a file), then rebuild gold.

### What each setting means

#### Paths
| Setting | Meaning |
|---|---|
| `IMAGE_ROOT` | Folder with the raw CCTV images: `raw_data/cctv_images/<camera_id>/<file>.jpg` |
| `BRONZE_MANIFEST` | Bronze table: one row per image (camera, timestamp, path, file size) |
| `SILVER_DIR` | Silver output: one parquet per camera with YOLO counts per image |
| `GOLD_PATH` | Gold output: all cameras combined, grouped into the columns used for analysis |

#### YOLO (silver) - changing these requires re-running silver
| Setting | Value | What it does | What we found |
|---|---|---|---|
| `MODEL_WEIGHTS` | `models/yolo11m_finetune.pt` | Which YOLO weights. Off-the-shelf sizes n < s < m < l < x: bigger sees more detail but is slower | Fine-tuned yolo11m beats off-the-shelf yolo11x (MAE 2.21 vs 2.82) and is faster (~14 min vs ~37 min for all images on an RTX 4060) |
| `CONF` | `0.1` | Minimum confidence to keep a box. Lower keeps more boxes (finds more vehicles, but also more false boxes) | The biggest lever. Fine-tuned model: tried 0.1/0.2/0.3/0.4, 0.1 best (still slightly under-counts). Off-the-shelf models: 0.075 best |
| `IOU` | `0.5` | Duplicate-box filter (NMS). If two boxes of the same class overlap more than this, only the more confident one is kept | 0.5 reduced double-counting vs. the default 0.7 |
| `IMGSZ` | `640` | Size YOLO resizes each image to before predicting | Our images are only 352x288; 960/1280 made results worse (upscaling adds no detail) |
| `BATCH` | `16` | How many images go to the GPU at once. Speed/memory only, does not change results | |
| `TRACKED` | 6 classes | COCO classes stored in silver: person, bicycle, car, motorcycle, bus, truck | The fine-tuned model keeps COCO class ids, so `TRACKED` did not change |

Note: `CONF` and `IOU` were chosen by comparing vehicle *counts* with hand counts, not by checking every
box. Some misses and false boxes can cancel out in the count.

#### Gold - changing these does NOT require re-running YOLO (`python silver_flow.py --gold-only`)
| Setting | What it does |
|---|---|
| `GOLD_GROUPS` | Maps silver classes to gold columns. Now: `car` = car, `motor` = motorcycle. bus/truck/person/bicycle are in silver but not used in gold yet. Pickups are labelled `truck`, so if a heavy-vehicle column is wanted: `"car": ["car", "truck"], "heavy": ["bus"]` |
| `NIGHT_START_HOUR`, `NIGHT_END_HOUR` | Gold's `day_night` column comes from the Bangkok hour of `ts_filename`: 20:00-05:59 = `night`. It used to be mean brightness < 70, but street lights keep 19-21h images at ~95 brightness, so only 403 of 61,319 images were `night` |

## Files

All scripts run from the repo root, e.g. `python DataSci/model_eval/02_train_finetune.py --help`.

| File | What it does |
|---|---|
| `00_benchmark_speed.py` | Measures speed of yolo11 s/m/l/x on your machine and estimates the full silver run time -> `speed_benchmark.csv` |
| `01_prepare_roboflow.py` | Fine-tuning step 1: picks ~600 images (6 per camera, half night, half busiest), pre-labels them with yolo11x, zips them for Roboflow |
| `02_train_finetune.py` | Fine-tuning step 2: trains YOLO on the Roboflow export, split by camera |
| `LABELING_RULES.md` | Labeling rules for the team in Roboflow (6 YOLO classes) |
| `viz.py` | Helper: upscales images before drawing so labels are sharp (does not affect predictions) |
| `100Imgs_Testing/make_count_sheet.py` | Picks the 100 test images (1 per camera, half night) and creates the hand-count sheet |
| `100Imgs_Testing/compare_configs.py` | Runs the models in `CONFIGS` on the 100 images -> `comparison/` (metrics + side-by-side images) |
| `100Imgs_Testing/make_charts.py` | Slide charts from `comparison/metrics.csv` -> `comparison/charts/` |

| Data / output | What it is | In git? |
|---|---|---|
| `count_set/counts.csv` | Hand counts (ground truth) for the 100 images. **Do not delete.** If you edit it in Excel, switch the keyboard to English first | yes |
| `count_set/RULES.txt` | Counting rules used for the hand counts | yes |
| `count_set/images/` | The 100 test images | no |
| `comparison/metrics.csv` | Error of each model vs. hand counts (overall / day / night) + speed | yes |
| `comparison/predicted_counts.csv` | Each model's car / motorcycle / heavy count per image | yes |
| `comparison/charts/*.png` | Slide charts | yes |
| `comparison/side_by_side/`, `comparison/<model>/` | Images with boxes drawn, all models next to each other | no |
| `tuning_history/round2/` | Stage 1 tuning results (all settings tried) | yes |
| `finetune/selection.csv` | Which images were sent to Roboflow and why (round 1, 600 images) | yes |
| `finetune/selection_round2.csv` | Round 2 picks (100 images): time group, reason, predicted vehicles, unsure boxes | yes |
| `finetune/` (everything else) | Roboflow upload/export, training dataset, training runs (`runs/yolo11m_ft/`) | no |
| `speed_benchmark.csv` | Speed results on RTX 4060 | yes |

Images that are not in git can be regenerated with the scripts, or see the team Google Drive.

## Fine-tuning with Roboflow (how the current model was made)

We keep **YOLO's own classes** (the 6 that silver stores: `person`, `bicycle`, `car`, `motorcycle`, `bus`,
`truck`), so the analysis groups can still be decided later in gold (`GOLD_GROUPS`) and the fine-tuned model
is a drop-in replacement for silver.

**Labeling rules: [`LABELING_RULES.md`](LABELING_RULES.md)** (6 YOLO classes, pickups = `truck`,
rider = `person` + `motorcycle`). `count_set/RULES.txt` is only for the hand counts used in evaluation.

1. **Pick + pre-label images** (~5 min on RTX 4060):
   `python DataSci/model_eval/01_prepare_roboflow.py`
   -> `finetune/roboflow_upload.zip` (600 images + yolo11x boxes, conf 0.15) and `finetune/selection.csv`.
   The 100 count-set images (and frames within 30 min of them) are never picked: they stay the test set.
   Re-running overwrites `selection.csv`.
2. **Roboflow: upload.** New project -> type *Object Detection* -> Upload Data -> drop the zip.
   Check that the boxes appear and the class names are exactly the 6 YOLO classes (don't rename them).
   Note: on the free plan, projects are public.
3. **Roboflow: fix the boxes** following `LABELING_RULES.md`: delete false boxes, draw missed ones (especially
   far away and at night), fix classes. Split the images between team members with Roboflow's *Assign*.
4. **Roboflow: generate a version.** Preprocessing: remove *Resize* (keep Auto-Orient). Augmentation: **none**
   (YOLO augments during training). The train/valid/test split here does not matter: step 5 re-splits by camera.
   Export -> format **YOLOv11** -> *Download zip to computer*.
5. **Train** (yolo11m, ~1.3 min per epoch on RTX 4060 with batch 16; stops early if val mAP does not improve
   for 20 epochs - our run stopped at epoch 34, best epoch 14):
   `python DataSci/model_eval/02_train_finetune.py --export "raw_data/Bangkok CCTV Vehicle Detection.v2i.yolov11.zip"`
   Labels are converted back to COCO ids (car=2, motorcycle=3, bus=5, truck=7, ...).
   Options: `--model yolo11l.pt` (more accurate, slower), `--batch 8` if GPU memory goes above 8 GB
   (it then spills into system RAM and gets slow), `--freeze 10` if you have fewer than ~300 labelled images,
   `--workers 0` (default; higher is faster but each worker reloads torch and can cause `MemoryError` on Windows).
   Interrupted? Resume with `yolo train resume model="DataSci/model_eval/finetune/runs/yolo11m_ft/weights/last.pt"`
   (do not re-run the script: it rebuilds `finetune/dataset/` and starts again from epoch 1).
6. **Evaluate on the 100 hand-counted images** at several conf values:
   `python DataSci/model_eval/100Imgs_Testing/compare_configs.py --configs yolo11x yolo11m_finetune --extra-model DataSci/model_eval/finetune/runs/yolo11m_ft/weights/best.pt`
   It tries conf 0.1/0.2/0.3/0.4 for the new model. Compare `total_mae` and `motorcycle_mae` (overall and night).
7. **Use it in silver** only if it beats the current model: copy `best.pt` to
   `DataEngineer/pipeline/object_detection_data/models/`, point `MODEL_WEIGHTS` at it, set `CONF` to the best
   conf from step 6, update the `yolo11m_finetune` entry in `compare_configs.py`, then re-run silver and gold.

### Round 2: 100 more images aimed at the weak spots

On the test set the fine-tuned model is worst on **busy night scenes** (16+ vehicles: MAE 4.8, misses ~4
vehicles per image) and round 1 had only 24 such images. Its errors are mostly missed vehicles.

1. **Pick + pre-label** (~5 min): `python DataSci/model_eval/01b_pick_more_images.py`
   -> `finetune/roboflow_upload_round2.zip` and `finetune/selection_round2.csv`.
   Default mix: night 50, dusk 15, busy day with motorcycles 25, most-unsure anywhere 10 (change with
   `--night --dusk --day --hard`). Never picks a test or round-1 image or anything within 30 min of one
   (same camera), and keeps new picks of one camera at least 30 min apart. `--max-per-cam 1` spreads the
   picks over more cameras (default 2: busy cameras get two).
2. **Upload into the SAME Roboflow project** as round 1, fix the boxes as before (same `LABELING_RULES.md`),
   then generate a NEW version containing all images (round 1 + round 2) and export it.
3. **Train** with a new run name so round 1's results are kept:
   `python DataSci/model_eval/02_train_finetune.py --export <new export.zip> --name yolo11m_ft_r2`
4. **Evaluate** against the current model (same 100 test images, so the numbers are directly comparable to 2.21):
   `python DataSci/model_eval/100Imgs_Testing/compare_configs.py --configs yolo11m_finetune --extra-model DataSci/model_eval/finetune/runs/yolo11m_ft_r2/weights/best.pt`
   Look at night `total_mae` first. Switch silver to it only if it is better (step 7 above).
