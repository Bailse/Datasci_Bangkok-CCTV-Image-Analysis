import hashlib
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import imagesize
import pandas as pd

BASE_PATH = Path('raw_data/cctv_images')
OUT_PATH = Path('DataEngineer/bronze_data/manifest.parquet')
VALID_FILE = {'.jpg', '.jpeg', '.png'}
TZ = "Asia/Bangkok"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

NAME_FORMAT = re.compile(
    r"(?P<camera_id>[^_]+)_(?P<camera_name>.+)_(?P<date>\d{8})_(?P<time>\d{6})"
)

def process(path : Path) -> dict :

    rel = path.relative_to(BASE_PATH)
    size_image = path.stat()
    stem = path.stem
    m = NAME_FORMAT.fullmatch(stem)

    width , height = imagesize.get(path)


    return {
        "image_id": hashlib.md5(rel.as_posix().encode("utf-8")).hexdigest()[:16], # create id
        "camera_id": m["camera_id"] if m else None, # camera id
        "camera_name": m["camera_name"] if m else None, # same as camera id
        "filename": path.name,  # original file name
        "rel_path": rel.as_posix(), # path of file
        "ext": path.suffix.lower(), # .type of files
        "file_size_bytes": size_image.st_size, # bytes of files
        "width": width, # width
        "height": height, # hight
        "ts_filename_raw": f"{m['date']}{m['time']}" if m else None, #date that appears in the filename
        "run_id": RUN_ID, # run id date of doing pipeline
    }

def main() -> None:

    files = [p for p in BASE_PATH.rglob("*")
             if p.is_file() and p.suffix.lower() in VALID_FILE]
    print(f"found {len(files):,} image files")

    rows = list(map(process, files))
    df = pd.DataFrame(rows)

    df["ingested_at"] = pd.Timestamp.now(tz="UTC")
    df = df.astype({
        "camera_id": "string", "camera_name": "string","width": "int32", "height": "int32", "file_size_bytes": "int64",
        "ext": "category",
    })
    df = df.sort_values(["camera_id", "ts_filename"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, OUT_PATH)
    print(f"wrote {len(df):,} rows -> {OUT_PATH}")

    
if __name__ == "__main__":
    main()
    