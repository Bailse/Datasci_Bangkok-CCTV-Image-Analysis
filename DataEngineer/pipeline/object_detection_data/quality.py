"""Mark bad frames in silver so gold leaves them out.

Two problems in the raw CCTV data (both still get a new filename/timestamp):
- no_signal: tiny "no signal" placeholder images (< NO_SIGNAL_MAX_BYTES).
- frozen:    the camera feed is stuck, so the image is a copy of the previous one.
             The first image of a stuck run is real and stays "ok"; only the copies are flagged.

Cheap (no YOLO): ~20 s for all 61k images.
"""
import numpy as np
import pandas as pd
import cv2

import config as cfg

BAD_STATUSES = ("no_signal", "frozen")


def _small_gray(path):
    """Downscaled grayscale image as float32, or None if unreadable.
    np.fromfile + imdecode works with Thai filenames on any Windows code page."""
    try:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    except OSError:
        return None
    if img is None:
        return None
    return cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def flag_bad_frames(df: pd.DataFrame, files: list[tuple[str, str, object]]) -> pd.DataFrame:
    """df = silver rows of ONE camera; files = [(filename, rel_path, ts), ...] for that camera.
    Adds file_bytes and frame_diff columns and sets status to no_signal / frozen.
    Safe to run again on already flagged data."""
    df = df.copy()
    df.loc[df.status.isin(BAD_STATUSES), "status"] = "ok"

    rel_paths = dict((fname, rel) for fname, rel, _ in files)
    size, diff = {}, {}
    prev = None
    for fname in sorted(rel_paths):                      # time order (filename ends with YYYYMMDD_HHMMSS)
        path = cfg.IMAGE_ROOT / rel_paths[fname]
        size[fname] = path.stat().st_size if path.exists() else None
        if size[fname] is None or size[fname] < cfg.NO_SIGNAL_MAX_BYTES:
            continue                                     # compare across no-signal gaps, not with them
        cur = _small_gray(path)
        if cur is not None:
            if prev is not None:
                diff[fname] = float(np.abs(cur - prev).mean())
            prev = cur

    df["file_bytes"] = df.filename.map(size)
    df["frame_diff"] = df.filename.map(diff)            # NaN for the first image of a camera

    ok = df.status == "ok"
    no_signal = ok & (df.file_bytes < cfg.NO_SIGNAL_MAX_BYTES)
    frozen = ok & ~no_signal & (df.frame_diff < cfg.FROZEN_MAX_DIFF)
    df.loc[no_signal, "status"] = "no_signal"
    df.loc[frozen, "status"] = "frozen"
    return df
