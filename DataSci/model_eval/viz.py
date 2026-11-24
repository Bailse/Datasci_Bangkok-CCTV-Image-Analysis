"""Sharp drawing of YOLO boxes on tiny CCTV frames (352x288).

Why: drawing text at native size and then zooming makes labels blurry.
We upscale the image first, then draw boxes/text at the larger size, with a filled
label background so "car 0.90" stays readable.
"""
import cv2
import numpy as np

COLORS = {"car": (0, 200, 255), "motorcycle": (0, 220, 0), "bus": (255, 0, 255),
          "truck": (255, 128, 0), "person": (60, 60, 255), "bicycle": (200, 200, 0),
          "heavy": (255, 0, 255)}


def draw_detections(img, boxes, names, confs, scale=3):
    """img: BGR array; boxes: list of (x1, y1, x2, y2) in original pixels."""
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    thick = max(1, scale // 2)
    font_scale = 0.18 * scale
    for (x1, y1, x2, y2), name, conf in zip(boxes, names, confs):
        color = COLORS.get(name, (255, 255, 255))
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(big, p1, p2, color, thick, cv2.LINE_AA)
        label = f"{name} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = p1[1] - th - base - 2 if p1[1] - th - base - 2 >= 0 else p1[1]   # inside box if no room above
        cv2.rectangle(big, (p1[0], top), (p1[0] + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(big, label, (p1[0] + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return big


def draw_result(res, scale=3):
    """Convenience wrapper for an ultralytics Result."""
    return draw_detections(res.orig_img, res.boxes.xyxy.tolist(),
                           [res.names[int(c)] for c in res.boxes.cls.tolist()],
                           res.boxes.conf.tolist(), scale)


def header(img, lines, scale=3):
    """Black bar with white text above an image, sized for the upscaled image."""
    fs, h = 0.18 * scale, int(7 * scale)
    bar = np.zeros((h * len(lines) + 6, img.shape[1], 3), dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (6, h * (i + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])
