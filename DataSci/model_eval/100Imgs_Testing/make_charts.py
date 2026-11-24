"""Slide charts from comparison/metrics.csv (made by compare_configs.py).

Run from the repo root:
    python DataSci/model_eval/100Imgs_Testing/make_charts.py

Output (DataSci/model_eval/comparison/charts/), 16:9-friendly PNGs at 200 dpi:
    1_total_mae.png        one bar per model: average counting error per image (headline chart)
    2_mae_by_vehicle.png   same, split into car / motorcycle / heavy (3 small panels)
    3_day_vs_night.png     day vs night error per model
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_EVAL = Path(__file__).resolve().parents[1]
METRICS = MODEL_EVAL / "comparison/metrics.csv"
OUT = MODEL_EVAL / "comparison/charts"

ORDER = ["yolo11m", "yolo11l", "yolo11x", "yolo11m_finetune"]          # small -> large -> fine-tuned
LABELS = {"yolo11m": "YOLO11m", "yolo11l": "YOLO11l", "yolo11x": "YOLO11x",
          "yolo11m_finetune": "YOLO11m\nfine-tuned"}
HIGHLIGHT = "yolo11m_finetune"

# palette (validated: light surface, categorical slots 1-2)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 12, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def style(ax, ymax):
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelcolor=MUTED)
    ax.tick_params(axis="x", length=0)


def title(fig, main, sub):
    fig.text(0.04, 0.95, main, fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=12, color=INK2, va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", OUT / name)


def bars(ax, values, colors, fmt="{:.2f}"):
    x = range(len(values))
    rects = ax.bar(x, values, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    for r, v, c in zip(rects, values, colors):
        bold = c != GRAY
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom",
                fontsize=12, color=INK if bold else INK2, fontweight="bold" if bold else "normal",
                transform=ax.transData, clip_on=False)
    return rects


def chart_total(m):
    a = m[m.day_night == "all"].set_index("config").loc[ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.14)
    colors = [BLUE if c == HIGHLIGHT else GRAY for c in ORDER]
    bars(ax, a.total_mae.values, colors)
    ax.set_xticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
    ax.set_ylabel("Avg. vehicles miscounted per image")
    style(ax, a.total_mae.max() * 1.18)
    best, m0, x = (a.loc[c, "total_mae"] for c in (HIGHLIGHT, "yolo11m", "yolo11x"))
    title(fig, "Fine-tuning beats a bigger model",
          f"Total counting error (MAE) on 100 hand-counted images, lower is better. Same conf/iou for 11m/l/x.\n"
          f"YOLO11m {m0:.2f} → {best:.2f} after fine-tuning ({(m0 - best) / m0:.0%} less error), "
          f"also below YOLO11x {x:.2f} ({(x - best) / x:.0%} less)")
    save(fig, "1_total_mae.png")


def chart_by_vehicle(m):
    a = m[m.day_night == "all"].set_index("config").loc[ORDER]
    cols = [("car_mae", "Car"), ("motorcycle_mae", "Motorcycle"), ("heavy_mae", "Heavy (bus/truck)")]
    ymax = max(a[c].max() for c, _ in cols) * 1.22
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.16, wspace=0.12)
    colors = [BLUE if c == HIGHLIGHT else GRAY for c in ORDER]
    short = {"yolo11m": "11m", "yolo11l": "11l", "yolo11x": "11x", "yolo11m_finetune": "11m\nFT"}
    for ax, (col, name) in zip(axes, cols):
        bars(ax, a[col].values, colors)
        ax.set_xticks(range(len(ORDER)), [short[c] for c in ORDER])
        ax.set_title(name, fontsize=13, color=INK, loc="left", pad=10)
        style(ax, ymax)
    axes[0].set_ylabel("Avg. miscounted per image")
    moto0, moto1 = a.loc["yolo11m", "motorcycle_mae"], a.loc[HIGHLIGHT, "motorcycle_mae"]
    title(fig, "Biggest gain: motorcycles",
          f"Counting error (MAE) by vehicle type, lower is better. FT = fine-tuned on 597 Bangkok CCTV images. "
          f"Motorcycle error {moto0:.2f} → {moto1:.2f}")
    save(fig, "2_mae_by_vehicle.png")


def chart_day_night(m):
    d = m[m.day_night != "all"].pivot(index="config", columns="day_night", values="total_mae").loc[ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.14)
    w = 0.36
    for i, (col, color, name) in enumerate([("day", ORANGE, "Day"), ("night", BLUE, "Night")]):
        xs = [k + (i - 0.5) * w for k in range(len(ORDER))]
        rects = ax.bar(xs, d[col].values, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        for r, v in zip(rects, d[col].values):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=INK2)
    ax.set_xticks(range(len(ORDER)), [LABELS[c] for c in ORDER])
    ax.set_ylabel("Avg. vehicles miscounted per image")
    style(ax, d.values.max() * 1.18)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=12, labelcolor=INK2)
    title(fig, "Night is harder for every model",
          "Total counting error (MAE) on 50 day + 50 night images, lower is better")
    save(fig, "3_day_vs_night.png")


def main():
    m = pd.read_csv(METRICS, encoding="utf-8-sig")
    missing = set(ORDER) - set(m.config)
    if missing:
        raise SystemExit(f"metrics.csv has no rows for {missing} - run compare_configs.py first")
    chart_total(m)
    chart_by_vehicle(m)
    chart_day_night(m)


if __name__ == "__main__":
    main()
