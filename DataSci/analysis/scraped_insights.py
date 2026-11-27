"""Reproducible analysis of the supplied six-class CCTV counts and scraped data.

Run from the repository root:
    python DataSci/analysis/scraped_insights.py --counts path/to/image_counts.parquet

Outputs are Parquet, tidy CSV tables, a Thai findings file and a PNG figure.
The existing two-class workbook and its output directory are left intact.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analyze_cctv import CAMERA, DATA, EXTERNAL, GRID, HERE, IMAGE_KEY, enrich, bem_months

CLASSES = ["car", "motor", "truck", "bus", "bicycle", "person"]
MOTOR_CLASSES = ["car", "motor", "truck", "bus"]
DEFAULT_OUT = HERE / "outputs" / "cctv_20261128"
MIN_PAIR_IMAGES = 3


def validate_counts(source):
    required = {*IMAGE_KEY, *CLASSES, "day_night"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Six-class input is required; missing columns: {sorted(missing)}")
    if source.empty or source[list(required)].isna().any().any():
        raise ValueError("Empty input or missing source values")
    if source.duplicated(IMAGE_KEY).any():
        raise ValueError("Duplicate image key")
    if not isinstance(source.ts_filename.dtype, pd.DatetimeTZDtype):
        raise ValueError("ts_filename must have an explicit timezone")
    if not source.camera_id.map(lambda x: isinstance(x, str)).all():
        raise ValueError("camera_id must be a string identifier")
    for col in CLASSES:
        if not pd.api.types.is_integer_dtype(source[col]) or source[col].lt(0).any():
            raise ValueError(f"{col} must contain nonnegative integer detections")
    if not source.day_night.isin(["day", "night"]).all():
        raise ValueError("day_night must contain day/night labels")


def add_features(images):
    """Keep legacy car+motor fields; use explicit four-class totals for this report."""
    out = images.copy()
    out["motor_vehicles_total"] = out[MOTOR_CLASSES].sum(axis=1)
    out["vehicles_including_bicycles"] = out.motor_vehicles_total + out.bicycle
    out["heavy_vehicles_total"] = out.truck + out.bus
    out["objects_total"] = out[CLASSES].sum(axis=1)
    denominator = out.motor_vehicles_total.where(out.motor_vehicles_total.ne(0))
    out["motor_share_all"] = out.motor / denominator
    out["heavy_vehicle_share"] = out.heavy_vehicles_total / denominator
    out["has_motor_vehicle"] = out.motor_vehicles_total.gt(0)
    out["weekday_number"] = out.date_local.dt.dayofweek
    hour = out.hour_local
    out["time_period"] = np.select(
        [hour.between(6, 8), hour.between(9, 15), hour.between(16, 18), hour.between(19, 21)],
        ["morning_06_09", "daytime_09_16", "evening_16_19", "late_19_22"], default="overnight_22_06",
    )
    out["is_commute_window"] = hour.between(6, 8) | hour.between(16, 18)
    out["clock_day_night"] = np.where(hour.between(6, 17), "day", "night")
    out["brightness_clock_disagree"] = out.day_night.ne(out.clock_day_night)
    out["rain_band"] = np.select(
        [out.rain_mm_hour.eq(0), out.rain_mm_hour.lt(1), out.rain_mm_hour.lt(5)],
        ["dry_0", "positive_lt1", "1_to_lt5"], default="5_or_more",
    )
    out["camera_mean_motor_vehicles"] = out.groupby(CAMERA).motor_vehicles_total.transform("mean")
    out["relative_to_camera_mean"] = out.motor_vehicles_total.div(
        out.camera_mean_motor_vehicles.where(out.camera_mean_motor_vehicles.ne(0)))
    return out


def summarize(images, keys):
    """Ratios are ratios of sums. Quantiles describe frames, not unique vehicles."""
    specs = {
        "images": ("filename", "size"), "cameras": (CAMERA, "nunique"),
        "dates": ("date_local", "nunique"),
        "motor_vehicle_detections": ("motor_vehicles_total", "sum"),
        "median_motor_vehicles": ("motor_vehicles_total", "median"),
        "p90_motor_vehicles": ("motor_vehicles_total", lambda x: x.quantile(.9)),
        "frames_with_motor_vehicles": ("has_motor_vehicle", "sum"),
        **{f"{c}_detections": (c, "sum") for c in CLASSES},
    }
    table = images.groupby(keys, as_index=False, observed=True, dropna=False).agg(**specs)
    table["mean_motor_vehicles"] = table.motor_vehicle_detections / table.images
    for col in CLASSES:
        table[f"mean_{col}"] = table[f"{col}_detections"] / table.images
    denom = table.motor_vehicle_detections.where(table.motor_vehicle_detections.ne(0))
    table["motor_share_all"] = table.motor_detections / denom
    table["heavy_vehicle_share"] = (table.truck_detections + table.bus_detections) / denom
    table["zero_motor_vehicle_fraction"] = 1 - table.frames_with_motor_vehicles / table.images
    return table


def matched_comparison(images, keys, category, baseline, comparison, min_images=MIN_PAIR_IMAGES):
    """Equal weight per shared stratum, with a visible minimum sample requirement."""
    grouped = images.groupby([*keys, category]).motor_vehicles_total.agg(["size", "mean"])
    wide = grouped.unstack(category).reindex(
        columns=pd.MultiIndex.from_product([["size", "mean"], [baseline, comparison]]))
    wide.columns = ["baseline_images", "comparison_images", "baseline_mean", "comparison_mean"]
    wide = wide.dropna().reset_index()
    wide = wide.loc[(wide.baseline_images >= min_images) & (wide.comparison_images >= min_images)].copy()
    wide[["baseline_images", "comparison_images"]] = wide[["baseline_images", "comparison_images"]].astype(int)
    wide["comparison_minus_baseline"] = wide.comparison_mean - wide.baseline_mean
    wide["baseline_category"] = baseline
    wide["comparison_category"] = comparison
    wide["min_images_per_category"] = min_images
    return wide


def coverage_tables(source, manifest):
    """Coverage against the collected inventory, not an assumed 24-hour schedule."""
    if manifest[IMAGE_KEY].isna().any().any() or manifest.duplicated(IMAGE_KEY).any():
        raise ValueError("Manifest must contain complete, unique image keys")
    audit = manifest[IMAGE_KEY].merge(source[IMAGE_KEY], on=IMAGE_KEY, how="outer",
                                    indicator=True, validate="one_to_one")
    if audit._merge.eq("right_only").any():
        raise ValueError("Supplied images not found in the Bronze inventory")
    audit["has_count"] = audit._merge.eq("both")
    local = audit.ts_filename.dt.tz_convert("Asia/Bangkok")
    audit["date_local"] = local.dt.tz_localize(None).dt.normalize()
    audit["hour_local"] = local.dt.hour
    result = {}
    for name, keys in {"coverage_daily": ["date_local"],
                       "coverage_camera": [CAMERA],
                       "coverage_camera_hour": [CAMERA, "date_local", "hour_local"]}.items():
        table = audit.groupby(keys, as_index=False).agg(
            inventory_images=("filename", "size"), analyzed_images=("has_count", "sum"))
        table["missing_counts"] = table.inventory_images - table.analyzed_images
        table["coverage_fraction"] = table.analyzed_images / table.inventory_images
        result[name] = table
    return result


def build_insight_tables(images, daily_mot, manifest):
    groupings = {
        "daily": ["date_local", "day_type"], "day_type": ["day_type"],
        "hourly": ["hour_local", "day_type"], "time_period": ["time_period", "day_type"],
        "cameras": [CAMERA, "location", "road_class_300m"],
        "camera_day": [CAMERA, "date_local", "day_type"],
        "camera_hour": [CAMERA, "date_local", "hour_local", "day_type", "rain_group"],
        "rain": ["rain_group"], "rain_band": ["rain_band"], "roads": ["road_class_300m"],
    }
    tables = {name: summarize(images, keys) for name, keys in groupings.items()}
    expected = [len(images), int(images.motor_vehicles_total.sum()), *images[CLASSES].sum().tolist()]
    columns = ["images", "motor_vehicle_detections", *[f"{c}_detections" for c in CLASSES]]
    for name, table in tables.items():
        if table[columns].sum().tolist() != expected:
            raise ValueError(f"Aggregate totals do not reconcile: {name}")
    tables["daily"] = tables["daily"].merge(daily_mot, on="date_local", validate="one_to_one")
    tables["cameras"] = tables["cameras"].sort_values("mean_motor_vehicles", ascending=False)
    # One observation per weather grid-hour prevents rain from being summed per image.
    weather = images[[*GRID, "hour_start", "rain_mm_hour", "rain_group", "temperature_2m_c"]].drop_duplicates()
    if weather.duplicated([*GRID, "hour_start"]).any():
        raise ValueError("Inconsistent weather within a grid-hour")
    weather["date_local"] = weather.hour_start.dt.tz_localize(None).dt.normalize()
    tables["weather_grid_hour"] = weather.sort_values([*GRID, "hour_start"])
    tables["weather_observed_daily"] = weather.groupby([*GRID, "date_local"], as_index=False).agg(
        observed_hours=("hour_start", "size"), rain_mm_observed_hours=("rain_mm_hour", "sum"),
        wet_hours=("rain_mm_hour", lambda s: s.gt(0).sum()), mean_temperature_c=("temperature_2m_c", "mean"))
    tables["rain"] = tables["rain"].merge(
        weather.groupby("rain_group").size().rename("grid_hours").reset_index(),
        on="rain_group", validate="one_to_one")
    tables["rain_matched"] = matched_comparison(images, [CAMERA, "hour_local", "day_type"], "rain_group", "dry", "wet")
    tables["weekend_matched"] = matched_comparison(images, [CAMERA, "hour_local"], "day_type", "weekday", "weekend")
    tables["bem_monthly"] = bem_months(images)
    tables["class_composition"] = pd.DataFrame({
        "object_class": CLASSES, "detections": [int(images[c].sum()) for c in CLASSES],
        "mean_per_image": [float(images[c].mean()) for c in CLASSES],
        "included_in_motor_total": [c in MOTOR_CLASSES for c in CLASSES],
    })
    tables.update(coverage_tables(images, manifest))
    # Include inventory hours with no counts, preserving unknown means as NaN.
    tables["coverage_camera_hour"] = tables["coverage_camera_hour"].merge(
        tables["camera_hour"][[CAMERA, "date_local", "hour_local", "mean_motor_vehicles"]],
        on=[CAMERA, "date_local", "hour_local"], how="left", validate="one_to_one")
    return tables


FIELD_NOTES = {
    "camera_id": "รหัสกล้องจากไฟล์ต้นทาง (เก็บเป็นข้อความ)",
    "ts_filename": "เวลาภาพพร้อม timezone; วิเคราะห์ตาม Asia/Bangkok",
    "filename": "ชื่อไฟล์ภาพ; ใช้ร่วมกับ camera_id และ ts_filename เป็นคีย์",
    "day_night": "ป้าย day/night จากต้นทาง; pipeline ใช้ความสว่าง ไม่ใช่เวลานาฬิกา",
    "vehicles_total": "สูตรเดิม car + motor เท่านั้น; รายงานใหม่ใช้ motor_vehicles_total",
    "motor_share": "สูตรเดิม motor / (car + motor); เป็น null เมื่อส่วนหารเป็นศูนย์",
    "date_local": "วันที่ไทยสำหรับเชื่อมปฏิทินและ aggregate รายวัน",
    "hour_local": "ชั่วโมงไทย 0–23", "hour_start": "เวลาต้นชั่วโมงไทย",
    "rain_interval_end": "hour_start + 1 ชั่วโมง; ฝนที่ประทับเวลาปลายช่วงสะสมหนึ่งชั่วโมง",
    "day_of_week": "ชื่อวันจากปฏิทินที่เก็บไว้", "is_weekend": "เสาร์หรืออาทิตย์",
    "is_holiday": "วันหยุดตามไฟล์ปฏิทิน; สถานะตรวจยืนยันทางการยัง incomplete",
    "day_type": "public_holiday ก่อน weekend และ weekday ตามลำดับ",
    "location": "ชื่อจุดกล้องจาก cctv_info.csv", "location_detail": "รายละเอียดตำแหน่งกล้อง",
    "latitude": "ละติจูดกล้อง (องศา)", "longitude": "ลองจิจูดกล้อง (องศา)",
    "osm_ways_300m": "จำนวนชิ้นส่วนทาง OSM ในบริเวณ 300 เมตร ไม่ใช่จำนวนถนนเต็มสาย",
    "osm_motor_ways_300m": "จำนวนชิ้นส่วนทางสำหรับยานยนต์ในบริเวณ 300 เมตร",
    "road_class_300m": "ลำดับชั้นถนนสูงสุดที่พบรอบกล้อง ไม่ยืนยันว่ากล้องหันไปยังถนนนั้น",
    "grid_latitude": "ละติจูดกริด ERA5 (องศา)", "grid_longitude": "ลองจิจูดกริด ERA5 (องศา)",
    "temperature_2m_c": "อุณหภูมิ ณ ต้นชั่วโมง (องศาเซลเซียส)",
    "relative_humidity_2m_percent": "ความชื้นสัมพัทธ์ ณ ต้นชั่วโมง (%)",
    "cloud_cover_percent": "เมฆปกคลุม ณ ต้นชั่วโมง (%)", "wind_speed_10m_kmh": "ลม ณ ต้นชั่วโมง (กม./ชม.)",
    "rain_mm_hour": "หยาดน้ำฟ้ารวมในช่วงชั่วโมงของภาพ (มม.); เป็นค่ากริด ไม่ใช่เซนเซอร์กล้อง",
    "rain_group": "wet เมื่อ rain_mm_hour > 0 มิฉะนั้น dry",
    "mot_reports_300m_hour": "จำนวนรายงาน MOT ที่มีพิกัดห่างไม่เกิน 300 ม. ในชั่วโมงภาพ; ห้ามบวกซ้ำข้ามภาพ",
    "motor_vehicles_total": "car + motor + truck + bus (การตรวจพบต่อภาพ)",
    "vehicles_including_bicycles": "motor_vehicles_total + bicycle",
    "heavy_vehicles_total": "truck + bus", "objects_total": "ผลรวมการตรวจพบทั้ง 6 ประเภท รวม person",
    "motor_share_all": "motor / motor_vehicles_total; null เมื่อไม่มีรถ",
    "heavy_vehicle_share": "(truck + bus) / motor_vehicles_total; null เมื่อไม่มีรถ",
    "has_motor_vehicle": "มีการตรวจพบรถอย่างน้อย 1 คันในภาพ",
    "weekday_number": "จันทร์ = 0 ถึงอาทิตย์ = 6",
    "time_period": "ช่วงวิเคราะห์ 06–ก่อน09, 09–ก่อน16, 16–ก่อน19, 19–ก่อน22, ที่เหลือ 22–ก่อน06",
    "is_commute_window": "ชั่วโมง 06–08 หรือ 16–18; ช่วงที่กำหนดเพื่อวิเคราะห์ ไม่ใช่หลักฐานว่ารถติด",
    "clock_day_night": "day เมื่อ 06:00–ก่อน18:00; night ช่วงอื่น เป็นเกณฑ์นาฬิกา ไม่ใช่เวลาพระอาทิตย์",
    "brightness_clock_disagree": "ป้ายต้นทางต่างจากเกณฑ์นาฬิกา; ไม่ได้หมายความว่าป้ายต้นทางผิด",
    "rain_band": "ช่วงฝน 0, มากกว่า0แต่น้อยกว่า1, 1ถึงก่อน5, ตั้งแต่5 มม./ชม.; ช่วงวิเคราะห์ที่กำหนดเอง",
    "camera_mean_motor_vehicles": "ค่าเฉลี่ยรถของกล้องเดียวกันตลอดชุดข้อมูลนี้",
    "relative_to_camera_mean": "รถในภาพ / ค่าเฉลี่ยของกล้อง; >1 คือมากกว่าค่าเฉลี่ยกล้อง; ใช้พรรณนาย้อนหลัง",
}
FIELD_NOTES.update({c: f"จำนวน {c} ที่ตรวจพบต่อภาพตามคอลัมน์ต้นทาง" for c in CLASSES})


def data_dictionary(images, source_columns):
    if set(images) != set(FIELD_NOTES):
        raise ValueError(f"Dictionary/schema mismatch: {set(images) ^ set(FIELD_NOTES)}")
    return pd.DataFrame([{"column": c, "dtype": str(images[c].dtype),
                          "source_column": c in source_columns, "description_th": FIELD_NOTES[c],
                          "null_rows": int(images[c].isna().sum())} for c in images])


def findings_text(images, tables, quality):
    daily = tables["daily"]
    day = tables["day_type"].set_index("day_type")
    rain = tables["rain"].set_index("rain_group")
    lines = [
        "ผลวิเคราะห์ CCTV และข้อมูลที่เก็บจากเว็บไซต์",
        f"ข้อมูล {len(images):,} ภาพ จาก {images.camera_id.nunique()} กล้อง ระหว่าง {images.date_local.min():%Y-%m-%d} ถึง {images.date_local.max():%Y-%m-%d}",
        "หน่วยหลัก: การตรวจพบรถต่อภาพ โดยรถ = car + motor + truck + bus",
        f"ค่าเฉลี่ยรวม {images.motor_vehicles_total.mean():.3f} คัน/ภาพ; motorcycle share = {images.motor.sum()/images.motor_vehicles_total.sum():.2%}" if images.motor_vehicles_total.sum() else "ไม่พบรถในชุดข้อมูลนี้; สัดส่วน motorcycle ไม่สามารถคำนวณได้",
        "",
        "1. วันและช่วงเวลา",
    ]
    if {"weekday", "weekend"}.issubset(day.index):
        a, b = day.loc["weekday", "mean_motor_vehicles"], day.loc["weekend", "mean_motor_vehicles"]
        relative = f" ({(b/a-1)*100:+.2f}%)" if a else " (ฐานเป็นศูนย์ จึงคำนวณ % ไม่ได้)"
        lines.append(f"วันธรรมดา {a:.3f} คัน/ภาพ เทียบเสาร์–อาทิตย์ {b:.3f} คัน/ภาพ{relative}")
    for kind, group in tables["hourly"].groupby("day_type"):
        peak = group.loc[group.mean_motor_vehicles.idxmax()]
        lines.append(f"{kind}: ชั่วโมงที่เฉลี่ยสูงสุด {int(peak.hour_local):02d}:00–{int(peak.hour_local)+1:02d}:00 = {peak.mean_motor_vehicles:.3f} คัน/ภาพ จาก {int(peak.images):,} ภาพ")
    lines.append("")
    lines.append("2. ฝนและการเปรียบเทียบกลุ่มที่ใกล้เคียงกัน")
    if {"dry", "wet"}.issubset(rain.index):
        lines.append(f"รวมทุกภาพ: dry {rain.loc['dry', 'mean_motor_vehicles']:.3f} เทียบ wet {rain.loc['wet', 'mean_motor_vehicles']:.3f} คัน/ภาพ")
    for name, label in [("rain_matched", "ฝน: กล้อง/ชั่วโมง/ประเภทวันเดียวกัน (wet - dry)"),
                        ("weekend_matched", "ประเภทวัน: กล้อง/ชั่วโมงเดียวกัน (weekend - weekday)")]:
        pairs = tables[name]
        if pairs.empty:
            lines.append(f"{label}: ไม่มีกลุ่มที่มีข้อมูลเพียงพอ")
        else:
            lines.append(f"{label}: ผลต่างเฉลี่ย {pairs.comparison_minus_baseline.mean():+.3f} คัน/ภาพ จาก {len(pairs):,} คู่กลุ่ม; ให้น้ำหนักคู่เท่ากันและอย่างน้อย {MIN_PAIR_IMAGES} ภาพต่อด้าน")
    lines += ["ผลนี้เป็นความสัมพันธ์เชิงพรรณนา ไม่ใช่ข้อพิสูจน์ว่าฝนทำให้รถเพิ่มหรือลด", "", "3. พื้นที่และประเภทวัตถุ"]
    for row in tables["cameras"].head(3).itertuples():
        lines.append(f"กล้อง {row.camera_id} ({row.location}): {row.mean_motor_vehicles:.3f} คัน/ภาพ จาก {row.images:,} ภาพ")
    extra = int(images.truck.sum() + images.bus.sum())
    lines += [f"truck + bus มีการตรวจพบรวม {extra:,} ครั้ง ซึ่งไม่รวมอยู่ในสูตรเดิม car + motor",
              "มุมกล้อง จำนวนช่องทาง และขอบเขตภาพต่างกัน จึงไม่ใช้การจัดอันดับนี้เป็นอันดับความรถติด", "", "4. ความครอบคลุมและข้อมูลภายนอก",
              f"จับคู่กับ Bronze ได้ {quality['rows']:,}/{quality['inventory_rows']:,} ภาพ ({quality['inventory_coverage']:.2%}); ไม่มีผลนับ {quality['missing_counts']:,} ภาพ โดยไฟล์นี้ไม่ระบุสาเหตุ",
              f"มี weather grid {quality['weather_grids']} กริด; จำนวนกล้องต่อกริด: {quality['cameras_per_weather_grid']}",
              f"มี {quality['low_sample_camera_hours']:,} กล้อง-วัน-ชั่วโมงที่มีผลนับน้อยกว่า 3 ภาพจากรายการภาพที่เก็บได้",
              f"เกณฑ์เวลา 06:00–ก่อน18:00 ต่างจาก day_night ต้นทาง {quality['brightness_clock_disagreements']:,} ภาพ; ใช้ hour_local วิเคราะห์เวลา",
              f"ช่วงนี้มีรายงาน MOT ในกรุงเทพฯ {int(daily.mot_reports_bkk.sum())} รายงาน โดยมีพิกัดใน 300 ม. รอบกล้อง {int(daily.mot_within_300m.sum())} รายงาน (ขอบเขตเฉพาะแหล่ง MOT)"]
    for row in tables["bem_monthly"].itertuples():
        if pd.notna(row.bem_thousand_trips_per_day):
            lines.append(f"BEM {row.month_start:%Y-%m}: {row.bem_thousand_trips_per_day:,.1f} พันเที่ยว/วัน; MoM {row.mom_change:+.2%}, YoY {row.yoy_change:+.2%}; ใช้เป็นบริบทรายเดือน ไม่ใช่การตรวจสอบยอด CCTV")
    lines += ["", "วิธีอ่านและข้อจำกัด",
              "- ภาพซ้ำช่วงเวลาอาจเห็นรถคันเดิม ผลรวม detection จึงไม่ใช่จำนวนรถผ่านหรือรถไม่ซ้ำ",
              "- Coverage เทียบกับรายการที่ scrape เก็บได้ ไม่ใช่ความครบถ้วนตลอด 24 ชั่วโมง; ชั่วโมง 21 มีข้อมูลเพียงบางส่วน",
              "- ค่าเฉลี่ยรวมถ่วงด้วยจำนวนภาพ ส่วน matched comparison ให้น้ำหนักคู่กลุ่มเท่ากัน",
              "- rain_mm_observed_hours รวมฝนครั้งเดียวต่อกริด-ชั่วโมงที่มีภาพ ไม่ใช่ยอดฝน 24 ชั่วโมง",
              "- ชุดข้อมูลมีหนึ่งสัปดาห์ จึงยังสรุปแนวโน้มระยะยาวหรือเหตุและผลไม่ได้",
              "- ไฟล์ต้นทางไม่มี metadata ยืนยันรุ่นโมเดล/threshold จึงไม่อ้างความแม่นยำหรือเทียบกับชุดเดิมว่าโมเดลดีขึ้น",
              "- OSM เป็นบริบทรอบกล้องจาก snapshot 2026-11-22; ปฏิทินใน repo ระบุ official_verification_status=incomplete",
              "- relative_to_camera_mean ใช้ค่าเฉลี่ยทั้งชุด เหมาะกับ EDA ย้อนหลัง; ห้ามใช้ตรง ๆ เป็น feature ทำนายโดยไม่แยก train/test",
              "", "แหล่งข้อมูลและการจับคู่",
              "CCTV: input_image_counts.parquet; กล้อง: data/cctv_info.csv; รายการภาพ: data/bronze_data/manifest.parquet",
              "อากาศ: data/external_data/processed/weather/weather_clean.parquet + camera_weather_grid_mapping.csv",
              "https://open-meteo.com/en/docs/historical-weather-api (ตรวจนิยาม 2026-11-28): precipitation เป็นผลรวมชั่วโมงก่อนเวลาที่ประทับ จึงใช้ภาพ 07:35 กับฝนที่ 08:00",
              "ปฏิทิน: data/external_data/processed/holidays/thailand_calendar_2026.parquet (join วันที่ไทย)",
              "OSM: data/external_data/processed/osm/camera_osm_ways.parquet (aggregate ก่อน join camera_id)",
              "MOT: data/external_data/processed/accidents/bangkok_accidents_2026.parquet (วัน/เวลาและระยะ Haversine 300 ม.)",
              "BEM: data/external_data/processed/bem/bem_monthly_traffic.parquet (แยกตารางรายเดือน)",
              "ไฟล์ที่ส่งออกเป็น snapshot; รันสคริปต์ซ้ำเพื่อคำนวณจากข้อมูลใหม่"]
    return "\n".join(lines) + "\n"


def plot_overview(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for day, group in tables["hourly"].groupby("day_type"):
        axes[0, 0].plot(group.hour_local, group.mean_motor_vehicles, marker="o", label=day)
    axes[0, 0].set(title="Hourly pattern by day type", xlabel="Hour (Asia/Bangkok)", ylabel="Motor vehicle detections / image")
    axes[0, 0].legend()
    day = tables["daily"]
    axes[0, 1].bar(day.date_local.dt.strftime("%d %b"), day.mean_motor_vehicles, color="#20888a")
    axes[0, 1].set(title="Daily means (observed images)", ylabel="Motor vehicle detections / image")
    classes = tables["class_composition"]
    axes[1, 0].bar(classes.object_class, classes.mean_per_image, color="#446c9b")
    axes[1, 0].set(title="Six detected classes", ylabel="Detections / image")
    cameras = tables["cameras"].head(10).iloc[::-1]
    axes[1, 1].barh(cameras.camera_id, cameras.mean_motor_vehicles, color="#d28b35")
    axes[1, 1].set(title="Top cameras by mean detections", xlabel="Motor vehicle detections / image", ylabel="Camera ID")
    period = f"{day.date_local.min():%d %b %Y} to {day.date_local.max():%d %b %Y}"
    fig.suptitle(f"Bangkok CCTV | {period}\nMotor vehicles = car + motor + truck + bus", fontsize=15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_analysis(counts_path, out=DEFAULT_OUT):
    counts_path, out = Path(counts_path).resolve(), Path(out).resolve()
    source = pd.read_parquet(counts_path)
    validate_counts(source)
    images, daily_mot = enrich(source)
    images = add_features(images)
    pd.testing.assert_frame_equal(images[source.columns], source.reset_index(drop=True), check_exact=True)
    manifest_path = DATA / "bronze_data/manifest.parquet"
    manifest = pd.read_parquet(manifest_path, columns=IMAGE_KEY)
    tables = build_insight_tables(images, daily_mot, manifest)
    dictionary = data_dictionary(images, source.columns)
    coverage = tables["coverage_daily"]
    source_hash = hashlib.sha256(counts_path.read_bytes()).hexdigest()
    grids = images[[CAMERA, *GRID]].drop_duplicates().groupby(GRID).size()
    quality = {
        "input_filename": counts_path.name, "input_sha256": source_hash,
        "rows": len(images), "source_columns": len(source.columns), "enriched_columns": len(images.columns),
        "cameras": int(images.camera_id.nunique()), "dates": int(images.date_local.nunique()),
        "inventory_rows": int(coverage.inventory_images.sum()), "missing_counts": int(coverage.missing_counts.sum()),
        "inventory_coverage": float(coverage.analyzed_images.sum()/coverage.inventory_images.sum()),
        "duplicate_image_keys": int(images.duplicated(IMAGE_KEY).sum()),
        "source_null_cells": int(source.isna().sum().sum()), "original_columns_unchanged": True,
        "aggregate_totals_reconciled": True, "weather_grids": len(grids),
        "cameras_per_weather_grid": {str(k): int(v) for k, v in grids.items()},
        "low_sample_camera_hours": int(tables["coverage_camera_hour"].analyzed_images.lt(3).sum()),
        "brightness_clock_disagreements": int(images.brightness_clock_disagree.sum()),
        "model_settings": "not embedded in supplied file",
        "motor_vehicle_definition": "car + motor + truck + bus",
        "source_hashes": {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [manifest_path, DATA / "cctv_info.csv",
                                    EXTERNAL / "weather/camera_weather_grid_mapping.csv", *sorted(EXTERNAL.rglob("*.parquet"))]},
    }
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "input_image_counts.parquet"
    if counts_path != snapshot:
        shutil.copy2(counts_path, snapshot)
    arrow = pa.Table.from_pandas(images, preserve_index=False)
    metadata = {**(arrow.schema.metadata or {}), b"counts_sha256": source_hash.encode(),
                b"grain": b"one source CCTV image", b"motor_vehicle_definition": b"car + motor + truck + bus"}
    pq.write_table(arrow.replace_schema_metadata(metadata), out / "image_counts_enriched.parquet", compression="zstd")
    pd.testing.assert_frame_equal(pd.read_parquet(out / "image_counts_enriched.parquet"), images, check_exact=True)
    for name, table in tables.items():
        table.to_csv(out / f"agg_{name}.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(out / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    (out / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "insights_th.txt").write_text(findings_text(images, tables, quality), encoding="utf-8")
    plot_overview(tables, out / "overview.png")
    return images, tables, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    images, tables, _ = run_analysis(args.counts, args.out)
    print(f"{len(images):,} images; {len(images.columns)} columns; {len(tables)} aggregate tables -> {args.out}")


if __name__ == "__main__":
    main()
