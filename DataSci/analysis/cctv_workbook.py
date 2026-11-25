"""Excel views for the Bangkok CCTV analysis. All detection rates use images as the denominator."""

from pathlib import Path

import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell


DAY = {"weekday": "วันธรรมดา", "weekend": "เสาร์–อาทิตย์", "public_holiday": "วันหยุดนักขัตฤกษ์"}
RAIN = {"dry": "ไม่มีฝน", "wet": "มีฝน"}
INK = "#193047"
BLUE = "#20506E"
TEAL = "#087E79"
AMBER = "#BD6B21"
START = 5  # Zero-based header row. All data tables begin in column B.

# Column caption and width in characters. The accompanying Parquet keeps the English field names.
COL = {
    "date_local": ("วันที่ (เวลาไทย)", 18), "month_start": ("เดือน", 18), "day_type": ("ประเภทวัน", 18),
    "hour_local": ("ชั่วโมงไทย", 15), "rain_group": ("ฝนของกริด", 18), "grid_hours": ("จำนวนกริด–ชั่วโมง", 22),
    "road_class_300m": ("ชั้นถนนสูงสุด\nในระยะ 300 ม.", 22), "camera_id": ("รหัสกล้อง", 15), "location": ("ตำแหน่งตามทะเบียนกล้อง", 36),
    "images": ("จำนวนภาพ", 15), "cameras": ("จำนวนกล้อง", 15), "dates": ("จำนวนวัน", 14),
    "cars": ("รถยนต์\nรวมทุกภาพ", 16), "motors": ("จักรยานยนต์\nรวมทุกภาพ", 17),
    "cars_per_image": ("รถยนต์\nคัน/ภาพ", 15), "motors_per_image": ("จักรยานยนต์\nคัน/ภาพ", 17),
    "vehicles_per_image": ("สองประเภท\nคัน/ภาพ", 16), "motor_share": ("สัดส่วน\nจักรยานยนต์", 16),
    "mot_reports_bkk": ("รายงาน MOT\nกรุงเทพฯ", 17), "mot_geocoded": ("รายงานที่\nมีพิกัด", 16), "mot_within_300m": ("รายงานใกล้กล้อง\nภายใน 300 ม.", 21),
    "bem_thousand_trips_per_day": ("เดือนที่วิเคราะห์", 20), "previous_month": ("เดือนก่อน", 18), "previous_year": ("เดือนเดียวกันปีก่อน", 23),
    "mom_change": ("เทียบเดือนก่อน", 20), "yoy_change": ("เทียบปีก่อน", 18),
    "dry_images": ("ภาพช่วงไม่มีฝน", 20), "wet_images": ("ภาพช่วงมีฝน", 19), "dry_vehicles_per_image": ("ไม่มีฝน\nคัน/ภาพ", 17),
    "wet_vehicles_per_image": ("มีฝน\nคัน/ภาพ", 17), "wet_minus_dry": ("มีฝน ลบ ไม่มีฝน\nคัน/ภาพ", 22),
    "pairs": ("จำนวนกลุ่มที่จับคู่ได้", 24), "comparison": ("กลุ่มเปรียบเทียบ", 23),
}
RATES = ["cars_per_image", "motors_per_image", "vehicles_per_image", "motor_share"]


def styles(book):
    common = {"font_name": "Tahoma", "font_size": 10, "font_color": INK, "valign": "vcenter"}
    def fmt(**kw):
        return book.add_format({**common, **kw})
    return {
        "title": fmt(font_size=16, bold=True), "note": fmt(font_color="#4C6373", italic=True),
        "section": fmt(font_size=11, bold=True, font_color=BLUE, bottom=1, bottom_color="#9CB6C0"),
        "header": fmt(bg_color="#DCE9EF", font_color=INK, bold=True, text_wrap=True, align="center", bottom=1, bottom_color="#9CB6C0"),
        "text": fmt(align="left", indent=1), "wrap": fmt(text_wrap=True), "label": fmt(bold=True, font_color=TEAL),
        "int": fmt(num_format="#,##0"), "decimal": fmt(num_format="#,##0.00"), "percent": fmt(num_format="0.0%"),
        "change": fmt(num_format="+0.00%;-0.00%;0.00%"), "delta": fmt(num_format="+0.00;-0.00;0.00"),
        "date": fmt(num_format="dd/mm/yyyy"), "month": fmt(num_format="mm/yyyy"),
    }


def sheet_top(sheet, style, title, note, last_column):
    sheet.hide_gridlines(2)
    sheet.set_default_row(21)
    sheet.set_column(0, 0, 3)
    sheet.set_row(1, 30)
    sheet.merge_range(1, 1, 1, last_column, title, style["title"])
    sheet.merge_range(2, 1, 2, last_column, note, style["note"])
    sheet.set_zoom(90)
    sheet.set_landscape()
    sheet.fit_to_pages(1, 0)


def cell_format(key, style):
    if key == "date_local":
        return style["date"]
    if key == "month_start":
        return style["month"]
    if key == "motor_share":
        return style["percent"]
    if key in ("mom_change", "yoy_change"):
        return style["change"]
    if key == "wet_minus_dry":
        return style["delta"]
    if key.endswith("per_image") or key in ("bem_thousand_trips_per_day", "previous_month", "previous_year"):
        return style["decimal"]
    if key in ("day_type", "rain_group", "road_class_300m", "camera_id", "location", "comparison"):
        return style["text"]
    return style["int"]


def derived_formula(key, cells):
    if key in RATES and "cars" in cells and "motors" in cells:
        car, motor, images = cells["cars"], cells["motors"], cells["images"]
        if key == "motor_share":
            return f'=IF({car}+{motor}=0,"",{motor}/({car}+{motor}))'
        numerator = {"cars_per_image": car, "motors_per_image": motor, "vehicles_per_image": f"({car}+{motor})"}[key]
        return f"={numerator}/{images}"
    if key == "wet_minus_dry":
        return f"={cells['wet_vehicles_per_image']}-{cells['dry_vehicles_per_image']}"
    if key in ("mom_change", "yoy_change"):
        reference = cells["previous_month" if key == "mom_change" else "previous_year"]
        current = cells["bem_thousand_trips_per_day"]
        return f'=IF(OR({current}="",{reference}="",{reference}=0),"",{current}/{reference}-1)'
    return None


def table(sheet, style, data, columns, start=START, widths=True, filters=True):
    formats = [cell_format(key, style) for key in columns]
    for ci, key in enumerate(columns, 1):
        caption, width = COL[key]
        if widths:
            sheet.set_column(ci, ci, width)
        sheet.write_string(start, ci, caption, style["header"])
    sheet.set_row(start, 35)
    for row, record in enumerate(data[columns].itertuples(index=False, name=None), start + 1):
        cells = {key: xl_rowcol_to_cell(row, col) for col, key in enumerate(columns, 1)}
        for col, (key, value, fmt) in enumerate(zip(columns, record, formats), 1):
            formula = derived_formula(key, cells)
            if formula:
                sheet.write_formula(row, col, formula, fmt, "" if pd.isna(value) else float(value))
            elif pd.isna(value):
                sheet.write_blank(row, col, None, fmt)
            elif key == "day_type":
                sheet.write_string(row, col, DAY[value], fmt)
            elif key == "rain_group":
                sheet.write_string(row, col, RAIN[value], fmt)
            else:
                sheet.write(row, col, value, fmt)
    if filters and len(data):
        sheet.autofilter(start, 1, start + len(data), len(columns))


def prep_table(book, style, name, title, note, data, columns, freeze_id=False):
    sheet = book.add_worksheet(name)
    sheet_top(sheet, style, title, note, len(columns))
    table(sheet, style, data, columns)
    sheet.freeze_panes(START + 1, 2 if freeze_id else 0)
    sheet.repeat_rows(START)
    return sheet


def paired_summary(pairs):
    rows = []
    for label, group in [("ทุกประเภทวัน", pairs), *[(DAY[k], v) for k, v in pairs.groupby("day_type")]]:
        rows.append(dict(comparison=label, pairs=len(group), dry_images=group.dry_images.sum(),
                         wet_images=group.wet_images.sum(), dry_vehicles_per_image=group.dry_vehicles_per_image.mean(),
                         wet_vehicles_per_image=group.wet_vehicles_per_image.mean(), wet_minus_dry=group.wet_minus_dry.mean()))
    return pd.DataFrame(rows)


def hourly_chart(book, sheet, data, columns):
    chart = book.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
    for day_type, part in data.groupby("day_type", sort=False):
        start, end = START + 1 + part.index.min(), START + 1 + part.index.max()
        x, y = columns.index("hour_local") + 1, columns.index("vehicles_per_image") + 1
        color = {"weekday": BLUE, "weekend": AMBER, "public_holiday": TEAL}[day_type]
        chart.add_series({"name": DAY[day_type], "categories": ["รายชั่วโมง", start, x, end, x],
                          "values": ["รายชั่วโมง", start, y, end, y], "line": {"color": color, "width": 2.2},
                          "marker": {"type": "circle", "size": 4, "border": {"color": color}, "fill": {"color": color}}})
    font = {"name": "Tahoma", "size": 9, "color": INK}
    chart.set_title({"name": "จำนวนรถเฉลี่ยตามชั่วโมง", "name_font": {**font, "size": 12, "bold": True}})
    chart.set_x_axis({"name": "ชั่วโมง (เวลาไทย)", "min": int(data.hour_local.min()), "max": int(data.hour_local.max()),
                      "major_unit": 3, "num_format": "00", "num_font": font, "name_font": font,
                      "major_gridlines": {"visible": False}})
    chart.set_y_axis({"name": "คันต่อภาพ", "min": 0, "num_format": "0", "num_font": font, "name_font": font,
                      "major_gridlines": {"visible": True, "line": {"color": "#DFE7EB"}}})
    chart.set_legend({"position": "bottom", "font": font})
    chart.set_chartarea({"border": {"none": True}})
    chart.set_size({"width": 530, "height": 249})
    sheet.insert_chart("I5", chart)


FIELDS = [
    ("camera_id", "รหัสกล้อง", "รหัสข้อความ ใช้เชื่อมทะเบียนกล้องและแหล่งข้อมูลเชิงพื้นที่"),
    ("ts_filename", "วันเวลา", "เวลาของภาพพร้อมเขตเวลา Asia/Bangkok"),
    ("filename", "ชื่อไฟล์", "ชื่อภาพ JPEG จากไฟล์ที่ส่งมา"),
    ("car", "คันต่อภาพ", "จำนวนรถยนต์ที่ตรวจพบในภาพ"), ("motor", "คันต่อภาพ", "จำนวนจักรยานยนต์ที่ตรวจพบในภาพ"),
    ("day_night", "ป้ายเดิม", "ป้ายกลางวันหรือกลางคืนของภาพ โค้ด Gold ของโปรเจกต์แบ่งด้วยความสว่าง"),
    ("vehicles_total", "คันต่อภาพ", "รถยนต์บวกจักรยานยนต์ในภาพเดียวกัน"),
    ("motor_share", "สัดส่วน 0–1", "จักรยานยนต์หารด้วยรถทั้งสองประเภท เว้นว่างเมื่อภาพนั้นพบรถ 0 คัน"),
    ("date_local", "วันที่ไทย", "วันที่ตาม ts_filename ใช้เชื่อมปฏิทินและรายงานรายวัน"),
    ("hour_local", "0–23 น.", "ชั่วโมงตามนาฬิกาไทย ใช้เทียบรูปแบบรายชั่วโมง"),
    ("hour_start", "วันเวลาไทย", "ต้นชั่วโมงของภาพ ใช้จับคู่อุณหภูมิ ความชื้น เมฆ และลม"),
    ("rain_interval_end", "วันเวลาไทย", "ปลายชั่วโมง เช่น ภาพ 07:35 ใช้ปริมาณฝนของกริดที่ประทับเวลา 08:00"),
    ("day_of_week", "ชื่อวัน", "ชื่อวันภาษาอังกฤษจากปฏิทิน"), ("is_weekend", "จริง/เท็จ", "วันเสาร์หรือวันอาทิตย์"),
    ("is_holiday", "จริง/เท็จ", "วันหยุดตามปฏิทินไทยในชุดข้อมูล แหล่งข้อมูลระบุว่ายังตรวจสอบกับประกาศทางการไม่ครบ"),
    ("day_type", "ประเภทวัน", "public_holiday ถ้ามีวันหยุดตามปฏิทิน จากนั้น weekend สำหรับเสาร์–อาทิตย์ และ weekday สำหรับวันที่เหลือ"),
    ("location", "ข้อความ", "ชื่อตำแหน่งจากทะเบียนกล้อง เก็บข้อความตามต้นทาง"),
    ("location_detail", "ข้อความ", "รายละเอียดตำแหน่งหรือทิศทางของกล้องตามทะเบียน"),
    ("latitude", "องศา", "ละติจูดของกล้อง"), ("longitude", "องศา", "ลองจิจูดของกล้อง"),
    ("osm_ways_300m", "ชิ้นส่วนทาง", "จำนวน OSM way ID ที่ไม่ซ้ำรอบกล้อง รัศมี 300 เมตร รวมทางเดิน"),
    ("osm_motor_ways_300m", "ชิ้นส่วนทาง", "นับเฉพาะ way ที่มีค่า highway_group เป็น motor_road ถนนหนึ่งสายอาจมีหลาย way"),
    ("road_class_300m", "ชั้นถนน OSM", "เลือกชั้นสูงสุดในรัศมี 300 ม. ตาม motorway, trunk, primary, secondary, tertiary รวมทางเชื่อม _link ไว้กับชั้นแม่"),
    ("grid_latitude", "องศา", "ละติจูดศูนย์กลางกริด ERA5 ที่จับคู่กับกล้อง"), ("grid_longitude", "องศา", "ลองจิจูดศูนย์กลางกริด ERA5 ที่จับคู่กับกล้อง"),
    ("temperature_2m_c", "°C", "อุณหภูมิที่ความสูง 2 เมตร ณ ต้นชั่วโมง"),
    ("relative_humidity_2m_percent", "%", "ความชื้นสัมพัทธ์ที่ความสูง 2 เมตร ณ ต้นชั่วโมง"),
    ("cloud_cover_percent", "%", "เมฆปกคลุม ณ ต้นชั่วโมง"), ("wind_speed_10m_kmh", "กม./ชม.", "ความเร็วลมที่ความสูง 10 เมตร ณ ต้นชั่วโมง"),
    ("rain_mm_hour", "มม. ในชั่วโมง", "ปริมาณหยาดน้ำฟ้าสะสมของกริด ERA5 ตลอดช่วงชั่วโมงของภาพ"),
    ("rain_group", "wet / dry", "wet เมื่อ rain_mm_hour > 0 และ dry เมื่อเท่ากับ 0"),
    ("mot_reports_300m_hour", "รายงาน", "จำนวนรายงาน MOT ที่มีพิกัดในรัศมี 300 ม. และอยู่ในชั่วโมงไทยเดียวกับภาพ รายงานเดียวจับคู่ได้หลายภาพ"),
]


def dictionary(book, style, source_name, source_hash, manifest_matches, images):
    sheet = book.add_worksheet("คำอธิบาย")
    sheet_top(sheet, style, "พจนานุกรมข้อมูลระดับภาพ", "ชื่อคอลัมน์ตรงกับไฟล์ Parquet ที่เพิ่มข้อมูลแล้ว", 3)
    sheet.set_column("B:B", 36)
    sheet.set_column("C:C", 22)
    sheet.set_column("D:D", 101)
    sheet.write_row(START, 1, ["คอลัมน์", "หน่วยหรือรูปแบบ", "ความหมายและวิธีสร้าง"], style["header"])
    if set(images.columns) != {field for field, _, _ in FIELDS}:
        raise ValueError("The column dictionary and enriched Parquet schema differ")
    for row, triple in enumerate(FIELDS, START + 1):
        sheet.write_row(row, 1, triple, style["wrap"])
        sheet.set_row(row, 35 if len(triple[2]) > 90 else 24)
    row = START + len(FIELDS) + 3
    sheet.merge_range(row, 1, row, 3, "แหล่งข้อมูลและการจับคู่", style["section"])
    sources = [
        ("ผลนับรถต่อภาพ", source_name, "คีย์ภาพคือ camera_id, filename และ ts_filename"),
        ("SHA-256 ของไฟล์ต้นทาง", "", source_hash),
        ("รายการภาพ Bronze", "manifest.parquet", f"ตรงกับคีย์ต้นทาง {manifest_matches:,} จาก {len(images):,} ภาพ" if manifest_matches is not None else "ไม่มีไฟล์ manifest ในการรันครั้งนี้"),
        ("ทะเบียนกล้อง", "cctv_info.csv", "เชื่อม camera_id เพื่อเติมชื่อและพิกัด"),
        ("อากาศ Open-Meteo ERA5", "weather_clean.parquet", "จับคู่กริดด้วย camera_weather_grid_mapping.csv แล้วเชื่อมตามเวลาไทย"),
        ("ปฏิทิน", "thailand_calendar_2026.parquet", "เชื่อมวันที่ไทยเพื่อแยกประเภทวัน"),
        ("OpenStreetMap © contributors", "camera_osm_ways.parquet", "รวมชิ้นส่วนทางต่อกล้องก่อนเชื่อม ภาพรวมแผนที่ ณ 22 พ.ย. 2026 ใช้อธิบายพื้นที่รอบกล้อง"),
        ("อุบัติเหตุ MOT", "bangkok_accidents_2026.parquet", "นับรายงานตาม source_row ในกรุงเทพฯ บนโครงข่ายของกระทรวงฯ ตลอดวัน ระยะจากกล้องคำนวณแบบ Haversine"),
        ("ทางพิเศษ BEM", "bem_monthly_traffic.parquet", "ค่าเฉลี่ยรายวันของแต่ละเดือน หน่วยพันเที่ยวต่อวัน"),
    ]
    for r, (name, filename, meaning) in enumerate(sources, row + 1):
        sheet.write(r, 1, name, style["text"])
        sheet.write(r, 2, filename, style["wrap"])
        sheet.write(r, 3, meaning, style["wrap"])
        sheet.set_row(r, 39)
    sheet.write_url(row + len(sources) + 2, 1, "https://open-meteo.com/en/docs/historical-weather-api#hourly_parameter_definition", string="นิยามช่วงเวลาของตัวแปรอากาศ")
    sheet.write_url(row + len(sources) + 3, 1, "https://www.openstreetmap.org/copyright", string="OpenStreetMap © contributors")
    sheet.freeze_panes(START + 1, 0)
    sheet.set_tab_color("#8797A1")


def overview(sheet, style, images, tables, daily_columns):
    dates = images.date_local
    sheet_top(sheet, style, "รถยนต์และจักรยานยนต์ในภาพ CCTV กรุงเทพฯ",
              f"{dates.min():%d/%m/%Y} – {dates.max():%d/%m/%Y}  •  หน่วยวิเคราะห์คือภาพหนึ่งภาพ รถคันเดิมอาจปรากฏหลายครั้ง", 15)
    sheet.set_column("B:B", 23)
    sheet.set_column("C:G", 15)
    sheet.set_column("H:H", 3)
    sheet.set_column("I:P", 10)
    sheet.set_tab_color(BLUE)
    sheet.merge_range("B5:G5", "ภาพที่ใช้วิเคราะห์", style["section"])
    sheet.write_row("B6", ["จำนวนภาพ", "กล้อง", "วัน", "รถยนต์/ภาพ", "จยย./ภาพ", "รวม/ภาพ"], style["header"])
    n_days = len(tables["daily"])
    refs = {}
    for key in ["day_type", "images", "cars", "motors"]:
        col = xl_col_to_name(daily_columns.index(key) + 1)
        refs[key] = f"'รายวัน'!${col}${START + 2}:${col}${START + 1 + n_days}"
    n_cameras = len(tables["cameras"])
    totals = [(f"=SUM({refs['images']})", len(images)), (f"=COUNTA('รายกล้อง'!B{START + 2}:B{START + 1 + n_cameras})", n_cameras),
              (f"=COUNT('รายวัน'!B{START + 2}:B{START + 1 + n_days})", n_days),
              (f"=SUM({refs['cars']})/B7", images.car.mean()), (f"=SUM({refs['motors']})/B7", images.motor.mean()),
              ("=E7+F7", images.vehicles_total.mean())]
    for col, (formula, value) in enumerate(totals, 1):
        sheet.write_formula(6, col, formula, style["int" if col <= 3 else "decimal"], float(value))

    sheet.merge_range("B9:G9", "เฉลี่ยตามประเภทวัน", style["section"])
    sheet.write_row("B10", ["ประเภทวัน", "วัน", "ภาพ", "รถยนต์/ภาพ", "จยย./ภาพ", "รวม/ภาพ"], style["header"])
    for row, item in enumerate(tables["day_type"].itertuples(index=False), 10):
        label, n = f"B{row + 1}", f"D{row + 1}"
        sheet.write_string(row, 1, DAY[item.day_type], style["text"])
        numbers = [(f"=COUNTIFS({refs['day_type']},{label})", item.dates),
                   (f"=SUMIFS({refs['images']},{refs['day_type']},{label})", item.images),
                   (f"=SUMIFS({refs['cars']},{refs['day_type']},{label})/{n}", item.cars_per_image),
                   (f"=SUMIFS({refs['motors']},{refs['day_type']},{label})/{n}", item.motors_per_image),
                   (f"=E{row + 1}+F{row + 1}", item.vehicles_per_image)]
        for col, (formula, value) in enumerate(numbers, 2):
            sheet.write_formula(row, col, formula, style["int" if col <= 3 else "decimal"], float(value))

    notes = []
    hourly = tables["hourly"]
    peaks = hourly.loc[hourly.groupby("day_type").vehicles_per_image.idxmax()]
    peak_text = [f"{DAY[r.day_type]}สูงสุดเวลา {r.hour_local:02d}:00–{r.hour_local:02d}:59 น. เฉลี่ย {r.vehicles_per_image:.2f} คัน/ภาพ ({r.images:,} ภาพ)" for r in peaks.itertuples()]
    notes.append(("ช่วงเวลาที่เห็นรถมาก", " ส่วน".join(peak_text)))
    rain = tables["rain"].set_index("rain_group")
    pairs = tables["pairs"]
    if {"wet", "dry"}.issubset(rain.index) and len(pairs):
        pooled = rain.loc["wet", "vehicles_per_image"] - rain.loc["dry", "vehicles_per_image"]
        notes.append(("ฝนกับจำนวนรถ", f"ภาพในชั่วโมงที่มีฝนต่างจากชั่วโมงที่ไม่มีฝน {pooled:+.2f} คัน/ภาพ เมื่อเทียบกล้อง ชั่วโมง และประเภทวันเดียวกัน {len(pairs):,} กลุ่ม ผลต่างเป็น {pairs.wet_minus_dry.mean():+.2f} คัน/ภาพ"))
    roads = tables["roads"].set_index("road_class_300m")
    if {"motorway", "secondary"}.issubset(roads.index):
        m, s = roads.loc["motorway"], roads.loc["secondary"]
        notes.append(("ถนนรอบกล้อง", f"กล้องที่มี motorway หรือทางเชื่อมในรัศมี 300 ม. ({m.cameras:.0f} กล้อง) เฉลี่ย {m.vehicles_per_image:.2f} คัน/ภาพ กลุ่มที่ชั้นสูงสุดคือ secondary ({s.cameras:.0f} กล้อง) เฉลี่ย {s.vehicles_per_image:.2f} คัน/ภาพ"))
    daily = tables["daily"]
    notes.append(("อุบัติเหตุที่มีรายงาน", f"โครงข่ายกระทรวงคมนาคมในกรุงเทพฯ มี {daily.mot_reports_bkk.sum():,} รายงานในช่วงวันที่วิเคราะห์ มีพิกัด {daily.mot_geocoded.sum():,} รายงาน และอยู่ในรัศมี 300 ม. ของกล้อง {daily.mot_within_300m.sum():,} รายงาน"))
    if len(tables["bem"]) == 1:
        b = tables["bem"].iloc[0]
        if pd.notna(b.bem_thousand_trips_per_day) and pd.notna(b.mom_change):
            notes.append(("ทางพิเศษ BEM", f"เดือน {b.month_start:%m/%Y} ทางพิเศษมีการเดินทางเฉลี่ย {b.bem_thousand_trips_per_day:,.1f} พันเที่ยว/วัน เปลี่ยนแปลง {b.mom_change:+.2%} จากเดือนก่อน ตัวเลขนี้ครอบคลุมทั้งเดือน"))
    morning = images.loc[images.hour_local.between(6, 8)].groupby("date_local").size().reindex(sorted(dates.unique()), fill_value=0)
    if len(morning) > 1 and morning.max() > 0:
        day = morning.idxmin()
        others = morning.drop(day)
        notes.append(("ความครอบคลุมช่วงเช้า", f"เวลา 06:00–08:59 น. วันที่ {day:%d/%m/%Y} มี {morning.min():,} ภาพ เทียบกับ {others.min():,}–{others.max():,} ภาพต่อวันในวันที่เหลือ จำนวนภาพที่บันทึกมีผลต่อการเทียบยอดรวม"))
    sheet.merge_range("B16:P16", "สิ่งที่พบในช่วงที่สังเกต", style["section"])
    for row, (label, note) in enumerate(notes, 16):
        sheet.set_row(row, 36)
        sheet.write_string(row, 1, label, style["label"])
        sheet.merge_range(row, 2, row, 15, note, style["wrap"])


def write_workbook(path: Path, images, tables, source_name, source_hash, manifest_matches):
    options = {"default_date_format": "dd/mm/yyyy", "strings_to_urls": False, "strings_to_formulas": False}
    with xlsxwriter.Workbook(path, options) as book:
        book.set_properties({"title": "รถยนต์และจักรยานยนต์ในภาพ CCTV กรุงเทพฯ", "subject": "ผลวิเคราะห์จากภาพ CCTV และข้อมูลภายนอก"})
        style = styles(book)
        main = book.add_worksheet("ภาพรวม")
        daily_cols = ["date_local", "day_type", "images", "cameras", "cars", "motors", "vehicles_per_image", "motor_share",
                      "mot_reports_bkk", "mot_geocoded", "mot_within_300m"]
        day = prep_table(book, style, "รายวัน", "ผลนับจากกล้องและรายงานอุบัติเหตุรายวัน",
                         "MOT ครอบคลุมรายงานบนโครงข่ายกระทรวงฯ ในกรุงเทพฯ ตลอดวัน นับแต่ละรายงานหนึ่งครั้งแม้อยู่ใกล้หลายกล้อง", tables["daily"], daily_cols)
        bem_at = START + len(tables["daily"]) + 4
        day.merge_range(bem_at, 1, bem_at, 8, "ทางพิเศษ BEM  •  ค่าเฉลี่ยรายวันของแต่ละเดือน (พันเที่ยว/วัน)", style["section"])
        table(day, style, tables["bem"], ["month_start", "bem_thousand_trips_per_day", "previous_month", "previous_year", "mom_change", "yoy_change"],
              start=bem_at + 1, widths=False, filters=False)
        hourly_cols = ["hour_local", "day_type", "images", "dates", "cameras", "cars", "motors", *RATES]
        hourly = tables["hourly"].sort_values(["day_type", "hour_local"]).reset_index(drop=True)
        prep_table(book, style, "รายชั่วโมง", "จำนวนรถเฉลี่ยต่อภาพตามชั่วโมงและประเภทวัน",
                   f"ใช้เวลาไทยจากชื่อภาพ ภาพแรกและสุดท้ายตามนาฬิกาในชุดนี้อยู่ที่ {images.ts_filename.dt.time.min():%H:%M:%S} และ {images.ts_filename.dt.time.max():%H:%M:%S}", hourly, hourly_cols)
        grid_sizes = images.groupby(["grid_latitude", "grid_longitude"]).camera_id.nunique()
        grid_text = " และ ".join(str(n) for n in grid_sizes)
        rain = prep_table(book, style, "ฝน", "จำนวนรถในชั่วโมงที่กริดมีฝนและไม่มีฝน",
                          f"มีฝนเมื่อปริมาณสะสมของกริด ERA5 เกิน 0 มม. กล้องใช้กริดอากาศ {len(grid_sizes)} ช่อง ({grid_text} กล้องตามลำดับ)  •  กริด–ชั่วโมงนับเฉพาะช่วงที่มีภาพ",
                          tables["rain"], ["rain_group", "images", "grid_hours", "cameras", "cars", "motors", *RATES])
        match_at = START + len(tables["rain"]) + 4
        rain.merge_range(match_at, 1, match_at, 10, "เทียบกล้องเดิม ชั่วโมงตามนาฬิกาเดิม และประเภทวันเดียวกัน ข้ามวันที่ต่างกัน", style["section"])
        rain.merge_range(match_at + 1, 1, match_at + 1, 10, "ใช้เฉพาะกลุ่มที่มีทั้งสองสภาพฝน เฉลี่ยจำนวนรถต่อภาพในแต่ละสภาพภายในกลุ่ม แล้วให้น้ำหนักแต่ละกลุ่มเท่ากัน รายละเอียดอยู่ในชีตคู่เทียบฝน", style["note"])
        paired_cols = ["dry_images", "wet_images", "dry_vehicles_per_image", "wet_vehicles_per_image", "wet_minus_dry"]
        table(rain, style, paired_summary(tables["pairs"]), ["comparison", "pairs", *paired_cols], start=match_at + 3, widths=False, filters=False)
        rain.set_column("C:C", 25)
        prep_table(book, style, "ถนนรอบกล้อง", "จำนวนรถตามชั้นถนนสูงสุดที่พบรอบกล้อง",
                   "OpenStreetMap ณ 22 พ.ย. 2026  •  รัศมี 300 เมตร รวมทางเชื่อมในชั้นแม่ ชั้นถนนอธิบายบริเวณโดยรอบ ทิศกล้องและมุมมองอาจต่างกัน",
                   tables["roads"], ["road_class_300m", "cameras", "images", "cars", "motors", *RATES])
        prep_table(book, style, "รายกล้อง", "จำนวนรถเฉลี่ยต่อภาพแยกตามกล้อง",
                   "เรียงจากรถสองประเภทเฉลี่ยต่อภาพสูงสุด ชื่อตำแหน่งใช้ข้อความจากทะเบียนกล้อง",
                   tables["cameras"], ["camera_id", "location", "road_class_300m", "images", "dates", "cars", "motors", *RATES], freeze_id=True)
        prep_table(book, style, "คู่เทียบฝน", "กลุ่มที่มีภาพทั้งช่วงที่กริดมีฝนและไม่มีฝน",
                   "หนึ่งแถวต่อกล้อง × ชั่วโมงตามนาฬิกา × ประเภทวัน ผลต่างมีหน่วยคันต่อภาพ ค่าเฉลี่ยของผลต่างให้น้ำหนักทุกแถวเท่ากัน",
                   tables["pairs"], ["camera_id", "hour_local", "day_type", *paired_cols], freeze_id=True)
        dictionary(book, style, source_name, source_hash, manifest_matches, images)
        overview(main, style, images, tables, daily_cols)
        hourly_chart(book, main, hourly, hourly_cols)
