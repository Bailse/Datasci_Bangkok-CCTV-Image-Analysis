# CCTV insights จากข้อมูลที่ scrape เก็บไว้

เปิด `Test_parquet.ipynb` แล้วเลือก kernel **dsde** และ **Run All** เพื่อดูข้อมูลที่เพิ่มคอลัมน์ ตาราง aggregate กราฟ และ insight ภาษาไทย

ชุดที่วิเคราะห์รอบนี้คือไฟล์ `image_counts.parquet` ที่ให้มาเมื่อ 28 พ.ย. 2026: 60,230 ภาพ, 100 กล้อง, 25–31 ส.ค. 2026, 10 คอลัมน์ต้นทาง ส่วนผลลัพธ์ใหม่มี 51 คอลัมน์ โดยเก็บคอลัมน์เดิมและค่าเดิมครบทั้งหมด

## รันใหม่

รันใน PowerShell ที่ root ของโปรเจกต์ ต้องมี `pandas`, `numpy`, `pyarrow`, `matplotlib` และไฟล์ใน `data/external_data/processed`, `data/cctv_info.csv`, `data/bronze_data/manifest.parquet` ส่วน Notebook ต้องมี IPython/ipykernel

```powershell
& "C:\Users\Victus\miniconda3\envs\dsde\python.exe" DataSci\analysis\scraped_insights.py --counts "C:\Users\Victus\Downloads\image_counts.parquet"
```

หรือใช้สำเนาต้นทางที่อยู่ในโฟลเดอร์ผลลัพธ์ได้:

```powershell
& "C:\Users\Victus\miniconda3\envs\dsde\python.exe" DataSci\analysis\scraped_insights.py --counts DataSci\analysis\outputs\cctv_20261128\input_image_counts.parquet
```

ใช้ `--out <folder>` ถ้าต้องการเก็บรอบใหม่แยกโฟลเดอร์ สคริปต์ตรวจ schema แบบ 6 ประเภทวัตถุและตรวจการเชื่อมข้อมูล ถ้าคอลัมน์หรือข้อมูลภายนอกไม่ครอบคลุมจะหยุดพร้อมสาเหตุ

## ไฟล์ผลลัพธ์

อยู่ใน `outputs/cctv_20261128/`

| ไฟล์ | ใช้ทำอะไร |
| --- | --- |
| `image_counts_enriched.parquet` | ข้อมูลระดับภาพ 51 คอลัมน์ สำหรับวิเคราะห์ต่อ |
| `input_image_counts.parquet` | สำเนาต้นทางที่ไม่เปลี่ยนค่า |
| `data_dictionary.csv` | สูตร ความหมาย ชนิดข้อมูล และจำนวน null ของทุกคอลัมน์ |
| `agg_daily.csv`, `agg_day_type.csv`, `agg_hourly.csv`, `agg_time_period.csv` | เทียบวันและช่วงเวลา |
| `agg_cameras.csv`, `agg_camera_day.csv`, `agg_camera_hour.csv` | เปรียบเทียบกล้องและดูรายละเอียดแต่ละวัน/ชั่วโมง |
| `agg_rain.csv`, `agg_rain_band.csv`, `agg_roads.csv` | วิเคราะห์สภาพฝนและถนนรอบกล้อง |
| `agg_rain_matched.csv`, `agg_weekend_matched.csv` | เปรียบเทียบกลุ่มเดียวกัน โดยอย่างน้อย 3 ภาพต่อด้าน |
| `agg_coverage_daily.csv`, `agg_coverage_camera.csv`, `agg_coverage_camera_hour.csv` | ผลนับที่มี/ขาดเทียบรายการภาพ Bronze |
| `agg_weather_grid_hour.csv`, `agg_weather_observed_daily.csv` | สรุปอากาศโดยไม่บวกฝนซ้ำทุกภาพ |
| `agg_class_composition.csv`, `agg_bem_monthly.csv` | ประเภทวัตถุ และบริบทการเดินทาง BEM รายเดือน |
| `insights_th.txt`, `overview.png` | ข้อค้นพบภาษาไทย และกราฟสรุป |
| `quality_report.json` | จำนวนแถว การตรวจยอดรวม และ SHA-256 ของข้อมูลต้นทาง |

## คอลัมน์ที่ช่วยวิเคราะห์

- `motor_vehicles_total = car + motor + truck + bus`: รวมรถเครื่องยนต์ 4 ประเภท สูตรเดิม `vehicles_total` ยังคงหมายถึง `car + motor` เพื่อไม่เปลี่ยนความหมายของโค้ดเดิม
- `motor_share_all`, `heavy_vehicle_share`: สัดส่วนรถจักรยานยนต์ และรถบรรทุก/รถเมล์ ถ้าไม่มีรถจะเป็น null เพราะไม่มีตัวหาร
- `time_period`, `is_commute_window`, `weekday_number`, `clock_day_night`: แบ่งช่วงเวลาโดยเวลาไทย ขอบเขตช่วงระบุในพจนานุกรมข้อมูล
- `rain_group`, `rain_band`: แบ่งตามหยาดน้ำฟ้าในชั่วโมงของภาพ โดยช่วง rain_band เป็นช่วงที่กำหนดเพื่อวิเคราะห์
- `relative_to_camera_mean`: จำนวนรถเทียบค่าเฉลี่ยของกล้องเดียวกัน ใช้อ่าน EDA ย้อนหลัง ไม่ใช่ feature ที่พร้อมใช้ทำนาย
- ชื่อและพิกัดกล้อง ปฏิทิน สภาพอากาศ ถนนรอบกล้อง และรายงานอุบัติเหตุ ใช้ฟังก์ชัน join ใน `analyze_cctv.py` ที่ตรวจ many-to-one และรักษาจำนวนแถว

## การตีความ aggregate

`mean_motor_vehicles = sum(motor_vehicles_total) / จำนวนภาพ` และสัดส่วน aggregate คำนวณจากผลรวมตัวเศษ/ผลรวมตัวส่วน ไม่เฉลี่ยสัดส่วนรายภาพโดยตรง ตารางระบุจำนวนภาพ กล้อง วัน median และ p90 เพื่อประกอบการอ่าน

matched comparison หาค่าเฉลี่ยแยกสภาพในแต่ละกลุ่ม แล้วเฉลี่ยผลต่างโดยให้ทุกกลุ่มน้ำหนักเท่ากัน ฝนจับคู่กล้อง+ชั่วโมง+ประเภทวัน ส่วน weekday/weekend จับคู่กล้อง+ชั่วโมง ทั้งสองวิธีเป็นคำอธิบายความสัมพันธ์ ไม่ใช่ causal analysis

[Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) ระบุ precipitation เป็นผลรวมชั่วโมงก่อนเวลาที่ประทับ ดังนั้นภาพ 07:35 ใช้ฝนที่ 08:00 ส่วนอุณหภูมิใช้ 07:00 (ตรวจนิยาม 28 พ.ย. 2026) การรวมฝนใช้กริด-ชั่วโมงเพียงครั้งเดียว และสรุปเฉพาะชั่วโมงที่มีภาพ ไม่ใช่ฝน 24 ชั่วโมง

รถในหลายภาพอาจเป็นคันเดิม จึงไม่ถือผลรวม detection เป็นจำนวนรถผ่าน ไม่ใช้จัดอันดับความรถติดเพราะมุมกล้องต่างกัน และไม่อ้างความแม่นยำโมเดลจากข้อมูลจำนวนภาพ ป้าย `day_night` มาจากความสว่าง ส่วนชั่วโมงวิเคราะห์มาจาก timestamp

Coverage เทียบกับภาพที่ scrape เก็บไว้เท่านั้น ไม่ได้ตรวจว่าการ scrape ครบตามตารางเวลา ภาพที่ไม่มีผลนับไม่เท่ากับภาพที่ไม่มีรถ สาเหตุที่ขาดต้องตรวจ Silver/log เพิ่มเติม

แหล่งภายนอกและข้อจำกัดเพิ่มเติมอยู่ใน `insights_th.txt` และ `../../DataEngineer/README_external_data.md` ผลลัพธ์ Excel ใน `outputs/cctv_aug2026/` เป็นชุดเดิม ไม่ใช่ผลรอบ 60,230 ภาพนี้

## ตรวจสอบโค้ด

```powershell
& "C:\Users\Victus\miniconda3\envs\dsde\python.exe" -m unittest discover -s DataSci\analysis -p "test_scraped_insights.py" -v
```

ตัวอย่างทดสอบครอบคลุมการรวมประเภทวัตถุ การหารด้วยศูนย์ สัดส่วน aggregate ขอบเขตเวลา/ฝน การจับคู่กลุ่ม ภาพที่ขาด และการเชื่อม lookup ที่ซ้ำหรือไม่ครบ ส่วนการรันจริงตรวจยอดรวมของทุกตารางระดับภาพ และอ่าน Parquet กลับเพื่อตรวจว่าค่าไม่เปลี่ยน
