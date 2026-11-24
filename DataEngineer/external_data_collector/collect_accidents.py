from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import requests


# The MOT catalog lists this CSV under Buddhist year 2569 (2026).
CSV_URL = (
    "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/"
    "resource/4625b7aa-99a4-4dfb-9a0f-6402814d0f2c/download/accident2026.csv"
)
SAMPLE_COLUMNS = [
    "วันที่เกิดเหตุ", "เวลา", "จังหวัด", "สายทาง", "สภาพอากาศ", "รถคันที่1",
    "LATITUDE", "LONGITUDE", "รวมจำนวนผู้บาดเจ็บ", "จำนวนผู้เสียชีวิต",
]


def inspect_accidents(accidents):
    # The CSV uses day/month/year and local Thai dates.
    dates = pd.to_datetime(accidents["วันที่เกิดเหตุ"], format="%d/%m/%Y", errors="coerce")
    bangkok = accidents["จังหวัด"].eq("กรุงเทพมหานคร")
    observation_week = dates.between("2026-08-25", "2026-08-31")
    coordinates = accidents[["LATITUDE", "LONGITUDE"]]
    missing = accidents.isna()
    missing_by_column = missing.sum()

    print(f"Records: {len(accidents):,}")
    print(f"Accident date coverage: {dates.min().date()} to {dates.max().date()}")
    print(f"Missing or unparseable accident dates: {dates.isna().sum():,}")
    print(f"Bangkok records: {bangkok.sum():,}")
    print(f"August 25–31, 2026 records (Asia/Bangkok): {observation_week.sum():,}")
    print(f"Bangkok records in the observation week: {(observation_week & bangkok).sum():,}")
    print("\nCoordinates (non-missing count and raw range):")
    print(coordinates.agg(["count", "min", "max"]).to_string())
    print(f"Records with both coordinates: {coordinates.notna().all(axis=1).sum():,}")
    print(f"Records with latitude and longitude both zero: {coordinates.eq(0).all(axis=1).sum():,}")
    print(f"\nMissing cells (Pandas NA): {missing_by_column.sum():,}")
    print(f"Rows with missing cells: {missing.any(axis=1).sum():,}")
    print(f"Exact duplicate rows after the first occurrence: {accidents.duplicated().sum():,}")
    print("\nColumns, Pandas dtypes, and missing counts:")
    print(pd.DataFrame({"dtype": accidents.dtypes, "missing": missing_by_column}).to_string())
    print("\nFirst five records (selected columns):")
    print(accidents[SAMPLE_COLUMNS].head().to_string(index=False))


def collect_accidents():
    project_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = project_dir / "data" / "external_data" / "raw" / "accidents" / "accident2026.csv"
    if csv_path.exists():
        raise FileExistsError(f"File already exists: {csv_path}")

    response = requests.get(CSV_URL, timeout=(10, 60))
    response.raise_for_status()
    raw_csv = response.content
    accidents = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
    missing_columns = set(SAMPLE_COLUMNS) - set(accidents.columns)
    if accidents.empty or missing_columns:
        raise ValueError(f"Unexpected MOT CSV: {len(accidents)} rows, missing columns {sorted(missing_columns)}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("xb") as file:
        file.write(raw_csv)

    print(f"HTTP status: {response.status_code}")
    print(f"Saved original CSV ({len(raw_csv):,} bytes): {csv_path}")
    inspect_accidents(accidents)
    return accidents


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        collect_accidents()
    except (requests.RequestException, ValueError, OSError) as error:
        raise SystemExit(f"MOT accident collection failed: {error}")
