from io import StringIO
from pathlib import Path
import sys

import pandas as pd
import requests


def collect_bem():
    project_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = project_dir / "data" / "external_data" / "raw" / "bem"
    html_path = output_dir / "traffic_volume.html"
    csv_path = output_dir / "traffic_volume_raw.csv"

    for path in (html_path, csv_path):
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")

    url = "https://investor.bemplc.co.th/th/business-performance/traffic-volume"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    with html_path.open("xb") as file:
        file.write(response.content)

    tables = pd.read_html(
        StringIO(response.text),
        attrs={"class": "table table--white table--statistics table-hover"},
        flavor="lxml",
        converters={column: str for column in range(14)},
        thousands=None,
        keep_default_na=False,
        displayed_only=False,
    )
    if len(tables) != 1 or tables[0].shape[1] != 14:
        raise ValueError("The BEM table structure has changed; inspect the saved HTML")
    traffic = tables[0]
    if traffic.empty:
        raise ValueError("The BEM table contains no records")

    traffic.to_csv(csv_path, index=False, encoding="utf-8", mode="x")

    print(f"HTTP status: {response.status_code}")
    print("Measure: Average Traffic Volume per Day")
    print("Unit: Thousand Trips")
    print(f"Year rows: {len(traffic)}")
    print("Columns:", traffic.columns.tolist())
    print("\nFirst 5 records (year rows):")
    print(traffic.head(5).to_string(index=False))
    print(f"\nSaved original HTML: {html_path}")
    print(f"Saved raw table: {csv_path}")
    return traffic


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        traffic = collect_bem()
    except (requests.RequestException, ValueError, OSError, ImportError) as error:
        raise SystemExit(f"BEM collection failed: {error}")
