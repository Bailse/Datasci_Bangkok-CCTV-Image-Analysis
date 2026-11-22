# Bangkok CCTV External Data

The collection scripts cover Open-Meteo weather, Thailand public holidays,
BEM traffic volume, MOT road accidents, and OpenStreetMap road geometry.
They use simple functions, Requests, and Pandas. CCTV images and vehicle
detection are outside this work's scope.

## Folder structure

Paths in this document are relative to the repository root unless noted.

```text
data/
  cctv_info.csv
  external_data/
    raw/
      weather/       # Original weather JSON
      holidays/      # Original holiday JSON
      bem/           # Original HTML and extracted raw table CSV
      accidents/     # Original MOT accident CSV
      osm/           # Original Overpass JSON per camera
    processed/
      weather/       # Hourly table CSV
      holidays/      # 2026 calendar
      bem/           # Monthly and published annual traffic tables
      accidents/     # 2026 Bangkok accidents in CSV and Parquet
      osm/           # Camera–way tables and way geometry
DataEngineer/
  external_data_collector/
    collect_weather.py
    collect_holidays.py
    collect_bem.py
    collect_accidents.py
    collect_osm.py
  requirements.txt
  README_external_data.md
  bronze_data/       # Existing work, unchanged
  clean_process/
  pipeline/          # Existing work, unchanged
```

`data/cctv_info.csv` stays in place for existing notebooks and future camera
integration. Raw responses are preserved as original bytes. Collection does not
fill missing values or change the original dataset structures.

## Run

From the repository root:

```powershell
python -m pip install -r DataEngineer/requirements.txt
python DataEngineer/external_data_collector/collect_weather.py
python DataEngineer/external_data_collector/collect_holidays.py
python DataEngineer/external_data_collector/collect_bem.py
```

Alternatively, from `DataEngineer/`, run
`python external_data_collector/collect_weather.py`,
`python external_data_collector/collect_holidays.py`, or
`python external_data_collector/collect_bem.py`.
Pandas uses `lxml` to read the BEM HTML table and `pyarrow` to read and write
Parquet.

Each script uses `Path(__file__).resolve().parent.parent.parent` to find the repository
root. Output paths therefore work independently of the current working folder.
Existing output files cause a clear error before any network request; this is
expected when rerunning the supplied examples. Nothing is overwritten.

The weather example requests 2026-08-25 through 2026-08-31, inclusive, for Bangkok
latitude 13.7563, longitude 100.5018. This is an example location, not a
verified CCTV camera coordinate. Edit the function call to use another location.

## Open-Meteo: API and function

Documentation: https://open-meteo.com/en/docs/historical-weather-api

The Historical Weather API returns gridded weather estimates using historical
weather datasets. These values are not measurements taken by a CCTV camera.
The free noncommercial endpoint does not require an API key.

`requests.get()` sends an HTTP GET request to
`https://archive-api.open-meteo.com/v1/archive` with these parameters:

| Parameter | Purpose | Example |
| --- | --- | --- |
| latitude, longitude | Required location in decimal degrees | 13.7563, 100.5018 |
| start_date, end_date | Required inclusive date range, YYYY-MM-DD | 2026-08-25, 2026-08-31 |
| hourly | Select the hourly variables to retrieve | temperature_2m,precipitation |
| timezone | Return Bangkok local hours instead of the default GMT | Asia/Bangkok |
| temperature_unit | Explicit temperature unit | celsius |
| wind_speed_unit | Explicit wind speed unit | kmh |
| precipitation_unit | Explicit precipitation unit | mm |

The request has a 30-second timeout. `raise_for_status()` stops on HTTP errors.
`response.json()` reads the response as a Python dictionary. Its `hourly`
object has parallel lists: timestamps and one list per weather variable.
`pd.DataFrame(data["hourly"])` turns these lists into a table.
`weather.to_csv(..., index=False, mode="x")` writes it without a Pandas index
and refuses to overwrite a file.

## Outputs

- `data/external_data/raw/weather/weather_<coordinate>_<dates>.json`: unchanged HTTP response body,
  including returned grid coordinates, timezone, and units.
- `data/external_data/processed/weather/weather_<coordinate>_<dates>.csv`: hourly table extracted
  from the response, with no filling, rounding, or weather-value changes.

CSV columns:

| Column | Meaning / unit |
| --- | --- |
| time | ISO local timestamp in Asia/Bangkok (UTC+07:00) |
| temperature_2m | Air temperature at 2 metres, degrees Celsius |
| relative_humidity_2m | Relative humidity at 2 metres, percent |
| precipitation | Preceding-hour precipitation total, millimetres |
| cloud_cover | Total cloud cover, percent |
| wind_speed_10m | Wind speed at 10 metres, kilometres per hour |

The timestamps contain no explicit UTC offset; interpret them using
Asia/Bangkok when preparing future joins. The API may return grid coordinates
slightly different from the requested point; the original metadata remains
in the raw JSON. The default model selection is Open-Meteo's best match.

Seven full Bangkok days should contain 168 unique, consecutive hourly records.
Check the timestamps and missing values before future integration. A second run
with the same inputs stops because the output files already exist. Nothing is
overwritten. A separate clean_data.py is deferred because this first step only
collects weather and converts its hourly structure to CSV.

Credit Open-Meteo and the underlying weather data providers when using the data;
see the documentation's data sources and attribution guidance.

## Explicit ERA5 collection for the two audited weather grids

The original default weather example above uses Best Match, not explicit ERA5.
Its files are retained separately. The approved two-grid ERA5 collection uses:

```powershell
python DataEngineer/external_data_collector/collect_weather.py --era5-grids
```

The function makes one hourly request per grid, sequentially, with a one-second
gap and no automatic retries. All three output filenames are checked before
either request; existing files are never overwritten. Running this command
again after successful collection stops before making any API requests.

Endpoint: `https://archive-api.open-meteo.com/v1/archive`

| Setting | Value |
| --- | --- |
| models | era5 |
| timezone | Asia/Bangkok (UTC+07:00) |
| cell_selection | land |
| elevation | nan (disable elevation downscaling; use grid-cell height) |
| start_date | 2026-08-25 |
| end_date | 2026-08-31, inclusive |
| hourly | temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m |
| temperature_unit | celsius |
| wind_speed_unit | kmh |
| precipitation_unit | mm |

Percent units for relative humidity and cloud cover are verified from
`hourly_units` in each original response. Timestamps are local Bangkok clock
times without an explicit offset in the raw JSON. Do not add seven hours to
these timestamps during future cleaning; localize them to Asia/Bangkok once.

Collection verified on 2026-11-15 (Bangkok date):

| Grid | Representative camera | Requested latitude | Requested longitude | Returned grid latitude | Returned grid longitude | Returned grid elevation (m) | Assigned cameras | HTTP | Hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 915 | 13.72639 | 100.49213 | 13.75 | 100.50 | -2.0 | 99 | 200 | 168 |
| B | 1804 | 13.71021 | 100.62955 | 13.75 | 100.75 | 2.0 | 1 | 200 | 168 |

The numeric elevation returned in each response is the grid elevation; the
requested setting was `elevation=nan` for both grids. Each response passed checks
for expected grid coordinates, timezone, units, and all 168 consecutive local
hours. Missing-value counts were zero for timestamps and all five variables.

Original response bytes are saved without added metadata under
`data/external_data/raw/weather/`:

- `weather_era5_grid_A_2026-08-25_2026-08-31.json`
- `weather_era5_grid_B_2026-08-25_2026-08-31.json`

The audit-derived mapping is saved as
`data/external_data/processed/weather/camera_weather_grid_mapping.csv` with:

```text
camera_id,camera_latitude,camera_longitude,grid_latitude,grid_longitude
```

It contains all 100 metadata camera IDs exactly once. Camera 1804 maps to Grid B;
the other 99 audited cameras map to Grid A. Original camera coordinates are
copied from `data/cctv_info.csv`; the mapping assignments come from the verified
metadata-only ERA5 spatial audit, not 100 repeated hourly downloads. If the
camera metadata coordinates change, rerun the spatial audit before reusing
these assignments.

The weather cleaning pipeline now reads only these two ERA5 responses. It
concatenates them into `weather_clean.csv` and `weather_clean.parquet` under
`data/external_data/processed/weather/`: 336 rows, 168 per grid. The earlier
Best Match response remains in raw storage for provenance and is excluded from
these canonical weather outputs. Its two obsolete processed CSVs were removed.

Run the cleaner from the repository root:

```powershell
python DataEngineer/clean_process/clean_external_data.py
```

The weather key is `grid_latitude, grid_longitude, time`; timestamps may appear
once per grid. Parquet retains `Asia/Bangkok` timezone information. Both
requested camera coordinates and actual grid coordinates are retained.
The request settings are included as provenance columns; the raw JSON does not
echo the requested model, cell-selection or elevation-downscaling options, so
these come from the documented collection configuration. `elevation_setting`
is written as `nan (no downscaling)` so CSV readers do not mistake it for a
missing value. `grid_elevation_m` holds the numeric response elevation.

The cleaner checks the saved audit mapping against current camera metadata,
including all original coordinates, 100 unique camera IDs, and weather grid
availability. It keeps the mapping CSV unchanged when it matches. Changed
camera locations or assignments require a new spatial audit; they are never
silently reassigned.

Ordinary runs verify matching outputs and refuse to overwrite conflicting
files. To explicitly approve replacing only the canonical weather CSV/Parquet
pair, use `--replace-weather`. This flag never authorizes replacing holiday,
BEM, or conflicting mapping files. Holiday and BEM cleaning logic is unchanged.

## Thailand public holidays

`collect_holidays.py` downloads:
https://raw.githubusercontent.com/ppraserts/thailand-open-data/main/data/thai-public-holidays/2026.json

It saves the original response as
`data/external_data/raw/holidays/thailand_public_holidays_2026.json`.
`pd.DataFrame(data["holidays"])` creates an in-memory table and displays its
columns and first five records. The default cleaner writes the separate
`thailand_calendar_2026.csv` and `.parquet` files.

## BEM traffic volume

`collect_bem.py` downloads:
https://investor.bemplc.co.th/th/business-performance/traffic-volume

It saves these files under `data/external_data/raw/bem/`:

- `traffic_volume.html`: unchanged original response, including headings and units.
- `traffic_volume_raw.csv`: the table extracted with `pd.read_html()`.

The table measures **Average Traffic Volume per Day**, in **Thousand Trips**.
It retains all available year rows, the 12 monthly columns, and the separate
average column. Values remain text to preserve commas, decimal places, and
blank cells. Buddhist Era years and Thai column names remain unchanged.
BEM is a separate monthly dataset. The default cleaner writes monthly and
published annual average tables in CSV and Parquet.

## MOT accidents and OpenStreetMap cleaning

Run from the repository root:

```powershell
python DataEngineer/clean_process/clean_external_data.py --accidents-osm
```

This command reads the saved raw files and writes the accidents and OSM outputs
listed below. Reruns verify matching outputs and stop if an existing file
differs. The console reports column names, dtypes, missing values, duplicate
counts, coverage, and samples. It verifies CSV and PyArrow Parquet values after
reading them back.

### MOT road accidents

The [MOT Road Accident dataset](https://datagov.mot.go.th/th/dataset/roadaccident)
covers motorways and highways, rural roads, and expressways in the Ministry's
road network. The raw file is `raw/accidents/accident2026.csv` relative to
`data/external_data/`. The cleaner writes
`processed/accidents/bangkok_accidents_2026.csv` and `.parquet`.

| Columns | Meaning and units |
| --- | --- |
| `source_row` | One-based record number in this raw CSV, excluding the header |
| `accident_year`, `province` | Source Gregorian year and Thai province name |
| `accident_date_raw`, `accident_time_raw`, `report_date_raw`, `report_time_raw` | Original date and time text |
| `accident_date`, `accident_at` | Local event date and timestamp in Asia/Bangkok. CSV stores the timestamp with its UTC+07:00 offset. Parquet retains the timezone name. |
| `latitude`, `longitude`, `kilometer_marker` | Coordinates in decimal degrees and the reported kilometer marker in kilometers |
| `agency`, `road_network`, `route_code`, `road_name` | Original road authority and route identifiers. Route codes are text. |
| `road_layout`, `other_location_detail`, `suspected_cause`, `accident_type`, `weather` | Original source descriptions |
| `first_vehicle_type`, `vehicles_total`, `participants_total`, vehicle-category columns, `pedestrians` | First vehicle label and source counts by category, including pedestrian counts |
| `fatalities`, `serious_injuries`, `minor_injuries`, `injuries_total` | Source counts of people |

The Bangkok table contains all 2,228 records available in the source for
January 1 through October 31, 2026, including 53 during August 25–31.
Missing coordinates and repeated source records are preserved. `source_row`
identifies a row in this file version. Repeated event timestamps and coordinates
are flagged in validation as possible duplicates. Future spatial matching can
use `latitude` and `longitude` when both exist, alongside `accident_at` for time
alignment.

### OpenStreetMap

The inputs are Overpass JSON files in `raw/osm/camera_<id>.json`, collected
through the [Dresel Asia Overpass mirror](https://ovp.dresel.systems/) using
300-meter camera searches. The OSM snapshot is from November 22, 2026 Bangkok
time. The August 2026 road network may differ.

- `processed/osm/camera_osm_ways.csv` and `.parquet` contain one row per
  `camera_id, osm_id`. `road_name`, `highway`, `lanes`, `oneway`, `maxspeed`,
  and `surface` preserve the source tag text and missing values. `lanes` is the
  tagged lane count. `maxspeed` keeps the tagged speed text, including unit
  suffixes where present.
- `highway_group` groups common motor-road tags, path and walkway tags, and
  other values. It describes the mapped highway class. Legal access and
  camera visibility require separate checks.
- `processed/osm/osm_ways.geojson` stores one full LineString per `osm_id`.
  Coordinate pairs are `[longitude, latitude]` in decimal degrees. Full ways
  can extend beyond the 300-meter radius.

Use `osm_id` to look up geometry and `camera_id` to look up camera metadata.
The camera–way key describes which searches returned each way. The road visible
in each camera remains unverified. Data © OpenStreetMap contributors under the
[Open Database License](https://www.openstreetmap.org/copyright).
