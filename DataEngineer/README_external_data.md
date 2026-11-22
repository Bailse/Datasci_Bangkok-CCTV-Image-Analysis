# Bangkok CCTV External Data

The collection scripts cover Open-Meteo weather, Thailand public holidays,
and BEM traffic volume. They use simple functions, Requests, and Pandas.
CCTV images and vehicle detection are outside this work's scope.

The current project contains no OpenStreetMap collector or OSM dataset files.
If OSM collection is restored later, use `data/external_data/raw/osm/` for
original responses and keep any processed data separate. A current OSM download
represents the map snapshot at collection time, not the historical road network
from August 2026.

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
    processed/
      weather/       # Hourly table CSV
      holidays/      # Empty; no cleaning performed
      bem/           # Empty; no cleaning performed
DataEngineer/
  external_data_collector/
    collect_data.py
    collect_holidays.py
    collect_bem.py
  requirements.txt
  README_external_data.md
  bronze_data/       # Existing work, unchanged
  clean_process/     # Existing work, unchanged
  pipeline/          # Existing work, unchanged
```

`data/cctv_info.csv` stays in place for existing notebooks and future camera
integration. Raw responses are preserved as original bytes. Collection does not
fill missing values or change the original dataset structures.

## Run

From the repository root:

```powershell
python -m pip install -r DataEngineer/requirements.txt
python DataEngineer/external_data_collector/collect_data.py
python DataEngineer/external_data_collector/collect_holidays.py
python DataEngineer/external_data_collector/collect_bem.py
```

Alternatively, from `DataEngineer/`, run
`python external_data_collector/collect_data.py`,
`python external_data_collector/collect_holidays.py`, or
`python external_data_collector/collect_bem.py`.
Dependencies are `requests`, `pandas`, and `lxml`; Pandas uses `lxml` to read
the BEM HTML table. No BeautifulSoup dependency is needed.

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
python DataEngineer/external_data_collector/collect_data.py --era5-grids
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
columns and first five records. It retains all 2026 holidays and missing values;
no cleaning or processed holiday file is created.

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
BEM is a separate monthly dataset and is never expanded into hourly weather
records. No processed BEM file is created.
