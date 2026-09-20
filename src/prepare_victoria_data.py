"""Acquire and normalize the public Victoria electricity benchmark.

The source files are curated by the tidyverts ``tsibbledata`` project from
Australian Energy Market Operator demand and Melbourne temperature data. Raw
files are pinned to an immutable upstream commit and are not committed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

UPSTREAM_COMMIT = "ccfcb321bb4f0e12b7287f5a8f752097e35a20a5"
BASE_URL = (
    f"https://raw.githubusercontent.com/tidyverts/tsibbledata/{UPSTREAM_COMMIT}/data-raw/vic_elec/VIC2015"
)
SOURCES = {
    "demand.csv": {
        "url": f"{BASE_URL}/demand.csv",
        "sha256": "bbc2c0f75636cb95e974438962fd858d5ce23e533f80e292c1b6b95f22c5aa37",
    },
    "temperature.csv": {
        "url": f"{BASE_URL}/temperature.csv",
        "sha256": "c889eccb098ecf970d2c691a774081c7f4603f6423522f7c50694d25f8ab0453",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _acquire(raw_dir: Path, refresh: bool) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for filename, source in SOURCES.items():
        destination = raw_dir / filename
        if refresh or not destination.exists():
            print(f"Downloading {source['url']}")
            urllib.request.urlretrieve(str(source["url"]), destination)
        actual = _sha256(destination)
        expected = str(source["sha256"])
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {filename}: expected {expected}, got {actual}")
        hashes[filename] = actual
    return hashes


def prepare(raw_dir: Path, output: Path, refresh: bool = False) -> pd.DataFrame:
    """Download, join, filter, and hourly-resample Victoria data."""

    hashes = _acquire(raw_dir, refresh)
    demand = pd.read_csv(raw_dir / "demand.csv")
    temperature = pd.read_csv(raw_dir / "temperature.csv")
    merged = demand.merge(temperature, on=["Date", "Period"], validate="one_to_one")

    excel_date = pd.to_datetime(merged["Date"], unit="D", origin="1899-12-30")
    merged["datetime"] = excel_date + pd.to_timedelta((merged["Period"] - 1) * 30, unit="m")
    merged = merged.loc[merged["datetime"].dt.year.between(2012, 2014)].copy()
    merged = merged.set_index("datetime").sort_index()

    hourly = (
        merged[["OperationalLessIndustrial", "Temp"]]
        .resample("h")
        .mean()
        .rename(columns={"OperationalLessIndustrial": "load_mw", "Temp": "temperature_c"})
        .reset_index()
    )
    if len(hourly) != 26_304 or hourly.isna().any().any():
        raise ValueError("Unexpected row count or missing values after hourly resampling")
    if hourly["datetime"].diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise ValueError("Prepared data do not form a complete hourly grid")

    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    metadata = {
        "dataset": "Victoria operational electricity demand and Melbourne temperature",
        "period": "2012-01-01 through 2014-12-31",
        "rows": len(hourly),
        "source_repository": "https://github.com/tidyverts/tsibbledata",
        "source_commit": UPSTREAM_COMMIT,
        "source_license": "GPL-3",
        "primary_demand_source": "Australian Energy Market Operator",
        "temperature_site": "Melbourne BOM site 086071",
        "raw_sha256": hashes,
        "transformation": "Mean of each pair of half-hourly demand and temperature observations",
        "time_policy": "Timezone-naive local civil dates; fixed 48 source periods per source date",
        "weather_assumption": "Observed temperature; retrospective oracle-weather benchmark",
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(hourly):,} rows to {output}")
    print(f"Saved provenance to {metadata_path}")
    return hourly


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/victoria"))
    parser.add_argument("--output", type=Path, default=Path("data/victoria_hourly_load.csv"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output, args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
