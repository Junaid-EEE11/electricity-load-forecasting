"""Data loading and validation for hourly load time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("datetime", "load_mw", "temperature_c")


def load_hourly_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV and enforce the project's hourly-data contract.

    The strict checks are intentional: silent duplicate timestamps, gaps, or
    non-finite measurements can make a forecasting experiment look much better
    than it is.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame = frame.sort_values("datetime").reset_index(drop=True)

    if frame["datetime"].duplicated().any():
        duplicates = int(frame["datetime"].duplicated(keep=False).sum())
        raise ValueError(f"Found {duplicates} rows with duplicate timestamps")
    if frame[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise ValueError("Load and temperature columns must not contain missing values")
    numeric = frame[list(REQUIRED_COLUMNS[1:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Load and temperature columns must be finite")
    if (frame["load_mw"] <= 0).any():
        raise ValueError("load_mw must be strictly positive")

    intervals = frame["datetime"].diff().dropna()
    expected = pd.Timedelta(hours=1)
    irregular = intervals.ne(expected)
    if irregular.any():
        first_bad = intervals.index[irregular][0]
        previous = frame.loc[first_bad - 1, "datetime"]
        current = frame.loc[first_bad, "datetime"]
        raise ValueError(
            f"Expected a complete hourly time index; first irregular interval is {previous} -> {current}"
        )

    return frame


def data_fingerprint(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact provenance metadata for an experiment artifact."""

    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    import hashlib

    return {
        "rows": int(len(frame)),
        "start": frame["datetime"].min().isoformat(),
        "end": frame["datetime"].max().isoformat(),
        "sha256": hashlib.sha256(hashed).hexdigest(),
    }
