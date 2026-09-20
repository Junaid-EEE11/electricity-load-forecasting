from __future__ import annotations

import pandas as pd
import pytest

from load_forecasting.data import load_hourly_data


def _write(frame: pd.DataFrame, tmp_path) -> object:
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def test_loader_sorts_and_accepts_complete_hourly_data(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=4, freq="h")[::-1],
            "load_mw": [103.0, 102.0, 101.0, 100.0],
            "temperature_c": [23.0, 22.0, 21.0, 20.0],
        }
    )
    loaded = load_hourly_data(_write(frame, tmp_path))
    assert loaded["datetime"].is_monotonic_increasing
    assert len(loaded) == 4


def test_loader_rejects_gap(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2024-01-01 00:00", "2024-01-01 02:00"],
            "load_mw": [100.0, 101.0],
            "temperature_c": [20.0, 21.0],
        }
    )
    with pytest.raises(ValueError, match="complete hourly"):
        load_hourly_data(_write(frame, tmp_path))


def test_loader_rejects_duplicate_timestamp(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2024-01-01 00:00", "2024-01-01 00:00"],
            "load_mw": [100.0, 101.0],
            "temperature_c": [20.0, 21.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_hourly_data(_write(frame, tmp_path))
