from __future__ import annotations

import numpy as np
import pandas as pd

from load_forecasting.features import (
    ForecastSpec,
    assert_temporal_contract,
    build_supervised_frame,
)


def _frame(rows: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "load_mw": np.arange(rows, dtype=float) + 100.0,
            "temperature_c": np.linspace(15, 30, rows),
        }
    )


def test_lags_respect_24_hour_issue_time() -> None:
    data = _frame()
    spec = ForecastSpec(horizon_hours=24)
    supervised = build_supervised_frame(data, spec)
    row = supervised.loc[supervised["target_time"] == data.loc[500, "datetime"]].iloc[0]
    assert row["forecast_origin"] == data.loc[476, "datetime"]
    assert row["load_lag_24h"] == data.loc[476, "load_mw"]
    assert np.isclose(row["load_mean_24h"], data.loc[453:476, "load_mw"].mean())
    assert_temporal_contract(supervised, spec)


def test_future_load_changes_cannot_change_features_available_at_origin() -> None:
    original = _frame()
    perturbed = original.copy()
    perturbed.loc[477:, "load_mw"] += 1_000_000
    spec = ForecastSpec(horizon_hours=24)
    first = build_supervised_frame(original, spec)
    second = build_supervised_frame(perturbed, spec)
    target = original.loc[500, "datetime"]
    load_features = [column for column in first if column.startswith("load_") and column != "load_mw"]
    first_values = first.loc[first["target_time"] == target, load_features].to_numpy()
    second_values = second.loc[second["target_time"] == target, load_features].to_numpy()
    np.testing.assert_allclose(first_values, second_values)
