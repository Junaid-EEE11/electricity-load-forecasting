"""Leakage-safe feature engineering for direct-horizon forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastSpec:
    """Operational definition of a forecasting task.

    ``horizon_hours=24`` means each target at time *t* is issued at *t - 24h*.
    Target-time weather is treated as an available numerical weather forecast.
    With the bundled synthetic data it is realized weather, so reported scores
    represent an oracle-weather upper bound.
    """

    horizon_hours: int = 24
    seasonal_period: int = 168

    def __post_init__(self) -> None:
        if self.horizon_hours < 1:
            raise ValueError("horizon_hours must be at least 1")
        if self.seasonal_period < self.horizon_hours:
            raise ValueError("seasonal_period must be >= horizon_hours")


def _cyclic(values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    angle = 2.0 * np.pi * values.astype(float) / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(angle),
            f"{prefix}_cos": np.cos(angle),
        },
        index=values.index,
    )


def build_supervised_frame(data: pd.DataFrame, spec: ForecastSpec) -> pd.DataFrame:
    """Create predictors whose load inputs are known at forecast issue time.

    For a target at row ``t``, all load-derived features are shifted by at least
    ``horizon_hours``. This is the central anti-leakage invariant of the project.
    """

    frame = data.copy().sort_values("datetime").reset_index(drop=True)
    timestamp = pd.to_datetime(frame["datetime"])
    horizon = spec.horizon_hours

    result = pd.DataFrame(
        {
            "target_time": timestamp,
            "forecast_origin": timestamp - pd.to_timedelta(horizon, unit="h"),
            "load_mw": frame["load_mw"].astype(float),
            "temperature_c": frame["temperature_c"].astype(float),
            "is_weekend": timestamp.dt.dayofweek.isin([5, 6]).astype(int),
        }
    )

    result = pd.concat(
        [
            result,
            _cyclic(timestamp.dt.hour, 24, "hour"),
            _cyclic(timestamp.dt.dayofweek, 7, "dow"),
            _cyclic(timestamp.dt.dayofyear - 1, 365.2425, "year"),
        ],
        axis=1,
    )

    # Candidate target-relative lags. Filtering guarantees that every lag was
    # observed no later than the forecast origin.
    lags = sorted({horizon, 24, 48, spec.seasonal_period, 2 * spec.seasonal_period})
    for lag in (lag for lag in lags if lag >= horizon):
        result[f"load_lag_{lag}h"] = frame["load_mw"].shift(lag)

    observed_at_origin = frame["load_mw"].shift(horizon)
    result["load_mean_24h"] = observed_at_origin.rolling(24).mean()
    result["load_std_24h"] = observed_at_origin.rolling(24).std()
    result["load_mean_168h"] = observed_at_origin.rolling(168).mean()
    result["load_std_168h"] = observed_at_origin.rolling(168).std()

    return result.dropna().reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model predictors, excluding identifiers and the target."""

    excluded = {"target_time", "forecast_origin", "load_mw"}
    return [column for column in frame.columns if column not in excluded]


def assert_temporal_contract(frame: pd.DataFrame, spec: ForecastSpec) -> None:
    """Fail loudly if a load lag can occur after its forecast origin."""

    if not (frame["forecast_origin"] < frame["target_time"]).all():
        raise AssertionError("Forecast origins must precede target timestamps")
    for column in frame.columns:
        if column.startswith("load_lag_"):
            lag = int(column.removeprefix("load_lag_").removesuffix("h"))
            if lag < spec.horizon_hours:
                raise AssertionError(f"{column} violates the forecast horizon")
