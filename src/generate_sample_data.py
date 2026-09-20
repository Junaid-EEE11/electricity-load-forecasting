"""Generate a reproducible synthetic hourly load benchmark.

The data-generating process contains nonlinear weather sensitivity, multiple
seasonalities, trend, serially correlated disturbances, and rare demand shocks.
It is useful for testing a pipeline, but it is not evidence about a real grid.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _ar1_noise(rng: np.random.Generator, rows: int, phi: float, sigma: float) -> np.ndarray:
    innovations = rng.normal(0, sigma, rows)
    values = np.zeros(rows)
    for index in range(1, rows):
        values[index] = phi * values[index - 1] + innovations[index]
    return values


def generate_data(start: str = "2022-01-01", end: str = "2024-12-31 23:00", seed: int = 42) -> pd.DataFrame:
    """Return an hourly synthetic system-level demand series."""

    rng = np.random.default_rng(seed)
    datetime_index = pd.date_range(start=start, end=end, freq="h")
    frame = pd.DataFrame({"datetime": datetime_index})

    hour = frame["datetime"].dt.hour.to_numpy()
    day_of_week = frame["datetime"].dt.dayofweek.to_numpy()
    day_of_year = frame["datetime"].dt.dayofyear.to_numpy()
    weekend = day_of_week >= 5
    elapsed_years = np.arange(len(frame)) / (365.2425 * 24)

    annual_temperature = 26 + 7.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365.2425)
    diurnal_temperature = 3.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather_system = _ar1_noise(rng, len(frame), phi=0.94, sigma=0.45)
    temperature = annual_temperature + diurnal_temperature + weather_system

    morning_peak = 125 * np.exp(-0.5 * ((hour - 9) / 2.6) ** 2)
    evening_peak = 235 * np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)
    overnight_trough = -130 * np.exp(-0.5 * ((hour - 3) / 2.8) ** 2)
    weekly_effect = np.where(weekend, -95, 30) + np.where(day_of_week == 4, -15, 0)
    cooling = 20 * np.maximum(temperature - 25, 0) ** 1.15
    heating = 12 * np.maximum(18 - temperature, 0) ** 1.10
    smooth_seasonality = 65 * np.sin(2 * np.pi * (day_of_year - 80) / 365.2425)
    growth = 28 * elapsed_years
    correlated_demand_noise = _ar1_noise(rng, len(frame), phi=0.72, sigma=27)

    # Reproducible short stress events make robustness diagnostics meaningful.
    stress = np.zeros(len(frame))
    event_starts = rng.choice(np.arange(168, len(frame) - 48), size=10, replace=False)
    for event_start in event_starts:
        duration = int(rng.integers(6, 30))
        stress[event_start : event_start + duration] += rng.normal(90, 25)

    load = (
        890
        + morning_peak
        + evening_peak
        + overnight_trough
        + weekly_effect
        + cooling
        + heating
        + smooth_seasonality
        + growth
        + correlated_demand_noise
        + stress
    )
    frame["load_mw"] = np.round(load, 2)
    frame["temperature_c"] = np.round(temperature, 2)
    return frame[["datetime", "load_mw", "temperature_c"]]


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(output_dir / "sample_hourly_load.csv", index=False)
    print(f"Saved {len(df)} rows to {output_dir / 'sample_hourly_load.csv'}")
