from __future__ import annotations

import numpy as np
import pandas as pd

from load_forecasting.experiment import ExperimentConfig, experiment_features
from load_forecasting.features import ForecastSpec, build_supervised_frame


def test_no_temperature_ablation_has_no_future_weather_covariate() -> None:
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=1000, freq="h"),
            "load_mw": 1000 + np.sin(np.arange(1000)),
            "temperature_c": 20 + np.cos(np.arange(1000)),
        }
    )
    frame = build_supervised_frame(data, ForecastSpec())
    config = ExperimentConfig(use_temperature=False)
    features = experiment_features(frame, config)
    assert "temperature_c" not in features
