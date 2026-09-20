from __future__ import annotations

import numpy as np
import pandas as pd

from load_forecasting.experiment import ExperimentConfig, chronological_partitions


def test_temporal_partitions_are_disjoint_and_ordered() -> None:
    frame = pd.DataFrame(
        {
            "target_time": pd.date_range("2020-01-01", periods=3000, freq="h"),
            "load_mw": np.arange(3000),
        }
    )
    fit, calibration, test = chronological_partitions(frame, ExperimentConfig())
    assert len(fit) == 1920
    assert len(calibration) == 480
    assert len(test) == 600
    assert fit["target_time"].max() < calibration["target_time"].min()
    assert calibration["target_time"].max() < test["target_time"].min()
