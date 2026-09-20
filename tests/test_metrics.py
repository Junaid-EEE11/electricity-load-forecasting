from __future__ import annotations

import numpy as np

from load_forecasting.metrics import conformal_radius, diebold_mariano, point_metrics


def test_point_metrics_on_known_errors() -> None:
    actual = np.array([100.0, 200.0])
    predicted = np.array([90.0, 220.0])
    metrics = point_metrics(actual, predicted, mase_scale=10.0)
    assert metrics["mae_mw"] == 15.0
    assert np.isclose(metrics["rmse_mw"], np.sqrt(250))
    assert metrics["mase"] == 1.5
    assert metrics["bias_mw"] == 5.0


def test_conformal_radius_uses_conservative_finite_sample_rank() -> None:
    residuals = np.arange(1.0, 11.0)
    assert conformal_radius(residuals, alpha=0.2) == 9.0


def test_dm_sign_favors_more_accurate_first_model() -> None:
    actual = np.linspace(100, 200, 200)
    strong = actual + np.sin(np.arange(200))
    weak = actual + 10 * np.sin(np.arange(200))
    result = diebold_mariano(actual, strong, weak, horizon=1)
    assert result["dm_statistic"] < 0
    assert result["p_value"] < 0.05
