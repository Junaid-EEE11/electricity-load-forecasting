"""Point, interval, and forecast-comparison metrics."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np


def point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mase_scale: float,
) -> dict[str, float]:
    """Compute complementary metrics without hiding zero-demand behavior."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = actual - predicted
    nonzero = np.abs(actual) > np.finfo(float).eps
    mape = np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100 if nonzero.any() else np.nan
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 0,
    )
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
        "mape_pct": float(mape),
        "smape_pct": float(100 * np.mean(smape_terms)),
        "mase": float(np.mean(np.abs(error)) / mase_scale),
        "bias_mw": float(np.mean(predicted - actual)),
    }


def seasonal_mase_scale(training_values: np.ndarray, period: int) -> float:
    values = np.asarray(training_values, dtype=float)
    if len(values) <= period:
        raise ValueError("Training series is too short for the MASE seasonal period")
    scale = float(np.mean(np.abs(values[period:] - values[:-period])))
    if scale <= np.finfo(float).eps:
        raise ValueError("MASE is undefined for a constant seasonal training series")
    return scale


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    """Evaluate marginal coverage, width, and Winkler interval score."""

    actual = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    width = high - low
    penalty_low = (2 / alpha) * np.maximum(low - actual, 0)
    penalty_high = (2 / alpha) * np.maximum(actual - high, 0)
    return {
        "coverage_pct": float(100 * np.mean((actual >= low) & (actual <= high))),
        "mean_width_mw": float(np.mean(width)),
        "winkler_score": float(np.mean(width + penalty_low + penalty_high)),
    }


def conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal radius using the conservative order statistic."""

    scores = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - alpha))))
    return float(scores[rank - 1])


def diebold_mariano(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    horizon: int = 1,
) -> dict[str, float]:
    """Two-sided Diebold-Mariano test with a Bartlett HAC variance estimate.

    The loss differential is squared error of A minus squared error of B, so a
    negative statistic favors model A. The normal approximation is reported;
    this is diagnostic evidence, not a substitute for replication on real data.
    """

    actual = np.asarray(y_true, dtype=float)
    loss_a = np.square(actual - np.asarray(prediction_a, dtype=float))
    loss_b = np.square(actual - np.asarray(prediction_b, dtype=float))
    differential = loss_a - loss_b
    n = len(differential)
    centered = differential - differential.mean()
    max_lag = min(max(horizon - 1, 0), n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1 - lag / (max_lag + 1)
        long_run_variance += 2 * weight * covariance
    if long_run_variance <= 0:
        statistic = 0.0 if differential.mean() == 0 else float(np.sign(differential.mean()) * np.inf)
    else:
        statistic = float(differential.mean() / sqrt(long_run_variance / n))
    p_value = float(erfc(abs(statistic) / sqrt(2)))
    return {"dm_statistic": statistic, "p_value": p_value}
