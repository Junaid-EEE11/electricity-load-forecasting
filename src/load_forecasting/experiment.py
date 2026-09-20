"""End-to-end, auditable experimental protocol."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .data import data_fingerprint, load_hourly_data
from .features import ForecastSpec, assert_temporal_contract, build_supervised_frame, feature_columns
from .metrics import (
    conformal_radius,
    diebold_mariano,
    interval_metrics,
    point_metrics,
    seasonal_mase_scale,
)
from .models import model_registry


@dataclass(frozen=True)
class ExperimentConfig:
    """All choices that materially define an experiment."""

    horizon_hours: int = 24
    seasonal_period: int = 168
    test_fraction: float = 0.20
    calibration_fraction: float = 0.20
    cv_splits: int = 4
    interval_alpha: float = 0.10
    random_state: int = 42
    use_temperature: bool = True

    def __post_init__(self) -> None:
        for name in ("test_fraction", "calibration_fraction"):
            value = getattr(self, name)
            if not 0.05 <= value <= 0.4:
                raise ValueError(f"{name} must be between 0.05 and 0.4")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")


@dataclass
class ExperimentResult:
    champion: str
    metrics: pd.DataFrame
    cv_results: pd.DataFrame
    predictions: pd.DataFrame
    interval_summary: dict[str, float]
    dm_test: dict[str, float | str]
    feature_importance: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    metadata: dict[str, object]


def chronological_partitions(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fit, calibration, and untouched test partitions in time order."""

    n_rows = len(frame)
    test_start = int(n_rows * (1 - config.test_fraction))
    calibration_start = int(test_start * (1 - config.calibration_fraction))
    fit = frame.iloc[:calibration_start].copy()
    calibration = frame.iloc[calibration_start:test_start].copy()
    test = frame.iloc[test_start:].copy()
    if min(len(fit), len(calibration), len(test)) < 2 * config.seasonal_period:
        raise ValueError("Dataset is too short for the requested temporal partitions")
    if not (fit["target_time"].max() < calibration["target_time"].min() < test["target_time"].min()):
        raise AssertionError("Temporal partitions overlap or are out of order")
    return fit, calibration, test


def experiment_features(frame: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    """Resolve the declared information set, including weather ablations."""

    columns = feature_columns(frame)
    if not config.use_temperature:
        columns.remove("temperature_c")
    return columns


def _cross_validate(
    candidates: dict[str, object],
    fit_frame: pd.DataFrame,
    features: list[str],
    config: ExperimentConfig,
    mase_scale: float,
) -> pd.DataFrame:
    splitter = TimeSeriesSplit(
        n_splits=config.cv_splits,
        gap=config.horizon_hours,
    )
    rows: list[dict[str, object]] = []
    X = fit_frame[features]
    y = fit_frame["load_mw"]
    for model_name, estimator in candidates.items():
        for fold, (train_index, validation_index) in enumerate(splitter.split(X), start=1):
            fitted = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
            prediction = fitted.predict(X.iloc[validation_index])
            metrics = point_metrics(
                y.iloc[validation_index].to_numpy(),
                prediction,
                mase_scale=mase_scale,
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_end": fit_frame.iloc[train_index[-1]]["target_time"],
                    "validation_start": fit_frame.iloc[validation_index[0]]["target_time"],
                    "validation_end": fit_frame.iloc[validation_index[-1]]["target_time"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _subgroup_metrics(
    test: pd.DataFrame,
    predictions: np.ndarray,
    mase_scale: float,
) -> pd.DataFrame:
    diagnostic = pd.DataFrame(
        {
            "hour": test["target_time"].dt.hour.to_numpy(),
            "day_type": np.where(test["is_weekend"].to_numpy() == 1, "weekend", "weekday"),
            "actual": test["load_mw"].to_numpy(),
            "predicted": predictions,
        }
    )
    rows: list[dict[str, object]] = []
    for grouping in ("hour", "day_type"):
        for group, values in diagnostic.groupby(grouping, sort=True):
            rows.append(
                {
                    "dimension": grouping,
                    "group": str(group),
                    "n": len(values),
                    **point_metrics(
                        values["actual"].to_numpy(),
                        values["predicted"].to_numpy(),
                        mase_scale=mase_scale,
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_experiment(
    data_path: str | Path,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Execute model selection, calibration, and one-time holdout evaluation."""

    config = config or ExperimentConfig()
    spec = ForecastSpec(config.horizon_hours, config.seasonal_period)
    raw = load_hourly_data(data_path)
    supervised = build_supervised_frame(raw, spec)
    assert_temporal_contract(supervised, spec)
    features = experiment_features(supervised, config)
    fit, calibration, test = chronological_partitions(supervised, config)
    mase_scale = seasonal_mase_scale(fit["load_mw"].to_numpy(), config.seasonal_period)
    candidates = model_registry(config.random_state)
    cv_results = _cross_validate(candidates, fit, features, config, mase_scale)
    mean_cv = cv_results.groupby("model", sort=False)["rmse_mw"].mean()
    champion = str(mean_cv.idxmin())

    # Calibration is isolated from model fitting and final holdout testing. The
    # same fitted function centers calibration and test intervals, as required
    # by ordinary split conformal prediction.
    calibration_model = clone(candidates[champion]).fit(fit[features], fit["load_mw"])
    calibration_prediction = calibration_model.predict(calibration[features])
    radius = conformal_radius(
        calibration["load_mw"].to_numpy() - calibration_prediction,
        config.interval_alpha,
    )

    predictions = pd.DataFrame(
        {
            "target_time": test["target_time"].to_numpy(),
            "forecast_origin": test["forecast_origin"].to_numpy(),
            "actual_mw": test["load_mw"].to_numpy(),
        }
    )
    metrics_rows: list[dict[str, object]] = []

    seasonal_column = f"load_lag_{config.seasonal_period}h"
    persistence_column = f"load_lag_{config.horizon_hours}h"
    benchmark_predictions = {
        "Persistence": test[persistence_column].to_numpy(),
        "Seasonal Naive": test[seasonal_column].to_numpy(),
    }
    for name, prediction in benchmark_predictions.items():
        predictions[name] = prediction
        metrics_rows.append(
            {
                "model": name,
                **point_metrics(test["load_mw"].to_numpy(), prediction, mase_scale=mase_scale),
            }
        )

    fitted_models: dict[str, object] = {}
    for name, estimator in candidates.items():
        fitted = clone(estimator).fit(fit[features], fit["load_mw"])
        prediction = fitted.predict(test[features])
        fitted_models[name] = fitted
        predictions[name] = prediction
        metrics_rows.append(
            {
                "model": name,
                **point_metrics(test["load_mw"].to_numpy(), prediction, mase_scale=mase_scale),
            }
        )

    champion_prediction = predictions[champion].to_numpy()
    predictions["lower_90_mw"] = champion_prediction - radius
    predictions["upper_90_mw"] = champion_prediction + radius
    interval_summary = {
        "nominal_coverage_pct": 100 * (1 - config.interval_alpha),
        "conformal_radius_mw": radius,
        **interval_metrics(
            predictions["actual_mw"].to_numpy(),
            predictions["lower_90_mw"].to_numpy(),
            predictions["upper_90_mw"].to_numpy(),
            alpha=config.interval_alpha,
        ),
    }

    dm = diebold_mariano(
        predictions["actual_mw"].to_numpy(),
        champion_prediction,
        predictions["Seasonal Naive"].to_numpy(),
        horizon=config.horizon_hours,
    )
    dm_test: dict[str, float | str] = {
        "model_a": champion,
        "model_b": "Seasonal Naive",
        "loss": "squared_error",
        **dm,
    }

    # Permutation importance is measured only after all choices are fixed.
    sample_size = min(3000, len(test))
    rng = np.random.default_rng(config.random_state)
    sample_index = np.sort(rng.choice(len(test), size=sample_size, replace=False))
    importance = permutation_importance(
        fitted_models[champion],
        test[features].iloc[sample_index],
        test["load_mw"].iloc[sample_index],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=config.random_state,
        n_jobs=-1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    subgroup_metrics = _subgroup_metrics(test, champion_prediction, mase_scale)
    metrics = pd.DataFrame(metrics_rows).sort_values("rmse_mw", ignore_index=True)
    metadata: dict[str, object] = {
        "config": asdict(config),
        "data": data_fingerprint(raw),
        "features": features,
        "partition_rows": {
            "fit": len(fit),
            "calibration": len(calibration),
            "test": len(test),
        },
        "partition_boundaries": {
            "fit_end": fit["target_time"].max().isoformat(),
            "calibration_start": calibration["target_time"].min().isoformat(),
            "calibration_end": calibration["target_time"].max().isoformat(),
            "test_start": test["target_time"].min().isoformat(),
            "test_end": test["target_time"].max().isoformat(),
        },
        "selected_model": champion,
        "selection_rule": "lowest mean rolling-origin validation RMSE",
        "weather_assumption": (
            "target-time temperature treated as a perfect forecast"
            if config.use_temperature
            else "temperature excluded; load history and calendar features only"
        ),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return ExperimentResult(
        champion=champion,
        metrics=metrics,
        cv_results=cv_results,
        predictions=predictions,
        interval_summary=interval_summary,
        dm_test=dm_test,
        feature_importance=feature_importance,
        subgroup_metrics=subgroup_metrics,
        metadata=metadata,
    )


def save_result(result: ExperimentResult, output_dir: str | Path, figures_dir: str | Path) -> None:
    """Persist machine-readable evidence and publication-quality figures."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    result.metrics.to_csv(output_dir / "holdout_metrics.csv", index=False)
    result.cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    result.predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    result.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    result.subgroup_metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    (output_dir / "interval_metrics.json").write_text(
        json.dumps(result.interval_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "dm_test.json").write_text(json.dumps(result.dm_test, indent=2), encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")

    _plot_forecast(result, figures_dir / "holdout_forecast.png")
    _plot_model_comparison(result, figures_dir / "model_comparison.png")
    _plot_diagnostics(result, figures_dir / "residual_diagnostics.png")
    _plot_importance(result, figures_dir / "feature_importance.png")


def _plot_forecast(result: ExperimentResult, path: Path) -> None:
    frame = result.predictions.iloc[: 24 * 14]
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.fill_between(
        frame["target_time"],
        frame["lower_90_mw"],
        frame["upper_90_mw"],
        alpha=0.22,
        color="#4C78A8",
        label="90% split-conformal interval",
    )
    axis.plot(frame["target_time"], frame["actual_mw"], color="#222222", linewidth=1.4, label="Observed")
    axis.plot(
        frame["target_time"], frame[result.champion], color="#E45756", linewidth=1.2, label=result.champion
    )
    axis.set(
        title="24-hour-ahead load forecast on the untouched holdout", xlabel="Target time", ylabel="Load (MW)"
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_model_comparison(result: ExperimentResult, path: Path) -> None:
    ordered = result.metrics.sort_values("rmse_mw", ascending=True)
    colors = ["#E45756" if name == result.champion else "#4C78A8" for name in ordered["model"]]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(ordered["model"], ordered["rmse_mw"], color=colors)
    axis.set(title="Untouched-holdout model comparison", xlabel="RMSE (MW)", ylabel="")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostics(result: ExperimentResult, path: Path) -> None:
    frame = result.predictions.copy()
    residual = frame["actual_mw"] - frame[result.champion]
    hourly = (
        pd.DataFrame({"hour": frame["target_time"].dt.hour, "absolute_error": residual.abs()})
        .groupby("hour")["absolute_error"]
        .mean()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(frame["target_time"].iloc[:336], residual.iloc[:336], color="#4C78A8", linewidth=0.9)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set(title="Residual sequence", xlabel="Target time", ylabel="Observed - forecast (MW)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].hist(residual, bins=35, color="#72B7B2", edgecolor="white")
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set(title="Residual distribution", xlabel="Residual (MW)", ylabel="Count")
    axes[2].bar(hourly.index, hourly.values, color="#F58518")
    axes[2].set(title="Error by target hour", xlabel="Hour", ylabel="MAE (MW)")
    for axis in axes:
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(result: ExperimentResult, path: Path) -> None:
    top = result.feature_importance.head(12).sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#54A24B",
        alpha=0.9,
    )
    axis.set(
        title=f"Permutation importance: {result.champion}",
        xlabel="Increase in MAE after permutation (MW)",
        ylabel="",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
