# Electricity Load Forecasting

A leakage-safe and uncertainty-aware research pipeline for short-term electricity load forecasting. The project supports reproducible experiments on hourly demand data, with chronological validation, benchmark models, prediction intervals, diagnostics, and provenance tracking.

> **Research note:** The bundled synthetic dataset is designed for testing the pipeline and is not evidence about a real electricity grid. When using the bundled temperature feature, scores represent an oracle-weather upper bound because realized target-time temperature is used as a perfect forecast.

## Highlights

- Direct, configurable forecasts with a default 24-hour horizon.
- Forecast-origin contracts that prevent load information from leaking after the issue time.
- Chronological fit, calibration, and untouched holdout partitions.
- Rolling-origin time-series cross-validation with a configurable gap.
- Persistence and seasonal-naive benchmarks.
- Ridge, histogram gradient boosting, and random forest candidate models.
- Split-conformal 90% prediction intervals.
- Point metrics including MAE, RMSE, MAPE, sMAPE, MASE, and bias.
- Holdout diagnostics, subgroup metrics, permutation importance, and a Diebold–Mariano comparison against Seasonal Naive.
- Reproducible synthetic data generation and checksum-verified Victoria benchmark preparation.

## Requirements

- Python 3.11 or newer
- Runtime dependencies: NumPy, pandas, scikit-learn, and Matplotlib

## Installation

Create and activate a virtual environment, then install the package with development and notebook extras if needed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For runtime dependencies only, use:

```bash
python -m pip install -e .
```

## Quick start

Generate the reproducible three-year synthetic hourly dataset:

```bash
python src/generate_sample_data.py
```

Run the default 24-hour-ahead experiment:

```bash
load-forecast
```

The command is equivalent to:

```bash
python -m load_forecasting.cli \
  --data data/sample_hourly_load.csv \
  --output-dir artifacts \
  --figures-dir figures
```

The run prints untouched-holdout metrics, the model selected by rolling validation, 90% interval coverage and width, and the Diebold–Mariano diagnostic. Results are written to `artifacts/`, and plots are written to `figures/`.

### Useful options

```bash
load-forecast --horizon 24 --cv-splits 4 --seed 42
load-forecast --no-temperature
load-forecast --data path/to/hourly_load.csv --output-dir artifacts --figures-dir figures
```

Input data must contain these columns:

- `datetime`: hourly timestamps
- `load_mw`: electricity demand in megawatts
- `temperature_c`: temperature in degrees Celsius

## Victoria benchmark data

The repository includes a preparation script for the public Victoria electricity benchmark curated by the [`tsibbledata`](https://github.com/tidyverts/tsibbledata) project. Raw files are downloaded from a pinned upstream commit, verified with SHA-256 checksums, joined, and resampled from half-hourly to hourly observations.

```bash
python src/prepare_victoria_data.py
load-forecast --data data/victoria_hourly_load.csv
```

By default, the preparation script stores raw downloads in `data/raw/victoria`, produces `data/victoria_hourly_load.csv`, and writes accompanying provenance to `data/victoria_hourly_load.metadata.json`. Use `--refresh` to download the pinned sources again.

The Victoria metadata records the source, license, date range, checksums, transformation, timezone policy, and weather assumption. Confirm that the source license and intended use meet your requirements before redistributing derived data.

## Outputs

A completed experiment writes the following machine-readable artifacts:

- `holdout_metrics.csv` — benchmark and candidate-model holdout metrics
- `cv_results.csv` — rolling validation results used for model selection
- `holdout_predictions.csv` — predictions, observations, and conformal bounds
- `feature_importance.csv` — permutation importance for the selected model
- `subgroup_metrics.csv` — performance by target hour and weekday/weekend
- `interval_metrics.json` — coverage, interval width, and Winkler score
- `dm_test.json` — Diebold–Mariano diagnostic against Seasonal Naive
- `run_metadata.json` — configuration, data fingerprint, partition boundaries, and software versions

Generated figures include the holdout forecast, model comparison, residual diagnostics, and feature importance chart.

## Methodology

For a target at time `t`, the default forecast is issued at `t - 24 hours`. Load-derived predictors are shifted by at least the forecast horizon. The default feature set includes:

- Hour, day-of-week, and annual cyclic features
- Weekend indicator
- Horizon, daily, and weekly seasonal load lags
- Rolling 24-hour and 168-hour load statistics computed only from information available at the forecast origin
- Temperature, when enabled

Model selection uses the lowest mean rolling-origin validation RMSE. The selected model is then evaluated once on an untouched chronological holdout. A separate calibration partition is used to construct split-conformal prediction intervals.

This protocol separates model selection, calibration, and final assessment. It is intended for methodological experimentation and should not be treated as operational grid-forecasting software without additional validation, live weather forecasts, timezone handling, monitoring, and deployment controls.

## Project layout

```text
.
├── notebooks/                  # Exploratory baseline notebook
├── src/
│   ├── generate_sample_data.py # Reproducible synthetic benchmark generator
│   ├── prepare_victoria_data.py# Public Victoria data acquisition and preparation
│   ├── train_baseline_model.py # Backward-compatible launcher
│   └── load_forecasting/       # Forecasting package and CLI
├── tests/                      # Data, feature, protocol, metric, and ablation tests
├── pyproject.toml              # Package metadata and tool configuration
└── requirements.txt             # Runtime dependency list
```

## Development and tests

Run formatting checks, linting, and the test suite:

```bash
ruff check .
ruff format --check .
pytest
```

Please preserve the forecast-origin contract and the distinction between model selection, calibration, and final assessment. Add or update tests for methodological changes, especially changes to feature timing. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution and dataset-provenance guidance.

## License and data

No project license is currently declared in `pyproject.toml`. Add an appropriate license before distributing this repository or its software. Third-party data retain their upstream licensing and attribution requirements; see the generated Victoria metadata for the pinned source details.
