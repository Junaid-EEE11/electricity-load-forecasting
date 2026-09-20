"""Command-line entry point for the forecasting experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe, 24-hour-ahead electricity load forecasting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=Path("data/sample_hourly_load.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="Exclude target-time temperature for a history-plus-calendar ablation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        horizon_hours=args.horizon,
        cv_splits=args.cv_splits,
        random_state=args.seed,
        use_temperature=not args.no_temperature,
    )
    result = run_experiment(args.data, config)
    save_result(result, args.output_dir, args.figures_dir)

    print("\nUntouched holdout results")
    print("=" * 76)
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected by rolling validation: {result.champion}")
    print(
        "90% interval: "
        f"coverage={result.interval_summary['coverage_pct']:.2f}%, "
        f"mean width={result.interval_summary['mean_width_mw']:.2f} MW"
    )
    print(
        "DM test against Seasonal Naive: "
        f"statistic={result.dm_test['dm_statistic']:.3f}, "
        f"p={result.dm_test['p_value']:.4g}"
    )
    print(f"Artifacts: {args.output_dir.resolve()}")
    print(f"Figures:   {args.figures_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
