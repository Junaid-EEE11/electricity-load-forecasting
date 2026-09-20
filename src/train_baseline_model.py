"""Backward-compatible launcher for the research-grade experiment.

Prefer ``python -m load_forecasting.cli`` after installing the package.
"""

from load_forecasting.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
