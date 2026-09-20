"""Research-grade utilities for short-term electricity load forecasting."""

from .features import ForecastSpec, build_supervised_frame
from .metrics import point_metrics

__all__ = ["ForecastSpec", "build_supervised_frame", "point_metrics"]
__version__ = "1.0.0"
