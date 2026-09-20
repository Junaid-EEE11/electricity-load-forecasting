"""Deterministic model registry and benchmark forecasts."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_registry(random_state: int = 42) -> dict[str, object]:
    """Return interpretable linear and nonlinear candidate estimators."""

    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
