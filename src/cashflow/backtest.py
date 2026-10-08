"""Rolling-origin backtest: fit on everything before a date, forecast the next 30 days, move on 30 days."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def origins(series: pd.Series, folds: int = 6, horizon: int = 30) -> list[int]:
    """Start positions of the forecast windows; the last window ends on the last day of the series."""
    first = len(series) - folds * horizon
    if first <= 0:
        raise ValueError("series too short for this many folds")
    return [first + k * horizon for k in range(folds)]


def run(make_model: Callable, series: pd.Series, folds: int = 6, horizon: int = 30) -> pd.DataFrame:
    """One row per fold: daily MAE (£) and the error on the 30-day total (%)."""
    rows = []
    for start in origins(series, folds, horizon):
        train, test = series.iloc[:start], series.iloc[start:start + horizon]
        forecast = make_model().fit(train).predict(len(test))
        actual = test.to_numpy(dtype=float)
        rows.append({
            "fold_start": test.index[0].date(),
            "mae": float(np.mean(np.abs(forecast - actual))),
            "total_actual": float(actual.sum()),
            "total_forecast": float(forecast.sum()),
            "total_error_pct": float(100 * abs(forecast.sum() - actual.sum()) / actual.sum()),
            "forecast": forecast,
        })
    return pd.DataFrame(rows)
