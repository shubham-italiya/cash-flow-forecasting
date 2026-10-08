import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def weekly_series() -> pd.Series:
    """200 days with a weekly pattern (closed on Saturdays) plus a little noise."""
    days = pd.date_range("2021-01-04", periods=200, freq="D")
    pattern = np.array([30, 35, 33, 37, 28, 0, 17], dtype=float) * 1000
    rng = np.random.default_rng(0)
    values = pattern[days.dayofweek] * rng.normal(1, 0.05, len(days))
    values[days.dayofweek == 5] = 0.0
    return pd.Series(values, index=days, name="net_cash")
