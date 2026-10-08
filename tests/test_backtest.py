import numpy as np
import pandas as pd
import pytest

from cashflow import backtest
from cashflow.models import SeasonalNaive


def test_origins_end_at_the_last_day():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2021-01-01", periods=100))
    assert backtest.origins(s, folds=3, horizon=10) == [70, 80, 90]
    with pytest.raises(ValueError):
        backtest.origins(s, folds=10, horizon=10)


def test_naive_is_perfect_on_a_perfectly_weekly_series():
    days = pd.date_range("2021-01-04", periods=120)
    s = pd.Series(np.array([3.0, 4, 5, 6, 7, 1, 2])[days.dayofweek], index=days)
    result = backtest.run(SeasonalNaive, s, folds=3, horizon=14)
    assert len(result) == 3
    assert np.allclose(result["mae"], 0) and np.allclose(result["total_error_pct"], 0)
