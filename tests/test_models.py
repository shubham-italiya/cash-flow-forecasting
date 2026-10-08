import numpy as np
import pandas as pd
import pytest

from cashflow.models import LstmForecaster, Sarima, SeasonalNaive, load_forecaster


def test_seasonal_naive_repeats_last_week(weekly_series):
    pred = SeasonalNaive().fit(weekly_series).predict(10)
    last_week = weekly_series.to_numpy()[-7:]
    assert np.allclose(pred[:7], last_week) and np.allclose(pred[7:], last_week[:3])


def test_sarima_forecasts_the_weekly_shape(weekly_series):
    pred = Sarima((1, 1, 1), (0, 1, 1, 7)).fit(weekly_series).predict(14)
    assert pred.shape == (14,) and (pred >= 0).all()
    future = pd.date_range(weekly_series.index[-1] + pd.Timedelta(days=1), periods=14, freq="D")
    saturdays = future.dayofweek == 5
    assert pred[saturdays].max() < pred[~saturdays].min()   # it learned that Saturdays are closed


def test_lstm_trains_and_forecasts(weekly_series):
    model = LstmForecaster(window=14, horizon=7, units=8, epochs=3).fit(weekly_series)
    pred = model.predict(7)
    assert pred.shape == (7,) and (pred >= 0).all()
    with pytest.raises(ValueError):
        model.predict(8)


@pytest.mark.parametrize("make", [
    SeasonalNaive,
    lambda: Sarima((1, 1, 0), (0, 1, 1, 7)),
    lambda: LstmForecaster(window=14, horizon=7, units=8, epochs=2),
])
def test_save_and_load_give_the_same_forecast(make, weekly_series, tmp_path):
    model = make().fit(weekly_series)
    model.save(tmp_path)
    loaded = load_forecaster(tmp_path)
    assert loaded.last_date_ == weekly_series.index[-1]
    assert np.allclose(loaded.predict(7), model.predict(7), rtol=1e-5)
