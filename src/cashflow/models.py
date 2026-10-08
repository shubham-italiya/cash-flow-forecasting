"""Three forecasters with the same interface: fit(series) -> self, predict(days) -> array, save(dir), load(dir).

All of them forecast the days straight after the end of the series they were fitted on.
"""
from __future__ import annotations

import itertools
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


class SeasonalNaive:
    """Baseline: each day equals the same weekday last week."""

    kind = "seasonal_naive"

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, series: pd.Series) -> SeasonalNaive:
        self.last_season_ = series.to_numpy(dtype=float)[-self.season:]
        self.last_date_ = series.index[-1]
        return self

    def predict(self, days: int) -> np.ndarray:
        return np.resize(self.last_season_, days)

    def save(self, folder: Path) -> None:
        _write_meta(folder, self, {"season": self.season, "last_season": self.last_season_.tolist()})

    @classmethod
    def load(cls, folder: Path) -> SeasonalNaive:
        meta = _read_meta(folder)
        model = cls(meta["season"])
        model.last_season_ = np.array(meta["last_season"])
        model.last_date_ = pd.Timestamp(meta["last_date"])
        return model


class Sarima:
    """Seasonal ARIMA on log1p(cash) with weekly seasonality. Orders come from select_order()."""

    kind = "sarima"

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)):
        self.order, self.seasonal_order = tuple(order), tuple(seasonal_order)

    def fit(self, series: pd.Series) -> Sarima:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short folds
            self.result_ = SARIMAX(np.log1p(series.to_numpy(dtype=float)), order=self.order,
                                   seasonal_order=self.seasonal_order, enforce_stationarity=False,
                                   enforce_invertibility=False).fit(disp=False)
        self.last_date_ = series.index[-1]
        return self

    def predict(self, days: int) -> np.ndarray:
        return np.clip(np.expm1(self.result_.forecast(days)), 0, None)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.result_.save(folder / "sarima.pkl", remove_data=False)
        _write_meta(folder, self, {"order": self.order, "seasonal_order": self.seasonal_order})

    @classmethod
    def load(cls, folder: Path) -> Sarima:
        meta = _read_meta(folder)
        model = cls(meta["order"], meta["seasonal_order"])
        with open(folder / "sarima.pkl", "rb") as f:
            model.result_ = pickle.load(f)
        model.last_date_ = pd.Timestamp(meta["last_date"])
        return model

    @staticmethod
    def select_order(series: pd.Series, p=(0, 1, 2), q=(0, 1, 2), seasonal_q=(0, 1)) -> tuple[tuple, tuple]:
        """Small grid search by AIC, with d = D = 1 (see the ADF tests in reports/stationarity.md)."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.log1p(series.to_numpy(dtype=float))
        best = (np.inf, None, None)
        for pp, qq, sq in itertools.product(p, q, seasonal_q):
            order, seasonal = (pp, 1, qq), (0, 1, sq, 7)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aic = SARIMAX(y, order=order, seasonal_order=seasonal, enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False).aic
            if aic < best[0]:
                best = (aic, order, seasonal)
        return best[1], best[2]


class LstmForecaster:
    """LSTM that reads the last `window` days and outputs all `horizon` future days at once.

    Inputs per day: standardised log1p(cash) plus a one-hot day of the week (so it knows where Saturdays are).
    """

    kind = "lstm"

    def __init__(self, window: int = 56, horizon: int = 30, units: int = 64, epochs: int = 300,
                 patience: int = 25, seed: int = SEED):
        self.window, self.horizon, self.units = window, horizon, units
        self.epochs, self.patience, self.seed = epochs, patience, seed

    def _inputs(self, values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        dow = np.eye(7)[dates.dayofweek]
        return np.column_stack([values, dow]).astype("float32")

    def fit(self, series: pd.Series) -> LstmForecaster:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
        y = np.log1p(series.to_numpy(dtype=float))
        self.mean_, self.std_ = float(y.mean()), float(y.std()) or 1.0
        z = (y - self.mean_) / self.std_
        feats = self._inputs(z, series.index)
        n = len(z) - self.window - self.horizon + 1
        if n < 20:
            raise ValueError(f"series too short for window {self.window} + horizon {self.horizon}")
        X = np.stack([feats[i:i + self.window] for i in range(n)])
        Y = np.stack([z[i + self.window:i + self.window + self.horizon] for i in range(n)])

        self.model_ = tf.keras.Sequential([
            tf.keras.layers.Input((self.window, X.shape[2])),
            tf.keras.layers.LSTM(self.units),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.horizon),
        ])
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
        stop = tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        # Validation = the most recent 15% of windows (no shuffling across time).
        self.model_.fit(X, Y, epochs=self.epochs, batch_size=32, validation_split=0.15, shuffle=True,
                        callbacks=[stop], verbose=0)
        self.last_window_ = feats[-self.window:]
        self.last_date_ = series.index[-1]
        return self

    def predict(self, days: int) -> np.ndarray:
        if days > self.horizon:
            raise ValueError(f"this model forecasts at most {self.horizon} days")
        z = self.model_.predict(self.last_window_[None], verbose=0)[0][:days]
        return np.clip(np.expm1(z * self.std_ + self.mean_), 0, None)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.model_.save(folder / "lstm.keras")
        np.save(folder / "last_window.npy", self.last_window_)
        _write_meta(folder, self, {"window": self.window, "horizon": self.horizon, "units": self.units,
                                   "mean": self.mean_, "std": self.std_})

    @classmethod
    def load(cls, folder: Path) -> LstmForecaster:
        import tensorflow as tf

        meta = _read_meta(folder)
        model = cls(meta["window"], meta["horizon"], meta["units"])
        model.model_ = tf.keras.models.load_model(folder / "lstm.keras")
        model.last_window_ = np.load(folder / "last_window.npy")
        model.mean_, model.std_ = meta["mean"], meta["std"]
        model.last_date_ = pd.Timestamp(meta["last_date"])
        return model


KINDS = {cls.kind: cls for cls in (SeasonalNaive, Sarima, LstmForecaster)}


def load_forecaster(folder: Path):
    """Load whichever model was saved in `folder` (used by the Lambda handler)."""
    return KINDS[_read_meta(folder)["kind"]].load(folder)


def _write_meta(folder: Path, model, extra: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"kind": model.kind, "last_date": str(model.last_date_.date()), **extra}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))


def _read_meta(folder: Path) -> dict:
    return json.loads((Path(folder) / "metadata.json").read_text())
