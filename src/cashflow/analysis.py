"""Augmented Dickey-Fuller tests: which transformation makes the series stationary?"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def variants(series: pd.Series) -> dict[str, pd.Series]:
    log = np.log1p(series)
    return {
        "raw": series,
        "log1p": log,
        "log1p, weekly difference (lag 7)": log.diff(7).dropna(),
        "log1p, first + weekly difference": log.diff(7).diff().dropna(),
    }


def adf_table(series: pd.Series) -> pd.DataFrame:
    rows = []
    for name, s in variants(series).items():
        with warnings.catch_warnings():  # statsmodels >= 0.15 warns about a future return type
            warnings.simplefilter("ignore", FutureWarning)
            stat, p_value, *_ = adfuller(s, autolag="AIC")
        rows.append({"series": name, "adf_statistic": round(stat, 3), "p_value": round(p_value, 4),
                     "stationary_at_5pct": bool(p_value < 0.05)})
    return pd.DataFrame(rows)
