"""Turn about one million retail transactions into one daily cash series.

Dataset: "Online Retail II" (Chen, 2019), UCI Machine Learning Repository,
https://doi.org/10.24432/C5CG6D, licensed CC BY 4.0. A UK online gift retailer, Dec 2009 - Dec 2011.

Daily net cash inflow = product sales minus product refunds (cancelled invoices have negative quantities).
Postage, fees, bank charges and accounting adjustments are left out: they are not product sales.
"""
from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .paths import RAW_FILE, SERIES_FILE

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
PRODUCT_CODE = r"^\d{5}"  # real products have stock codes starting with five digits


def download(dest: Path = RAW_FILE) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "retail.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            name = next(n for n in z.namelist() if n.endswith(".xlsx"))
            dest.write_bytes(z.read(name))
    return dest


def read_transactions(path: Path = RAW_FILE) -> pd.DataFrame:
    """Both yearly sheets, with exact duplicate rows removed."""
    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True).drop_duplicates()


def daily_net_cash(tx: pd.DataFrame) -> pd.Series:
    """Net product sales per calendar day; days with no trading (e.g. Saturdays) are 0."""
    products = tx[tx["StockCode"].astype(str).str.match(PRODUCT_CODE) & (tx["Price"] > 0)]
    amount = products["Quantity"] * products["Price"]
    daily = amount.groupby(pd.to_datetime(products["InvoiceDate"]).dt.normalize()).sum()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(days, fill_value=0.0).rename("net_cash").rename_axis("date")


def load_series(refresh: bool = False) -> pd.Series:
    """The daily series, built once from the raw file and then read from a small CSV."""
    if SERIES_FILE.exists() and not refresh:
        s = pd.read_csv(SERIES_FILE, index_col="date", parse_dates=["date"])["net_cash"]
        return s.asfreq("D")
    s = daily_net_cash(read_transactions(download()))
    s.to_csv(SERIES_FILE)
    return s


def main() -> None:
    s = load_series(refresh=True)
    print(f"{s.index[0]:%d %b %Y} - {s.index[-1]:%d %b %Y}: {len(s)} days, "
          f"{(s > 0).sum()} trading days, mean £{s.mean():,.0f} per day")
    print(f"Saved {SERIES_FILE}")


if __name__ == "__main__":
    main()
