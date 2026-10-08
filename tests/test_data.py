import pandas as pd

from cashflow.data import daily_net_cash


def test_net_cash_counts_products_minus_refunds_and_fills_closed_days():
    tx = pd.DataFrame({
        "Invoice": ["1", "1", "C2", "3", "4"],
        "StockCode": ["85048", "POST", "85048", "21232", "22423"],
        "Quantity": [10, 1, -2, 5, 3],
        "Price": [2.0, 18.0, 2.0, 1.0, 0.0],
        "InvoiceDate": ["2021-01-04 09:00", "2021-01-04 09:00", "2021-01-04 15:00", "2021-01-06 10:00",
                        "2021-01-06 11:00"],
    })
    s = daily_net_cash(tx)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2021-01-04", "2021-01-05", "2021-01-06"]
    assert s.iloc[0] == 10 * 2.0 - 2 * 2.0   # postage left out, refund subtracted
    assert s.iloc[1] == 0.0                  # no trading that day
    assert s.iloc[2] == 5.0                  # zero-price line left out
