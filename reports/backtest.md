## Backtest: 6 rolling 30-day windows (last six months)

| Model | Mean daily MAE | vs baseline | Error on 30-day total |
|---|---|---|---|
| Seasonal naive (baseline) | £11,871 | +0% | 20.7% |
| SARIMA | £12,787 | +8% | 19.4% |
| LSTM | £11,673 | -2% | 24.8% |

## Per window

| Window start | Seasonal naive (baseline) MAE | SARIMA MAE | LSTM MAE |
|---|---|---|---|
| 2011-06-13 | £12,991 | £7,876 | £8,069 |
| 2011-07-13 | £7,813 | £9,198 | £7,214 |
| 2011-08-12 | £9,786 | £8,932 | £6,983 |
| 2011-09-11 | £15,918 | £21,245 | £15,832 |
| 2011-10-11 | £13,772 | £15,466 | £14,542 |
| 2011-11-10 | £10,945 | £14,003 | £17,396 |

SARIMA order: (2, 1, 2) x (0, 1, 1, 7). Best model: LSTM.
