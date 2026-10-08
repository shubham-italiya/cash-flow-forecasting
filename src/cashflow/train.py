"""Run the analysis and the backtest, then save the best model:  python -m cashflow.train

1. ADF tests on raw, log and differenced series -> reports/stationarity.md
2. SARIMA orders chosen by AIC on data before the first backtest window only (no peeking at test data).
3. Rolling backtest (6 x 30 days over the last six months) for the baseline, SARIMA and the LSTM.
4. The model with the lowest mean daily MAE is refitted on all data and saved to artifacts/model/.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import backtest  # noqa: E402
from .analysis import adf_table  # noqa: E402
from .data import load_series  # noqa: E402
from .models import LstmForecaster, Sarima, SeasonalNaive  # noqa: E402
from .paths import ARTIFACT_DIR, REPORTS_DIR  # noqa: E402

FOLDS, HORIZON = 6, 30
NAMES = {"seasonal_naive": "Seasonal naive (baseline)", "sarima": "SARIMA", "lstm": "LSTM"}


def main() -> None:
    started = time.time()
    series = load_series()
    REPORTS_DIR.mkdir(exist_ok=True)
    print(f"{len(series)} days, {series.index[0]:%d %b %Y} - {series.index[-1]:%d %b %Y}")

    adf = adf_table(series)
    table = ["| Series | ADF statistic | p-value | Stationary at 5% |", "|---|---|---|---|"]
    table += [f"| {r.series} | {r.adf_statistic} | {r.p_value} | {'yes' if r.stationary_at_5pct else 'no'} |"
              for r in adf.itertuples()]
    (REPORTS_DIR / "stationarity.md").write_text("\n".join(table) + "\n")
    print(adf.to_string(index=False))

    first_test = backtest.origins(series, FOLDS, HORIZON)[0]
    order, seasonal = Sarima.select_order(series.iloc[:first_test])
    print(f"SARIMA order chosen on pre-backtest data: {order} x {seasonal}")

    factories = {
        "seasonal_naive": SeasonalNaive,
        "sarima": lambda: Sarima(order, seasonal),
        "lstm": lambda: LstmForecaster(horizon=HORIZON),
    }
    results = {}
    for kind, make in factories.items():
        t = time.time()
        results[kind] = backtest.run(make, series, FOLDS, HORIZON)
        r = results[kind]
        print(f"  {NAMES[kind]:<26} MAE £{r['mae'].mean():>7,.0f}"
              f"   30-day total error {r['total_error_pct'].mean():5.1f}%   ({time.time() - t:.0f}s)")

    best = min(results, key=lambda k: results[k]["mae"].mean())
    base = results["seasonal_naive"]["mae"].mean()
    write_reports(series, results, best, base, order, seasonal)

    final = factories[best]().fit(series)
    final.save(ARTIFACT_DIR)
    print(f"Best: {NAMES[best]} ({100 * (1 - results[best]['mae'].mean() / base):.0f}% lower MAE than the baseline). "
          f"Saved to {ARTIFACT_DIR}. Done in {time.time() - started:.0f}s.")


def write_reports(series, results, best, base_mae, order, seasonal) -> None:
    rows = ["| Model | Mean daily MAE | vs baseline | Error on 30-day total |", "|---|---|---|---|"]
    for k, r in results.items():
        mae = r["mae"].mean()
        change = 100 * (mae / base_mae - 1)
        rows.append(f"| {NAMES[k]} | £{mae:,.0f} | {change:+.0f}% | {r['total_error_pct'].mean():.1f}% |")
    folds = ["| Window start | " + " | ".join(NAMES[k] + " MAE" for k in results) + " |",
             "|---|" + "---|" * len(results)]
    for i in range(len(next(iter(results.values())))):
        start = next(iter(results.values()))["fold_start"].iloc[i]
        folds.append(f"| {start} | " + " | ".join(f"£{r['mae'].iloc[i]:,.0f}" for r in results.values()) + " |")
    text = ["## Backtest: 6 rolling 30-day windows (last six months)", "", *rows, "",
            "## Per window", "", *folds, "", f"SARIMA order: {order} x {seasonal}. Best model: {NAMES[best]}."]
    (REPORTS_DIR / "backtest.md").write_text("\n".join(text) + "\n")
    metrics = {k: {"mean_daily_mae": float(r["mae"].mean()),
                   "mean_total_error_pct": float(r["total_error_pct"].mean()),
                   "per_fold_mae": [float(x) for x in r["mae"]]} for k, r in results.items()}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({"best": best, "sarima_order": [order, seasonal],
                                                         "models": metrics}, indent=2))

    # Plot: the last four windows, actual vs forecasts.
    fig, ax = plt.subplots(figsize=(10, 4))
    starts = backtest.origins(series, len(next(iter(results.values()))), HORIZON)
    window = series.iloc[starts[-4]:]
    ax.plot(window.index, window.to_numpy() / 1000, color="black", lw=1.4, label="Actual")
    colours = {"seasonal_naive": "#a8a29e", "sarima": "#2563eb", "lstm": "#ea580c"}
    for k, r in results.items():
        for j, start in enumerate(starts[-4:]):
            idx = series.index[start:start + HORIZON]
            ax.plot(idx, r["forecast"].iloc[len(starts) - 4 + j] / 1000, color=colours[k], lw=1,
                    alpha=0.9, label=NAMES[k] if j == 0 else None)
    for start in starts[-4:]:
        ax.axvline(series.index[start], color="grey", lw=0.5, ls=":")
    ax.set_ylabel("Net cash inflow per day (£ thousands)")
    ax.set_title("30-day forecasts in the last four backtest windows")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "backtest_forecasts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.to_numpy() / 1000, lw=0.8, color="#2563eb")
    ax.set_ylabel("£ thousands per day")
    ax.set_title("Daily net cash inflow (product sales minus refunds)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "series.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
