import json

import lambda_function
import pytest

from cashflow.models import SeasonalNaive


@pytest.fixture
def handler(weekly_series, tmp_path, monkeypatch):
    SeasonalNaive().fit(weekly_series).save(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(lambda_function, "_MODEL", None)
    return lambda_function.handler


def test_direct_invoke(handler, weekly_series):
    resp = handler({"days": 5})
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200 and len(body["forecast"]) == 5
    assert body["trained_until"] == str(weekly_series.index[-1].date())
    assert body["total"] == pytest.approx(sum(d["net_cash"] for d in body["forecast"]), abs=0.05)


def test_function_url_query_string(handler):
    body = json.loads(handler({"queryStringParameters": {"days": "3"}})["body"])
    assert [d["date"] for d in body["forecast"]] == sorted(d["date"] for d in body["forecast"])
    assert len(body["forecast"]) == 3


@pytest.mark.parametrize("days", [0, 31, "abc"])
def test_bad_input_is_rejected(handler, days):
    assert handler({"days": days})["statusCode"] == 400
