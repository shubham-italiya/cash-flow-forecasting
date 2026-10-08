"""AWS Lambda handler: the next 1-30 days of forecast daily cash inflow.

Works with a direct invoke ({"days": 7}) or a Lambda Function URL (?days=7).
Model files are read from MODEL_DIR (local folder, used in tests) or MODEL_S3_URI (s3://bucket/prefix/),
downloaded once per container and cached in memory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists():  # running from the repository; in the container image the package sits next to this file
    sys.path.insert(0, str(_SRC))

from cashflow.models import load_forecaster  # noqa: E402

_MODEL = None


def _model_dir() -> Path:
    if os.environ.get("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"])
    import boto3

    bucket, _, prefix = os.environ["MODEL_S3_URI"].removeprefix("s3://").partition("/")
    dest = Path("/tmp/model")
    dest.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        name = obj["Key"][len(prefix):].lstrip("/")
        if name:
            s3.download_file(bucket, obj["Key"], str(dest / name))
    return dest


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_forecaster(_model_dir())
    return _MODEL


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def handler(event, context=None):
    params = (event or {}).get("queryStringParameters") or event or {}
    try:
        days = int(params.get("days", 30))
    except (TypeError, ValueError):
        return _response(400, {"error": "days must be a whole number"})
    if not 1 <= days <= 30:
        return _response(400, {"error": "days must be between 1 and 30"})

    model = _get_model()
    values = model.predict(days)
    dates = pd.date_range(model.last_date_ + pd.Timedelta(days=1), periods=days, freq="D")
    return _response(200, {
        "model": model.kind,
        "trained_until": str(model.last_date_.date()),
        "forecast": [{"date": str(d.date()), "net_cash": round(float(v), 2)}
                     for d, v in zip(dates, values, strict=True)],
        "total": round(float(values.sum()), 2),
    })
