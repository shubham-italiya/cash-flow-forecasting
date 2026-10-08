"""Project paths, resolved from this file so commands work from any folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "online_retail_II.xlsx"
SERIES_FILE = DATA_DIR / "daily_net_cash.csv"
REPORTS_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts" / "model"
