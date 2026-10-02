"""Central configuration for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "car_dataset.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "meta.json"

CURRENT_YEAR = 2026
TARGET = "price_lakh"

CATEGORICAL = ["brand", "fuel_type", "transmission", "ownership", "insurance"]
TARGET_ENC = ["model"]            # high-cardinality -> target encoded inside the CV pipeline
NUMERIC = ["car_age", "kms_driven", "seats", "engine_cc"]
FEATURES = CATEGORICAL + TARGET_ENC + NUMERIC

OWNER_ORDER = ["First Owner", "Second Owner", "Third Owner", "Fourth Owner", "Fifth Owner"]
INSURANCE_MAP = {
    "Comprehensive": "Comprehensive",
    "Zero Dep": "Zero Dep",
    "Third Party": "Third Party",
    "Third Party insurance": "Third Party",
    "Not Available": "Not Available",
}
