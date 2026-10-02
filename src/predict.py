"""Prediction logic: price estimate, range, what-if analysis, deal check, EMI, similar listings."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from .config import CURRENT_YEAR, FEATURES, META_PATH, MODEL_PATH
from .data import load_clean


@dataclass
class CarSpec:
    brand: str
    model: str
    fuel_type: str
    transmission: str
    ownership: str
    insurance: str
    car_age: int
    kms_driven: int
    seats: int
    engine_cc: float

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])[FEATURES]

    @property
    def title(self) -> str:
        return f"{CURRENT_YEAR - self.car_age} {self.brand} {self.model}"


class Predictor:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception:  # missing file or sklearn version mismatch -> retrain transparently
            from . import train
            train.main()
            self.model = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.data = load_clean()

    # ---------- core ----------
    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame)

    def estimate(self, spec: CarSpec) -> dict:
        z = float(self._raw(spec.frame())[0])
        q10, q90 = self.meta["interval_log_q10"], self.meta["interval_log_q90"]
        return {"price": float(np.expm1(z)), "low": float(np.expm1(z + q10)), "high": float(np.expm1(z + q90))}

    def _price_many(self, specs: list[CarSpec]) -> np.ndarray:
        frame = pd.concat([s.frame() for s in specs], ignore_index=True)
        return np.expm1(self._raw(frame))

    # ---------- analysis ----------
    def depreciation(self, spec: CarSpec, max_age: int = 15) -> pd.DataFrame:
        """Estimated price by age, assuming ~10,000 km/year of usage from the car's current pace."""
        pace = spec.kms_driven / max(spec.car_age, 1)
        pace = float(np.clip(pace, 4_000, 20_000))
        ages = list(range(1, max_age + 1))
        specs = [CarSpec(**{**asdict(spec), "car_age": a, "kms_driven": int(pace * a)}) for a in ages]
        return pd.DataFrame({"age": ages, "price": self._price_many(specs)})

    def what_if(self, spec: CarSpec) -> pd.DataFrame:
        base = self.estimate(spec)["price"]
        variants = {
            "10,000 km more driven": {"kms_driven": spec.kms_driven + 10_000},
            "10,000 km less driven": {"kms_driven": max(spec.kms_driven - 10_000, 0)},
            "One year older": {"car_age": spec.car_age + 1},
            "One owner more": {"ownership": _next_owner(spec.ownership, +1)},
            "One owner fewer": {"ownership": _next_owner(spec.ownership, -1)},
            "Switch transmission": {"transmission": "Automatic" if spec.transmission == "Manual" else "Manual"},
        }
        rows = []
        for label, change in variants.items():
            new = CarSpec(**{**asdict(spec), **change})
            if new == spec:
                continue
            delta = float(self._price_many([new])[0]) - base
            rows.append({"Change": label, "Price impact (lakh)": delta})
        return pd.DataFrame(rows).sort_values("Price impact (lakh)")

    def future_values(self, spec: CarSpec, years=(1, 2, 3, 5)) -> dict[int, float]:
        pace = float(np.clip(spec.kms_driven / max(spec.car_age, 1), 4_000, 20_000))
        specs = [CarSpec(**{**asdict(spec), "car_age": spec.car_age + y,
                            "kms_driven": int(spec.kms_driven + pace * y)}) for y in years]
        return dict(zip(years, self._price_many(specs)))

    def similar(self, spec: CarSpec, n: int = 6) -> pd.DataFrame:
        d = self.data
        pool = d[d["model"] == spec.model]
        if len(pool) < n:
            pool = d[d["brand"] == spec.brand]
        score = (pool["car_age"] - spec.car_age).abs() / 3 + (pool["kms_driven"] - spec.kms_driven).abs() / 40_000
        score += (pool["fuel_type"] != spec.fuel_type) * 0.5 + (pool["transmission"] != spec.transmission) * 0.5
        out = pool.assign(_s=score).sort_values("_s").head(n)
        return out[["car_name", "fuel_type", "transmission", "kms_driven", "ownership", "price_lakh"]]

    # ---------- catalogue for UI defaults ----------
    def catalogue(self) -> dict:
        d = self.data
        cat = {}
        for brand, g in d.groupby("brand"):
            cat[brand] = {}
            for model, m in g.groupby("model"):
                cat[brand][model] = {
                    "fuels": m["fuel_type"].value_counts().index.tolist(),
                    "transmissions": m["transmission"].value_counts().index.tolist(),
                    "seats": int(m["seats"].mode().iloc[0]),
                    "engine": int(round(m["engine_cc"].median(), -1)),
                    "median_price": float(m["price_lakh"].median()),
                    "n": int(len(m)),
                }
        return cat


def _next_owner(current: str, step: int) -> str:
    from .config import OWNER_ORDER
    i = int(np.clip(OWNER_ORDER.index(current) + step, 0, len(OWNER_ORDER) - 1))
    return OWNER_ORDER[i]


def emi(principal_lakh: float, annual_rate_pct: float, years: int) -> dict:
    p = principal_lakh * 100_000
    r = annual_rate_pct / 12 / 100
    n = years * 12
    monthly = p / n if r == 0 else p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return {"monthly": monthly, "total": monthly * n, "interest": monthly * n - p}


def deal_verdict(asking: float, est: dict) -> tuple[str, str, str]:
    """Return (label, explanation, tone) for an asking price against the estimated fair range."""
    p, lo, hi = est["price"], est["low"], est["high"]
    gap = (asking - p) / p * 100
    if asking < lo:
        return ("Below market", f"{abs(gap):.0f}% under the estimate. Inspect carefully for hidden issues or paperwork gaps.", "good")
    if asking <= p * 1.05:
        return ("Good deal", f"Within {abs(gap):.0f}% of the estimate, a fair price for this car.", "good")
    if asking <= hi:
        return ("Fair, negotiable", f"{gap:.0f}% above the estimate but inside the normal range. Offer near ₹{p:.2f} lakh.", "warn")
    return ("Overpriced", f"{gap:.0f}% above the estimate and outside the usual range. Aim for ₹{p:.2f} lakh or less.", "bad")
