"""Data loading and cleaning."""
import re
import pandas as pd
from .config import CURRENT_YEAR, DATA_PATH, INSURANCE_MAP

_TWO_WORD_MODELS = {"Grand", "Wagon", "Innova", "Alto", "Santro", "Range"}


def _registration_year(value) -> int:
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    yy = int(s.split("-")[1])
    return 2000 + yy if yy <= 30 else 1900 + yy


def _split_name(name: str):
    tokens = name.split()[1:]  # drop leading year
    brand = tokens[0]
    rest = tokens[1:]
    if brand == "Land" and rest and rest[0] == "Rover":
        brand, rest = "Land Rover", rest[1:]
    if not rest:
        return brand, "Other"
    model = rest[0]
    if brand in ("BMW", "Mini") and model.isdigit():
        model = f"{model} Series"
    elif model in _TWO_WORD_MODELS and len(rest) > 1 and not re.match(r"^\d\.\d|^[A-Z]{2,}$", rest[1]):
        model = f"{model} {rest[1]}"
    return brand, model


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    """Return a cleaned dataframe ready for modelling / analytics.

    Cleaning decisions (see README): mileage/engine/power/torque columns are
    corrupted in many rows (values shifted between columns) so they are not
    used; duplicate listings and absurd prices are removed.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"ownsership": "ownership", "insurance_validity": "insurance",
                            "price(in lakhs)": "price_lakh"})
    # engine(cc) is valid in ~91% of rows; elsewhere the column holds shifted garbage -> NaN, then imputed
    df["engine_cc"] = df["engine(cc)"].where(df["engine(cc)"].between(600, 6500))
    df["reg_year"] = df["registration_year"].map(_registration_year)
    df["car_age"] = (CURRENT_YEAR - df["reg_year"]).clip(lower=0)
    df["insurance"] = df["insurance"].map(INSURANCE_MAP).fillna("Not Available")
    df["ownership"] = df["ownership"].str.strip()
    names = df["car_name"].map(_split_name)
    df["brand"] = names.str[0]
    df["model"] = names.str[1]
    df["engine_cc"] = (df["engine_cc"].fillna(df.groupby("model")["engine_cc"].transform("median"))
                       .fillna(df.groupby("brand")["engine_cc"].transform("median"))
                       .fillna(df["engine_cc"].median()))
    df = df.drop_duplicates(subset=["car_name", "reg_year", "kms_driven", "price_lakh", "fuel_type",
                                    "transmission", "ownership"])
    df = df[df["price_lakh"].between(0.5, 150)]
    df = df[df["kms_driven"] < 500_000]
    keep = ["car_name", "brand", "model", "fuel_type", "transmission", "ownership", "insurance",
            "car_age", "kms_driven", "seats", "engine_cc", "reg_year", "price_lakh"]
    return df[keep].reset_index(drop=True)
