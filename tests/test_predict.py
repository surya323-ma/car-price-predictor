import pytest
from src.predict import CarSpec, Predictor, deal_verdict, emi


@pytest.fixture(scope="module")
def P():
    return Predictor()


def spec(**kw):
    base = dict(brand="Maruti", model="Swift", fuel_type="Petrol", transmission="Manual", ownership="First Owner",
                insurance="Comprehensive", car_age=6, kms_driven=40_000, seats=5, engine_cc=1197)
    return CarSpec(**{**base, **kw})


def test_range_contains_estimate(P):
    e = P.estimate(spec())
    assert 0 < e["low"] < e["price"] < e["high"]


def test_older_car_is_not_pricier(P):
    assert P.estimate(spec(car_age=12))["price"] <= P.estimate(spec(car_age=4))["price"]


def test_more_km_is_not_pricier(P):
    assert P.estimate(spec(kms_driven=150_000))["price"] <= P.estimate(spec(kms_driven=10_000))["price"]


def test_depreciation_curve_is_monotone(P):
    prices = P.depreciation(spec())["price"].tolist()
    assert all(a >= b - 1e-9 for a, b in zip(prices, prices[1:]))


def test_luxury_costs_more_than_hatchback(P):
    lux = P.estimate(spec(brand="Mercedes-Benz", model="C-Class", engine_cc=1950))["price"]
    assert lux > P.estimate(spec())["price"]


def test_emi_zero_interest():
    assert emi(6, 0, 5)["monthly"] == pytest.approx(10_000)


def test_deal_verdicts(P):
    e = P.estimate(spec())
    assert deal_verdict(e["price"], e)[2] == "good"
    assert deal_verdict(e["high"] * 1.5, e)[2] == "bad"
