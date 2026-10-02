"""Train, compare and persist the price model.   Run:  python -m src.train"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from .config import CATEGORICAL, FEATURES, META_PATH, MODEL_PATH, NUMERIC, TARGET, TARGET_ENC
from .data import load_clean
from .estimators import MonotoneExtraTrees, MonotoneRandomForest


def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),                       # car_age, kms_driven first (monotone constraints)
        ("te", TargetEncoder(random_state=42), TARGET_ENC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("est", estimator)])


CANDIDATES = {
    # baselines (no behavioural guarantees)
    "Ridge (baseline)": Ridge(alpha=1.0),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                                                   subsample=0.7, min_samples_leaf=5, random_state=42),
    # production candidates: price must fall as age / kms rise
    "Monotone Random Forest": MonotoneRandomForest(n_estimators=300, min_samples_leaf=2, max_features=0.4,
                                                   n_jobs=-1, random_state=42),
    "Monotone Extra Trees": MonotoneExtraTrees(n_estimators=400, min_samples_leaf=2, max_features=1.0,
                                               n_jobs=-1, random_state=42),
}
SELECTABLE = ["Monotone Random Forest", "Monotone Extra Trees"]


def cv_score(estimator, X, y, seeds=(1, 2, 3)):
    """Mean R2 / MAE over repeated 5-fold CV (more stable than one split on ~1k rows)."""
    r2, mae = [], []
    for seed in seeds:
        pred = cross_val_predict(build_pipeline(estimator), X, y, cv=KFold(5, shuffle=True, random_state=seed))
        r2.append(r2_score(y, pred))
        mae.append(mean_absolute_error(np.expm1(y), np.expm1(pred)))
    return float(np.mean(r2)), float(np.mean(mae))


def main():
    df = load_clean()
    X, y = df[FEATURES], np.log1p(df[TARGET])          # log target: prices are heavily right-skewed
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(5, shuffle=True, random_state=42)

    results = {}
    for name, est in CANDIDATES.items():
        r2, mae = cv_score(est, X_tr, y_tr)
        results[name] = {"cv_r2": r2, "cv_mae_lakh": mae}
        print(f"{name:24s} CV R2={results[name]['cv_r2']:.3f}  MAE={results[name]['cv_mae_lakh']:.2f} lakh")

    best = max(SELECTABLE, key=lambda k: results[k]["cv_r2"])
    # out-of-fold residuals on the training set give an honest prediction interval
    oof = cross_val_predict(build_pipeline(CANDIDATES[best]), X_tr, y_tr, cv=cv)
    resid = (y_tr - oof).to_numpy()
    lo_q, hi_q = np.quantile(resid, [0.10, 0.90])

    model = build_pipeline(CANDIDATES[best]).fit(X_tr, y_tr)
    test_pred = model.predict(X_te)
    test = {"r2": float(r2_score(y_te, test_pred)),
            "mae_lakh": float(mean_absolute_error(np.expm1(y_te), np.expm1(test_pred))),
            "median_ape_pct": float(np.median(np.abs(np.expm1(y_te) - np.expm1(test_pred)) / np.expm1(y_te)) * 100)}
    print(f"\nBest: {best}  | hold-out R2={test['r2']:.3f}  MAE={test['mae_lakh']:.2f} lakh  median error={test['median_ape_pct']:.1f}%")

    final = build_pipeline(CANDIDATES[best]).fit(X, y)  # refit on all data for deployment
    names = final.named_steps["pre"].get_feature_names_out()
    est = final.named_steps["est"]
    imp = getattr(est, "feature_importances_", None)
    importance = {}
    if imp is not None:
        for n, v in zip(names, imp):
            key = n.split("__")[1]
            grp = next((c for c in sorted(FEATURES, key=len, reverse=True) if key == c or key.startswith(c + "_")), key)
            importance[grp] = importance.get(grp, 0.0) + float(v)

    oof_all = cross_val_predict(build_pipeline(CANDIDATES[best]), X, y, cv=KFold(5, shuffle=True, random_state=7))
    oof_pairs = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(np.expm1(y), np.expm1(oof_all))]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps({
        "best_model": best, "cv_results": results, "holdout": test,
        "interval_log_q10": float(lo_q), "interval_log_q90": float(hi_q),
        "importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
        "n_rows": int(len(df)),
        "oof_actual_pred": oof_pairs,
    }, indent=2))
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()
