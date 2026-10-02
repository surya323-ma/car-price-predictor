"""Tree ensembles that enforce: older age and more kilometres can never raise the predicted price."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


def _constraints(n_features: int) -> np.ndarray:
    c = np.zeros(n_features, dtype=int)
    c[0] = -1  # car_age     (first column of the preprocessor output)
    c[1] = -1  # kms_driven  (second column)
    return c


class MonotoneRandomForest(RandomForestRegressor):
    def fit(self, X, y, sample_weight=None):
        self.monotonic_cst = _constraints(X.shape[1])
        return super().fit(X, y, sample_weight)


class MonotoneExtraTrees(ExtraTreesRegressor):
    def fit(self, X, y, sample_weight=None):
        self.monotonic_cst = _constraints(X.shape[1])
        return super().fit(X, y, sample_weight)
