import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from src.models.base import BaseModel
from src.masking import date_seed

def _trend_design(k, T):
    """Simple smile trend in z = k/sqrt(T): level, slope, curvature, and term structure."""
    z = k / np.sqrt(T)
    return np.column_stack([np.ones_like(k), z, z ** 2, np.sqrt(T), z * np.sqrt(T)])

class GPModel(BaseModel):
    """M3: Gaussian process fitted per day (no history).

    IV = trend(k, T) + GP residual. The trend (least squares in z = k/sqrt(T)) carries the
    model into the wings, where a zero-mean GP would fall back to a flat level.
    GP inputs: (k, sqrt(T)); kernel: Matern 5/2 with one length scale per input + noise.
    For speed the GP uses at most max_points visible quotes (random subsample).
    """
    name = 'gp'
    max_points = 800

    def fit(self, visible, history=None):
        k, T, iv = (visible[c].to_numpy(float) for c in ('k', 'T', 'iv'))
        self.beta = np.linalg.lstsq(_trend_design(k, T), iv, rcond=None)[0]
        resid = iv - _trend_design(k, T) @ self.beta

        rng = np.random.default_rng(date_seed(visible['quote_date'].iloc[0]))
        idx = rng.choice(len(k), size=min(self.max_points, len(k)), replace=False)
        X = np.column_stack([k, np.sqrt(T)])[idx]

        kernel = (ConstantKernel(1e-3, (1e-6, 1.0))
                  * Matern(length_scale=[0.1, 0.2], length_scale_bounds=(1e-3, 10.0), nu=2.5)
                  + WhiteKernel(1e-5, (1e-9, 1e-2)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                           n_restarts_optimizer=0, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            self.gp.fit(X, resid[idx])
        return self

    def predict(self, k, T, return_std=False):
        k, T = np.asarray(k, float), np.asarray(T, float)
        X = np.column_stack([k, np.sqrt(T)])
        mean = _trend_design(k, T) @ self.beta
        if return_std:
            r, sd = self.gp.predict(X, return_std=True)
            return np.maximum(mean + r, 0.01), sd
        return np.maximum(mean + self.gp.predict(X), 0.01)
