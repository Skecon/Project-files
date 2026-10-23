import numpy as np
from scipy.optimize import minimize
from src.models.base import BaseModel

# ---------- formula + butterfly check ----------
def svi_w(k, a, b, rho, m, sigma):
    """Raw SVI total variance."""
    x = k - m
    return a + b * (rho * x + np.sqrt(x**2 + sigma**2))

def svi_g(k, a, b, rho, m, sigma):
    """Gatheral-Jacquier butterfly function. g(k) >= 0 everywhere = no butterfly arbitrage."""
    x = k - m
    r = np.sqrt(x**2 + sigma**2)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)                 # dw/dk
    w2 = b * sigma**2 / r**3               # d2w/dk2
    return (1 - k * w1 / (2 * w))**2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2

# ---------- one-slice fit ----------
STARTS = [(-0.5, 0.0, 0.1), (-0.7, -0.05, 0.05), (-0.3, 0.05, 0.2)]   # (rho, m, sigma)

def fit_slice(k, w, rel=False):
    """Fit raw SVI to one expiry. Returns (params [a, b, rho, m, sigma], rel_rmse).
    rel=False: minimise squared error in w.  rel=True: squared *relative* error (w_model-w)/w."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    n = len(k)
    scale = 1.0 if rel else np.mean(w) ** 2             # rel loss is already unit-free
    wt = 1 / w if rel else np.ones_like(w)              # per-point residual weight

    def loss_and_grad(p):
        a, b, rho, m, sigma = p
        x = k - m
        R = np.sqrt(x**2 + sigma**2)
        r = ((a + b * (rho * x + R)) - w) * wt          # (weighted) residuals
        dw = np.stack([np.ones_like(k),                 # d/da
                       rho * x + R,                     # d/db
                       b * x,                           # d/drho
                       -b * (rho + x / R),              # d/dm
                       b * sigma / R]) * wt             # d/dsigma, weighted too
        return np.mean(r**2) / scale, (2 / n / scale) * (dw @ r)

    cons = [
        {'type': 'ineq', 'fun': lambda p: p[0] + p[1] * p[4] * np.sqrt(1 - p[2]**2)},  # min w >= 0
        {'type': 'ineq', 'fun': lambda p: 2 - p[1] * (1 + p[2])},                       # Lee, right wing
        {'type': 'ineq', 'fun': lambda p: 2 - p[1] * (1 - p[2])},                       # Lee, left wing
    ]
    bounds = [(-1, max(w.max(), 1e-4)), (0, 2), (-0.999, 0.999), (-1, 1), (1e-4, 2)]

    best = None
    for rho0, m0, s0 in STARTS:
        p0 = [w.min() * 0.9, 0.1, rho0, m0, s0]
        res = minimize(loss_and_grad, p0, jac=True, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 200})
        if best is None or res.fun < best.fun:
            best = res
        if best.fun < 1e-6:
            break
    return best.x, np.sqrt(best.fun)

# ---------- model ----------
class SVIModel(BaseModel):
    """B1: raw SVI per expiry; linear-in-T interpolation of w for missing expiries."""
    name = 'svi'
    min_points = 5
    rel = False

    def fit(self, visible, history=None):
        Ts, params, diag = [], [], []
        for T, g in visible.groupby('T', sort=True):
            if len(g) < self.min_points:
                continue                                # too few to fit 5 params
            w = (g['iv'] ** 2 * T).to_numpy()
            p, err = fit_slice(g['k'].to_numpy(), w, rel=self.rel)
            Ts.append(T); params.append(p); diag.append((T, len(g), err))
        self.Ts = np.array(Ts)
        self.params = np.array(params)
        self.diag = diag                                # (T, n_points, rel_rmse) per slice
        self.fallback_iv = visible['iv'].mean()         # used only if nothing could be fit
        return self

    def _iv_at(self, k, t):
        Ts, P = self.Ts, self.params
        j = np.searchsorted(Ts, t)
        for jj in (j - 1, j):                           # exact slice?
            if 0 <= jj < len(Ts) and np.isclose(Ts[jj], t):
                return np.sqrt(np.maximum(svi_w(k, *P[jj]), 0) / t)
        if j == 0:                                      # below shortest slice: flat IV
            return np.sqrt(np.maximum(svi_w(k, *P[0]), 0) / Ts[0])
        if j == len(Ts):                                # above longest slice: flat IV
            return np.sqrt(np.maximum(svi_w(k, *P[-1]), 0) / Ts[-1])
        t0, t1 = Ts[j - 1], Ts[j]                       # between: interpolate w in T
        lam = (t - t0) / (t1 - t0)
        w = (1 - lam) * svi_w(k, *P[j - 1]) + lam * svi_w(k, *P[j])
        return np.sqrt(np.maximum(w, 0) / t)

    def predict(self, k, T):
        k, T = np.asarray(k, float), np.asarray(T, float)
        if len(self.Ts) == 0:
            return np.full(len(k), self.fallback_iv)
        out = np.empty_like(k)
        for t in np.unique(T):                          # one call per maturity
            idx = T == t
            out[idx] = self._iv_at(k[idx], t)
        return out

class SVIRelModel(SVIModel):
    """SVI fitted on relative error in w."""
    name = 'svi_rel'
    rel = True