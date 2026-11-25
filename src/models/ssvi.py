import numpy as np
from scipy.optimize import least_squares
from src.models.base import BaseModel

# ---------- formula ----------
def ssvi_phi(theta, eta, gamma):
    """Power-law curvature function (Gatheral & Jacquier 2014, eq. 4.5)."""
    return eta / (theta ** gamma * (1 + theta) ** (1 - gamma))

def ssvi_w(k, theta, rho, eta, gamma):
    """SSVI total variance at log-moneyness k for an ATM total variance theta."""
    phi = ssvi_phi(theta, eta, gamma)
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))

def atm_theta(k, w):
    """ATM total variance of one slice: linear interpolation of w at k = 0 (flat outside)."""
    order = np.argsort(k)
    return float(np.interp(0.0, np.asarray(k)[order], np.asarray(w)[order]))

# ---------- model ----------
class SSVIModel(BaseModel):
    """B2: one SSVI surface per day.

    Free parameters: rho, eta, gamma (shared by all expiries) and one ATM total variance
    theta per visible expiry. theta is built as a cumulative sum of positive increments,
    so it always increases with T (no calendar arbitrage in theta). The Gatheral-Jacquier
    butterfly conditions theta*phi*(1+|rho|) <= 4 and theta*phi^2*(1+|rho|) <= 4 are added
    as penalty residuals. Missing maturities: theta is interpolated linearly in T.
    """
    name = 'ssvi'
    min_points = 3          # an expiry needs this many quotes to get its own theta
    penalty = 10.0          # weight of the butterfly-condition penalties

    def fit(self, visible, history=None):
        slices = [(T, g['k'].to_numpy(), (g['iv'] ** 2 * T).to_numpy())
                  for T, g in visible.groupby('T', sort=True) if len(g) >= self.min_points]
        self.fallback_iv = visible['iv'].mean()
        if not slices:
            self.Ts = np.array([])
            return self

        Ts = np.array([s[0] for s in slices])
        theta0 = np.maximum.accumulate([max(atm_theta(k, w), 1e-6) for _, k, w in slices])
        inc0 = np.diff(theta0, prepend=0.0)
        x0 = np.concatenate([[-0.6, 1.0, 0.4], np.log(np.maximum(inc0, 1e-7))])

        k_all = np.concatenate([s[1] for s in slices])
        T_all = np.concatenate([np.full(len(s[1]), s[0]) for s in slices])
        iv_all = np.sqrt(np.concatenate([s[2] for s in slices]) / T_all)
        slice_id = np.concatenate([np.full(len(s[1]), i) for i, s in enumerate(slices)])

        def residuals(x):
            rho, eta, gamma = x[:3]
            theta = np.cumsum(np.exp(x[3:]))
            th = theta[slice_id]
            w = ssvi_w(k_all, th, rho, eta, gamma)
            fit_res = np.sqrt(np.maximum(w, 1e-12) / T_all) - iv_all
            phi = ssvi_phi(theta, eta, gamma)
            c1 = np.maximum(theta * phi * (1 + abs(rho)) - 4, 0)
            c2 = np.maximum(theta * phi ** 2 * (1 + abs(rho)) - 4, 0)
            return np.concatenate([fit_res, self.penalty * c1, self.penalty * c2])

        lb = np.concatenate([[-0.999, 1e-4, 0.01], np.full(len(Ts), -25.0)])
        ub = np.concatenate([[0.999, 10.0, 0.99], np.full(len(Ts), 3.0)])
        res = least_squares(residuals, np.clip(x0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub),
                            method='trf', max_nfev=300, x_scale='jac')

        self.rho, self.eta, self.gamma = res.x[:3]
        self.Ts = Ts
        self.thetas = np.cumsum(np.exp(res.x[3:]))
        return self

    def theta_at(self, T):
        """theta(T): linear in T between expiries; constant ATM vol (theta ∝ T) outside."""
        T = np.asarray(T, float)
        out = np.interp(T, self.Ts, self.thetas)
        lo, hi = T < self.Ts[0], T > self.Ts[-1]
        out[lo] = self.thetas[0] * T[lo] / self.Ts[0]
        out[hi] = self.thetas[-1] * T[hi] / self.Ts[-1]
        return out

    def predict_w(self, k, T):
        return ssvi_w(np.asarray(k, float), self.theta_at(T), self.rho, self.eta, self.gamma)

    def predict(self, k, T):
        k, T = np.asarray(k, float), np.asarray(T, float)
        if len(self.Ts) == 0:
            return np.full(len(k), self.fallback_iv)
        return np.sqrt(np.maximum(self.predict_w(k, T), 0) / T)
