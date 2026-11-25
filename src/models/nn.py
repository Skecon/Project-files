import numpy as np
import torch
from torch import nn
from src.models.base import BaseModel
from src.models.ssvi import SSVIModel
from src.masking import date_seed

class _Net(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))
        nn.init.zeros_(self.body[-1].weight)          # start exactly at the SSVI prior
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, k, T):
        return self.body(torch.stack([k / 0.3, torch.sqrt(T)], dim=-1)).squeeze(-1)

class NNModel(BaseModel):
    """M4: neural network on top of an SSVI prior, fitted per day (as in Deep Smoothing,
    Ackerer et al. 2020).

        w(k, T) = w_SSVI(k, T) * exp(net(k, T))

    Loss = squared IV error on visible quotes (in vol points)
         + lam * (butterfly penalty + calendar penalty) on a grid of (k, T) points.
    Butterfly: Gatheral-Jacquier g(k) >= 0. Calendar: dw/dT >= 0. Derivatives by autograd.
    lam = 0 gives the same network without the no-arbitrage penalty.
    """
    name = 'nn'
    lam = 1.0
    steps = 300
    lr = 5e-3
    seed = 0

    def fit(self, visible, history=None):
        torch.set_num_threads(1)
        torch.manual_seed(date_seed(visible['quote_date'].iloc[0], self.seed))
        self.prior = SSVIModel().fit(visible)
        p = self.prior
        if len(p.Ts) == 0:
            self.net = None
            self.fallback_iv = visible['iv'].mean()
            return self
        self._Ts = torch.tensor(p.Ts, dtype=torch.float64)
        self._th = torch.tensor(p.thetas, dtype=torch.float64)

        k = torch.tensor(visible['k'].to_numpy(), dtype=torch.float64)
        T = torch.tensor(visible['T'].to_numpy(), dtype=torch.float64)
        iv = torch.tensor(visible['iv'].to_numpy(), dtype=torch.float64)

        # penalty grid: visible k range (+ margin) x all visible maturities and midpoints
        kg = torch.linspace(float(k.min()) - 0.05, float(k.max()) + 0.05, 30, dtype=torch.float64)
        Tv = np.sort(visible['T'].unique())
        Tg = torch.tensor(np.unique(np.concatenate([Tv, (Tv[1:] + Tv[:-1]) / 2])), dtype=torch.float64)
        KG, TG = torch.meshgrid(kg, Tg, indexing='ij')
        KG, TG = KG.reshape(-1).clone().requires_grad_(True), TG.reshape(-1).clone().requires_grad_(True)

        self.net = _Net().double()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            fit_loss = (((torch.sqrt(self._w(k, T) / T) - iv) * 100) ** 2).mean()
            loss = fit_loss + (self.lam * self._penalty(KG, TG) if self.lam > 0 else 0.0)
            loss.backward()
            opt.step()
        return self

    # ----- SSVI prior in torch (same formula as src.models.ssvi) -----
    def _theta(self, T):
        Ts, th = self._Ts, self._th
        j = torch.clamp(torch.searchsorted(Ts, T), 1, len(Ts) - 1) if len(Ts) > 1 else None
        if j is None:
            return th[0] * T / Ts[0]
        t0, t1, y0, y1 = Ts[j - 1], Ts[j], th[j - 1], th[j]
        inside = y0 + (y1 - y0) * (T - t0) / (t1 - t0)
        return torch.where(T < Ts[0], th[0] * T / Ts[0],
                           torch.where(T > Ts[-1], th[-1] * T / Ts[-1], inside))

    def _w(self, k, T):
        p = self.prior
        th = self._theta(T)
        phi = p.eta / (th ** p.gamma * (1 + th) ** (1 - p.gamma))
        w0 = th / 2 * (1 + p.rho * phi * k + torch.sqrt((phi * k + p.rho) ** 2 + 1 - p.rho ** 2))
        return w0 * torch.exp(self.net(k, T))

    def _penalty(self, k, T):
        w = self._w(k, T)
        wk, wT = torch.autograd.grad(w.sum(), (k, T), create_graph=True)
        wkk = torch.autograd.grad(wk.sum(), k, create_graph=True)[0]
        g = (1 - k * wk / (2 * w)) ** 2 - wk ** 2 / 4 * (1 / w + 0.25) + wkk / 2
        return torch.relu(-g).mean() + 10 * torch.relu(-wT).mean()

    def predict(self, k, T):
        k, T = np.asarray(k, float), np.asarray(T, float)
        if self.net is None:
            return np.full(len(k), self.fallback_iv)
        with torch.no_grad():
            kt, Tt = torch.tensor(k, dtype=torch.float64), torch.tensor(T, dtype=torch.float64)
            return torch.sqrt(torch.clamp(self._w(kt, Tt), min=1e-10) / Tt).numpy()

class NNNoPenaltyModel(NNModel):
    """Same network, no arbitrage penalty (ablation)."""
    name = 'nn_nopen'
    lam = 0.0
