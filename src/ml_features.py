"""Features for the learned models, computed ONLY from a day's visible quotes.

The same function builds the training table (past days, masked the same way as test days)
and the features at test time, so training and testing see identical inputs.
"""
import os
from functools import partial
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.masking import random_mask, wing_mask, expiry_mask
from src.models.interp import LinearInterpModel
from src.models.svi import SVIModel
from src.models.ssvi import atm_theta

TRAIN_MASKS = {                      # different seeds from the test masks (seed=0)
    'random': partial(random_mask, seed=1),
    'wing':   wing_mask,
    'expiry': partial(expiry_mask, seed=1),
}
FEATURES = ['k', 'T', 'sqrt_T', 'z', 'atm_T', 'atm_30d', 'term_slope', 'skew',
            'dT_near', 'has_slice', 'dk_near', 'extrap_lo', 'extrap_hi', 'lin_pred']

def _atm_curve(visible):
    """ATM IV of every visible expiry, as (Ts, atm_iv), sorted by T."""
    rows = [(T, np.sqrt(max(atm_theta(g['k'].to_numpy(), (g['iv'] ** 2 * T).to_numpy()), 1e-8) / T))
            for T, g in visible.groupby('T', sort=True)]
    Ts, atm = map(np.array, zip(*rows))
    return Ts, atm

class DayContext:
    """Everything the features need from one day's visible quotes."""

    def __init__(self, visible, with_svi=False):
        self.Ts, self.atm = _atm_curve(visible)
        self.atm_30d = np.interp(30 / 365, self.Ts, self.atm)
        self.term_slope = np.interp(1.0, self.Ts, self.atm) - self.atm_30d
        z = visible['k'] / np.sqrt(visible['T'])
        near = z.abs() < 1
        self.skew = np.polyfit(z[near], visible['iv'][near], 1)[0] if near.sum() > 5 else 0.0
        self.k_by_T = {T: np.sort(g['k'].to_numpy()) for T, g in visible.groupby('T')}
        self.linear = LinearInterpModel().fit(visible)
        self.svi = SVIModel().fit(visible) if with_svi else None

    def features(self, k, T):
        k, T = np.asarray(k, float), np.asarray(T, float)
        f = pd.DataFrame({'k': k, 'T': T, 'sqrt_T': np.sqrt(T), 'z': k / np.sqrt(T)})
        f['atm_T'] = np.interp(T, self.Ts, self.atm)
        f['atm_30d'] = self.atm_30d
        f['term_slope'] = self.term_slope
        f['skew'] = self.skew

        # where is the point relative to the visible quotes?
        j = np.abs(T[:, None] - self.Ts[None, :]).argmin(axis=1)     # nearest visible expiry
        T_near = self.Ts[j]
        f['dT_near'] = np.abs(T - T_near)
        f['has_slice'] = np.isclose(T, T_near).astype(float)
        dk, lo, hi = np.empty(len(k)), np.empty(len(k)), np.empty(len(k))
        for t in np.unique(T_near):
            idx = T_near == t
            ks = self.k_by_T[t]
            pos = np.clip(np.searchsorted(ks, k[idx]), 1, len(ks) - 1)
            dk[idx] = np.minimum(np.abs(k[idx] - ks[pos - 1]), np.abs(k[idx] - ks[pos]))
            lo[idx] = np.maximum(ks[0] - k[idx], 0)
            hi[idx] = np.maximum(k[idx] - ks[-1], 0)
        f['dk_near'], f['extrap_lo'], f['extrap_hi'] = dk, lo, hi

        f['lin_pred'] = self.linear.predict(k, T)
        if self.svi is not None:
            f['svi_pred'] = self.svi.predict(k, T)
        return f

# ---------- training table ----------
def _day_rows(day, with_svi):
    out = []
    for mask_name, fn in TRAIN_MASKS.items():
        m = fn(day)
        visible, hidden = day[~m], day[m]
        if len(hidden) == 0 or visible['T'].nunique() < 2:
            continue
        ctx = DayContext(visible, with_svi=with_svi)
        f = ctx.features(hidden['k'].to_numpy(), hidden['T'].to_numpy())
        f['iv'] = hidden['iv'].to_numpy()
        f['date'] = day['quote_date'].iloc[0]
        f['mask'] = mask_name
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else None

def build_training_table(data, path, with_svi=True, n_jobs=-1):
    """Masked features + targets for every day in data; cached at path (parquet)."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    days = [g for _, g in data.groupby('quote_date')]
    parts = Parallel(n_jobs=n_jobs)(delayed(_day_rows)(d, with_svi) for d in days)
    table = pd.concat([p for p in parts if p is not None], ignore_index=True)
    table.to_parquet(path, index=False)
    return table
