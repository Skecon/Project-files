import numpy as np
import pandas as pd
from scipy import stats

def dm_test(loss_a, loss_b, lags=5):
    """Diebold-Mariano test on daily losses (same days, same order).
    d = loss_a - loss_b; negative mean d means model a is better.
    Variance uses Newey-West weights up to `lags` days. Returns (DM statistic, two-sided p)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dc = d - d.mean()
    var = np.dot(dc, dc) / n
    for L in range(1, lags + 1):
        var += 2 * (1 - L / (lags + 1)) * np.dot(dc[L:], dc[:-L]) / n
    dm = d.mean() / np.sqrt(var / n)
    return dm, 2 * stats.norm.sf(abs(dm))

def block_bootstrap(values, stat=np.mean, block=10, n_boot=2000, seed=0):
    """Moving-block bootstrap of a daily series (keeps short-run dependence).
    Returns the (2.5%, 97.5%) interval of stat."""
    x = np.asarray(values, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = stat(x[idx])
    return np.percentile(out, [2.5, 97.5])

def compare_to_benchmark(scores, benchmark='svi', lags=5, block=10, n_boot=2000):
    """For every model and mask: improvement over the benchmark and its significance.

    scores: concatenated scores.csv files (columns date, model, mask, rmse).
    Daily loss = RMSE^2 (mean squared error of that day's hidden points).
    improvement_% = 1 - mean RMSE(model) / mean RMSE(benchmark), with a block-bootstrap CI.
    """
    rows = []
    wide = scores.pivot_table(index=['mask', 'date'], columns='model', values='rmse')
    for mask, tab in wide.groupby(level='mask'):
        if benchmark not in tab:
            continue
        for model in tab.columns:
            if model == benchmark:
                continue
            pair = tab[[model, benchmark]].dropna()
            a, b = pair[model].to_numpy(), pair[benchmark].to_numpy()
            dm, p = dm_test(a ** 2, b ** 2, lags)
            ratio = lambda idx: 1 - a[idx].mean() / b[idx].mean()
            lo, hi = block_bootstrap(np.arange(len(a)), stat=lambda i: ratio(i.astype(int)),
                                     block=block, n_boot=n_boot)
            rows.append({'mask': mask, 'model': model, 'days': len(pair),
                         'rmse_model': a.mean(), 'rmse_' + benchmark: b.mean(),
                         'improvement_%': 100 * ratio(np.arange(len(a))),
                         'ci_low_%': 100 * lo, 'ci_high_%': 100 * hi,
                         'dm_stat': dm, 'p_value': p,
                         'win_rate_%': 100 * (a < b).mean()})
    return pd.DataFrame(rows).sort_values(['mask', 'rmse_model']).reset_index(drop=True)
