import numpy as np

def check_surface(model, day, n_k=41):
    """Static-arbitrage check of a fitted model on a grid.

    Grid: n_k log-moneyness points between the 1st and 99th percentile of the day's k,
    and every maturity listed that day (visible and hidden).
    Butterfly: Gatheral-Jacquier g(k) >= 0, derivatives by central finite differences.
    Calendar: total variance w must not decrease from one maturity to the next.
    Returns rates (share of grid points violating) and mean violation size.
    """
    k = np.linspace(*np.quantile(day['k'], [0.01, 0.99]), n_k)
    Ts = np.sort(day['T'].unique())
    K, T = np.meshgrid(k, Ts)                                   # rows = maturities
    iv = np.asarray(model.predict(K.ravel(), T.ravel()), float).reshape(K.shape)
    w = iv ** 2 * T

    h = k[1] - k[0]
    w1 = (w[:, 2:] - w[:, :-2]) / (2 * h)
    w2 = (w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]) / h ** 2
    wc, kc = w[:, 1:-1], K[:, 1:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (1 - kc * w1 / (2 * wc)) ** 2 - w1 ** 2 / 4 * (1 / wc + 0.25) + w2 / 2
    g = g[np.isfinite(g)]
    bfly = g < -1e-6

    dw = np.diff(w, axis=0) if len(Ts) > 1 else np.zeros((0, n_k))
    cal = dw < -1e-6

    return {
        'butterfly_rate': bfly.mean() if g.size else np.nan,
        'butterfly_size': -g[bfly].mean() if bfly.any() else 0.0,
        'calendar_rate': cal.mean() if dw.size else np.nan,
        'calendar_size': -dw[cal].mean() if cal.any() else 0.0,
    }
