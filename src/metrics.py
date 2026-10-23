import numpy as np
import pandas as pd

Z_BINS = [-np.inf, -0.4, -0.1, 0.1, 0.4, np.inf]
T_BINS = [0, 0.25, 1, np.inf]

def _err(y_true, y_pred):
    return np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)

def rmse(y_true, y_pred):
    return 100 * np.sqrt(np.mean(_err(y_true, y_pred) ** 2))

def mae(y_true, y_pred):
    return 100 * np.mean(np.abs(_err(y_true, y_pred)))

def max_error(y_true, y_pred):
    return 100 * np.max(np.abs(_err(y_true, y_pred)))

def regional_rmse(df):
    """df needs columns k, T, iv_true, iv_pred.
    Rows: z = k/sqrt(T) buckets (deep OTM put .. deep OTM call). Cols: T buckets.
    Returns (rmse_table, count_table), RMSE in vol points."""
    z = df['k'] / np.sqrt(df['T'])
    z_b = pd.cut(z, bins=Z_BINS)
    t_b = pd.cut(df['T'], bins=T_BINS)
    g = df.groupby([z_b, t_b], observed=False)
    table = g.apply(lambda x: rmse(x['iv_true'], x['iv_pred']) if len(x) else np.nan).unstack()
    counts = g.size().unstack()
    table.index.name = counts.index.name = 'z = k/sqrt(T)'
    return table, counts