import numpy as np
import pandas as pd

K_BINS = [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf]
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
    Returns (rmse_table, count_table), each k-bucket x T-bucket, RMSE in vol points."""
    k_b = pd.cut(df['k'], bins=K_BINS)
    t_b = pd.cut(df['T'], bins=T_BINS)
    g = df.groupby([k_b, t_b], observed=False)
    table = g.apply(lambda x: rmse(x['iv_true'], x['iv_pred']) if len(x) else np.nan).unstack()
    counts = g.size().unstack()
    return table, counts