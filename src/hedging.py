"""One-day delta-hedging test.

For every hidden option, take the delta implied by a model's IV, hold the option against
-delta of SPY for one trading day, and measure the hedging error:

    error = (mid_next - mid_today) - delta * (S_next - S_today)

The same is done with the market's own IV ('true' delta). A better surface should give
hedging errors close to those of the market IV. Deltas use Black's formula on the forward
(discounting ignored: D ~ 1 over these horizons).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

def black_delta(F, K, T, iv, is_call):
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1)

def hedge_errors(preds, data):
    """preds: one model's preds.parquet (needs date, expire_date, strike, type, iv_pred, iv_true).
    data: options_features table. Returns one row per hedgeable option."""
    days = np.sort(data['quote_date'].unique())
    next_day = dict(zip(days[:-1], days[1:]))
    key = ['expire_date', 'strike', 'type']

    today = data[['quote_date'] + key + ['mid', 'F', 'underlying_last']]
    p = preds.merge(today, left_on=['date'] + key, right_on=['quote_date'] + key, how='inner')
    p['next_date'] = p['date'].map(next_day)

    tomorrow = data[['quote_date'] + key + ['mid', 'underlying_last']].rename(
        columns={'quote_date': 'next_date', 'mid': 'mid_next', 'underlying_last': 'S_next'})
    p = p.merge(tomorrow, on=['next_date'] + key, how='inner')

    is_call = (p['type'] == 'call').to_numpy()
    dS = p['S_next'] - p['underlying_last']
    dV = p['mid_next'] - p['mid']
    p['delta_model'] = black_delta(p['F'], p['strike'], p['T'], p['iv_pred'], is_call)
    p['delta_true'] = black_delta(p['F'], p['strike'], p['T'], p['iv_true'], is_call)
    p['err_model'] = dV - p['delta_model'] * dS
    p['err_true'] = dV - p['delta_true'] * dS
    p['err_unhedged'] = dV
    return p

def hedge_summary(errors):
    """RMSE of hedging error by mask, in dollars per option (lower = better)."""
    return errors.groupby('mask').apply(lambda g: pd.Series({
        'n': len(g),
        'rmse_model': np.sqrt((g['err_model'] ** 2).mean()),
        'rmse_true_iv': np.sqrt((g['err_true'] ** 2).mean()),
        'rmse_unhedged': np.sqrt((g['err_unhedged'] ** 2).mean()),
        'mean_abs_delta_gap': (g['delta_model'] - g['delta_true']).abs().mean(),
    }))
