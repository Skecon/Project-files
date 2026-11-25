from functools import partial
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.metrics import rmse, mae, max_error
from src.regimes import regime_of
from src.arbitrage import check_surface

def history_for(data, test_day):
    """All rows strictly before the first day of test_day's month (walk-forward)."""
    month_start = pd.Timestamp(test_day).replace(day=1)
    return data[data['quote_date'] < month_start]

def _eval_day(model_factory, day, history, mask_fn, mask_name, d, arbitrage=True):
    m = mask_fn(day)
    visible, hidden = day[~m], day[m]
    if len(hidden) == 0:
        return None, None
    model = model_factory().fit(visible, history)
    pred = model.predict(hidden['k'].to_numpy(), hidden['T'].to_numpy())
    y = hidden['iv'].to_numpy()
    score = {'date': d, 'model': model.name, 'mask': mask_name,
             'n_visible': len(visible), 'n_hidden': len(hidden),
             'rmse': rmse(y, pred), 'mae': mae(y, pred), 'max_error': max_error(y, pred)}
    if arbitrage:
        score.update(check_surface(model, day))
    preds = pd.DataFrame({'date': d, 'model': model.name, 'mask': mask_name,
                          'expire_date': hidden['expire_date'].to_numpy(),
                          'strike': hidden['strike'].to_numpy(),
                          'type': hidden['type'].to_numpy(),
                          'k': hidden['k'].to_numpy(), 'T': hidden['T'].to_numpy(),
                          'iv_true': y, 'iv_pred': pred})
    return score, preds

def make_factories(model_class, data, test_days):
    """For each test day, a no-argument function that builds a fresh model.

    Models with a `train` classmethod (learned models) are trained once per month on the
    walk-forward history; every day of that month reuses the trained state."""
    test_days = [pd.Timestamp(d) for d in test_days]
    if not hasattr(model_class, 'train'):
        return {d: model_class for d in test_days}
    out = {}
    for month in sorted({d.replace(day=1) for d in test_days}):
        state = model_class.train(history_for(data, month))
        for d in test_days:
            if d.replace(day=1) == month:
                out[d] = partial(model_class, state=state)
    return out

def evaluate(model_class, data, test_days, mask_fn, mask_name,
             n_jobs=-1, use_history=False, return_preds=False, arbitrage=True,
             factories=None):
    """One score row per day. With return_preds=True also returns every hidden point."""
    test_days = [pd.Timestamp(d) for d in test_days]
    sub = data[data['quote_date'].isin(test_days)]
    by_day = {d: g for d, g in sub.groupby('quote_date')}
    if factories is None:
        factories = make_factories(model_class, data, test_days)

    jobs = [delayed(_eval_day)(factories[d], by_day[d],
                               history_for(data, d) if use_history else None,
                               mask_fn, mask_name, d, arbitrage)
            for d in test_days if d in by_day]
    out = [o for o in Parallel(n_jobs=n_jobs)(jobs) if o[0] is not None]

    scores = pd.DataFrame([s for s, _ in out])
    if len(scores):
        scores['regime'] = regime_of(scores['date'])
    if not return_preds:
        return scores
    preds = pd.concat([p for _, p in out], ignore_index=True)
    preds['regime'] = regime_of(preds['date'])
    return scores, preds
