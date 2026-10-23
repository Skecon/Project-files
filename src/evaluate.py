import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.metrics import rmse, mae, max_error

def history_for(data, test_day):
    """All rows strictly before the first day of test_day's month (walk-forward)."""
    month_start = pd.Timestamp(test_day).replace(day=1)
    return data[data['quote_date'] < month_start]

def _eval_day(model_class, day, history, mask_fn, mask_name, d):
    """Mask one day, fit on visible, score on hidden. Returns one results dict."""
    m = mask_fn(day)
    visible, hidden = day[~m], day[m]
    if len(hidden) == 0:
        return None
    model = model_class().fit(visible, history)
    pred = model.predict(hidden['k'].to_numpy(), hidden['T'].to_numpy())
    y = hidden['iv'].to_numpy()
    return {'date': d, 'model': model.name, 'mask': mask_name,
            'n_visible': len(visible), 'n_hidden': len(hidden),
            'rmse': rmse(y, pred), 'mae': mae(y, pred), 'max_error': max_error(y, pred)}

def evaluate(model_class, data, test_days, mask_fn, mask_name,
             n_jobs=-1, use_history=False):
    """Run model_class over test_days with mask_fn. One results row per day.
    n_jobs=-1 uses all CPU cores; use_history=True only for learned models."""
    test_days = [pd.Timestamp(d) for d in test_days]
    sub = data[data['quote_date'].isin(test_days)]
    by_day = {d: g for d, g in sub.groupby('quote_date')}     # split once, not per day

    jobs = [delayed(_eval_day)(model_class, by_day[d],
                               history_for(data, d) if use_history else None,
                               mask_fn, mask_name, d)
            for d in test_days if d in by_day]
    rows = Parallel(n_jobs=n_jobs)(jobs)
    return pd.DataFrame([r for r in rows if r is not None])

