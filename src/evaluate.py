import numpy as np
import pandas as pd
from src.metrics import rmse, mae, max_error

def history_for(data, test_day):
    """All rows strictly before the first day of test_day's month (walk-forward)."""
    month_start = pd.Timestamp(test_day).replace(day=1)
    return data[data['quote_date'] < month_start]

def evaluate(model_class, data, test_days, mask_fn, mask_name):
    """mask_fn(day) -> boolean Series (True = hidden). Returns one row per test day."""
    rows = []
    cached_month, history = None, None

    for d in test_days:
        d = pd.Timestamp(d)
        month = (d.year, d.month)
        if month != cached_month:                  # compute history once per month
            history = history_for(data, d)
            cached_month = month

        day = data[data['quote_date'] == d]
        m = mask_fn(day)
        visible, hidden = day[~m], day[m]
        if len(hidden) == 0:
            continue

        model = model_class().fit(visible, history)
        pred = model.predict(hidden['k'].to_numpy(), hidden['T'].to_numpy())
        y = hidden['iv'].to_numpy()

        rows.append({
            'date': d, 'model': model.name, 'mask': mask_name,
            'n_visible': len(visible), 'n_hidden': len(hidden),
            'rmse': rmse(y, pred), 'mae': mae(y, pred), 'max_error': max_error(y, pred),
        })

    return pd.DataFrame(rows)