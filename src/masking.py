import numpy as np
import pandas as pd

def date_seed(date, seed = 42) :
    return int(pd.Timestamp(date).strftime('%Y%m%d')) + seed

def random_mask(day, frac = 0.2, seed = 42, min_visible = 5) :
    rng = np.random.default_rng(date_seed(day['quote_date'].iloc[0], seed))
    mask = pd.Series(False, index = day.index)

    for _, chunk in day.groupby('expire_date', sort = True) :
        n = len(chunk)
        n_hide = int(round(frac * n))
        n_hide = min(n_hide, max(n - min_visible, 0))
        if n_hide == 0 :
            continue
        pos = rng.choice(n, size  = n_hide, replace = False)
        mask.loc[chunk.index[pos]] = True

    return mask

def wing_mask(day, frac_side = 0.10, min_visible = 5) :
    mask = pd.Series(False, index = day.index)

    for _, chunk in day.groupby('expire_date', sort = True) :
        n = len(chunk)
        chunk = chunk.sort_values('k')
        n_side = int(round(frac_side * n))
        n_side = min(n_side, max((n - min_visible)//2, 0))
        if n_side <= 0 :
            continue

        first, last = chunk.head(n_side), chunk.tail(n_side)
        mask.loc[first.index] = True
        mask.loc[last.index] = True
        
    return mask

def expiry_mask(day, seed = 42) :
    rng = np.random.default_rng(date_seed(day['quote_date'].iloc[0], seed))
    expires = np.sort(day['expire_date'].unique())
    if len(expires) < 3 :
        mask = pd.Series(False, index = day.index)
    else : 
        expires = expires[1:-1]
        chosen = rng.choice(expires)
        mask = day['expire_date'] == chosen

    return mask
