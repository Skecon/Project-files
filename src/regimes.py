import numpy as np
import pandas as pd

REGIMES = [                                   # (name, start, end) inclusive
    ('covid_crash', '2020-02-20', '2020-04-30'),
    ('recovery',    '2020-05-01', '2020-12-31'),
    ('calm_2021',   '2021-01-01', '2021-12-31'),
    ('bear_2022',   '2022-01-01', '2022-12-31'),
]

def regime_of(dates):
    """Map dates (Series/array) to regime names. Before 2020-02-20 -> 'pre_covid'."""
    d = pd.to_datetime(pd.Series(dates))
    out = pd.Series('pre_covid', index=d.index)
    for name, start, end in REGIMES:
        out[(d >= start) & (d <= end)] = name
    return out.to_numpy()