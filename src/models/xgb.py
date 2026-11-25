import os
import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.base import BaseModel
from src.ml_features import DayContext, FEATURES

TABLE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed',
                          'ml_train_table.parquet')
_TABLE = None

def _table():
    global _TABLE
    if _TABLE is None:
        if not os.path.exists(TABLE_PATH):
            raise FileNotFoundError('Run src.ml_features.build_training_table first: ' + TABLE_PATH)
        _TABLE = pd.read_parquet(TABLE_PATH)
    return _TABLE

PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, tree_method='hist', n_jobs=-1)

class XGBModel(BaseModel):
    """Learned model: XGBoost trained once a month on past days (walk-forward).

    base = 'none'   -> predicts IV directly                         (M1a)
    base = 'linear' -> predicts the error of linear interpolation    (M1b)
    base = 'svi'    -> predicts the error of SVI                     (M2)
    Training rows come from past days masked the same way as test days.
    """
    name = 'xgb'
    base = 'none'

    def __init__(self, state=None):
        self.booster = state

    @classmethod
    def _cols(cls):
        return FEATURES + (['svi_pred'] if cls.base == 'svi' else [])

    @classmethod
    def _baseline(cls, f):
        if cls.base == 'none':
            return 0.0
        return f['lin_pred'].to_numpy() if cls.base == 'linear' else f['svi_pred'].to_numpy()

    @classmethod
    def train(cls, history):
        """Fit on every training-table row dated before the test month."""
        if len(history) == 0:
            return None
        tab = _table()
        tab = tab[tab['date'] <= history['quote_date'].max()].dropna(subset=cls._cols())
        target = tab['iv'].to_numpy() - cls._baseline(tab)
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(tab[cls._cols()], target)
        model.set_params(n_jobs=1)            # predictions run inside parallel workers
        return model

    def fit(self, visible, history=None):
        self.ctx = DayContext(visible, with_svi=(self.base == 'svi'))
        return self

    def predict(self, k, T):
        f = self.ctx.features(k, T)
        if self.booster is None:              # no history yet: fall back to the baseline
            return f['lin_pred'].to_numpy() if self.base != 'svi' else f['svi_pred'].to_numpy()
        out = self.booster.predict(f[self._cols()]) + self._baseline(f)
        return np.maximum(out, 0.01)

class XGBLinearModel(XGBModel):
    name = 'xgb_lin'
    base = 'linear'

class XGBSVIModel(XGBModel):
    name = 'xgb_svi'
    base = 'svi'
