import numpy as np
from src.models.base import BaseModel

class MeanModel(BaseModel):
    name = 'mean'

    def fit(self, visible, history=None):
        self.mean_iv = visible['iv'].mean()
        return self

    def predict(self, k, T):
        return np.full(len(k), self.mean_iv)