import numpy as np
from scipy.interpolate import griddata
from src.models.base import BaseModel

class LinearInterpModel(BaseModel):
    """B0: linear interpolation in (k, T); nearest-neighbour fill outside the data."""
    name = 'linear'

    def __init__(self, use_w=True):
        self.use_w = use_w              # interpolate total variance w instead of IV

    def fit(self, visible, history=None):
        self.points = visible[['k', 'T']].to_numpy()
        iv, T = visible['iv'].to_numpy(), visible['T'].to_numpy()
        self.values = iv**2 * T if self.use_w else iv
        return self

    def predict(self, k, T):
        xi = np.column_stack([k, T])
        out = griddata(self.points, self.values, xi, method='linear')
        nan = np.isnan(out)
        if nan.any():                                    # outside the visible hull
            out[nan] = griddata(self.points, self.values, xi[nan], method='nearest')
        if self.use_w:
            out = np.sqrt(np.maximum(out, 0) / np.asarray(T))  # w -> iv; clip w<0
        return out


class LinearInterpIVModel(LinearInterpModel):
    """Same, but interpolating IV directly (for comparison)."""
    name = 'linear_iv'
    def __init__(self):
        super().__init__(use_w=False)