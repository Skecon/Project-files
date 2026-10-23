class BaseModel:
    name = 'base'

    def fit(self, visible, history=None):
        """visible = today's unmasked quotes; history = past days (learned models only)."""
        raise NotImplementedError

    def predict(self, k, T):
        """Return predicted IVs (array) at points (k, T)."""
        raise NotImplementedError