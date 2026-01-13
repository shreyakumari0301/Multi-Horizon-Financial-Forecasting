"""Ridge regression model for time series forecasting."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from .registry import register_model


class _SeqMaker:
    """Build fixed-length sequences with left-padding so we predict every time step."""
    def __init__(self, seq_len: int):
        self.seq_len = int(seq_len)

    def build(self, X: np.ndarray, y: np.ndarray | None = None):
        # X: (N, F), y: (N,) or (N,O) or None
        N, F = X.shape
        L = self.seq_len
        X_pad = np.vstack([np.repeat(X[0:1], L-1, axis=0), X])  # left-pad with first row
        X_seq = np.lib.stride_tricks.sliding_window_view(X_pad, (L, F))[:, 0, :]
        # X_seq: (N, L, F)
        if y is None:
            return X_seq, None
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        return X_seq, y


@register_model("ridge")
class RidgeRegressor:
    """
    Ridge regression with left-padded sliding windows.
    Flattens sequences to use all temporal information.
    .fit(X, y) and .predict(X) accept 2D arrays (N,F) and return aligned predictions (N,) or (N,O).
    """
    def __init__(self,
                 seq_len: int = 32,
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 val_frac: float = 0.1,
                 seed: int = 0):
        self.seq_len = seq_len
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.val_frac = val_frac
        self.seed = seed
        self._model = None
        self._seq = _SeqMaker(seq_len)
        self.out_dim_ = None
        self.in_dim_ = None
        self._fitted = False
        np.random.seed(self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        self.out_dim_ = y.shape[1]
        self.in_dim_ = X.shape[1]

        X_seq, y_seq = self._seq.build(X, y)  # (N, L, F), (N, O)

        # Flatten sequences: (N, L, F) -> (N, L*F)
        N, L, F = X_seq.shape
        X_flat = X_seq.reshape(N, L * F)

        # simple chronological split: last val_frac as validation
        n_val = max(1, int(self.val_frac * N))
        n_tr = N - n_val
        X_tr, Y_tr = X_flat[:n_tr], y_seq[:n_tr]
        X_va, Y_va = X_flat[n_tr:], y_seq[n_tr:]

        # Fit Ridge model
        model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, random_state=self.seed)
        model.fit(X_tr, Y_tr)

        # Evaluate on validation set
        y_va_pred = model.predict(X_va)
        va_mse = mean_squared_error(Y_va, y_va_pred)

        self._model = model
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32)
        X_seq, _ = self._seq.build(X, None)  # (N, L, F)
        
        # Flatten sequences: (N, L, F) -> (N, L*F)
        N, L, F = X_seq.shape
        X_flat = X_seq.reshape(N, L * F)
        
        y_hat = self._model.predict(X_flat)
        return y_hat.ravel() if self.out_dim_ == 1 else y_hat

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("RidgeRegressor is not fitted.")
