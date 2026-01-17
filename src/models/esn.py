"""
Echo State Network (ESN) for time series forecasting.

ESN is a reservoir computing approach that uses a randomly initialized
recurrent network (reservoir) and only trains the output layer with Ridge regression.
"""
import numpy as np
from sklearn.linear_model import Ridge
from .registry import register_model


class _SeqMaker:
    """Build fixed-length sequences with left-padding."""
    def __init__(self, seq_len: int):
        self.seq_len = int(seq_len)

    def build(self, X: np.ndarray, y: np.ndarray | None = None):
        N, F = X.shape
        L = self.seq_len
        X_pad = np.vstack([np.repeat(X[0:1], L-1, axis=0), X])
        X_seq = np.lib.stride_tricks.sliding_window_view(X_pad, (L, F))[:, 0, :]
        if y is None:
            return X_seq, None
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        return X_seq, y


@register_model("esn")
class ESNRegressor:
    """
    Echo State Network (ESN) for time series forecasting.
    
    Uses a randomly initialized reservoir and trains only the output layer.
    .fit(X, y) and .predict(X) accept 2D arrays (N,F) and return aligned predictions (N,) or (N,O).
    """
    def __init__(self,
                 seq_len: int = 32,
                 hidden_size: int = 400,
                 spectral_radius: float = 0.85,
                 leak_rate: float = 0.3,
                 ridge_alpha: float = 0.3,
                 washout: int = 100,
                 density: float = 0.1,
                 state_clip: float | None = None,
                 val_frac: float = 0.1,
                 seed: int = 0):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.washout = washout
        self.density = density
        self.state_clip = state_clip
        self.val_frac = val_frac
        self.seed = seed
        
        self._model = None
        self._seq = _SeqMaker(seq_len)
        self._reservoir_W = None  # Reservoir weight matrix
        self._input_W = None  # Input weight matrix
        self.out_dim_ = None
        self.in_dim_ = None
        self._fitted = False
        
        np.random.seed(self.seed)
    
    def _initialize_reservoir(self):
        """Initialize reservoir and input weight matrices."""
        # Input weight matrix (sparse, random)
        self._input_W = np.random.randn(self.hidden_size, self.in_dim_) * 0.1
        
        # Reservoir weight matrix (sparse, random)
        self._reservoir_W = np.random.randn(self.hidden_size, self.hidden_size)
        
        # Make sparse
        mask = np.random.rand(self.hidden_size, self.hidden_size) < self.density
        self._reservoir_W[~mask] = 0
        
        # Normalize to desired spectral radius
        eigenvals = np.linalg.eigvals(self._reservoir_W)
        max_eigenval = np.max(np.abs(eigenvals))
        if max_eigenval > 0:
            self._reservoir_W = self._reservoir_W * (self.spectral_radius / max_eigenval)
    
    def _compute_reservoir_states(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Compute reservoir states for input sequences.
        
        Args:
            X_seq: Input sequences (N, L, F)
        
        Returns:
            Reservoir states (N, hidden_size)
        """
        N, L, F = X_seq.shape
        
        # Initialize states
        states = np.zeros((N, L, self.hidden_size))
        
        # Process each sequence
        for n in range(N):
            # Initialize reservoir state
            reservoir_state = np.zeros(self.hidden_size)
            
            for t in range(L):
                # Input to reservoir
                input_vec = X_seq[n, t, :].reshape(-1, 1)
                input_activation = self._input_W @ input_vec
                
                # Reservoir update
                reservoir_state = (1 - self.leak_rate) * reservoir_state + \
                                 self.leak_rate * np.tanh(
                                     self._reservoir_W @ reservoir_state.reshape(-1, 1) + 
                                     input_activation
                                 ).ravel()
                
                # State clipping (optional)
                if self.state_clip is not None:
                    reservoir_state = np.clip(reservoir_state, -self.state_clip, self.state_clip)
                
                # Store state (after washout period)
                if t >= self.washout:
                    states[n, t, :] = reservoir_state
        
        # Return final states (after washout) or mean of states
        if L > self.washout:
            # Use states from after washout period
            final_states = states[:, self.washout:, :].mean(axis=1)  # (N, hidden_size)
        else:
            # If sequence too short, use last state
            final_states = states[:, -1, :]  # (N, hidden_size)
        
        return final_states
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32).copy()
        y = np.asarray(y, dtype=np.float32).copy()
        if y.ndim == 1:
            y = y[:, None]
        
        self.out_dim_ = y.shape[1]
        self.in_dim_ = X.shape[1]
        
        # Initialize reservoir with correct input dimension (only if not already initialized)
        if self._reservoir_W is None or self._input_W is None or self._input_W.shape[1] != self.in_dim_:
            self._initialize_reservoir()
        
        X_seq, y_seq = self._seq.build(X, y)  # (N, L, F), (N, O)
        
        # Compute reservoir states
        reservoir_states = self._compute_reservoir_states(X_seq)  # (N, hidden_size)
        
        # Chronological split
        N = reservoir_states.shape[0]
        n_val = max(1, int(self.val_frac * N))
        n_tr = N - n_val
        
        X_tr = reservoir_states[:n_tr]
        Y_tr = y_seq[:n_tr]
        X_va = reservoir_states[n_tr:]
        Y_va = y_seq[n_tr:]
        
        # Train Ridge regression on reservoir states
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True, random_state=self.seed)
        model.fit(X_tr, Y_tr)
        
        # Evaluate on validation
        y_va_pred = model.predict(X_va)
        va_mse = np.mean((Y_va - y_va_pred) ** 2)
        
        self._model = model
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32).copy()
        X_seq, _ = self._seq.build(X, None)  # (N, L, F)
        
        # Compute reservoir states
        reservoir_states = self._compute_reservoir_states(X_seq)  # (N, hidden_size)
        
        # Predict using Ridge model
        y_hat = self._model.predict(reservoir_states)
        return y_hat.ravel() if self.out_dim_ == 1 else y_hat
    
    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("ESNRegressor is not fitted.")
