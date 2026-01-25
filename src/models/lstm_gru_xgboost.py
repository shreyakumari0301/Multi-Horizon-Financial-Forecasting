"""LSTM+GRU+XGBoost hybrid model for time series forecasting."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBRegressor
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


class _LSTMGRUFeatureExtractor(nn.Module):
    """Extract features using LSTM and GRU in parallel."""
    def __init__(self, in_dim, lstm_hidden=128, gru_hidden=128, 
                 lstm_layers=2, gru_layers=2, dropout=0.0):
        super().__init__()
        # LSTM branch
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=lstm_hidden, 
                           num_layers=lstm_layers,
                           dropout=(dropout if lstm_layers > 1 else 0.0),
                           batch_first=True)
        # GRU branch
        self.gru = nn.GRU(input_size=in_dim, hidden_size=gru_hidden, 
                         num_layers=gru_layers,
                         dropout=(dropout if gru_layers > 1 else 0.0),
                         batch_first=True)
        
    def forward(self, x):  # x: (B, L, F)
        # Process through LSTM
        lstm_out, (lstm_hn, _) = self.lstm(x)  # lstm_hn: (lstm_layers, B, H)
        lstm_last = lstm_hn[-1]  # (B, lstm_hidden)
        
        # Process through GRU
        gru_out, gru_hn = self.gru(x)  # gru_hn: (gru_layers, B, H)
        gru_last = gru_hn[-1]  # (B, gru_hidden)
        
        # Also extract sequence-level statistics from outputs
        lstm_mean = lstm_out.mean(dim=1)  # (B, lstm_hidden)
        lstm_std = lstm_out.std(dim=1)    # (B, lstm_hidden)
        gru_mean = gru_out.mean(dim=1)    # (B, gru_hidden)
        gru_std = gru_out.std(dim=1)      # (B, gru_hidden)
        
        # Concatenate all features
        features = torch.cat([
            lstm_last, lstm_mean, lstm_std,
            gru_last, gru_mean, gru_std
        ], dim=1)  # (B, 2*lstm_hidden + 2*gru_hidden + 2*lstm_hidden + 2*gru_hidden)
        # Actually: (B, lstm_hidden + lstm_hidden + lstm_hidden + gru_hidden + gru_hidden + gru_hidden)
        # = (B, 3*lstm_hidden + 3*gru_hidden)
        
        return features


@register_model("lstm_gru_xgboost")
class LSTMGRUXGBoostRegressor:
    """
    Hybrid LSTM+GRU+XGBoost model for time series forecasting.
    
    Architecture:
    1. LSTM and GRU extract sequence features in parallel
    2. Features from both networks are concatenated
    3. XGBoost uses these features + original features for final prediction
    
    .fit(X, y) and .predict(X) accept 2D arrays (N,F) and return aligned predictions (N,) or (N,O).
    """
    def __init__(self,
                 seq_len: int = 32,
                 lstm_hidden: int = 128,
                 gru_hidden: int = 128,
                 lstm_layers: int = 2,
                 gru_layers: int = 2,
                 dropout: float = 0.0,
                 xgb_n_estimators: int = 100,
                 xgb_max_depth: int = 6,
                 xgb_learning_rate: float = 0.1,
                 xgb_subsample: float = 0.8,
                 xgb_colsample_bytree: float = 0.8,
                 use_delta_target: bool = False,
                 val_frac: float = 0.1,
                 seed: int = 0,
                 device: str | None = None):
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        self.gru_hidden = gru_hidden
        self.lstm_layers = lstm_layers
        self.gru_layers = gru_layers
        self.dropout = dropout
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.use_delta_target = use_delta_target
        self.val_frac = val_frac
        self.seed = seed
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_extractor = None
        self._xgb_model = None
        self._seq = _SeqMaker(seq_len)
        self.out_dim_ = None
        self.in_dim_ = None
        self._fitted = False
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _extract_features(self, X_seq: np.ndarray) -> np.ndarray:
        """Extract features using LSTM+GRU network."""
        self._feature_extractor.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            features = self._feature_extractor(X_tensor).cpu().numpy()
        return features

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        
        # Convert to delta target if requested (y[t] - y[t-1])
        if self.use_delta_target:
            y_delta = np.zeros_like(y)
            y_delta[1:] = y[1:] - y[:-1]
            y_delta[0] = y[0]  # First value stays same
            y = y_delta
        
        self.out_dim_ = y.shape[1]
        self.in_dim_ = X.shape[1]

        X_seq, y_seq = self._seq.build(X, y)  # (N, L, F), (N, O)

        # Chronological split: last val_frac as validation
        N = X_seq.shape[0]
        n_val = max(1, int(self.val_frac * N))
        n_tr = N - n_val
        X_tr_seq, Y_tr = X_seq[:n_tr], y_seq[:n_tr]
        X_va_seq, Y_va = X_seq[n_tr:], y_seq[n_tr:]
        X_tr_flat = X[:n_tr]  # Original features for XGBoost
        X_va_flat = X[n_tr:]

        # Step 1: Train LSTM+GRU feature extractor
        print("  Training LSTM+GRU feature extractor...")
        feature_extractor = _LSTMGRUFeatureExtractor(
            in_dim=self.in_dim_,
            lstm_hidden=self.lstm_hidden,
            gru_hidden=self.gru_hidden,
            lstm_layers=self.lstm_layers,
            gru_layers=self.gru_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Split training data for feature extractor training
        n_fe_tr = max(1, int(0.8 * len(X_tr_seq)))
        X_fe_tr = X_tr_seq[:n_fe_tr]
        Y_fe_tr = Y_tr[:n_fe_tr]
        X_fe_va = X_tr_seq[n_fe_tr:]
        Y_fe_va = Y_tr[n_fe_tr:]
        
        opt = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        loss_fn = nn.MSELoss()
        
        def make_loader(Xa, Ya, bs, shuffle):
            Xa = np.asarray(Xa, dtype=np.float32).copy()
            Ya = np.asarray(Ya, dtype=np.float32).copy()
            ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(Ya))
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)
        
        tr_loader = make_loader(X_fe_tr, Y_fe_tr, 128, True)
        va_loader = make_loader(X_fe_va, Y_fe_va, 128, False)
        
        # Simple linear head for feature extractor training
        linear_head = nn.Linear(3 * self.lstm_hidden + 3 * self.gru_hidden, self.out_dim_).to(self.device)
        opt_head = torch.optim.Adam(linear_head.parameters(), lr=1e-3)
        
        best_va = np.inf
        best_state = None
        patience_counter = 0
        early_stop_patience = 10
        
        for ep in range(30):  # Train feature extractor for fewer epochs
            feature_extractor.train()
            linear_head.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                opt_head.zero_grad()
                feats = feature_extractor(xb)
                pred = linear_head(feats)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(feature_extractor.parameters()) + list(linear_head.parameters()), max_norm=1.0)
                opt.step()
                opt_head.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_loader.dataset)
            
            feature_extractor.eval()
            linear_head.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    feats = feature_extractor(xb)
                    pred = linear_head(feats)
                    va_loss += loss_fn(pred, yb).item() * len(xb)
            va_loss /= len(va_loader.dataset)
            
            scheduler.step(va_loss)
            
            if va_loss < best_va:
                best_va = va_loss
                best_state = {k: v.cpu().clone() for k, v in feature_extractor.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
        
        if best_state is not None:
            feature_extractor.load_state_dict(best_state)
        
        self._feature_extractor = feature_extractor
        
        # Step 2: Extract features from all training data
        print("  Extracting LSTM+GRU features...")
        X_tr_features = self._extract_features(X_tr_seq)  # (n_tr, feature_dim)
        X_va_features = self._extract_features(X_va_seq)  # (n_val, feature_dim)
        
        # Step 3: Combine with original features
        X_tr_combined = np.hstack([X_tr_flat, X_tr_features])  # (n_tr, F + feature_dim)
        X_va_combined = np.hstack([X_va_flat, X_va_features])  # (n_val, F + feature_dim)
        
        # Step 4: Train XGBoost on combined features
        print("  Training XGBoost on combined features...")
        xgb_model = XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=0
        )
        
        # Handle multi-output
        if self.out_dim_ == 1:
            xgb_model.fit(X_tr_combined, Y_tr.ravel())
        else:
            # Multi-output: train separate models or use MultiOutputRegressor
            from sklearn.multioutput import MultiOutputRegressor
            xgb_model = MultiOutputRegressor(xgb_model)
            xgb_model.fit(X_tr_combined, Y_tr)
        
        self._xgb_model = xgb_model
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32)
        X_seq, _ = self._seq.build(X, None)  # (N, L, F)
        
        # Extract features
        X_features = self._extract_features(X_seq)  # (N, feature_dim)
        
        # Combine with original features
        X_combined = np.hstack([X, X_features])  # (N, F + feature_dim)
        
        # Predict with XGBoost
        y_hat = self._xgb_model.predict(X_combined)
        return y_hat.ravel() if self.out_dim_ == 1 else y_hat

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("LSTMGRUXGBoostRegressor is not fitted.")
