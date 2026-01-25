"""Temporal Convolutional Network (TCN) model for time series forecasting."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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


class Chomp1d(nn.Module):
    """Remove padding from the right side."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal block with dilated convolution, normalization, and dropout."""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCNHead(nn.Module):
    def __init__(self, in_dim, channels=(64, 64), kernel_size=3, dropout=0.0, out_dim=1):
        super().__init__()
        layers = []
        num_levels = len(channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            
            layers += [
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], out_dim)
        )

    def forward(self, x):  # x: (B, L, F)
        x = x.transpose(1, 2)  # (B, F, L) - TCN expects (batch, features, seq_len)
        x = self.network(x)  # (B, C, L)
        h_last = x[:, :, -1]  # (B, C) - take last timestep
        return self.head(h_last)  # (B, O)


@register_model("tcn")
class TCNRegressor:
    """
    TCN model with left-padded sliding windows.
    .fit(X, y) and .predict(X) accept 2D arrays (N,F) and return aligned predictions (N,) or (N,O).
    """
    def __init__(self,
                 seq_len: int = 32,
                 channels: tuple = (64, 64),
                 kernel_size: int = 3,
                 dropout: float = 0.0,
                 epochs: int = 10,
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 use_delta_target: bool = False,
                 val_frac: float = 0.1,
                 seed: int = 0,
                 device: str | None = None):
        self.seq_len = seq_len
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_delta_target = use_delta_target
        self.val_frac = val_frac
        self.seed = seed
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._seq = _SeqMaker(seq_len)
        self.out_dim_ = None
        self.in_dim_ = None
        self._fitted = False
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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

        # simple chronological split: last val_frac as validation
        N = X_seq.shape[0]
        n_val = max(1, int(self.val_frac * N))
        n_tr  = N - n_val
        X_tr, Y_tr = X_seq[:n_tr], y_seq[:n_tr]
        X_va, Y_va = X_seq[n_tr:], y_seq[n_tr:]

        model = _TCNHead(in_dim=self.in_dim_, channels=self.channels,
                         kernel_size=self.kernel_size, dropout=self.dropout,
                         out_dim=self.out_dim_).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )
        loss_fn = nn.MSELoss()

        def make_loader(Xa, Ya, bs, shuffle):
            # Make arrays writable by copying to avoid PyTorch warnings
            Xa = np.asarray(Xa, dtype=np.float32).copy()
            Ya = np.asarray(Ya, dtype=np.float32).copy()
            ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(Ya))
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

        tr_loader = make_loader(X_tr, Y_tr, self.batch_size, True)
        va_loader = make_loader(X_va, Y_va, self.batch_size, False)

        best_va = np.inf
        best_state = None
        patience_counter = 0
        early_stop_patience = 10
        
        for ep in range(self.epochs):
            model.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_loader.dataset)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(self.device); yb = yb.to(self.device)
                    pred = model(xb)
                    va_loss += loss_fn(pred, yb).item() * len(xb)
            va_loss /= len(va_loader.dataset)
            
            scheduler.step(va_loss)

            if va_loss < best_va:
                best_va = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        self._model.eval()  # Ensure eval mode for faster inference
        X = np.asarray(X, dtype=np.float32)
        X_seq, _ = self._seq.build(X, None)  # (N, L, F)
        with torch.inference_mode():  # Faster than no_grad
            y_hat = self._model(torch.from_numpy(X_seq).to(self.device)).cpu().numpy()
        return y_hat.ravel() if self.out_dim_ == 1 else y_hat

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("TCNRegressor is not fitted.")
