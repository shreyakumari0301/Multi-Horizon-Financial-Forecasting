"""Multi-Scale Temporal Fusion Network with Cross-Attention (MSTF-CA).

Novel architecture for time series forecasting that:
1. Processes sequences at multiple temporal scales (LSTM, GRU, TCN, Transformer)
2. Uses cross-attention to dynamically fuse multi-scale representations
3. Adaptively weights contributions based on temporal patterns

This is a novel contribution for research publication.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class _MultiScaleTemporalFusion(nn.Module):
    """
    Multi-Scale Temporal Fusion Network with Cross-Attention.
    
    Architecture:
    - LSTM: Captures short-term dependencies (2 layers)
    - GRU: Captures medium-term patterns (2 layers)
    - TCN: Multi-scale temporal patterns via dilated convolutions
    - Transformer: Long-range dependencies via self-attention
    - Cross-Attention Fusion: Dynamically combines all representations
    """
    def __init__(self, in_dim, hidden=128, d_model=128, nhead=4, 
                 num_transformer_layers=2, tcn_channels=(64, 128), 
                 dropout=0.1, out_dim=1):
        super().__init__()
        self.hidden = hidden
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, d_model)
        
        # Multi-scale encoders
        # 1. LSTM (short-term)
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=hidden, 
            num_layers=2,
            dropout=(dropout if 2 > 1 else 0.0),
            batch_first=True
        )
        
        # 2. GRU (medium-term)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=2,
            dropout=(dropout if 2 > 1 else 0.0),
            batch_first=True
        )
        
        # 3. TCN (multi-scale via dilation)
        self.tcn_layers = nn.ModuleList()
        for i, out_ch in enumerate(tcn_channels):
            in_ch = d_model if i == 0 else tcn_channels[i-1]
            dilation = 2 ** i
            padding = (3 - 1) * dilation
            self.tcn_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, dilation=dilation, padding=padding)
            )
        self.tcn_norm = nn.LayerNorm(tcn_channels[-1])
        
        # 4. Transformer (long-term)
        self.max_len = 1000
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pos_encoder', pe)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Cross-Attention Fusion Module
        # Query: Transformer output (long-term context)
        # Keys/Values: LSTM, GRU, TCN outputs (multi-scale features)
        self.cross_attn_lstm = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_gru = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_tcn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        
        # Projection layers to align dimensions
        self.lstm_proj = nn.Linear(hidden, d_model)
        self.gru_proj = nn.Linear(hidden, d_model)
        self.tcn_proj = nn.Linear(tcn_channels[-1], d_model)
        
        # Adaptive fusion weights (learned)
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)  # LSTM, GRU, TCN, Transformer
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim)
        )
        
    def forward(self, x):  # x: (B, L, F)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.input_proj(x)  # (B, L, d_model)
        
        # 1. LSTM encoding (short-term)
        lstm_out, (lstm_hn, _) = self.lstm(x_proj)  # (B, L, hidden)
        lstm_last = lstm_hn[-1].unsqueeze(1)  # (B, 1, hidden)
        lstm_proj = self.lstm_proj(lstm_last)  # (B, 1, d_model)
        lstm_seq = self.lstm_proj(lstm_out)  # (B, L, d_model)
        
        # 2. GRU encoding (medium-term)
        gru_out, gru_hn = self.gru(x_proj)  # (B, L, hidden)
        gru_last = gru_hn[-1].unsqueeze(1)  # (B, 1, hidden)
        gru_proj = self.gru_proj(gru_last)  # (B, 1, d_model)
        gru_seq = self.gru_proj(gru_out)  # (B, L, d_model)
        
        # 3. TCN encoding (multi-scale)
        x_tcn = x_proj.transpose(1, 2)  # (B, d_model, L)
        for tcn_layer in self.tcn_layers:
            x_tcn = F.relu(tcn_layer(x_tcn))
        x_tcn = x_tcn.transpose(1, 2)  # (B, L, tcn_channels[-1])
        x_tcn = self.tcn_norm(x_tcn)
        tcn_seq = self.tcn_proj(x_tcn)  # (B, L, d_model)
        tcn_last = tcn_seq[:, -1:, :]  # (B, 1, d_model)
        
        # 4. Transformer encoding (long-term)
        x_trans = x_proj + self.pos_encoder[:, :seq_len, :]
        trans_out = self.transformer(x_trans)  # (B, L, d_model)
        trans_last = trans_out[:, -1:, :]  # (B, 1, d_model) - query for cross-attention
        
        # Cross-Attention Fusion
        # Use Transformer output as query, attend to other encoders
        attn_lstm, _ = self.cross_attn_lstm(trans_last, lstm_seq, lstm_seq)  # (B, 1, d_model)
        attn_gru, _ = self.cross_attn_gru(trans_last, gru_seq, gru_seq)  # (B, 1, d_model)
        attn_tcn, _ = self.cross_attn_tcn(trans_last, tcn_seq, tcn_seq)  # (B, 1, d_model)
        
        # Adaptive weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (
            weights[0] * attn_lstm +
            weights[1] * attn_gru +
            weights[2] * attn_tcn +
            weights[3] * trans_last
        )  # (B, 1, d_model)
        
        # Final prediction
        output = self.head(fused.squeeze(1))  # (B, out_dim)
        return output


@register_model("mstf_ca")
class MSTFCARegressor:
    """
    Multi-Scale Temporal Fusion Network with Cross-Attention.
    
    Novel architecture combining:
    - LSTM (short-term dependencies)
    - GRU (medium-term patterns)
    - TCN (multi-scale temporal patterns)
    - Transformer (long-range dependencies)
    - Cross-Attention Fusion (dynamic multi-scale integration)
    
    This architecture is novel and suitable for research publication.
    
    .fit(X, y) and .predict(X) accept 2D arrays (N,F) and return aligned predictions (N,) or (N,O).
    """
    def __init__(self,
                 seq_len: int = 64,
                 hidden: int = 128,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_transformer_layers: int = 2,
                 tcn_channels: tuple = (64, 128),
                 dropout: float = 0.1,
                 epochs: int = 40,
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 use_delta_target: bool = False,
                 direction_loss_weight: float = 0.5,
                 val_frac: float = 0.1,
                 seed: int = 0,
                 device: str | None = None):
        self.seq_len = seq_len
        self.hidden = hidden
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.tcn_channels = tcn_channels
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_delta_target = use_delta_target
        self.direction_loss_weight = direction_loss_weight
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
        
        # Convert to delta target if requested
        if self.use_delta_target:
            y_delta = np.zeros_like(y)
            y_delta[1:] = y[1:] - y[:-1]
            y_delta[0] = y[0]
            y = y_delta
        
        self.out_dim_ = y.shape[1]
        self.in_dim_ = X.shape[1]

        X_seq, y_seq = self._seq.build(X, y)  # (N, L, F), (N, O)

        # Chronological split
        N = X_seq.shape[0]
        n_val = max(1, int(self.val_frac * N))
        n_tr = N - n_val
        X_tr, Y_tr = X_seq[:n_tr], y_seq[:n_tr]
        X_va, Y_va = X_seq[n_tr:], y_seq[n_tr:]

        model = _MultiScaleTemporalFusion(
            in_dim=self.in_dim_,
            hidden=self.hidden,
            d_model=self.d_model,
            nhead=self.nhead,
            num_transformer_layers=self.num_transformer_layers,
            tcn_channels=self.tcn_channels,
            dropout=self.dropout,
            out_dim=self.out_dim_
        ).to(self.device)
        
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )
        mse_loss = nn.MSELoss()
        
        def direction_loss(pred, target):
            """Penalize wrong direction predictions."""
            pred_sign = torch.sign(pred)
            target_sign = torch.sign(target)
            wrong_dir = (pred_sign != target_sign).float()
            return wrong_dir.mean()
        
        def combined_loss(pred, target):
            mse = mse_loss(pred, target)
            if self.direction_loss_weight > 0:
                dir_loss = direction_loss(pred, target)
                return mse + self.direction_loss_weight * dir_loss
            return mse
        
        loss_fn = combined_loss

        def make_loader(Xa, Ya, bs, shuffle):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_loader.dataset)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
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

        # Optimize model for inference
        model.eval()
        # Use torch.compile for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
                print("  Model compiled for faster inference")
        except Exception:
            pass  # Fallback if compilation fails
        
        self._model = model
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        self._model.eval()  # Ensure eval mode
        
        X = np.asarray(X, dtype=np.float32)
        X_seq, _ = self._seq.build(X, None)  # (N, L, F)
        
        # Use inference_mode for faster inference (slightly faster than no_grad)
        with torch.inference_mode():
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            y_hat = self._model(X_tensor).cpu().numpy()
        
        return y_hat.ravel() if self.out_dim_ == 1 else y_hat

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("MSTFCARegressor is not fitted.")
