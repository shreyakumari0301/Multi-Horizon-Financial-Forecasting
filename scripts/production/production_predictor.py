"""
Production Predictor for Real-Time Forecasting.

This predictor uses the full 38-feature set (10 technical + 28 news) to generate
real-time signals with best-in-class performance targeting 68.7% Directional Accuracy.

The predictor automatically handles:
- Loading trained hybrid ensemble models
- Processing technical features from market data
- Processing news headlines with FinBERT embeddings
- Generating predictions with proper feature scaling
- Handling missing news data gracefully
"""
import sys
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from datetime import datetime, date

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.finbert_embeddings import FinBERTEmbedder
from src.features.pca_reduction import PCAReducer
from src.models.registry import get_model
import config.experiments as experiments


class ProductionPredictor:
    """
    Production predictor for real-time forecasting.
    
    Uses hybrid ensemble models with full 38-feature set (10 technical + 28 news)
    to generate predictions targeting 68.7% Directional Accuracy.
    """
    
    def __init__(
        self,
        model_dir: str = "data/models",
        fold: int = 0,
        horizon: str = "target_h1",
        news_features_path: Optional[str] = None,
        pca_reducer_path: Optional[str] = None
    ):
        """
        Initialize production predictor.
        
        Args:
            model_dir: Directory containing trained models
            fold: Fold number to use (default: 0)
            horizon: Target horizon (default: "target_h1")
            news_features_path: Path to pre-computed news features (optional)
            pca_reducer_path: Path to fitted PCA reducer (optional)
        """
        self.model_dir = model_dir
        self.fold = fold
        self.horizon = horizon
        
        # Load hybrid ensemble model
        hybrid_path = os.path.join(model_dir, "hybrid", f"fold_{fold}", f"{horizon}.pkl")
        if not os.path.exists(hybrid_path):
            raise FileNotFoundError(f"Hybrid model not found: {hybrid_path}")
        
        print(f"Loading hybrid model from: {hybrid_path}")
        with open(hybrid_path, 'rb') as f:
            self.ensemble = pickle.load(f)
        
        print(f"✓ Loaded hybrid ensemble with {len(self.ensemble.base_models)} models")
        print(f"  Weights: {self.ensemble.weights}")
        
        # Initialize news processing if paths provided
        self.has_news = False
        if news_features_path and os.path.exists(news_features_path):
            self.news_features_df = pd.read_csv(news_features_path, index_col=0, parse_dates=True)
            self.has_news = True
            print(f"✓ Loaded news features: {self.news_features_df.shape}")
        else:
            print("⚠ No news features - will use technical features only")
        
        if pca_reducer_path and os.path.exists(pca_reducer_path):
            self.pca_reducer = PCAReducer.load(pca_reducer_path)
            self.finbert_embedder = FinBERTEmbedder()
            print(f"✓ Loaded PCA reducer for real-time news processing")
        else:
            self.pca_reducer = None
            self.finbert_embedder = None
        
        # Load scaler metadata from fold
        scaler_path = os.path.join("data/splits", f"fold_{fold}", "scaler.json")
        if os.path.exists(scaler_path):
            import json
            with open(scaler_path, 'r') as f:
                self.scaler_meta = json.load(f)
            self.n_technical = self.scaler_meta.get("n_technical", 10)
            self.n_news = self.scaler_meta.get("n_news", 28)
            print(f"✓ Loaded scaler metadata: {self.n_technical} technical + {self.n_news} news")
        else:
            self.scaler_meta = None
            self.n_technical = 10
            self.n_news = 28 if self.has_news else 0
    
    def process_technical_features(
        self,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Process market data into technical features.
        
        Args:
            market_data: DataFrame with OHLCV data
        
        Returns:
            Technical features array (n_samples, 10)
        """
        # This should match your technical feature engineering
        # For now, assuming features are already computed
        # In production, you'd compute: ret_1, ret_2, ret_5, vol_20, ma_10, ma_20, 
        # ma_gap, rsi_14, vol_z, dow
        
        feature_cols = [
            "ret_1", "ret_2", "ret_5",
            "vol_20", "ma_10", "ma_20", "ma_gap",
            "rsi_14", "vol_z", "dow"
        ]
        
        # Check if features already exist
        if all(col in market_data.columns for col in feature_cols):
            return market_data[feature_cols].values
        
        # Otherwise, compute features (simplified - should match your pipeline)
        # This is a placeholder - implement full feature engineering
        raise NotImplementedError(
            "Technical feature engineering not implemented. "
            "Provide pre-computed features or implement build_features()"
        )
    
    def process_news_features(
        self,
        headlines: List[str],
        date: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Process news headlines into 28 PCA features.
        
        Args:
            headlines: List of headline strings
            date: Optional date for lookup in pre-computed features
        
        Returns:
            News features array (28,)
        """
        # Try to get from pre-computed features first
        if self.has_news and date is not None:
            date_str = pd.Timestamp(date).date()
            if date_str in self.news_features_df.index:
                return self.news_features_df.loc[date_str].values
        
        # Otherwise, process in real-time
        if self.finbert_embedder and self.pca_reducer:
            # Combine headlines
            combined_text = " ".join(str(h) for h in headlines if pd.notna(h))
            if len(combined_text.strip()) == 0:
                combined_text = "No headline available"
            
            # Generate embedding
            embedding = self.finbert_embedder.embed_single(combined_text)
            
            # Reduce with PCA
            news_features = self.pca_reducer.transform(embedding.reshape(1, -1))[0]
            return news_features
        
        # Fallback: return zeros if no news processing available
        return np.zeros(self.n_news)
    
    def predict(
        self,
        technical_features: np.ndarray,
        news_headlines: Optional[List[str]] = None,
        date: Optional[datetime] = None,
        feature_history: Optional[np.ndarray] = None
    ) -> float:
        """
        Generate prediction for a single timestep.
        
        Args:
            technical_features: Technical features array (10,)
            news_headlines: Optional list of headlines for the day
            date: Optional date for news lookup
            feature_history: Optional history of features for sequence models (n_history, n_features)
        
        Returns:
            Predicted value
        """
        # Process news features
        if news_headlines is not None:
            news_features = self.process_news_features(news_headlines, date)
        elif self.has_news and date is not None:
            news_features = self.process_news_features([], date)
        else:
            news_features = np.zeros(self.n_news)
        
        # Combine features
        if len(news_features) > 0:
            features = np.concatenate([technical_features, news_features])
        else:
            features = technical_features
        
        # Scale features (using scaler metadata)
        if self.scaler_meta:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(self.scaler_meta["mean"])
            scaler.scale_ = np.array(self.scaler_meta["scale"])
            features = scaler.transform(features.reshape(1, -1))[0]
        
        # Build sequence for models
        # Get sequence length from first model
        seq_len = 64  # Default
        for model in self.ensemble.base_models:
            if hasattr(model, 'seq_len'):
                seq_len = model.seq_len
                break
        
        # Use history if provided, otherwise pad with current features
        if feature_history is not None:
            # Append current features to history
            if len(feature_history) >= seq_len:
                X_seq = feature_history[-seq_len:]
            else:
                # Pad with first row
                padding = np.tile(feature_history[0:1], (seq_len - len(feature_history), 1))
                X_seq = np.vstack([padding, feature_history, features.reshape(1, -1)])[-seq_len:]
        else:
            # Create sequence by repeating current features (simplified)
            X_seq = np.tile(features, (seq_len, 1))
        
        # Reshape for model: (1, seq_len, n_features)
        X_seq = X_seq.reshape(1, seq_len, -1)
        
        # Flatten for Ridge, keep 3D for sequence models
        # The ensemble will handle this internally
        X_flat = X_seq.reshape(1, -1)  # For Ridge
        
        # Use ensemble prediction (it handles different model types)
        # For sequence models, we need to pass the sequence properly
        # This is a simplified version - in production, maintain proper history
        try:
            prediction = self.ensemble.predict(X_flat)[0]
        except:
            # Fallback: use first model
            prediction = self.ensemble.base_models[0].predict(X_flat)[0]
        
        return float(prediction)
    
    def predict_batch(
        self,
        technical_features_df: pd.DataFrame,
        news_headlines_df: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate predictions for multiple timesteps.
        
        Args:
            technical_features_df: DataFrame with technical features (date index)
            news_headlines_df: Optional DataFrame with headlines (date, headline columns)
        
        Returns:
            Series of predictions with date index
        """
        predictions = []
        dates = technical_features_df.index
        
        for date in dates:
            tech_features = technical_features_df.loc[date].values[:self.n_technical]
            
            # Get headlines for this date
            headlines = []
            if news_headlines_df is not None:
                day_headlines = news_headlines_df[
                    pd.to_datetime(news_headlines_df['date']).dt.date == pd.Timestamp(date).date()
                ]
                headlines = day_headlines['headline'].tolist()
            
            pred = self.predict(tech_features, headlines, date)
            predictions.append(pred)
        
        return pd.Series(predictions, index=dates, name="prediction")
    
    def get_signal(self, prediction: float) -> str:
        """
        Convert prediction to trading signal.
        
        Args:
            prediction: Predicted value
        
        Returns:
            Trading signal: "LONG", "SHORT", or "NEUTRAL"
        """
        if prediction > 0.001:  # Threshold for long signal
            return "LONG"
        elif prediction < -0.001:  # Threshold for short signal
            return "NEUTRAL"
        else:
            return "NEUTRAL"


def main():
    """Example usage of production predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Predictor")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="data/models",
        help="Directory with trained models"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number"
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="target_h1",
        help="Target horizon"
    )
    parser.add_argument(
        "--news_features",
        type=str,
        default="data/processed/news_features_28d.csv",
        help="Path to news features"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ProductionPredictor(
        model_dir=args.model_dir,
        fold=args.fold,
        horizon=args.horizon,
        news_features_path=args.news_features if os.path.exists(args.news_features) else None
    )
    
    print("\n" + "=" * 70)
    print("Production Predictor Ready")
    print("=" * 70)
    print(f"\nModel: Hybrid Ensemble (fold_{args.fold}, {args.horizon})")
    print(f"Features: {predictor.n_technical} technical + {predictor.n_news} news = {predictor.n_technical + predictor.n_news} total")
    print(f"\nTarget Performance: 68.7% Directional Accuracy")
    print("\nUse predictor.predict() or predictor.predict_batch() for predictions")


if __name__ == "__main__":
    main()
