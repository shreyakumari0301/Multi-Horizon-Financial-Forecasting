"""
PCA Dimensionality Reduction for FinBERT Embeddings.

Reduces high-dimensional FinBERT embeddings (typically 768 dimensions) to
28 key features using Principal Component Analysis (PCA).
"""
import os
import numpy as np
import pandas as pd
from typing import Optional
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAReducer:
    """
    PCA reducer for FinBERT embeddings.
    
    Reduces embeddings from high-dimensional space (e.g., 768) to 28 features
    while preserving the most important variance.
    """
    
    def __init__(self, n_components: int = 28, random_state: int = 0):
        """
        Initialize PCA reducer.
        
        Args:
            n_components: Number of PCA components (default: 28)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, embeddings: np.ndarray):
        """
        Fit PCA on training embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        """
        # Scale embeddings before PCA
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit PCA
        self.pca.fit(embeddings_scaled)
        self.fitted = True
        
        # Print explained variance
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"PCA fitted: {embeddings.shape[1]} → {self.n_components} components")
        print(f"Explained variance: {explained_var:.2%}")
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted PCA.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        if not self.fitted:
            raise ValueError("PCA reducer must be fitted before transform")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.pca.transform(embeddings_scaled)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and transform embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def save(self, save_path: str):
        """Save fitted PCA reducer to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'scaler': self.scaler,
                'n_components': self.n_components,
            }, f)
        print(f"Saved PCA reducer to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'PCAReducer':
        """Load fitted PCA reducer from disk."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        reducer = cls(n_components=data['n_components'])
        reducer.pca = data['pca']
        reducer.scaler = data['scaler']
        reducer.fitted = True
        return reducer


def reduce_embeddings(
    embeddings_df: pd.DataFrame,
    n_components: int = 28,
    train_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Reduce embeddings using PCA, fitting on training data only.
    
    Args:
        embeddings_df: DataFrame with date index and embedding columns
        n_components: Number of PCA components (default: 28)
        train_dates: Training date range (if None, uses all data for fitting)
        save_path: Optional path to save PCA reducer
    
    Returns:
        DataFrame with reduced embeddings (n_components columns)
    """
    reducer = PCAReducer(n_components=n_components)
    
    # Fit on training data if provided, otherwise use all
    if train_dates is not None:
        train_embeddings = embeddings_df.loc[train_dates].values
        reducer.fit(train_embeddings)
    else:
        reducer.fit(embeddings_df.values)
    
    # Transform all embeddings
    reduced = reducer.transform(embeddings_df.values)
    
    # Create DataFrame
    reduced_df = pd.DataFrame(
        reduced,
        index=embeddings_df.index,
        columns=[f"news_pc{i+1}" for i in range(n_components)]
    )
    
    # Save reducer if path provided
    if save_path:
        reducer.save(save_path)
    
    return reduced_df
