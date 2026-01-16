"""Feature engineering modules for FinBERT embeddings and dimensionality reduction."""
from src.features.finbert_embeddings import FinBERTEmbedder, generate_news_embeddings
from src.features.pca_reduction import PCAReducer, reduce_embeddings

__all__ = [
    "FinBERTEmbedder",
    "generate_news_embeddings",
    "PCAReducer",
    "reduce_embeddings",
]
