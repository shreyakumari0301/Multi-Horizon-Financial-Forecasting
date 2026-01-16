"""
FinBERT Embeddings for Financial News Headlines.

Uses FinBERT (a BERT model pre-trained on financial communication) to generate
sentence embeddings from S&P 500 news headlines. This captures sentiment and
market events that technical indicators often miss.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")


class FinBERTEmbedder:
    """
    FinBERT-based sentence embedder for financial news headlines.
    
    Uses the ProsusAI/finbert model which is pre-trained on financial communication
    and fine-tuned for financial sentiment analysis.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 32
    ):
        """
        Initialize FinBERT embedder.
        
        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing headlines
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load tokenizer and model
        print(f"Loading FinBERT model: {model_name} on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"✓ FinBERT model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load FinBERT model: {e}")
            print("Falling back to generic BERT model")
            self.model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of headline strings
        
        Returns:
            Array of shape (n_sentences, embedding_dim)
        """
        if len(sentences) == 0:
            return np.array([])
        
        # Clean sentences
        sentences = [str(s).strip() if pd.notna(s) else "" for s in sentences]
        sentences = [s if len(s) > 0 else "No headline available" for s in sentences]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def embed_single(self, sentence: str) -> np.ndarray:
        """
        Generate embedding for a single sentence.
        
        Args:
            sentence: Single headline string
        
        Returns:
            Array of shape (embedding_dim,)
        """
        return self.embed_sentences([sentence])[0]


def generate_news_embeddings(
    news_df: pd.DataFrame,
    headline_col: str = "headline",
    date_col: str = "date",
    model_name: str = "ProsusAI/finbert",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate FinBERT embeddings for news headlines and aggregate by date.
    
    Args:
        news_df: DataFrame with headlines and dates
        headline_col: Column name containing headlines
        date_col: Column name containing dates
        model_name: FinBERT model name
        save_path: Optional path to save embeddings
    
    Returns:
        DataFrame with date index and embedding columns
    """
    # Initialize embedder
    embedder = FinBERTEmbedder(model_name=model_name)
    
    # Ensure date column is datetime
    if date_col in news_df.columns:
        news_df[date_col] = pd.to_datetime(news_df[date_col], errors="coerce")
    
    # Group by date and combine headlines
    print("Aggregating headlines by date...")
    daily_headlines = news_df.groupby(date_col)[headline_col].apply(
        lambda x: " ".join(str(h) for h in x if pd.notna(h))
    ).reset_index()
    
    # Generate embeddings
    print(f"Generating embeddings for {len(daily_headlines)} days...")
    embeddings = embedder.embed_sentences(daily_headlines[headline_col].tolist())
    
    # Create DataFrame with date index
    embedding_df = pd.DataFrame(
        embeddings,
        index=pd.to_datetime(daily_headlines[date_col]),
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )
    embedding_df = embedding_df.sort_index()
    
    if save_path:
        embedding_df.to_csv(save_path)
        print(f"Saved embeddings to {save_path}")
    
    return embedding_df


def load_news_embeddings(embeddings_path: str) -> pd.DataFrame:
    """
    Load pre-computed news embeddings from CSV.
    
    Args:
        embeddings_path: Path to embeddings CSV file
    
    Returns:
        DataFrame with date index and embedding columns
    """
    df = pd.read_csv(embeddings_path, index_col=0, parse_dates=True)
    return df.sort_index()
