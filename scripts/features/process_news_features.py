"""
Process news headlines and generate FinBERT embeddings with PCA reduction.

This script:
1. Loads news headlines (CSV format with 'date' and 'headline' columns)
2. Generates FinBERT embeddings for each day
3. Reduces embeddings to 28 features using PCA
4. Saves processed features for integration with technical features
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.finbert_embeddings import generate_news_embeddings
from src.features.pca_reduction import reduce_embeddings


def process_news_data(
    news_path: str,
    output_dir: str = "data/processed",
    n_components: int = 28,
    model_name: str = "ProsusAI/finbert"
):
    """
    Process news headlines into 28 PCA-reduced FinBERT features.
    
    Args:
        news_path: Path to news CSV file (columns: date, headline)
        output_dir: Directory to save processed features
        n_components: Number of PCA components (default: 28)
        model_name: FinBERT model name
    """
    print("=" * 70)
    print("Processing News Headlines with FinBERT")
    print("=" * 70)
    
    # Load news data
    print(f"\nLoading news data from: {news_path}")
    news_df = pd.read_csv(news_path)
    
    # Check required columns
    required_cols = ["date", "headline"]
    missing = [c for c in required_cols if c not in news_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"Loaded {len(news_df)} news headlines")
    print(f"Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    
    # Generate FinBERT embeddings
    print("\n" + "=" * 70)
    print("Step 1: Generating FinBERT Embeddings")
    print("=" * 70)
    
    embeddings_path = os.path.join(output_dir, "news_embeddings_raw.csv")
    embeddings_df = generate_news_embeddings(
        news_df,
        headline_col="headline",
        date_col="date",
        model_name=model_name,
        save_path=embeddings_path
    )
    
    print(f"\nGenerated embeddings: {embeddings_df.shape}")
    print(f"Embedding dimension: {embeddings_df.shape[1]}")
    
    # Reduce to 28 features using PCA
    print("\n" + "=" * 70)
    print("Step 2: Reducing to 28 Features with PCA")
    print("=" * 70)
    
    pca_path = os.path.join(output_dir, "news_pca_reducer.pkl")
    reduced_df = reduce_embeddings(
        embeddings_df,
        n_components=n_components,
        train_dates=None,  # Fit on all data (can be changed for time-series splits)
        save_path=pca_path
    )
    
    # Save reduced features
    output_path = os.path.join(output_dir, "news_features_28d.csv")
    reduced_df.to_csv(output_path)
    
    print(f"\n✓ Reduced to {n_components} features")
    print(f"✓ Saved to: {output_path}")
    print(f"\nFeature columns: {list(reduced_df.columns)}")
    
    print("\n" + "=" * 70)
    print("News Processing Complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - Raw embeddings: {embeddings_path}")
    print(f"  - Reduced features (28D): {output_path}")
    print(f"  - PCA reducer: {pca_path}")
    
    return reduced_df


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process news headlines with FinBERT")
    parser.add_argument(
        "--news_path",
        type=str,
        default="data/raw/news_headlines.csv",
        help="Path to news headlines CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed features"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=28,
        help="Number of PCA components (default: 28)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        help="FinBERT model name"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process news data
    process_news_data(
        news_path=args.news_path,
        output_dir=args.output_dir,
        n_components=args.n_components,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
