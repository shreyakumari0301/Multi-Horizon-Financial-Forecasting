"""
Process news headlines using sentence-transformers (dual model approach).

This script:
1. Loads news headlines (CSV format with 'date' and 'headline' columns)
2. Generates embeddings with two models (small + large)
3. Reduces with PCA (12 + 14 = 26 features)
4. Aggregates by date and saves processed features
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_headlines(csv_path: str) -> pd.DataFrame:
    """
    Load headlines from CSV and normalize dates.
    Supports both formats:
    - 'date' and 'headline' columns
    - 'published_utc' and 'title' columns
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Headlines file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Handle different column name formats
    if 'published_utc' in df.columns and 'title' in df.columns:
        df["date"] = pd.to_datetime(df["published_utc"]).dt.tz_convert(None).dt.normalize()
        df = df.rename(columns={'title': 'headline'})
    elif 'date' in df.columns and 'headline' in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    else:
        raise ValueError(f"CSV must have either ('date', 'headline') or ('published_utc', 'title') columns")
    
    # Debug: Show what columns we have
    print(f"  CSV columns: {list(df.columns)}")
    print(f"  Total rows: {len(df)}")
    
    df = df[["date", "headline"]].dropna(subset=["headline"])
    print(f"  After dropna: {len(df)} rows")
    
    # Convert headline to string and remove empty headlines
    df["headline"] = df["headline"].astype(str)
    df = df[df["headline"].str.strip() != ""]  # Remove empty headlines
    print(f"  After removing empty: {len(df)} rows")
    
    if len(df) == 0:
        raise ValueError(
            f"No valid headlines found in {csv_path}.\n"
            f"  - Check that the CSV has 'date' and 'headline' columns\n"
            f"  - Check that headlines are not empty\n"
            f"  - Try fetching news again: python scripts/data/fetch_news.py"
        )
    
    print(f"Loaded {len(df)} headlines from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def get_embeddings(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence-transformers.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # fallback if tqdm not available
    
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


def make_daily_pca(
    model_name: str,
    n_components: int,
    headlines_df: pd.DataFrame,
    random_state: int = 42,
    agg_method: str = "mean"
) -> Tuple[pd.DataFrame, object]:
    """
    Embed headlines, reduce with PCA, and aggregate by date.
    
    Returns:
        (daily_pca_df, fitted_pca_object)
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "sentence-transformers and scikit-learn required for embeddings. "
            f"Install with: pip install sentence-transformers scikit-learn\n{e}"
        )
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = headlines_df["headline"].tolist()
    emb = get_embeddings(model, texts)
    
    # Adjust n_components if we have fewer samples
    n_samples = emb.shape[0]
    actual_components = min(n_components, n_samples, emb.shape[1])
    
    if actual_components < n_components:
        print(f"  ⚠ Adjusting PCA: {n_components} → {actual_components} (only {n_samples} samples)")
    
    pca = PCA(n_components=actual_components, random_state=random_state)
    reduced = pca.fit_transform(emb)
    
    reduced_df = pd.DataFrame(
        reduced, 
        columns=[f"pca_{i+1}" for i in range(actual_components)]
    )
    reduced_df["date"] = headlines_df["date"].values
    
    # Aggregate multiple headlines per day
    daily = reduced_df.groupby("date").agg(agg_method).reset_index()
    
    # Pad to requested number of components with zeros
    if actual_components < n_components:
        for i in range(actual_components, n_components):
            daily[f"pca_{i+1}"] = 0.0
    
    explained_var = pca.explained_variance_ratio_.sum() if actual_components > 0 else 0.0
    print(f"PCA: {actual_components} components explain {explained_var:.2%} variance")
    if actual_components < n_components:
        print(f"  Padded to {n_components} components (added {n_components - actual_components} zero columns)")
    
    return daily, pca


def align_with_dates(
    pca_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    suffix: str = ""
) -> pd.DataFrame:
    """
    Reindex PCA features to match trading calendar.
    Missing dates filled with 0.0, and add 'has_news' indicator.
    """
    pca_df = pca_df.set_index("date").reindex(target_dates)
    pca_df.index.name = "date"
    
    # Fill missing with 0
    pca_df = pca_df.fillna(0.0)
    
    # Add has_news indicator
    has_news_col = f"has_news{suffix}"
    pca_df[has_news_col] = (pca_df.sum(axis=1) != 0).astype(int)
    
    # Rename columns with suffix if provided
    if suffix:
        rename_map = {col: f"{col}{suffix}" for col in pca_df.columns if col.startswith("pca_")}
        pca_df = pca_df.rename(columns=rename_map)
    
    return pca_df.reset_index()


def process_news_data(
    news_path: str,
    output_dir: str = "data/processed",
    small_model: str = "all-MiniLM-L6-v2",
    large_model: str = "sentence-transformers/all-mpnet-base-v2",
    small_pca_dim: int = 12,
    large_pca_dim: int = 14,
    random_state: int = 42,
    agg_method: str = "mean",
    use_finbert: bool = False
) -> pd.DataFrame:
    """
    Process news headlines using sentence-transformers dual model approach.
    
    Args:
        news_path: Path to news CSV file (columns: date, headline or published_utc, title)
        output_dir: Directory to save processed features
        small_model: Small sentence-transformers model
        large_model: Large sentence-transformers model
        small_pca_dim: PCA dimensions for small model (default: 12)
        large_pca_dim: PCA dimensions for large model (default: 14)
        random_state: Random seed
        agg_method: Aggregation method for daily headlines (mean, max, etc.)
    
    Returns:
        DataFrame with processed features (26 PCA + 2 has_news = 28 total)
    """
    if use_finbert:
        # Use FinBERT approach (original)
        print("=" * 70)
        print("Processing News Headlines with FinBERT")
        print("=" * 70)
        
        from src.features.finbert_embeddings import generate_news_embeddings
        from src.features.pca_reduction import reduce_embeddings
        
        # Load headlines
        headlines_df = load_headlines(news_path)
        
        # Generate FinBERT embeddings
        print("\nGenerating FinBERT embeddings...")
        embeddings_df = generate_news_embeddings(
            headlines_df,
            headline_col="headline",
            date_col="date",
            model_name="ProsusAI/finbert"
        )
        
        # Reduce with PCA
        print("\nReducing to 28 features with PCA...")
        
        # Check if we have enough samples for PCA
        n_samples = len(embeddings_df)
        n_components = 28
        
        if n_samples < n_components:
            print(f"⚠ Warning: Only {n_samples} samples but {n_components} PCA components requested")
            print(f"  Reducing to {n_samples} components (or fewer if needed)")
            n_components = min(n_samples, n_components)
            if n_components < 1:
                raise ValueError(f"Not enough data for PCA. Need at least 1 sample, got {n_samples}")
        
        reduced_df = reduce_embeddings(
            embeddings_df,
            n_components=n_components,
            train_dates=None
        )
        
        # If we had fewer components, pad with zeros to get 28 features
        if len(reduced_df.columns) < 28:
            for i in range(len(reduced_df.columns), 28):
                reduced_df[f'news_pc{i+1}'] = 0.0
            print(f"  Padded to 28 features (added {28 - len(reduced_df.columns)} zero columns)")
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "news_features_28d.csv")
        reduced_df.to_csv(output_path)
        
        print(f"\n✓ Saved {len(reduced_df.columns)} FinBERT features to {output_path}")
        return reduced_df
    
    # Use sentence-transformers approach (dual model)
    print("=" * 70)
    print("Processing News Headlines with Sentence-Transformers (Dual Model)")
    print("=" * 70)
    
    # Load headlines
    headlines_df = load_headlines(news_path)
    
    # Get date range from headlines
    all_dates = pd.date_range(
        start=headlines_df['date'].min(),
        end=headlines_df['date'].max(),
        freq='D'
    )
    
    # Small model
    print(f"\n--- Small model ({small_model}) ---")
    small_daily, _ = make_daily_pca(
        small_model, small_pca_dim, headlines_df, random_state, agg_method
    )
    small_aligned = align_with_dates(small_daily, all_dates, suffix="")
    
    # Large model
    print(f"\n--- Large model ({large_model}) ---")
    large_daily, _ = make_daily_pca(
        large_model, large_pca_dim, headlines_df, random_state, agg_method
    )
    large_aligned = align_with_dates(large_daily, all_dates, suffix="_large")
    
    # Merge
    merged = small_aligned.merge(large_aligned, on="date", how="outer")
    merged = merged.set_index("date").sort_index()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "news_features_28d.csv")
    merged.to_csv(output_path)
    
    print("\n" + "=" * 70)
    print("News Processing Complete!")
    print("=" * 70)
    print(f"✓ Generated {len(merged.columns)} headline features for {len(merged)} dates")
    print(f"  Features: {small_pca_dim} small + {large_pca_dim} large + 2 has_news = {len(merged.columns)} total")
    print(f"✓ Saved to: {output_path}")
    print("=" * 70)
    
    return merged


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process news headlines with sentence-transformers")
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
        "--small_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Small sentence-transformers model"
    )
    parser.add_argument(
        "--large_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Large sentence-transformers model"
    )
    parser.add_argument(
        "--small_pca_dim",
        type=int,
        default=12,
        help="PCA dimensions for small model (default: 12)"
    )
    parser.add_argument(
        "--large_pca_dim",
        type=int,
        default=14,
        help="PCA dimensions for large model (default: 14)"
    )
    parser.add_argument(
        "--use_finbert",
        action="store_true",
        help="Use FinBERT instead of sentence-transformers (default: sentence-transformers)"
    )
    
    args = parser.parse_args()
    
    process_news_data(
        news_path=args.news_path,
        output_dir=args.output_dir,
        small_model=args.small_model,
        large_model=args.large_model,
        small_pca_dim=args.small_pca_dim,
        large_pca_dim=args.large_pca_dim,
        use_finbert=args.use_finbert
    )


if __name__ == "__main__":
    main()
