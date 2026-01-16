"""
Unified feature processing pipeline.

This script processes both technical and news features, integrating them into
a single pipeline that creates the full 38-feature set (10 technical + 28 news).

If news data is available, it automatically processes and integrates it.
If not, it falls back to technical features only.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional


def process_all_features(
    processed_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    news_path: Optional[str] = None,
    n_components: int = 28
):
    """
    Process all features (technical + news) and integrate into splits.
    
    Args:
        processed_dir: Directory for processed features
        splits_dir: Directory with train/test splits
        news_path: Optional path to news headlines CSV (if None, skips news)
        n_components: Number of PCA components for news (default: 28)
    """
    print("=" * 70)
    print("Unified Feature Processing Pipeline")
    print("=" * 70)
    
    # Step 1: Process news features if available
    news_features_path = os.path.join(processed_dir, "news_features_28d.csv")
    
    if news_path and os.path.exists(news_path):
        print("\n" + "=" * 70)
        print("Step 1: Processing News Features")
        print("=" * 70)
        
        # Import here to avoid circular imports
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "process_news_features",
            os.path.join(project_root, "scripts", "features", "process_news_features.py")
        )
        process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(process_module)
        
        process_module.process_news_data(
            news_path=news_path,
            output_dir=processed_dir,
            n_components=n_components
        )
        
        print(f"\n✓ News features processed: {news_features_path}")
        has_news = True
    elif os.path.exists(news_features_path):
        print(f"\n✓ Using existing news features: {news_features_path}")
        has_news = True
    else:
        print("\n⚠ No news data available - using technical features only")
        print("  To add news features, provide --news_path")
        has_news = False
    
    # Step 2: Integrate news features with splits
    if has_news:
        print("\n" + "=" * 70)
        print("Step 2: Integrating News Features with Technical Features")
        print("=" * 70)
        
        # Import here to avoid circular imports
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "integrate_news_features",
            os.path.join(project_root, "scripts", "features", "integrate_news_features.py")
        )
        integrate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(integrate_module)
        
        integrate_module.integrate_news_features(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            news_features_path=news_features_path
        )
        
        print("\n✓ All features integrated: 10 technical + 28 news = 38 total")
    else:
        print("\n✓ Using technical features only: 10 features")
    
    print("\n" + "=" * 70)
    print("Feature Processing Complete!")
    print("=" * 70)
    print(f"\nFeatures available in: {splits_dir}")
    print(f"  - Technical features: 10")
    if has_news:
        print(f"  - News features: 28")
        print(f"  - Total: 38 features")
    else:
        print(f"  - Total: 10 features (news not available)")


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process all features (technical + news)"
    )
    parser.add_argument(
        "--news_path",
        type=str,
        default=None,
        help="Path to news headlines CSV (optional)"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory for processed features"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory with train/test splits"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=28,
        help="Number of PCA components for news (default: 28)"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)
    
    # Process all features
    process_all_features(
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        news_path=args.news_path,
        n_components=args.n_components
    )


if __name__ == "__main__":
    main()
