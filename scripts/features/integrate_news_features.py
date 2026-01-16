"""
Integrate news features (28 PCA components) with technical features (10 features).

This script:
1. Loads technical features from processed data
2. Loads news features (28 PCA components from FinBERT)
3. Merges them by date
4. Updates train/test splits with combined features (38 total)
"""
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def integrate_news_features(
    processed_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    news_features_path: str = "data/processed/news_features_28d.csv"
):
    """
    Integrate news features with technical features in train/test splits.
    
    Args:
        processed_dir: Directory with processed technical features
        splits_dir: Directory with train/test splits
        news_features_path: Path to news features CSV (28 PCA components)
    """
    print("=" * 70)
    print("Integrating News Features with Technical Features")
    print("=" * 70)
    
    # Load news features
    print(f"\nLoading news features from: {news_features_path}")
    if not os.path.exists(news_features_path):
        print(f"Warning: News features not found at {news_features_path}")
        print("Run scripts/features/process_news_features.py first")
        return
    
    news_df = pd.read_csv(news_features_path, index_col=0, parse_dates=True)
    print(f"Loaded news features: {news_df.shape}")
    print(f"Date range: {news_df.index.min()} to {news_df.index.max()}")
    
    # Load split configuration
    splits_json = os.path.join(splits_dir, "splits.json")
    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"Split configuration not found: {splits_json}")
    
    with open(splits_json, 'r') as f:
        split_plan = json.load(f)
    
    folds = split_plan["folds"]
    print(f"\nProcessing {len(folds)} folds...")
    
    # Process each fold
    for k, fold in enumerate(folds):
        fold_dir = os.path.join(splits_dir, f"fold_{k}")
        train_path = os.path.join(fold_dir, "train.csv")
        test_path = os.path.join(fold_dir, "test.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"  Skipping fold_{k} (files not found)")
            continue
        
        print(f"\n  Processing fold_{k}...")
        
        # Load train/test data
        train = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Get technical features (z_* columns)
        tech_cols = [c for c in train.columns if c.startswith("z_")]
        target_cols = [c for c in train.columns if c.startswith("target_")]
        other_cols = [c for c in train.columns if c not in tech_cols + target_cols]
        
        print(f"    Technical features: {len(tech_cols)}")
        print(f"    Target columns: {len(target_cols)}")
        
        # Merge news features
        train_news = news_df.reindex(train.index, fill_value=0.0)
        test_news = news_df.reindex(test.index, fill_value=0.0)
        
        # Check if news features exist for this date range
        train_news_available = train_news.sum(axis=1).sum() > 0
        test_news_available = test_news.sum(axis=1).sum() > 0
        
        if not train_news_available and not test_news_available:
            print(f"    Warning: No news features available for fold_{k} date range")
            print(f"    Using zero-filled news features")
        
        # Combine technical and news features
        train_combined = pd.concat([
            train[tech_cols],
            train_news,
            train[target_cols + other_cols]
        ], axis=1)
        
        test_combined = pd.concat([
            test[tech_cols],
            test_news,
            test[target_cols + other_cols]
        ], axis=1)
        
        # Re-scale features (including news features)
        # Get all feature columns
        all_feature_cols = tech_cols + list(news_df.columns)
        
        # Scale features
        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_combined[all_feature_cols].values)
        test_X_scaled = scaler.transform(test_combined[all_feature_cols].values)
        
        # Create scaled feature columns
        scaled_cols = [f"z_{c}" if not c.startswith("z_") else c for c in tech_cols]
        scaled_cols += [f"z_{c}" for c in news_df.columns]
        
        # Create output DataFrames
        train_out = pd.DataFrame(
            train_X_scaled,
            index=train_combined.index,
            columns=scaled_cols
        )
        test_out = pd.DataFrame(
            test_X_scaled,
            index=test_combined.index,
            columns=scaled_cols
        )
        
        # Add targets and other columns
        for col in target_cols + other_cols:
            train_out[col] = train_combined[col].values
            test_out[col] = test_combined[col].values
        
        # Save updated splits
        train_out.to_csv(train_path)
        test_out.to_csv(test_path)
        
        # Update scaler metadata
        scaler_meta_path = os.path.join(fold_dir, "scaler.json")
        if os.path.exists(scaler_meta_path):
            with open(scaler_meta_path, 'r') as f:
                scaler_meta = json.load(f)
            
            scaler_meta["features"] = all_feature_cols
            scaler_meta["mean"] = scaler.mean_.tolist()
            scaler_meta["scale"] = scaler.scale_.tolist()
            scaler_meta["n_technical"] = len(tech_cols)
            scaler_meta["n_news"] = len(news_df.columns)
            scaler_meta["n_total"] = len(all_feature_cols)
            
            with open(scaler_meta_path, 'w') as f:
                json.dump(scaler_meta, f, indent=2)
        
        print(f"    ✓ Updated fold_{k}: {len(tech_cols)} technical + {len(news_df.columns)} news = {len(all_feature_cols)} total features")
    
    print("\n" + "=" * 70)
    print("Integration Complete!")
    print("=" * 70)
    print(f"\nAll folds updated with combined features:")
    print(f"  - Technical features: {len(tech_cols)}")
    print(f"  - News features: {len(news_df.columns)}")
    print(f"  - Total features: {len(tech_cols) + len(news_df.columns)}")


def main():
    """Main integration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate news features with technical features")
    parser.add_argument(
        "--news_features",
        type=str,
        default="data/processed/news_features_28d.csv",
        help="Path to news features CSV"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory with train/test splits"
    )
    
    args = parser.parse_args()
    
    integrate_news_features(
        splits_dir=args.splits_dir,
        news_features_path=args.news_features
    )


if __name__ == "__main__":
    main()
