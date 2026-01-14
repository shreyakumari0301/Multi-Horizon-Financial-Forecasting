"""
Comprehensive evaluation script.

This script:
1. Aggregates metrics across all folds
2. Creates metrics tables for cross-model comparison
3. Generates cumulative PnL charts
4. Creates residual plots
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval import aggregate_all_folds
from src.visualisation.plots import create_comprehensive_report


def main():
    print("=" * 60)
    print("Comprehensive Evaluation Pipeline")
    print("=" * 60)
    
    results_dir = "data/experiments"
    output_dir = "data/experiments/reports"
    
    # Step 1: Aggregate metrics across all folds
    print("\n" + "=" * 60)
    print("Step 1: Aggregating Metrics Across All Folds")
    print("=" * 60)
    
    aggregated = aggregate_all_folds(
        results_dir=results_dir,
        output_dir="data/experiments/aggregated",
        model_names=None,  # All models
        folds=None,  # All folds
        horizons=None,  # All horizons
    )
    
    # Step 2: Create comprehensive visualization report
    print("\n" + "=" * 60)
    print("Step 2: Creating Comprehensive Visualization Report")
    print("=" * 60)
    
    create_comprehensive_report(
        results_dir=results_dir,
        output_dir=output_dir,
        folds=None,  # All folds
        horizons=None,  # All horizons
        cost_per_trade=0.0001,  # 1 bp per trade
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - Aggregated metrics: data/experiments/aggregated/")
    print(f"  - Reports: {output_dir}/")
    print(f"    - Tables: {output_dir}/tables/")
    print(f"    - PnL charts: {output_dir}/pnl/")
    print(f"    - Residual plots: {output_dir}/residuals/")


if __name__ == "__main__":
    main()
