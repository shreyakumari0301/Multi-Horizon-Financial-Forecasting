# Evaluation and Aggregation

This module provides utilities for aggregating and summarizing experiment results across multiple folds.

## Functions

### `load_all_results()`
Load all experiment results from JSON files across all folds, models, and horizons.

### `aggregate_metrics()`
Aggregate metrics across folds, computing mean, std, min, max, and median statistics.

### `create_metrics_table()`
Create formatted metrics tables for cross-model comparison with mean ± std formatting.

### `aggregate_all_folds()`
Comprehensive aggregation function that:
- Loads all results
- Aggregates metrics by model and horizon
- Creates metrics tables for all metrics and splits
- Saves summary statistics

## Usage

```python
from src.eval.aggregate import aggregate_all_folds

# Aggregate all results
results = aggregate_all_folds(
    results_dir="data/experiments",
    output_dir="data/experiments/aggregated"
)
```

## Output

- `aggregated_metrics.csv`: Aggregated statistics across folds
- `{split}_{metric}_table.csv`: Formatted tables for each metric
- `summary.json`: Summary of available models, folds, and horizons
