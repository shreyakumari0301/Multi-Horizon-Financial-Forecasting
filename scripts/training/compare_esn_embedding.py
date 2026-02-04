"""
Compare ESN with vs without news embeddings.
Shows DirAcc, Sharpe, and prediction magnitude (RMSE, MAE) for ESN; then DirAcc for all models.
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import config.experiments as experiments
from src.models import get_model
from src.train.runner import run_experiment


def check_news_embeddings(splits_dir: str = "data/splits", fold: int = 0) -> None:
    """
    Load fold data and report whether news columns contain real values or are zero-filled.
    """
    fold_dir = os.path.join(splits_dir, f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train.csv")
    if not os.path.exists(train_path):
        print("News check: train.csv not found, skipping.")
        return
    train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    news_cols = [c for c in train.columns if c.startswith("z_news_pc")]
    if not news_cols:
        print("News check: No news columns (z_news_pc*) in fold data. News not integrated.")
        return
    # Check if any news column has non-zero values
    news_block = train[news_cols]
    total_abs = news_block.abs().sum().sum()
    non_zero_count = (news_block != 0).sum().sum()
    n_cells = news_block.size
    min_val = float(news_block.min().min())
    max_val = float(news_block.max().max())
    std_val = float(news_block.std().mean())  # mean std across columns
    print("\n" + "=" * 60)
    print("News embeddings check (fold {} train set)".format(fold))
    print("=" * 60)
    print("  News columns: {} (z_news_pc1 .. z_news_pc{})".format(len(news_cols), len(news_cols)))
    print("  Rows: {}".format(len(train)))
    if total_abs < 1e-12 and non_zero_count == 0:
        print("  Result: ZERO-FILLED — all news values are 0. No real embeddings.")
        print("  Reason: No news dates overlap this fold, or news_features_28d.csv was empty.")
    else:
        print("  Result: REAL EMBEDDINGS — non-zero values present.")
        print("  Sum of absolute values: {:.6f}".format(total_abs))
        print("  Non-zero cells: {} / {}".format(non_zero_count, n_cells))
        print("  Min / Max: {:.4f} / {:.4f}".format(min_val, max_val))
        print("  Mean std across news cols: {:.4f}".format(std_val))
    print("=" * 60 + "\n")


def get_grid_params(grid: Dict[str, list], index: int = 0) -> Dict[str, Any]:
    return {k: v[index] if isinstance(v, list) else v for k, v in grid.items()}


def create_model(
    model_name: str,
    grid: Dict[str, list],
    grid_index: int = 0,
    horizon: str = None,
    **override_kwargs
):
    params = get_grid_params(grid, grid_index)
    if horizon and hasattr(experiments, "HORIZON_SPECIFIC_CONFIG"):
        horizon_config = experiments.HORIZON_SPECIFIC_CONFIG.get(horizon, {})
        if model_name in horizon_config:
            params.update(horizon_config[model_name])
    params.update(override_kwargs)
    return get_model(model_name, **params)


def main():
    splits_dir = "data/splits"
    folds = getattr(experiments, "FOLDS", [0])
    horizons = getattr(experiments, "HORIZONS", ["target_h1"])
    grid_map = getattr(experiments, "MODEL_GRIDS", {})

    # Check if news embeddings are actually present (or zero-filled)
    check_news_embeddings(splits_dir=splits_dir, fold=folds[0] if folds else 0)

    # ESN-only: (display_name, fold, horizon, DirAcc, Sharpe, RMSE, MAE)
    esn_rows: List[Tuple[str, int, str, float, float, float, float]] = []
    # Full comparison: (display_name, fold, horizon, dir_acc)
    rows = []
    baseline_diracc = {}

    # 1) BASELINE: ESN without embedding (technical features only)
    print("Training BASELINE: ESN (no embedding, technical only)...")
    for fold in folds:
        for horizon in horizons:
            model = create_model("esn", grid_map["esn"], grid_index=0, horizon=horizon)
            res = run_experiment(
                model=model,
                fold=fold,
                horizon=horizon,
                splits_dir=splits_dir,
                results_dir=None,
                save_predictions=False,
                include_news=False,
            )
            tm = res["test_metrics"]
            da, sh, rmse, mae = tm["dir_acc"], tm["sharpe"], tm["rmse"], tm["mae"]
            baseline_diracc[(fold, horizon)] = da
            esn_rows.append(("ESN (no embedding) [BASELINE]", fold, horizon.replace("target_", ""), da, sh, rmse, mae))
            rows.append(("ESN (no embedding) [BASELINE]", fold, horizon.replace("target_", ""), da))
            print(f"  Fold {fold} {horizon}: DirAcc = {da:.3f} | Sharpe = {sh:.3f} | RMSE = {rmse:.6f} | MAE = {mae:.6f}")

    # 2) ESN with embedding (technical + news)
    print("Training ESN (with embedding, technical + news)...")
    for fold in folds:
        for horizon in horizons:
            model = create_model("esn", grid_map["esn"], grid_index=0, horizon=horizon)
            res = run_experiment(
                model=model,
                fold=fold,
                horizon=horizon,
                splits_dir=splits_dir,
                results_dir=None,
                save_predictions=False,
                include_news=True,
            )
            tm = res["test_metrics"]
            da, sh, rmse, mae = tm["dir_acc"], tm["sharpe"], tm["rmse"], tm["mae"]
            esn_rows.append(("ESN (with embedding)", fold, horizon.replace("target_", ""), da, sh, rmse, mae))
            rows.append(("ESN (with embedding)", fold, horizon.replace("target_", ""), da))
            print(f"  Fold {fold} {horizon}: DirAcc = {da:.3f} | Sharpe = {sh:.3f} | RMSE = {rmse:.6f} | MAE = {mae:.6f}")

    # ESN-only table: DirAcc, Sharpe, prediction magnitude (RMSE, MAE)
    print("\n" + "=" * 90)
    print("ESN with vs without embedding — DirAcc, Sharpe, prediction magnitude (test set)")
    print("=" * 90)
    print(f"{'model':<32}  {'fold':<6}  {'horizon':<8}  {'DirAcc':<8}  {'Sharpe':<8}  {'RMSE':<10}  {'MAE':<10}")
    print("-" * 90)
    for name, fold, hz, da, sh, rmse, mae in esn_rows:
        print(f"{name:<32}  {fold:<6}  {hz:<8}  {da:<8.3f}  {sh:<8.3f}  {rmse:<10.6f}  {mae:<10.6f}")
    print("=" * 90)

    # Why both ESN rows can be identical
    if len(esn_rows) >= 2:
        no_emb = esn_rows[0]   # (name, fold, hz, da, sh, rmse, mae)
        w_emb = esn_rows[1]
        same = (abs(no_emb[3] - w_emb[3]) < 1e-6 and abs(no_emb[4] - w_emb[4]) < 1e-6 and
                abs(no_emb[5] - w_emb[5]) < 1e-9 and abs(no_emb[6] - w_emb[6]) < 1e-9)
        if same:
            print("\nWhy are both scores the same?")
            print("  For this fold there are no real news features: the pipeline uses zero-filled")
            print("  news columns (see integration: 'No news features available for fold date range').")
            print("  So 'with embedding' = 10 technical + 28 zeros. The ESN's input weights times")
            print("  zero give zero, so the reservoir sees the same effective input as 'no embedding'.")
            print("  Same input -> same predictions -> same DirAcc, Sharpe, RMSE, MAE.")
            print("  To see a real difference, ensure news_features_28d.csv has dates overlapping")
            print("  the fold (e.g. run fetch_news + process_news_features for your symbol/dates).")
            print("")

    # 3) Other models (all with embedding)
    other_models = [m for m in grid_map.keys() if m != "esn"]
    for model_name in other_models:
        print(f"Training {model_name.upper()} (with embedding)...")
        for fold in folds:
            for horizon in horizons:
                model = create_model(model_name, grid_map[model_name], grid_index=0, horizon=horizon)
                res = run_experiment(
                    model=model,
                    fold=fold,
                    horizon=horizon,
                    splits_dir=splits_dir,
                    results_dir=None,
                    save_predictions=False,
                    include_news=True,
                )
                da = res["test_metrics"]["dir_acc"]
                rows.append((model_name, fold, horizon.replace("target_", ""), da))
                print(f"  Fold {fold} {horizon}: DirAcc = {da:.3f}")

    # Print directional accuracy comparison table (baseline first)
    print("\n" + "=" * 70)
    print("Directional accuracy comparison (test set only)")
    print("Baseline = ESN (no embedding). All others use technical + news unless noted.")
    print("=" * 70)
    col_w = {"model": 32, "fold": 6, "horizon": 8, "DirAcc": 8}
    header = f"{'model'.ljust(col_w['model'])}  {'fold'.ljust(col_w['fold'])}  {'horizon'.ljust(col_w['horizon'])}  {'DirAcc'.ljust(col_w['DirAcc'])}"
    print(header)
    print("-" * 70)
    for display_name, fold, hz, da in rows:
        print(f"{display_name.ljust(col_w['model'])}  {str(fold).ljust(col_w['fold'])}  {hz.ljust(col_w['horizon'])}  {da:.3f}")
    print("=" * 70)

    # Explain why ESN DirAcc can be the same with and without embedding
    for fold in folds:
        for horizon in horizons:
            base_da = baseline_diracc.get((fold, horizon))
            hz_short = horizon.replace("target_", "")
            with_emb = next((r[3] for r in rows if r[0] == "ESN (with embedding)" and r[1] == fold and r[2] == hz_short), None)
            if base_da is not None and with_emb is not None and abs(base_da - with_emb) < 0.001:
                print("\nWhy is DirAcc the same for ESN with vs without embedding?")
                print("  For this fold, news features are likely zero-filled or missing (no news dates overlap the fold).")
                print("  So 'with embedding' = 10 technical + 28 zeros → no extra signal → DirAcc matches the baseline (no embedding).")
                print("  To see a difference, ensure news_features_28d.csv has real data over the fold's date range.")
                break
        else:
            continue
        break


if __name__ == "__main__":
    main()
