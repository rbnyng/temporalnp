"""
Shared utilities for disturbance analysis and stratified evaluation.

Provides consistent metrics computation across baseline and main models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_disturbance_analysis(
    gedi_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pre_years: List[int],
    post_years: List[int],
    test_year: int,
    predictions: Optional[np.ndarray] = None
) -> dict:
    """
    Compute per-tile disturbance metrics and optionally stratified R².

    Disturbance intensity = (expected - observed) / expected
    where expected = (pre_mean + post_mean) / 2, or just pre_mean if no post.

    Args:
        gedi_df: Full GEDI DataFrame with all years
        test_df: Test set DataFrame (test year only)
        pre_years: Years before the event
        post_years: Years after the event
        test_year: The held-out test year
        predictions: Optional predictions aligned with test_df for error analysis

    Returns:
        Dictionary with per_tile stats, correlation, quartile_rmse,
        stratified_r2, and summary statistics.
    """
    from scipy import stats

    # Get test tiles
    test_tiles = test_df['tile_id'].unique()

    # Compute per-tile statistics
    tile_stats = []
    for tile_id in test_tiles:
        # Pre-event mean for this tile
        pre_data = gedi_df[(gedi_df['tile_id'] == tile_id) &
                           (gedi_df['year'].isin(pre_years))]
        pre_mean = pre_data['agbd'].mean() if len(pre_data) > 0 else np.nan

        # Post-event mean for this tile
        post_data = gedi_df[(gedi_df['tile_id'] == tile_id) &
                            (gedi_df['year'].isin(post_years))]
        post_mean = post_data['agbd'].mean() if len(post_data) > 0 else np.nan

        # Test year mean for this tile
        test_tile_df = test_df[test_df['tile_id'] == tile_id]
        test_mean = test_tile_df['agbd'].mean()
        n_test_shots = len(test_tile_df)

        # Prediction error for this tile (if predictions provided)
        tile_rmse = np.nan
        tile_mae = np.nan
        if predictions is not None:
            tile_mask = test_df['tile_id'] == tile_id
            tile_preds = predictions[tile_mask.values]
            tile_targets = test_tile_df['agbd'].values
            if len(tile_preds) > 0 and not np.any(np.isnan(tile_preds)):
                tile_rmse = np.sqrt(np.mean((tile_preds - tile_targets) ** 2))
                tile_mae = np.mean(np.abs(tile_preds - tile_targets))

        # Expected value (linear interpolation assumption)
        if not np.isnan(pre_mean) and not np.isnan(post_mean):
            expected = (pre_mean + post_mean) / 2
        elif not np.isnan(pre_mean):
            expected = pre_mean
        elif not np.isnan(post_mean):
            expected = post_mean
        else:
            expected = np.nan

        if not np.isnan(expected) and expected > 0:
            disturbance = (expected - test_mean) / expected
            change_from_pre = (pre_mean - test_mean) / pre_mean if not np.isnan(pre_mean) and pre_mean > 0 else np.nan
        else:
            disturbance = np.nan
            change_from_pre = np.nan

        tile_stats.append({
            'tile_id': tile_id,
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'test_mean': test_mean,
            'expected': expected,
            'disturbance': disturbance,
            'change_from_pre': change_from_pre,
            'tile_rmse': tile_rmse,
            'tile_mae': tile_mae,
            'n_test_shots': n_test_shots,
            'n_pre_shots': len(pre_data),
            'n_post_shots': len(post_data)
        })

    tile_df = pd.DataFrame(tile_stats)

    # Summary statistics
    valid_mask = ~tile_df['disturbance'].isna()

    # Compute stratified R² if predictions are provided
    stratified_r2 = None
    if predictions is not None:
        stratified_r2 = compute_stratified_r2(test_df, predictions, tile_df)

    # Compute correlation between disturbance and error (if predictions provided)
    correlation = {'pearson_r': None, 'p_value': None}
    quartile_rmse = {}
    if predictions is not None:
        valid_error_mask = valid_mask & (~tile_df['tile_rmse'].isna())
        if valid_error_mask.sum() >= 3:
            corr, p_value = stats.pearsonr(
                tile_df.loc[valid_error_mask, 'disturbance'],
                tile_df.loc[valid_error_mask, 'tile_rmse']
            )
            correlation = {
                'pearson_r': float(corr) if not np.isnan(corr) else None,
                'p_value': float(p_value) if not np.isnan(p_value) else None
            }

        # Quartile breakdown
        valid_tiles = tile_df[valid_error_mask].copy()
        if len(valid_tiles) >= 4:
            valid_tiles['disturbance_quartile'] = pd.qcut(
                valid_tiles['disturbance'], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high']
            )
            quartile_rmse = valid_tiles.groupby('disturbance_quartile')['tile_rmse'].mean().to_dict()
            quartile_rmse = {k: float(v) for k, v in quartile_rmse.items()}

    result = {
        'per_tile': tile_df.to_dict('records'),
        'pre_years': pre_years,
        'post_years': post_years,
        'correlation': correlation,
        'quartile_rmse': quartile_rmse,
        'summary': {
            'mean_disturbance': float(tile_df['disturbance'].mean()) if valid_mask.any() else None,
            'std_disturbance': float(tile_df['disturbance'].std()) if valid_mask.any() else None,
            'n_tiles_with_loss': int((tile_df['disturbance'] > 0).sum()),
            'n_tiles_with_gain': int((tile_df['disturbance'] < 0).sum()),
            'pct_tiles_major_loss': float((tile_df['disturbance'] > 0.3).mean() * 100) if valid_mask.any() else None,
            'mean_change_from_pre': float(tile_df['change_from_pre'].mean()) if valid_mask.any() else None
        }
    }

    if stratified_r2 is not None:
        result['stratified_r2'] = stratified_r2

    return result


def compute_stratified_r2(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    tile_df: pd.DataFrame
) -> dict:
    """
    Compute R² separately for stable forest vs fire-affected tiles.

    Stratification:
    - Stable tiles: disturbance < 0.2 (less than 20% biomass change)
    - Moderate tiles: 0.2 <= disturbance <= 0.5
    - Fire tiles: disturbance > 0.5 (more than 50% biomass loss)

    Args:
        test_df: Test DataFrame with tile_id and agbd columns
        predictions: Predictions aligned with test_df
        tile_df: DataFrame with tile_id and disturbance columns

    Returns:
        Dictionary with r2, rmse, n_shots, n_tiles for each stratum
    """
    from sklearn.metrics import r2_score

    # Create tile_id -> disturbance mapping
    tile_disturbance = dict(zip(tile_df['tile_id'], tile_df['disturbance']))

    # Map disturbance to each shot
    test_disturbance = test_df['tile_id'].map(tile_disturbance)

    # Define masks for each stratum
    stable_mask = ((test_disturbance < 0.2) & (~test_disturbance.isna())).values
    moderate_mask = ((test_disturbance >= 0.2) & (test_disturbance <= 0.5) & (~test_disturbance.isna())).values
    fire_mask = ((test_disturbance > 0.5) & (~test_disturbance.isna())).values

    results = {
        'stable': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'moderate': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'fire': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0}
    }

    targets = test_df['agbd'].values

    def compute_stratum_metrics(mask, stratum_name):
        if mask.sum() < 10:
            return None
        stratum_preds = predictions[mask]
        stratum_targets = targets[mask]
        valid = ~np.isnan(stratum_preds)
        if valid.sum() < 10:
            return None
        r2 = r2_score(stratum_targets[valid], stratum_preds[valid])
        rmse = np.sqrt(np.mean((stratum_preds[valid] - stratum_targets[valid]) ** 2))
        n_tiles = test_df.loc[mask, 'tile_id'].nunique()
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'n_shots': int(valid.sum()),
            'n_tiles': int(n_tiles)
        }

    stable_result = compute_stratum_metrics(stable_mask, 'stable')
    if stable_result:
        results['stable'] = stable_result

    moderate_result = compute_stratum_metrics(moderate_mask, 'moderate')
    if moderate_result:
        results['moderate'] = moderate_result

    fire_result = compute_stratum_metrics(fire_mask, 'fire')
    if fire_result:
        results['fire'] = fire_result

    return results


def print_disturbance_analysis(disturbance_analysis: dict, indent: str = "  ") -> None:
    """Print disturbance analysis results in a consistent format."""
    summary = disturbance_analysis['summary']

    print(f"\n{indent}Disturbance Analysis:")
    if summary['mean_disturbance'] is not None:
        print(f"{indent}  Mean disturbance: {summary['mean_disturbance']:.1%}")
        print(f"{indent}  Tiles with biomass loss: {summary['n_tiles_with_loss']}")
        print(f"{indent}  Tiles with major loss (>30%): {summary['pct_tiles_major_loss']:.1f}%")

        corr = disturbance_analysis['correlation']
        if corr['pearson_r'] is not None:
            print(f"{indent}  Error-disturbance correlation: r={corr['pearson_r']:.3f} "
                  f"(p={corr['p_value']:.3f})")

        if disturbance_analysis['quartile_rmse']:
            print(f"{indent}  RMSE by disturbance quartile:")
            for q, rmse_val in disturbance_analysis['quartile_rmse'].items():
                print(f"{indent}    {q}: {rmse_val:.2f} Mg/ha")
    else:
        print(f"{indent}  No valid disturbance data")


def print_stratified_r2(stratified_r2: dict, indent: str = "  ") -> None:
    """Print stratified R² results in a consistent format."""
    print(f"\n{indent}Stratified R² by Disturbance Level:")

    if stratified_r2['stable']['r2'] is not None:
        s = stratified_r2['stable']
        print(f"{indent}  Stable (<20% change):   R²={s['r2']:.4f}, "
              f"RMSE={s['rmse']:.2f} Mg/ha ({s['n_shots']} shots, {s['n_tiles']} tiles)")
    else:
        print(f"{indent}  Stable (<20% change):   Not enough data")

    if stratified_r2['moderate']['r2'] is not None:
        m = stratified_r2['moderate']
        print(f"{indent}  Moderate (20-50%):      R²={m['r2']:.4f}, "
              f"RMSE={m['rmse']:.2f} Mg/ha ({m['n_shots']} shots, {m['n_tiles']} tiles)")
    else:
        print(f"{indent}  Moderate (20-50%):      Not enough data")

    if stratified_r2['fire']['r2'] is not None:
        f = stratified_r2['fire']
        print(f"{indent}  Fire (>50% loss):       R²={f['r2']:.4f}, "
              f"RMSE={f['rmse']:.2f} Mg/ha ({f['n_shots']} shots, {f['n_tiles']} tiles)")
    else:
        print(f"{indent}  Fire (>50% loss):       Not enough data")


def aggregate_stratified_r2(results_list: list) -> dict:
    """
    Aggregate stratified R² metrics across multiple experiment runs.

    Args:
        results_list: List of result dictionaries, each with 'stratified_r2' key

    Returns:
        Aggregated statistics with mean, std for each stratum
    """
    strata = ['stable', 'moderate', 'fire']
    aggregated = {}

    for stratum in strata:
        r2_values = [r['stratified_r2'][stratum]['r2']
                     for r in results_list
                     if r.get('stratified_r2') and r['stratified_r2'].get(stratum)
                     and r['stratified_r2'][stratum].get('r2') is not None]
        rmse_values = [r['stratified_r2'][stratum]['rmse']
                       for r in results_list
                       if r.get('stratified_r2') and r['stratified_r2'].get(stratum)
                       and r['stratified_r2'][stratum].get('rmse') is not None]

        aggregated[stratum] = {
            'r2_mean': float(np.mean(r2_values)) if r2_values else None,
            'r2_std': float(np.std(r2_values)) if r2_values else None,
            'rmse_mean': float(np.mean(rmse_values)) if rmse_values else None,
            'rmse_std': float(np.std(rmse_values)) if rmse_values else None,
            'r2_values': r2_values,
            'rmse_values': rmse_values
        }

    return aggregated


def print_aggregated_stratified_r2(aggregated: dict, indent: str = "") -> None:
    """Print aggregated stratified R² results."""
    print(f"\n{indent}Stratified R² by Disturbance Level:")

    for stratum, label in [('stable', 'Stable (<20% change)'),
                           ('moderate', 'Moderate (20-50%)'),
                           ('fire', 'Fire (>50% loss)')]:
        s = aggregated[stratum]
        if s['r2_mean'] is not None:
            print(f"{indent}  {label:22s} R²={s['r2_mean']:.4f} ± {s['r2_std']:.4f}, "
                  f"RMSE={s['rmse_mean']:.2f} ± {s['rmse_std']:.2f}")
        else:
            print(f"{indent}  {label:22s} Not enough data")
