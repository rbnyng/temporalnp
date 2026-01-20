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
    tile_df: pd.DataFrame,
    thresholds: dict = None
) -> dict:
    """
    Compute R² separately for stable forest vs disturbed tiles.

    Default stratification (based on empirical Dixie Fire observations):
    - Stable tiles: disturbance < 0.1 (less than 10% biomass change)
    - Moderate tiles: 0.1 <= disturbance <= 0.3
    - Disturbed tiles: disturbance > 0.3 (more than 30% biomass loss)

    Args:
        test_df: Test DataFrame with tile_id and agbd columns
        predictions: Predictions aligned with test_df
        tile_df: DataFrame with tile_id and disturbance columns
        thresholds: Optional dict with 'stable_max' and 'disturbed_min' keys

    Returns:
        Dictionary with r2, rmse, n_shots, n_tiles for each stratum
    """
    from sklearn.metrics import r2_score

    # Default thresholds based on empirical fire observations
    if thresholds is None:
        thresholds = {'stable_max': 0.1, 'disturbed_min': 0.3}

    stable_max = thresholds.get('stable_max', 0.1)
    disturbed_min = thresholds.get('disturbed_min', 0.3)

    # Create tile_id -> disturbance mapping
    tile_disturbance = dict(zip(tile_df['tile_id'], tile_df['disturbance']))

    # Map disturbance to each shot
    test_disturbance = test_df['tile_id'].map(tile_disturbance)

    # Define masks for each stratum
    stable_mask = ((test_disturbance < stable_max) & (~test_disturbance.isna())).values
    moderate_mask = ((test_disturbance >= stable_max) & (test_disturbance <= disturbed_min) & (~test_disturbance.isna())).values
    disturbed_mask = ((test_disturbance > disturbed_min) & (~test_disturbance.isna())).values

    results = {
        'stable': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'moderate': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'disturbed': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'thresholds': thresholds
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

    disturbed_result = compute_stratum_metrics(disturbed_mask, 'disturbed')
    if disturbed_result:
        results['disturbed'] = disturbed_result

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
    thresholds = stratified_r2.get('thresholds', {'stable_max': 0.1, 'disturbed_min': 0.3})
    stable_max = int(thresholds['stable_max'] * 100)
    disturbed_min = int(thresholds['disturbed_min'] * 100)

    print(f"\n{indent}Stratified R² by Disturbance Level:")

    if stratified_r2['stable']['r2'] is not None:
        s = stratified_r2['stable']
        print(f"{indent}  Stable (<{stable_max}% change):    R²={s['r2']:.4f}, "
              f"RMSE={s['rmse']:.2f} Mg/ha ({s['n_shots']} shots, {s['n_tiles']} tiles)")
    else:
        print(f"{indent}  Stable (<{stable_max}% change):    Not enough data")

    if stratified_r2['moderate']['r2'] is not None:
        m = stratified_r2['moderate']
        print(f"{indent}  Moderate ({stable_max}-{disturbed_min}%):      R²={m['r2']:.4f}, "
              f"RMSE={m['rmse']:.2f} Mg/ha ({m['n_shots']} shots, {m['n_tiles']} tiles)")
    else:
        print(f"{indent}  Moderate ({stable_max}-{disturbed_min}%):      Not enough data")

    # Support both 'disturbed' (new) and 'fire' (legacy) keys
    disturbed_key = 'disturbed' if 'disturbed' in stratified_r2 else 'fire'
    if stratified_r2.get(disturbed_key, {}).get('r2') is not None:
        d = stratified_r2[disturbed_key]
        print(f"{indent}  Disturbed (>{disturbed_min}% loss): R²={d['r2']:.4f}, "
              f"RMSE={d['rmse']:.2f} Mg/ha ({d['n_shots']} shots, {d['n_tiles']} tiles)")
    else:
        print(f"{indent}  Disturbed (>{disturbed_min}% loss): Not enough data")


def aggregate_stratified_r2(results_list: list) -> dict:
    """
    Aggregate stratified R² metrics across multiple experiment runs.

    Note: This simple aggregation averages R² values across seeds, but each seed
    may have different tiles in the test set. For more robust analysis, use
    compute_pooled_stratified_r2() which pools predictions across all seeds.

    Args:
        results_list: List of result dictionaries, each with 'stratified_r2' key

    Returns:
        Aggregated statistics with mean, std for each stratum
    """
    # Support both 'disturbed' (new) and 'fire' (legacy) keys
    strata = ['stable', 'moderate', 'disturbed']
    aggregated = {}

    for stratum in strata:
        # Try both new and legacy key names
        keys_to_try = [stratum] if stratum != 'disturbed' else ['disturbed', 'fire']

        r2_values = []
        rmse_values = []
        for r in results_list:
            if not r.get('stratified_r2'):
                continue
            for key in keys_to_try:
                if r['stratified_r2'].get(key) and r['stratified_r2'][key].get('r2') is not None:
                    r2_values.append(r['stratified_r2'][key]['r2'])
                    if r['stratified_r2'][key].get('rmse') is not None:
                        rmse_values.append(r['stratified_r2'][key]['rmse'])
                    break

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
    thresholds = aggregated.get('thresholds', {'stable_max': 0.1, 'disturbed_min': 0.3})
    stable_max = int(thresholds.get('stable_max', 0.1) * 100)
    disturbed_min = int(thresholds.get('disturbed_min', 0.3) * 100)

    print(f"\n{indent}Stratified R² by Disturbance Level:")

    for stratum, label in [('stable', f'Stable (<{stable_max}% change)'),
                           ('moderate', f'Moderate ({stable_max}-{disturbed_min}%)'),
                           ('disturbed', f'Disturbed (>{disturbed_min}% loss)')]:
        s = aggregated.get(stratum, {})
        if s.get('r2_mean') is not None:
            n_seeds = len(s.get('r2_values', []))
            print(f"{indent}  {label:24s} R²={s['r2_mean']:.4f} ± {s['r2_std']:.4f}, "
                  f"RMSE={s['rmse_mean']:.2f} ± {s['rmse_std']:.2f} (n={n_seeds} seeds)")
        else:
            print(f"{indent}  {label:24s} Not enough data")


def compute_pooled_stratified_r2(
    output_dir,
    gedi_df: pd.DataFrame,
    pre_years: List[int],
    post_years: List[int],
    thresholds: dict = None
) -> dict:
    """
    Compute stratified R² by pooling predictions across all seeds.

    This is more robust than averaging R² values from different seeds because
    each tile appears exactly once in the pooled predictions.

    Args:
        output_dir: Path to experiment output directory containing seed subdirs
        gedi_df: Full GEDI DataFrame with all years (for computing disturbance)
        pre_years: Years before the event
        post_years: Years after the event
        thresholds: Optional dict with 'stable_max' and 'disturbed_min' keys

    Returns:
        Dictionary with pooled stratified R² metrics
    """
    from pathlib import Path
    from sklearn.metrics import r2_score

    if thresholds is None:
        thresholds = {'stable_max': 0.1, 'disturbed_min': 0.3}

    output_dir = Path(output_dir)

    # Collect predictions from all seeds
    all_predictions = []

    for seed_dir in sorted(output_dir.glob('seed_*')):
        pred_file = seed_dir / 'test_predictions.parquet'
        if pred_file.exists():
            pred_df = pd.read_parquet(pred_file)
            all_predictions.append(pred_df)

    if not all_predictions:
        return {'error': 'No prediction files found'}

    # Concatenate all predictions
    pooled_df = pd.concat(all_predictions, ignore_index=True)

    # Compute disturbance per tile using pre/post data
    tile_stats = []
    for tile_id in pooled_df['tile_id'].unique():
        pre_data = gedi_df[(gedi_df['tile_id'] == tile_id) &
                           (gedi_df['year'].isin(pre_years))]
        pre_mean = pre_data['agbd'].mean() if len(pre_data) > 0 else np.nan

        post_data = gedi_df[(gedi_df['tile_id'] == tile_id) &
                            (gedi_df['year'].isin(post_years))]
        post_mean = post_data['agbd'].mean() if len(post_data) > 0 else np.nan

        # Expected value (linear interpolation assumption)
        if not np.isnan(pre_mean) and not np.isnan(post_mean):
            expected = (pre_mean + post_mean) / 2
        elif not np.isnan(pre_mean):
            expected = pre_mean
        elif not np.isnan(post_mean):
            expected = post_mean
        else:
            expected = np.nan

        # Test mean from pooled predictions
        tile_data = pooled_df[pooled_df['tile_id'] == tile_id]
        test_mean = tile_data['agbd'].mean()

        if not np.isnan(expected) and expected > 0:
            disturbance = (expected - test_mean) / expected
        else:
            disturbance = np.nan

        tile_stats.append({
            'tile_id': tile_id,
            'disturbance': disturbance,
            'n_shots': len(tile_data)
        })

    tile_df = pd.DataFrame(tile_stats)

    # Map disturbance to each shot
    tile_disturbance = dict(zip(tile_df['tile_id'], tile_df['disturbance']))
    pooled_df['disturbance'] = pooled_df['tile_id'].map(tile_disturbance)

    # Define strata
    stable_max = thresholds['stable_max']
    disturbed_min = thresholds['disturbed_min']

    stable_mask = (pooled_df['disturbance'] < stable_max) & (~pooled_df['disturbance'].isna())
    moderate_mask = (pooled_df['disturbance'] >= stable_max) & (pooled_df['disturbance'] <= disturbed_min) & (~pooled_df['disturbance'].isna())
    disturbed_mask = (pooled_df['disturbance'] > disturbed_min) & (~pooled_df['disturbance'].isna())

    results = {
        'stable': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'moderate': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'disturbed': {'r2': None, 'rmse': None, 'n_shots': 0, 'n_tiles': 0},
        'thresholds': thresholds,
        'pooled': True,
        'total_tiles': len(tile_df),
        'total_shots': len(pooled_df)
    }

    def compute_stratum(mask, name):
        subset = pooled_df[mask]
        if len(subset) < 10:
            return None
        valid = ~subset['pred'].isna()
        if valid.sum() < 10:
            return None
        preds = subset.loc[valid, 'pred'].values
        targets = subset.loc[valid, 'agbd'].values
        r2 = r2_score(targets, preds)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'n_shots': int(valid.sum()),
            'n_tiles': int(subset['tile_id'].nunique())
        }

    for mask, name in [(stable_mask, 'stable'), (moderate_mask, 'moderate'), (disturbed_mask, 'disturbed')]:
        result = compute_stratum(mask, name)
        if result:
            results[name] = result

    return results
