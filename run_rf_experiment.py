"""
Run RF/XGBoost baseline experiments for comparison with Neural Process models.

This script trains traditional ML baselines (Random Forest, XGBoost) using:
- Flattened GeoTessera embedding patches as features
- Spatial coordinates (lon, lat)
- Optional temporal encoding (sin_doy, cos_doy, norm_time)

Usage:
    python run_rf_experiment.py \
        --region_bbox -122.5 40.5 -121.5 41.5 \
        --train_years 2019 2020 2022 2023 \
        --test_year 2021 \
        --model rf \
        --n_seeds 5 \
        --output_dir ./results/rf_baseline
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from time import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from baselines.models import QuantileRandomForest, QuantileXGBoost
from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import compute_temporal_encoding
from data.spatial_cv import SpatiotemporalSplitter
from utils.config import save_config, _make_serializable
from utils.evaluation import compute_calibration_metrics
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd
from utils.disturbance import (
    compute_disturbance_analysis,
    print_disturbance_analysis,
    print_stratified_r2,
    aggregate_stratified_r2,
    print_aggregated_stratified_r2,
    compute_pooled_stratified_r2
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run RF/XGBoost Baseline Experiment')

    # Region and temporal arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--fire_shapefile', type=str, default=None,
                        help='Optional: Path to fire boundary shapefile to filter GEDI shots')
    parser.add_argument('--train_years', type=int, nargs='+', required=True,
                        help='Years to use for training')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Year to hold out for testing')
    parser.add_argument('--test_months', type=int, nargs='+', default=None,
                        help='Optional: Filter test year to specific months')

    # Model arguments
    parser.add_argument('--model', type=str, default='rf',
                        choices=['rf', 'xgb', 'quantile_rf', 'quantile_xgb'],
                        help='Model type: rf (Random Forest), xgb (XGBoost), quantile_rf (Quantile RF), quantile_xgb (Quantile XGBoost)')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees/estimators')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum tree depth (default: 6 for better generalization)')
    parser.add_argument('--min_samples_leaf', type=int, default=5,
                        help='Minimum samples per leaf node')
    parser.add_argument('--include_temporal', action='store_true',
                        help='Include temporal encoding in features (5D coords instead of 2D)')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')

    # Experiment arguments
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds to run')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting seed value')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    # Infrastructure
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='GEDI cache directory')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Embeddings directory')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for RF (-1 for all cores)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--buffer_size', type=float, default=0.1,
                        help='Buffer size in degrees for spatial CV')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def filter_shots_by_shapefile(df: pd.DataFrame, shapefile_path: str) -> pd.DataFrame:
    """Filter GEDI shots to those inside a shapefile boundary."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')

    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
        crs='EPSG:4326'
    )

    geometry = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    within_mask = points.within(geometry)

    filtered_df = df[within_mask.values].copy()
    print(f"Filtered to {len(filtered_df)} shots inside fire perimeter (from {len(df)})")

    return filtered_df


def prepare_features(
    df: pd.DataFrame,
    global_bounds: Tuple[float, float, float, float],
    temporal_bounds: Optional[Tuple[float, float]] = None,
    include_temporal: bool = False
) -> np.ndarray:
    """
    Prepare feature matrix from GEDI data.

    Features:
    - Flattened embedding patches (patch_size * patch_size * 128)
    - Normalized spatial coordinates (lon, lat)
    - Optional: temporal encoding (sin_doy, cos_doy, norm_time)

    Args:
        df: DataFrame with embedding_patch, longitude, latitude, time columns
        global_bounds: (lon_min, lat_min, lon_max, lat_max) for normalization
        temporal_bounds: (t_min, t_max) as unix timestamps
        include_temporal: Whether to include temporal encoding

    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    # Flatten embeddings
    embeddings = np.stack(df['embedding_patch'].values)
    n_samples = embeddings.shape[0]
    embedding_features = embeddings.reshape(n_samples, -1)

    # Normalize spatial coordinates
    coords = df[['longitude', 'latitude']].values
    coords_norm = normalize_coords(coords, global_bounds)

    # Combine features
    if include_temporal and temporal_bounds is not None:
        temporal = compute_temporal_encoding(df['time'], temporal_bounds)
        features = np.concatenate([embedding_features, coords_norm, temporal], axis=1)
    else:
        features = np.concatenate([embedding_features, coords_norm], axis=1)

    return features


def train_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    n_jobs: int = -1,
    random_state: int = 42
) -> object:
    """
    Train a Random Forest or XGBoost model.

    Args:
        X_train: Training features
        y_train: Training targets
        model_type: 'rf', 'xgb', 'quantile_rf', or 'quantile_xgb'
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
        n_jobs: Number of parallel jobs
        random_state: Random seed

    Returns:
        Trained model
    """
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0
        )
        model.fit(X_train, y_train.ravel())

    elif model_type == 'quantile_rf':
        # Proper Quantile RF using quantile-forest package (Meinshausen 2006)
        # Default quantiles (0.025, 0.975) for 95% prediction interval -> std estimation
        model = QuantileRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            default_quantiles=(0.025, 0.975)
        )
        model.fit(X_train, y_train.ravel())

    elif model_type == 'xgb':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else 6,
            learning_rate=0.1,
            min_child_weight=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0
        )
        model.fit(X_train, y_train.ravel())

    elif model_type == 'quantile_xgb':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        # Proper Quantile XGBoost using native quantile regression
        model = QuantileXGBoost(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else 6,
            learning_rate=0.1,
            min_child_weight=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            quantiles=(0.159, 0.5, 0.841)
        )
        model.fit(X_train, y_train.ravel())

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def predict_with_uncertainty(
    model: object,
    X: np.ndarray,
    model_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions with uncertainty estimates.

    For RF: Use tree predictions variance (ensemble disagreement)
    For quantile_rf: Use proper quantile predictions (lower/upper quantiles)
    For XGBoost: Use tree predictions variance
    For quantile_xgb: Use proper quantile predictions

    Returns:
        (predictions, uncertainties) where uncertainty is standard deviation estimate
    """
    if model_type == 'rf':
        # Get predictions from all trees - use ensemble disagreement
        tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        predictions = tree_predictions.mean(axis=0)
        uncertainties = tree_predictions.std(axis=0)

    elif model_type == 'quantile_rf':
        # Proper quantile-based uncertainty using quantile-forest package
        # Uses default_quantiles (0.025, 0.975) to estimate std via z-scores
        predictions, uncertainties = model.predict_with_std(X)

    elif model_type == 'xgb':
        predictions = model.predict(X)
        # For standard XGB, we don't have good uncertainty estimates
        # Use a constant relative uncertainty based on training residuals
        # This is a placeholder - users should use quantile_xgb for proper UQ
        uncertainties = np.abs(predictions) * 0.2  # 20% relative uncertainty

    elif model_type == 'quantile_xgb':
        # Proper quantile-based uncertainty from native XGBoost quantile regression
        quantile_preds = model.predict_quantiles(X, quantiles=[0.159, 0.5, 0.841])
        predictions = quantile_preds[:, 1]  # Median
        uncertainties = (quantile_preds[:, 2] - quantile_preds[:, 0]) / 2.0

    else:
        predictions = model.predict(X)
        uncertainties = np.zeros_like(predictions)

    return predictions, uncertainties


def run_single_seed(
    seed: int,
    gedi_df: pd.DataFrame,
    args,
    global_bounds: Tuple[float, float, float, float],
    temporal_bounds: Tuple[float, float],
    seed_output_dir: Path
) -> dict:
    """Run training and evaluation for a single seed."""
    set_seed(seed)

    seed_output_dir.mkdir(parents=True, exist_ok=True)

    # Spatiotemporal split
    splitter = SpatiotemporalSplitter(
        gedi_df,
        train_years=args.train_years,
        test_year=args.test_year,
        buffer_size=args.buffer_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=seed
    )
    train_df, val_df, test_df = splitter.split()

    # Filter test data to specific months if specified
    if args.test_months:
        test_df['month'] = pd.to_datetime(test_df['time']).dt.month
        original_count = len(test_df)
        test_df = test_df[test_df['month'].isin(args.test_months)].copy()
        test_df = test_df.drop(columns=['month'])
        print(f"  Filtered test data to months {args.test_months}: {len(test_df)} shots (from {original_count})")

    print(f"  Train: {len(train_df)} shots, Val: {len(val_df)} shots, Test: {len(test_df)} shots")

    # Prepare features
    X_train = prepare_features(train_df, global_bounds, temporal_bounds, args.include_temporal)
    X_val = prepare_features(val_df, global_bounds, temporal_bounds, args.include_temporal)
    X_test = prepare_features(test_df, global_bounds, temporal_bounds, args.include_temporal)

    # Prepare targets (log-normalized AGBD)
    y_train = normalize_agbd(train_df['agbd'].values[:, None])
    y_val = normalize_agbd(val_df['agbd'].values[:, None])
    y_test_log = normalize_agbd(test_df['agbd'].values[:, None])
    y_test_linear = test_df['agbd'].values

    print(f"  Feature dimensions: {X_train.shape[1]}")

    # Train model
    start_time = time()
    model = train_rf_model(
        X_train, y_train,
        model_type=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=seed
    )
    train_time = time() - start_time
    print(f"  Training time: {train_time:.1f}s")

    # Predictions
    pred_log, unc_log = predict_with_uncertainty(model, X_test, args.model)
    pred_linear = denormalize_agbd(pred_log)
    unc_linear = denormalize_agbd(pred_log + unc_log) - pred_linear  # Approximate

    # Compute metrics (log space)
    log_r2 = r2_score(y_test_log.ravel(), pred_log)
    log_rmse = np.sqrt(mean_squared_error(y_test_log.ravel(), pred_log))

    # Compute metrics (linear space)
    linear_r2 = r2_score(y_test_linear, pred_linear)
    linear_rmse = np.sqrt(mean_squared_error(y_test_linear, pred_linear))
    linear_mae = mean_absolute_error(y_test_linear, pred_linear)

    # Calibration metrics
    calibration = compute_calibration_metrics(pred_log, y_test_log.ravel(), unc_log)

    # Split train years into pre/post relative to test year
    pre_years = [y for y in args.train_years if y < args.test_year]
    post_years = [y for y in args.train_years if y > args.test_year]

    # Disturbance analysis
    disturbance_analysis = compute_disturbance_analysis(
        gedi_df, test_df, pre_years, post_years, args.test_year,
        predictions=pred_linear
    )

    # Save model
    with open(seed_output_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save predictions
    test_df_out = test_df[['latitude', 'longitude', 'agbd', 'time', 'tile_id']].copy()
    test_df_out['pred'] = pred_linear
    test_df_out['unc'] = unc_linear
    test_df_out['residual'] = y_test_linear - pred_linear
    test_df_out.to_parquet(seed_output_dir / 'test_predictions.parquet')

    # Save tile disturbance
    tile_disturbance_df = pd.DataFrame(disturbance_analysis['per_tile'])
    tile_disturbance_df.to_parquet(seed_output_dir / 'tile_disturbance.parquet')

    # Results
    results = {
        'seed': seed,
        'status': 'success',
        'train_time': train_time,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'test_metrics': {
            'log_r2': float(log_r2),
            'log_rmse': float(log_rmse),
            'linear_r2': float(linear_r2),
            'linear_rmse': float(linear_rmse),
            'linear_mae': float(linear_mae),
            'z_mean': calibration.get('z_mean'),
            'z_std': calibration.get('z_std'),
            'coverage_1sigma': calibration.get('coverage_1sigma'),
            'coverage_2sigma': calibration.get('coverage_2sigma'),
            'coverage_3sigma': calibration.get('coverage_3sigma'),
        },
        'disturbance': {
            'pre_years': pre_years,
            'post_years': post_years,
            'correlation': disturbance_analysis['correlation'],
            'quartile_rmse': disturbance_analysis['quartile_rmse'],
            'summary': disturbance_analysis['summary']
        },
        'stratified_r2': disturbance_analysis.get('stratified_r2', {})
    }

    with open(seed_output_dir / 'results.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)

    return results


def aggregate_results(all_results: list) -> dict:
    """Aggregate results across seeds."""
    successful = [r for r in all_results if r.get('status') == 'success']

    if not successful:
        return {
            'error': 'No successful runs',
            'n_seeds': len(all_results),
            'n_successful': 0
        }

    # Extract metrics
    log_r2 = [r['test_metrics'].get('log_r2', 0) for r in successful if r.get('test_metrics')]
    log_rmse = [r['test_metrics'].get('log_rmse', 0) for r in successful if r.get('test_metrics')]
    linear_r2 = [r['test_metrics'].get('linear_r2', 0) for r in successful if r.get('test_metrics')]
    linear_rmse = [r['test_metrics'].get('linear_rmse', 0) for r in successful if r.get('test_metrics')]
    linear_mae = [r['test_metrics'].get('linear_mae', 0) for r in successful if r.get('test_metrics')]
    train_times = [r.get('train_time', 0) for r in successful]

    # Extract calibration metrics
    z_mean = [r['test_metrics'].get('z_mean') for r in successful
              if r.get('test_metrics') and r['test_metrics'].get('z_mean') is not None]
    z_std = [r['test_metrics'].get('z_std') for r in successful
             if r.get('test_metrics') and r['test_metrics'].get('z_std') is not None]
    coverage_1sigma = [r['test_metrics'].get('coverage_1sigma') for r in successful
                       if r.get('test_metrics') and r['test_metrics'].get('coverage_1sigma') is not None]
    coverage_2sigma = [r['test_metrics'].get('coverage_2sigma') for r in successful
                       if r.get('test_metrics') and r['test_metrics'].get('coverage_2sigma') is not None]

    # Extract disturbance metrics
    error_disturbance_corr = [r['disturbance']['correlation']['pearson_r']
                               for r in successful if r.get('disturbance') and
                               r['disturbance'].get('correlation') and
                               r['disturbance']['correlation'].get('pearson_r') is not None]

    # Use shared utility for stratified R² aggregation
    stratified_r2_agg = aggregate_stratified_r2(successful)

    aggregated = {
        'n_seeds': len(all_results),
        'n_successful': len(successful),
        'log_r2': {
            'mean': float(np.mean(log_r2)) if log_r2 else None,
            'std': float(np.std(log_r2)) if log_r2 else None,
            'min': float(np.min(log_r2)) if log_r2 else None,
            'max': float(np.max(log_r2)) if log_r2 else None,
            'values': log_r2
        },
        'log_rmse': {
            'mean': float(np.mean(log_rmse)) if log_rmse else None,
            'std': float(np.std(log_rmse)) if log_rmse else None,
            'values': log_rmse
        },
        'linear_r2': {
            'mean': float(np.mean(linear_r2)) if linear_r2 else None,
            'std': float(np.std(linear_r2)) if linear_r2 else None,
            'values': linear_r2
        },
        'linear_rmse': {
            'mean': float(np.mean(linear_rmse)) if linear_rmse else None,
            'std': float(np.std(linear_rmse)) if linear_rmse else None,
            'values': linear_rmse
        },
        'linear_mae': {
            'mean': float(np.mean(linear_mae)) if linear_mae else None,
            'std': float(np.std(linear_mae)) if linear_mae else None,
            'values': linear_mae
        },
        'train_time': {
            'mean': float(np.mean(train_times)) if train_times else None,
            'total': float(np.sum(train_times)) if train_times else None
        },
        'calibration': {
            'z_mean': {
                'mean': float(np.mean(z_mean)) if z_mean else None,
                'std': float(np.std(z_mean)) if z_mean else None,
            },
            'z_std': {
                'mean': float(np.mean(z_std)) if z_std else None,
                'std': float(np.std(z_std)) if z_std else None,
            },
            'coverage_1sigma': {
                'mean': float(np.mean(coverage_1sigma)) if coverage_1sigma else None,
                'std': float(np.std(coverage_1sigma)) if coverage_1sigma else None,
            },
            'coverage_2sigma': {
                'mean': float(np.mean(coverage_2sigma)) if coverage_2sigma else None,
                'std': float(np.std(coverage_2sigma)) if coverage_2sigma else None,
            }
        },
        'disturbance': {
            'error_disturbance_correlation': {
                'mean': float(np.mean(error_disturbance_corr)) if error_disturbance_corr else None,
                'std': float(np.std(error_disturbance_corr)) if error_disturbance_corr else None,
            }
        },
        'stratified_r2': stratified_r2_agg
    }

    return aggregated


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model names for display
    model_names = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'quantile_rf': 'Quantile Random Forest',
        'quantile_xgb': 'Quantile XGBoost'
    }

    print("=" * 80)
    print(f"{model_names[args.model]} Baseline Experiment")
    print("=" * 80)
    print(f"Region: {args.region_bbox}")
    print(f"Train years: {args.train_years}")
    print(f"Test year: {args.test_year}")
    print(f"Model: {args.model} (n_estimators={args.n_estimators})")
    print(f"Include temporal: {args.include_temporal}")
    print(f"Seeds: {args.n_seeds} (starting from {args.start_seed})")
    print(f"Output: {output_dir}")
    print()

    # Save experiment config
    experiment_config = {
        'region_bbox': args.region_bbox,
        'fire_shapefile': args.fire_shapefile,
        'train_years': args.train_years,
        'test_year': args.test_year,
        'test_months': args.test_months,
        'model': args.model,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_leaf': args.min_samples_leaf,
        'include_temporal': args.include_temporal,
        'patch_size': args.patch_size,
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'started_at': datetime.now().isoformat()
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)

    # Query GEDI data
    all_years = sorted(set(args.train_years + [args.test_year]))
    start_year = min(all_years)
    end_year = max(all_years)

    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=f'{start_year}-01-01',
        end_time=f'{end_year}-12-31',
        max_agbd=500.0
    )
    gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
    print(f"Retrieved {len(gedi_df)} shots across {gedi_df['tile_id'].nunique()} tiles")

    # Apply fire shapefile filter if specified
    if args.fire_shapefile:
        print(f"\nApplying fire perimeter filter from: {args.fire_shapefile}")
        gedi_df = filter_shots_by_shapefile(gedi_df, args.fire_shapefile)
        if len(gedi_df) == 0:
            print("No GEDI shots inside fire perimeter. Exiting.")
            return

    print(f"Shots per year: {dict(gedi_df['year'].value_counts().sort_index())}")

    # Extract embeddings
    print("\nStep 2: Extracting GeoTessera embeddings...")
    extractor = EmbeddingExtractor(
        year=all_years[0],
        patch_size=args.patch_size,
        embeddings_dir=args.embeddings_dir
    )

    all_dfs = []
    for year in all_years:
        year_df = gedi_df[gedi_df['year'] == year].copy()
        if len(year_df) == 0:
            print(f"  Year {year}: No shots found, skipping")
            continue
        extractor.set_year(year)
        year_df = extractor.extract_patches_batch(year_df, verbose=True, cache_dir=args.cache_dir)
        all_dfs.append(year_df)

    gedi_df = pd.concat(all_dfs, ignore_index=True)
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"\nRetained {len(gedi_df)} shots with valid embeddings")

    # Save processed data
    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    # Compute global bounds
    global_bounds = (
        gedi_df['longitude'].min(),
        gedi_df['latitude'].min(),
        gedi_df['longitude'].max(),
        gedi_df['latitude'].max()
    )

    # Compute temporal bounds
    timestamps = pd.to_datetime(gedi_df['time'])
    unix_time = timestamps.astype(np.int64) / 1e9
    temporal_bounds = (unix_time.min(), unix_time.max())

    # Run experiments
    print(f"\nStep 3: Running {args.n_seeds} seeds...")
    all_results = []
    seeds = [args.start_seed + i for i in range(args.n_seeds)]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_output_dir = output_dir / f'seed_{seed}'
        try:
            result = run_single_seed(
                seed, gedi_df, args, global_bounds, temporal_bounds, seed_output_dir
            )
            all_results.append(result)
            print(f"  Log R²: {result['test_metrics']['log_r2']:.4f}, "
                  f"Linear R²: {result['test_metrics']['linear_r2']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({'seed': seed, 'status': 'failed', 'error': str(e)})

        # Save intermediate results
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(_make_serializable(all_results), f, indent=2)

    # Aggregate results
    aggregated = aggregate_results(all_results)
    aggregated['experiment_config'] = experiment_config
    aggregated['completed_at'] = datetime.now().isoformat()

    # Compute pooled stratified R²
    try:
        pre_years = [y for y in args.train_years if y < args.test_year]
        post_years = [y for y in args.train_years if y > args.test_year]
        pooled_stratified = compute_pooled_stratified_r2(
            output_dir, gedi_df, pre_years, post_years
        )
        aggregated['pooled_stratified_r2'] = pooled_stratified
    except Exception as e:
        print(f"\nWarning: Could not compute pooled stratified R²: {e}")

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(_make_serializable(aggregated), f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Model: {model_names[args.model]}")
    print(f"Successful runs: {aggregated['n_successful']}/{aggregated['n_seeds']}")

    if aggregated.get('log_r2', {}).get('mean') is not None:
        print(f"\nLog-space metrics:")
        print(f"  R²:   {aggregated['log_r2']['mean']:.4f} ± {aggregated['log_r2']['std']:.4f}")
        print(f"  RMSE: {aggregated['log_rmse']['mean']:.4f} ± {aggregated['log_rmse']['std']:.4f}")

    if aggregated.get('linear_r2', {}).get('mean') is not None:
        print(f"\nLinear-space metrics:")
        print(f"  R²:   {aggregated['linear_r2']['mean']:.4f} ± {aggregated['linear_r2']['std']:.4f}")
        print(f"  RMSE: {aggregated['linear_rmse']['mean']:.2f} ± {aggregated['linear_rmse']['std']:.2f} Mg/ha")
        print(f"  MAE:  {aggregated['linear_mae']['mean']:.2f} ± {aggregated['linear_mae']['std']:.2f} Mg/ha")

    if aggregated.get('train_time', {}).get('mean') is not None:
        print(f"\nTraining time: {aggregated['train_time']['mean']:.1f}s per seed")

    if aggregated.get('calibration', {}).get('coverage_1sigma', {}).get('mean') is not None:
        print(f"\nUncertainty Calibration:")
        print(f"  Z-score mean: {aggregated['calibration']['z_mean']['mean']:.3f} (ideal: 0.0)")
        print(f"  Z-score std:  {aggregated['calibration']['z_std']['mean']:.3f} (ideal: 1.0)")
        print(f"  Coverage 1σ:  {aggregated['calibration']['coverage_1sigma']['mean']:.1f}% (ideal: 68.3%)")
        print(f"  Coverage 2σ:  {aggregated['calibration']['coverage_2sigma']['mean']:.1f}% (ideal: 95.4%)")

    # Print stratified R² (per-seed)
    if aggregated.get('stratified_r2'):
        print("\n(Per-seed averaged)")
        print_aggregated_stratified_r2(aggregated['stratified_r2'])

    # Print pooled stratified R²
    if aggregated.get('pooled_stratified_r2') and not aggregated['pooled_stratified_r2'].get('error'):
        pooled = aggregated['pooled_stratified_r2']
        thresholds = pooled.get('thresholds', {'stable_max': 0.1, 'disturbed_min': 0.3})
        stable_max = int(thresholds['stable_max'] * 100)
        disturbed_min = int(thresholds['disturbed_min'] * 100)

        print(f"\nPooled Stratified R² (all seeds combined, {pooled['total_tiles']} tiles):")
        for stratum, label in [('stable', f'Stable (<{stable_max}% change)'),
                               ('moderate', f'Moderate ({stable_max}-{disturbed_min}%)'),
                               ('disturbed', f'Disturbed (>{disturbed_min}% loss)')]:
            s = pooled.get(stratum, {})
            if s.get('r2') is not None:
                print(f"  {label:24s} R²={s['r2']:.4f}, RMSE={s['rmse']:.2f} Mg/ha "
                      f"({s['n_shots']} shots, {s['n_tiles']} tiles)")
            else:
                print(f"  {label:24s} Not enough data")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
