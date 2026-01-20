"""
Temporal Baselines for Disturbance Detection.

Supports multiple baseline modes:
- 'interpolation': Train pre/post models, linearly interpolate by timestamp
- 'pre_only': Train only on pre-event data, predict using historical context
- 'post_only': Train only on post-event data (oracle baseline)
- 'mean': Average of pre and post predictions (no temporal weighting)

All baselines use tile-based context selection (same as main model) for fair comparison.
"""

import argparse
import json
from pathlib import Path
import pickle
from typing import Tuple, Optional
from time import time

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from data.spatial_cv import SpatiotemporalSplitter
from models.neural_process import GEDINeuralProcess, neural_process_loss
from utils.config import save_config, _make_serializable
from utils.evaluation import evaluate_model, compute_calibration_metrics
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd, denormalize_std
from utils.disturbance import (
    compute_disturbance_analysis,
    print_disturbance_analysis,
    print_stratified_r2
)


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Baselines for Disturbance Detection')

    # Baseline mode
    parser.add_argument('--mode', type=str, default='interpolation',
                        choices=['interpolation', 'pre_only', 'post_only', 'mean'],
                        help='Baseline mode: interpolation (linear blend), pre_only (historical), '
                             'post_only (oracle), mean (simple average)')

    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box')
    parser.add_argument('--fire_shapefile', type=str, default=None,
                        help='Optional: Path to fire boundary shapefile (.shp) to filter GEDI shots')
    parser.add_argument('--pre_years', type=int, nargs='+', required=True,
                        help='Years before event (e.g., 2019 2020)')
    parser.add_argument('--post_years', type=int, nargs='+', required=True,
                        help='Years after event (e.g., 2022 2023)')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Event year to test on (e.g., 2021)')
    parser.add_argument('--test_months', type=int, nargs='+', default=None,
                        help='Optional: Filter test year to specific months (e.g., 8 9 10 11 12 for Aug-Dec)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    # Model args
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_context_shots', type=int, default=1024,
                        help='Max context shots per tile during training/inference')
    parser.add_argument('--max_target_shots', type=int, default=1024,
                        help='Max target shots per tile during training/inference')
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='LR scheduler patience')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='LR scheduler reduction factor')

    # Infrastructure
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_shots_by_shapefile(df: pd.DataFrame, shapefile_path: str) -> pd.DataFrame:
    """Filter GEDI shots to those inside a shapefile boundary."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(shapefile_path)
    print(f"  Shapefile CRS: {gdf.crs}")
    print(f"  Shapefile bounds (original): {gdf.total_bounds}")

    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')

    print(f"  Shapefile bounds (WGS84): {gdf.total_bounds}")
    print(f"  GEDI lon range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    print(f"  GEDI lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")

    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
        crs='EPSG:4326'
    )

    geometry = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    within_mask = points.within(geometry)

    filtered_df = df[within_mask.values].copy()
    print(f"Filtered to {len(filtered_df)} shots inside fire perimeter (from {len(df)})")

    return filtered_df


def train_spatial_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    global_bounds: Tuple[float, float, float, float],
    args,
    model_name: str,
    max_context_shots: int = 1024,
    max_target_shots: int = 1024
) -> Tuple[GEDINeuralProcess, dict]:
    """Train a spatial-only (coord_dim=2) model with runtime subsampling."""

    train_dataset = GEDINeuralProcessDataset(
        train_df,
        min_shots_per_tile=10,
        agbd_scale=200.0,
        log_transform_agbd=True,
        augment_coords=True,
        global_bounds=global_bounds
    )
    val_dataset = GEDINeuralProcessDataset(
        val_df,
        min_shots_per_tile=10,
        agbd_scale=200.0,
        log_transform_agbd=True,
        augment_coords=False,
        global_bounds=global_bounds
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_neural_process
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_neural_process
    )

    # Spatial-only model (coord_dim=2)
    model = GEDINeuralProcess(
        hidden_dim=args.hidden_dim,
        architecture_mode='anp',
        coord_dim=2  # Spatial only
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )

    best_val_loss = float('inf')
    best_r2 = -float('inf')
    best_state = None
    epochs_without_improvement = 0

    print(f"\nTraining {model_name}...")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            optimizer.zero_grad()
            batch_loss = 0
            n_tiles = 0

            for i in range(len(batch['context_coords'])):
                context_coords = batch['context_coords'][i].to(args.device)
                context_embeddings = batch['context_embeddings'][i].to(args.device)
                context_agbd = batch['context_agbd'][i].to(args.device)
                target_coords = batch['target_coords'][i].to(args.device)
                target_embeddings = batch['target_embeddings'][i].to(args.device)
                target_agbd = batch['target_agbd'][i].to(args.device)

                if len(target_coords) == 0:
                    continue

                # Runtime subsampling of context if too large
                n_context = len(context_coords)
                if n_context > max_context_shots:
                    indices = torch.randperm(n_context, device=args.device)[:max_context_shots]
                    context_coords = context_coords[indices]
                    context_embeddings = context_embeddings[indices]
                    context_agbd = context_agbd[indices]

                # Runtime subsampling of targets if too large
                n_targets = len(target_coords)
                if n_targets > max_target_shots:
                    indices = torch.randperm(n_targets, device=args.device)[:max_target_shots]
                    target_coords = target_coords[indices]
                    target_embeddings = target_embeddings[indices]
                    target_agbd = target_agbd[indices]

                pred_mean, pred_log_var, z_mu_c, z_log_s_c, z_mu_a, z_log_s_a = model(
                    context_coords, context_embeddings, context_agbd,
                    target_coords, target_embeddings, target_agbd, training=True
                )

                loss, _ = neural_process_loss(
                    pred_mean, pred_log_var, target_agbd,
                    z_mu_c, z_log_s_c, z_mu_a, z_log_s_a, kl_weight=0.01
                )

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss += loss
                    n_tiles += 1

            if n_tiles > 0:
                batch_loss = batch_loss / n_tiles
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += batch_loss.item()
                n_batches += 1

            # Clear cache after each batch
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()

        # Validate
        _, _, _, val_metrics, val_loss_dict = evaluate_model(
            model, val_loader, args.device,
            compute_loss=True, kl_weight=0.01,
            max_context_shots=max_context_shots,
            max_targets_per_chunk=max_target_shots
        )

        val_loss = val_loss_dict['loss']
        val_r2 = val_metrics.get('log_r2', -float('inf'))
        scheduler.step(val_loss)

        # Track best by loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

        # Track best by R² (save this model)
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss/max(n_batches,1):.4e}, "
                  f"val={val_loss:.4e}, R²={val_r2:.4f}")

        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (best R²={best_r2:.4f})")
            break

    model.load_state_dict(best_state)
    return model, {'best_val_loss': best_val_loss, 'best_r2': best_r2}


def predict_with_tile_context(
    model: GEDINeuralProcess,
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    global_bounds: Tuple[float, float, float, float],
    device: str,
    max_context_shots: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions using tile-based context selection.

    For each test tile, uses all context points from the same tile as context.
    This matches the main model's evaluation approach for fair comparison.
    """
    model.eval()

    # Group test points by tile
    test_tiles = test_df['tile_id'].unique()

    all_predictions = []
    all_uncertainties = []
    all_indices = []

    lon_min, lat_min, lon_max, lat_max = global_bounds

    with torch.no_grad():
        for tile_id in tqdm(test_tiles, desc='Predicting by tile'):
            # Get test points for this tile
            tile_test = test_df[test_df['tile_id'] == tile_id]
            tile_indices = tile_test.index.tolist()

            # Get context points for this tile (from context_df)
            tile_context = context_df[context_df['tile_id'] == tile_id]

            if len(tile_context) == 0:
                # No context for this tile, use NaN predictions
                all_predictions.extend([np.nan] * len(tile_test))
                all_uncertainties.extend([np.nan] * len(tile_test))
                all_indices.extend(tile_indices)
                continue

            # Prepare context
            ctx_coords = tile_context[['longitude', 'latitude']].values
            ctx_embeddings = np.stack(tile_context['embedding_patch'].values)
            ctx_agbd = tile_context['agbd'].values[:, None]

            # Subsample context if too large
            if len(ctx_coords) > max_context_shots:
                indices = np.random.choice(len(ctx_coords), max_context_shots, replace=False)
                ctx_coords = ctx_coords[indices]
                ctx_embeddings = ctx_embeddings[indices]
                ctx_agbd = ctx_agbd[indices]

            # Normalize
            ctx_coords_norm = normalize_coords(ctx_coords, global_bounds)
            ctx_agbd_norm = normalize_agbd(ctx_agbd)

            # Prepare targets
            tgt_coords = tile_test[['longitude', 'latitude']].values
            tgt_embeddings = np.stack(tile_test['embedding_patch'].values)
            tgt_coords_norm = normalize_coords(tgt_coords, global_bounds)

            # To tensors
            ctx_coords_t = torch.from_numpy(ctx_coords_norm).float().to(device)
            ctx_embeddings_t = torch.from_numpy(ctx_embeddings).float().to(device)
            ctx_agbd_t = torch.from_numpy(ctx_agbd_norm).float().to(device)
            tgt_coords_t = torch.from_numpy(tgt_coords_norm).float().to(device)
            tgt_embeddings_t = torch.from_numpy(tgt_embeddings).float().to(device)

            # Predict
            pred_mean, pred_std = model.predict(
                ctx_coords_t, ctx_embeddings_t, ctx_agbd_t,
                tgt_coords_t, tgt_embeddings_t
            )

            all_predictions.extend(pred_mean.cpu().numpy().flatten().tolist())
            all_uncertainties.extend(pred_std.cpu().numpy().flatten().tolist())
            all_indices.extend(tile_indices)

            # Clear GPU cache
            if 'cuda' in str(device):
                torch.cuda.empty_cache()

    # Reorder to match original test_df order
    result_df = pd.DataFrame({
        'idx': all_indices,
        'pred': all_predictions,
        'unc': all_uncertainties
    }).set_index('idx')

    # Align with test_df index
    predictions = result_df.loc[test_df.index, 'pred'].values
    uncertainties = result_df.loc[test_df.index, 'unc'].values

    return predictions, uncertainties


def temporal_interpolation(
    pred_pre: np.ndarray,
    pred_post: np.ndarray,
    unc_pre: np.ndarray,
    unc_post: np.ndarray,
    test_timestamps: pd.Series,
    pre_end: pd.Timestamp,
    post_start: pd.Timestamp
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearly interpolate predictions and uncertainties based on timestamp.

    Returns:
        (pred_interp, unc_interp, alpha)
    """
    test_times = pd.to_datetime(test_timestamps)

    # Compute interpolation weight α ∈ [0, 1]
    # α = 0 means use pre prediction, α = 1 means use post prediction
    pre_end_ts = pre_end.timestamp()
    post_start_ts = post_start.timestamp()
    test_ts = test_times.astype(np.int64) / 1e9

    alpha = (test_ts - pre_end_ts) / (post_start_ts - pre_end_ts)
    alpha = np.clip(alpha, 0, 1)

    # Linear interpolation of predictions
    pred_interp = (1 - alpha) * pred_pre + alpha * pred_post

    # Weighted interpolation of variances
    # More weight on temporally closer prediction's uncertainty
    var_interp = (1 - alpha) * unc_pre**2 + alpha * unc_post**2
    unc_interp = np.sqrt(var_interp)

    return pred_interp, unc_interp, alpha


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_years = sorted(set(args.pre_years + args.post_years + [args.test_year]))
    train_years = args.pre_years + args.post_years

    mode_names = {
        'interpolation': 'Temporal Linear Interpolation',
        'pre_only': 'Pre-Event Only (Historical)',
        'post_only': 'Post-Event Only (Oracle)',
        'mean': 'Mean of Pre/Post'
    }

    print("=" * 80)
    print(f"Baseline: {mode_names[args.mode]}")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Pre-event years: {args.pre_years}")
    print(f"Post-event years: {args.post_years}")
    print(f"Test year: {args.test_year}")
    print()

    # Query GEDI data
    print("Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=f'{min(all_years)}-01-01',
        end_time=f'{max(all_years)}-12-31',
        max_agbd=500.0
    )
    gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
    print(f"Retrieved {len(gedi_df)} shots")

    # Apply fire shapefile filter if specified
    if args.fire_shapefile:
        print(f"\nApplying fire perimeter filter from: {args.fire_shapefile}")
        gedi_df = filter_shots_by_shapefile(gedi_df, args.fire_shapefile)
        if len(gedi_df) == 0:
            print("No GEDI shots inside fire perimeter. Exiting.")
            return

    # Extract embeddings (reuse same extractor to avoid reinitializing GeoTessera)
    print("\nExtracting embeddings...")
    extractor = EmbeddingExtractor(year=all_years[0], embeddings_dir=args.embeddings_dir)

    all_dfs = []
    for year in all_years:
        year_df = gedi_df[gedi_df['year'] == year].copy()
        if len(year_df) == 0:
            continue
        extractor.set_year(year)
        year_df = extractor.extract_patches_batch(year_df, verbose=True, cache_dir=args.cache_dir)
        all_dfs.append(year_df)

    gedi_df = pd.concat(all_dfs, ignore_index=True)
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]

    # Spatiotemporal split (for test tiles)
    splitter = SpatiotemporalSplitter(
        gedi_df, train_years=train_years, test_year=args.test_year,
        buffer_size=0.1, random_state=args.seed
    )
    train_df_full, val_df_full, test_df = splitter.split()

    # Filter test data to specific months if specified
    if args.test_months:
        test_df['month'] = pd.to_datetime(test_df['time']).dt.month
        original_count = len(test_df)
        test_df = test_df[test_df['month'].isin(args.test_months)].copy()
        test_df = test_df.drop(columns=['month'])
        print(f"Filtered test data to months {args.test_months}: {len(test_df)} shots (from {original_count})")

    # Split train data into pre and post
    pre_train_df = train_df_full[train_df_full['year'].isin(args.pre_years)]
    pre_val_df = val_df_full[val_df_full['year'].isin(args.pre_years)]
    post_train_df = train_df_full[train_df_full['year'].isin(args.post_years)]
    post_val_df = val_df_full[val_df_full['year'].isin(args.post_years)]

    print(f"\nPre-event data: {len(pre_train_df)} train, {len(pre_val_df)} val")
    print(f"Post-event data: {len(post_train_df)} train, {len(post_val_df)} val")
    print(f"Test data: {len(test_df)} shots in {args.test_year}")

    global_bounds = (
        gedi_df['longitude'].min(), gedi_df['latitude'].min(),
        gedi_df['longitude'].max(), gedi_df['latitude'].max()
    )

    # Train models based on mode
    pre_model, post_model = None, None
    pred_pre, unc_pre = None, None
    pred_post, unc_post = None, None

    # Determine which models to train
    need_pre = args.mode in ['interpolation', 'pre_only', 'mean']
    need_post = args.mode in ['interpolation', 'post_only', 'mean']

    if need_pre:
        pre_model, pre_metrics = train_spatial_model(
            pre_train_df, pre_val_df, global_bounds, args, "Pre-event model",
            max_context_shots=args.max_context_shots, max_target_shots=args.max_target_shots
        )
        torch.save(pre_model.state_dict(), output_dir / 'pre_model.pt')

        # Generate predictions using tile-based context
        print("\nGenerating pre-event predictions (tile-based context)...")
        pre_context = gedi_df[gedi_df['year'].isin(args.pre_years)]
        pred_pre, unc_pre = predict_with_tile_context(
            pre_model, pre_context, test_df, global_bounds, args.device,
            max_context_shots=args.max_context_shots
        )

    if need_post:
        post_model, post_metrics = train_spatial_model(
            post_train_df, post_val_df, global_bounds, args, "Post-event model",
            max_context_shots=args.max_context_shots, max_target_shots=args.max_target_shots
        )
        torch.save(post_model.state_dict(), output_dir / 'post_model.pt')

        # Generate predictions using tile-based context
        print("\nGenerating post-event predictions (tile-based context)...")
        post_context = gedi_df[gedi_df['year'].isin(args.post_years)]
        pred_post, unc_post = predict_with_tile_context(
            post_model, post_context, test_df, global_bounds, args.device,
            max_context_shots=args.max_context_shots
        )

    # Combine predictions based on mode
    if args.mode == 'interpolation':
        # Linear interpolation based on timestamp
        pre_end = pd.Timestamp(f'{max(args.pre_years)}-12-31')
        post_start = pd.Timestamp(f'{min(args.post_years)}-01-01')
        pred_final, unc_final, alpha = temporal_interpolation(
            pred_pre, pred_post, unc_pre, unc_post,
            test_df['time'], pre_end, post_start
        )
    elif args.mode == 'pre_only':
        # Only use pre-event predictions
        pred_final = pred_pre
        unc_final = unc_pre
        alpha = np.zeros(len(test_df))
    elif args.mode == 'post_only':
        # Only use post-event predictions (oracle)
        pred_final = pred_post
        unc_final = unc_post
        alpha = np.ones(len(test_df))
    elif args.mode == 'mean':
        # Simple average of pre and post (no temporal weighting)
        pred_final = (pred_pre + pred_post) / 2
        # Combine variances: Var((X+Y)/2) = (Var(X) + Var(Y)) / 4
        unc_final = np.sqrt((unc_pre**2 + unc_post**2) / 4)
        alpha = np.full(len(test_df), 0.5)

    # Handle NaN predictions (tiles with no context)
    valid_mask = ~np.isnan(pred_final)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"\nWarning: {n_invalid} predictions are NaN (tiles without context)")

    # Denormalize predictions
    pred_final_linear = denormalize_agbd(pred_final)
    target_linear = test_df['agbd'].values
    unc_final_linear = denormalize_std(unc_final, pred_final)

    # Also denormalize pre/post for saving
    if pred_pre is not None:
        pred_pre_linear = denormalize_agbd(pred_pre)
        unc_pre_linear = denormalize_std(unc_pre, pred_pre)
    if pred_post is not None:
        pred_post_linear = denormalize_agbd(pred_post)
        unc_post_linear = denormalize_std(unc_post, pred_post)

    # Compute metrics (only on valid predictions)
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    valid_pred = pred_final_linear[valid_mask]
    valid_target = target_linear[valid_mask]
    valid_unc = unc_final_linear[valid_mask]

    rmse = np.sqrt(mean_squared_error(valid_target, valid_pred))
    mae = mean_absolute_error(valid_target, valid_pred)
    r2 = r2_score(valid_target, valid_pred)

    # Log-space metrics
    target_log = normalize_agbd(target_linear)
    valid_target_log = target_log[valid_mask]
    valid_pred_log = pred_final[valid_mask]
    valid_unc_log = unc_final[valid_mask]

    log_rmse = np.sqrt(mean_squared_error(valid_target_log, valid_pred_log))
    log_r2 = r2_score(valid_target_log, valid_pred_log)

    # UQ calibration metrics (in log space)
    calibration = compute_calibration_metrics(valid_pred_log, valid_target_log, valid_unc_log)

    # Disturbance analysis (using shared utility, log-space for stratified R²)
    disturbance_analysis = compute_disturbance_analysis(
        gedi_df, test_df, args.pre_years, args.post_years, args.test_year,
        predictions=pred_final_linear,
        predictions_log=pred_final,
        targets_log=target_log
    )

    print("\n" + "=" * 80)
    print(f"RESULTS: {mode_names[args.mode]} Baseline")
    print("=" * 80)
    print(f"Test year: {args.test_year}")
    print(f"Valid predictions: {valid_mask.sum()}/{len(test_df)}")
    print(f"Linear R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} Mg/ha")
    print(f"MAE: {mae:.2f} Mg/ha")
    print(f"Log R²: {log_r2:.4f}")
    print(f"Log RMSE: {log_rmse:.4f}")
    print(f"\nUncertainty Calibration:")
    print(f"  Z-score mean: {calibration['z_mean']:.3f} (ideal: 0.0)")
    print(f"  Z-score std:  {calibration['z_std']:.3f} (ideal: 1.0)")
    print(f"  Coverage 1σ:  {calibration['coverage_1sigma']:.1f}% (ideal: 68.3%)")
    print(f"  Coverage 2σ:  {calibration['coverage_2sigma']:.1f}% (ideal: 95.4%)")
    print(f"  Coverage 3σ:  {calibration['coverage_3sigma']:.1f}% (ideal: 99.7%)")

    # Print disturbance analysis using shared utility
    print_disturbance_analysis(disturbance_analysis, indent="  ")

    # Print stratified R² using shared utility
    if 'stratified_r2' in disturbance_analysis:
        print_stratified_r2(disturbance_analysis['stratified_r2'], indent="  ")

    # Save results
    results = {
        'method': args.mode,
        'method_name': mode_names[args.mode],
        'pre_years': args.pre_years,
        'post_years': args.post_years,
        'test_year': args.test_year,
        'context_selection': 'tile_based',
        'max_context_shots': args.max_context_shots,
        'metrics': {
            'linear_r2': float(r2),
            'linear_rmse': float(rmse),
            'linear_mae': float(mae),
            'log_r2': float(log_r2),
            'log_rmse': float(log_rmse)
        },
        'calibration': calibration,
        'n_test_shots': len(test_df),
        'n_valid_predictions': int(valid_mask.sum()),
        'alpha_stats': {
            'mean': float(np.nanmean(alpha)),
            'std': float(np.nanstd(alpha)),
            'min': float(np.nanmin(alpha)),
            'max': float(np.nanmax(alpha))
        },
        'disturbance': {
            'correlation': disturbance_analysis['correlation'],
            'quartile_rmse': disturbance_analysis['quartile_rmse'],
            'summary': disturbance_analysis['summary']
        },
        'stratified_r2': disturbance_analysis['stratified_r2']
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions for analysis
    test_df_out = test_df[['latitude', 'longitude', 'agbd', 'time', 'tile_id']].copy()
    test_df_out['pred_final'] = pred_final_linear
    test_df_out['unc_final'] = unc_final_linear
    test_df_out['alpha'] = alpha
    test_df_out['residual'] = target_linear - pred_final_linear

    # Add pre/post predictions if available
    if pred_pre is not None:
        test_df_out['pred_pre'] = pred_pre_linear
        test_df_out['unc_pre'] = unc_pre_linear
    if pred_post is not None:
        test_df_out['pred_post'] = pred_post_linear
        test_df_out['unc_post'] = unc_post_linear

    test_df_out.to_parquet(output_dir / 'predictions.parquet')

    # Save per-tile disturbance analysis
    tile_disturbance_df = pd.DataFrame(disturbance_analysis['per_tile'])
    tile_disturbance_df.to_parquet(output_dir / 'tile_disturbance.parquet')

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
