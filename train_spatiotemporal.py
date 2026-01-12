"""
Train Neural Process with spatiotemporal features for fire detection / temporal holdout.

This script trains a model that can detect biomass changes by:
1. Training on years surrounding a disturbance event (e.g., 2019-2020, 2022-2023)
2. Testing on the disturbance year (e.g., 2021 for McFarland Fire)
3. Using spatiotemporal features (location + temporal encoding)

Usage:
    python train_spatiotemporal.py \
        --region_bbox -122.5 40.5 -121.5 41.5 \
        --train_years 2019 2020 2022 2023 \
        --test_year 2021 \
        --output_dir ./results/mcfarland
"""

import argparse
import json
from pathlib import Path
import pickle
from time import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import (
    GEDISpatiotemporalDataset,
    CrossYearSpatiotemporalDataset,
    collate_neural_process,
    compute_temporal_encoding
)
from data.spatial_cv import SpatiotemporalSplitter
from models.neural_process import (
    GEDINeuralProcess,
    neural_process_loss,
)
from utils.config import save_config, _make_serializable
from utils.evaluation import evaluate_model
from utils.disturbance import (
    compute_disturbance_analysis,
    print_disturbance_analysis,
    print_stratified_r2
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Spatiotemporal Neural Process')

    # Region and temporal arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--fire_shapefile', type=str, default=None,
                        help='Optional: Path to fire boundary shapefile (.shp) to filter GEDI shots')
    parser.add_argument('--train_years', type=int, nargs='+', required=True,
                        help='Years to use for training (e.g., 2019 2020 2022 2023)')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Year to hold out for testing (e.g., 2021)')
    parser.add_argument('--test_months', type=int, nargs='+', default=None,
                        help='Optional: Filter test year to specific months (e.g., 8 9 10 11 12 for Aug-Dec)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching GEDI query results')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Directory where geotessera stores embedding tiles')

    # Model arguments
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--embedding_feature_dim', type=int, default=1024,
                        help='Embedding feature dimension')
    parser.add_argument('--context_repr_dim', type=int, default=256,
                        help='Context representation dimension')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent variable dimension')
    parser.add_argument('--architecture_mode', type=str, default='anp',
                        choices=['deterministic', 'latent', 'anp', 'cnp'],
                        help='Architecture mode')
    parser.add_argument('--num_attention_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--no_temporal_encoding', action='store_true',
                        help='Disable temporal encoding (spatial-only baseline with 2D coords)')
    parser.add_argument('--temporal_context', action='store_true',
                        help='Use train years as context for test prediction (true temporal prediction). '
                             'Without this flag, test year data is used as context (spatial interpolation).')
    parser.add_argument('--cross_year_training', action='store_true',
                        help='Train with cross-year context/target splits. For each tile, randomly '
                             'holds out one year as target and uses other years as context. '
                             'This trains the model for the --temporal_context evaluation task.')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (number of tiles)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--buffer_size', type=float, default=0.1,
                        help='Buffer size in degrees for spatial CV')
    parser.add_argument('--min_shots_per_tile', type=int, default=10,
                        help='Minimum GEDI shots per tile')
    parser.add_argument('--max_context_shots', type=int, default=20000,
                        help='Maximum context shots per tile (subsampled if exceeded for memory)')
    parser.add_argument('--max_target_shots', type=int, default=5000,
                        help='Maximum target shots per tile (subsampled if exceeded)')
    parser.add_argument('--target_chunk_size', type=int, default=2000,
                        help='Process targets in chunks of this size for memory efficiency')
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='LR scheduler patience')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='LR scheduler reduction factor')
    parser.add_argument('--kl_weight_max', type=float, default=0.01,
                        help='Maximum KL weight')
    parser.add_argument('--kl_warmup_epochs', type=int, default=10,
                        help='KL warmup epochs')

    # Dataset arguments
    parser.add_argument('--agbd_scale', type=float, default=200.0,
                        help='AGBD scale factor for normalization')
    parser.add_argument('--log_transform_agbd', type=lambda x: x.lower() == 'true', default=True,
                        help='Apply log transform to AGBD')
    parser.add_argument('--augment_coords', action='store_true', default=True,
                        help='Add coordinate augmentation')
    parser.add_argument('--coord_noise_std', type=float, default=0.01,
                        help='Standard deviation for coordinate noise')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def filter_shots_by_shapefile(df: pd.DataFrame, shapefile_path: str) -> pd.DataFrame:
    """Filter GEDI shots to those inside a shapefile boundary."""
    import geopandas as gpd
    from shapely.geometry import Point

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')

    # Create points from GEDI shots
    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
        crs='EPSG:4326'
    )

    # Check which points are within the geometry
    geometry = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
    within_mask = points.within(geometry)

    filtered_df = df[within_mask.values].copy()
    print(f"Filtered to {len(filtered_df)} shots inside fire perimeter (from {len(df)})")

    return filtered_df


def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0,
                max_context_shots=20000, max_target_shots=5000, target_chunk_size=2000):
    """Train for one epoch with chunked target processing to manage memory.

    Args:
        model: The neural process model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        kl_weight: Weight for KL divergence term
        max_context_shots: Maximum context shots (subsampled if exceeded)
        max_target_shots: Maximum target shots per tile (subsampled if exceeded)
        target_chunk_size: Process targets in chunks of this size for memory efficiency
    """
    model.train()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    n_tiles = 0

    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()

        batch_loss = 0
        batch_nll = 0
        batch_kl = 0
        n_tiles_in_batch = 0

        for i in range(len(batch['context_coords'])):
            context_coords = batch['context_coords'][i].to(device)
            context_embeddings = batch['context_embeddings'][i].to(device)
            context_agbd = batch['context_agbd'][i].to(device)
            target_coords = batch['target_coords'][i].to(device)
            target_embeddings = batch['target_embeddings'][i].to(device)
            target_agbd = batch['target_agbd'][i].to(device)

            if len(target_coords) == 0:
                continue

            # Runtime subsampling of context if too large (prevents OOM in attention)
            n_context = len(context_coords)
            if n_context > max_context_shots:
                indices = torch.randperm(n_context, device=device)[:max_context_shots]
                context_coords = context_coords[indices]
                context_embeddings = context_embeddings[indices]
                context_agbd = context_agbd[indices]

            # Runtime subsampling of targets if too large
            n_targets = len(target_coords)
            if n_targets > max_target_shots:
                indices = torch.randperm(n_targets, device=device)[:max_target_shots]
                target_coords = target_coords[indices]
                target_embeddings = target_embeddings[indices]
                target_agbd = target_agbd[indices]
                n_targets = max_target_shots

            # Process targets in chunks if still too large for memory
            if n_targets > target_chunk_size:
                # Chunked processing - accumulate losses
                chunk_losses = []
                chunk_nlls = []
                chunk_kls = []

                # Shuffle targets for chunking
                perm = torch.randperm(n_targets, device=device)
                target_coords = target_coords[perm]
                target_embeddings = target_embeddings[perm]
                target_agbd = target_agbd[perm]

                for chunk_start in range(0, n_targets, target_chunk_size):
                    chunk_end = min(chunk_start + target_chunk_size, n_targets)
                    chunk_target_coords = target_coords[chunk_start:chunk_end]
                    chunk_target_embeddings = target_embeddings[chunk_start:chunk_end]
                    chunk_target_agbd = target_agbd[chunk_start:chunk_end]

                    pred_mean, pred_log_var, z_mu_context, z_log_sigma_context, z_mu_all, z_log_sigma_all = model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        chunk_target_coords,
                        chunk_target_embeddings,
                        query_agbd=chunk_target_agbd,
                        training=True
                    )

                    loss, loss_dict = neural_process_loss(
                        pred_mean, pred_log_var, chunk_target_agbd,
                        z_mu_context, z_log_sigma_context,
                        z_mu_all, z_log_sigma_all,
                        kl_weight
                    )

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        chunk_losses.append(loss * len(chunk_target_coords))
                        chunk_nlls.append(loss_dict['nll'] * len(chunk_target_coords))
                        chunk_kls.append(loss_dict['kl'] * len(chunk_target_coords))

                if chunk_losses:
                    # Weighted average by chunk size
                    total_chunk_targets = sum(len(target_coords[i:i+target_chunk_size])
                                             for i in range(0, n_targets, target_chunk_size))
                    tile_loss = sum(chunk_losses) / total_chunk_targets
                    tile_nll = sum(chunk_nlls) / total_chunk_targets
                    tile_kl = sum(chunk_kls) / total_chunk_targets

                    batch_loss += tile_loss
                    batch_nll += tile_nll
                    batch_kl += tile_kl
                    n_tiles_in_batch += 1
            else:
                # Standard processing for smaller tiles
                pred_mean, pred_log_var, z_mu_context, z_log_sigma_context, z_mu_all, z_log_sigma_all = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings,
                    query_agbd=target_agbd,
                    training=True
                )

                loss, loss_dict = neural_process_loss(
                    pred_mean, pred_log_var, target_agbd,
                    z_mu_context, z_log_sigma_context,
                    z_mu_all, z_log_sigma_all,
                    kl_weight
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected! Skipping batch.")
                    continue

                batch_loss += loss
                batch_nll += loss_dict['nll']
                batch_kl += loss_dict['kl']
                n_tiles_in_batch += 1

        if n_tiles_in_batch > 0:
            batch_loss = batch_loss / n_tiles_in_batch
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            total_nll += batch_nll / n_tiles_in_batch
            total_kl += batch_kl / n_tiles_in_batch
            n_tiles += n_tiles_in_batch

        # Clear cache after each batch to reduce memory fragmentation
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    return {
        'loss': total_loss / max(n_tiles, 1),
        'nll': total_nll / max(n_tiles, 1),
        'kl': total_kl / max(n_tiles, 1)
    }


def predict_on_test_df(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    context_df: pd.DataFrame,
    global_bounds: tuple,
    temporal_bounds: tuple,
    device: str,
    max_context_shots: int = 1024,
    include_temporal: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions on test_df while preserving shot order.

    Uses tile-based context selection (same tile from context_df).
    Returns predictions and uncertainties aligned with test_df index.

    Args:
        include_temporal: If True, use 5D coords (lon, lat, sin_doy, cos_doy, norm_time).
                         If False, use 2D coords (lon, lat) for spatial-only baseline.
    """
    model.eval()

    from utils.normalization import normalize_agbd

    # Group test points by tile
    test_tiles = test_df['tile_id'].unique()

    all_predictions = []
    all_uncertainties = []
    all_indices = []

    with torch.no_grad():
        for tile_id in tqdm(test_tiles, desc='Predicting by tile'):
            # Get test points for this tile
            tile_test = test_df[test_df['tile_id'] == tile_id]
            tile_indices = tile_test.index.tolist()

            # Get context points for this tile
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
            ctx_time = pd.to_datetime(tile_context['time'])

            # Subsample context if too large
            if len(ctx_coords) > max_context_shots:
                indices = np.random.choice(len(ctx_coords), max_context_shots, replace=False)
                ctx_coords = ctx_coords[indices]
                ctx_embeddings = ctx_embeddings[indices]
                ctx_agbd = ctx_agbd[indices]
                ctx_time = ctx_time.iloc[indices]

            # Normalize spatial coords
            lon_min, lat_min, lon_max, lat_max = global_bounds
            ctx_coords_norm = np.zeros_like(ctx_coords)
            ctx_coords_norm[:, 0] = (ctx_coords[:, 0] - lon_min) / (lon_max - lon_min + 1e-8)
            ctx_coords_norm[:, 1] = (ctx_coords[:, 1] - lat_min) / (lat_max - lat_min + 1e-8)

            # Build coordinate vector: spatial only or spatiotemporal
            if include_temporal:
                ctx_temporal = compute_temporal_encoding(ctx_time, temporal_bounds)
                ctx_final_coords = np.concatenate([ctx_coords_norm, ctx_temporal], axis=1)
            else:
                ctx_final_coords = ctx_coords_norm

            ctx_agbd_norm = normalize_agbd(ctx_agbd)

            # Prepare targets
            tgt_coords = tile_test[['longitude', 'latitude']].values
            tgt_embeddings = np.stack(tile_test['embedding_patch'].values)
            tgt_time = pd.to_datetime(tile_test['time'])

            tgt_coords_norm = np.zeros_like(tgt_coords)
            tgt_coords_norm[:, 0] = (tgt_coords[:, 0] - lon_min) / (lon_max - lon_min + 1e-8)
            tgt_coords_norm[:, 1] = (tgt_coords[:, 1] - lat_min) / (lat_max - lat_min + 1e-8)

            # Build target coordinate vector
            if include_temporal:
                tgt_temporal = compute_temporal_encoding(tgt_time, temporal_bounds)
                tgt_final_coords = np.concatenate([tgt_coords_norm, tgt_temporal], axis=1)
            else:
                tgt_final_coords = tgt_coords_norm

            # To tensors
            ctx_coords_t = torch.from_numpy(ctx_final_coords).float().to(device)
            ctx_embeddings_t = torch.from_numpy(ctx_embeddings).float().to(device)
            ctx_agbd_t = torch.from_numpy(ctx_agbd_norm).float().to(device)
            tgt_coords_t = torch.from_numpy(tgt_final_coords).float().to(device)
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


def validate(model, dataloader, device, kl_weight=1.0, agbd_scale=200.0, log_transform_agbd=True,
              max_context_shots=1024, max_targets_per_chunk=1024):
    """Validate model using evaluate_model with same limits as training."""
    _, _, _, metrics, loss_dict = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        max_context_shots=max_context_shots,
        max_targets_per_chunk=max_targets_per_chunk,
        compute_loss=True,
        kl_weight=kl_weight,
        agbd_scale=agbd_scale,
        log_transform_agbd=log_transform_agbd,
        denormalize_for_reporting=False
    )
    return loss_dict, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine all years we need data for
    all_years = sorted(set(args.train_years + [args.test_year]))
    start_year = min(all_years)
    end_year = max(all_years)

    config = vars(args)
    config['all_years'] = all_years
    save_config(config, output_dir / 'config.json')

    print("=" * 80)
    print("Spatiotemporal Neural Process Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Region: {args.region_bbox}")
    print(f"Train years: {args.train_years}")
    print(f"Test year: {args.test_year}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Query GEDI data for all years
    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=f'{start_year}-01-01',
        end_time=f'{end_year}-12-31',
        max_agbd=500.0
    )
    print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")

    # Apply fire shapefile filter if specified
    if args.fire_shapefile:
        print(f"\nApplying fire perimeter filter from: {args.fire_shapefile}")
        gedi_df = filter_shots_by_shapefile(gedi_df, args.fire_shapefile)
        if len(gedi_df) == 0:
            print("No GEDI shots inside fire perimeter. Exiting.")
            return

    # Add year column
    gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
    print(f"Shots per year: {dict(gedi_df['year'].value_counts().sort_index())}")
    print()

    if len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    # Step 2: Extract embeddings for each year (reuse same extractor to avoid reinitializing GeoTessera)
    print("Step 2: Extracting GeoTessera embeddings...")
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

        extractor.set_year(year)  # Switch year without reinitializing GeoTessera
        year_df = extractor.extract_patches_batch(year_df, verbose=True, cache_dir=args.cache_dir)
        all_dfs.append(year_df)

    gedi_df = pd.concat(all_dfs, ignore_index=True)
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"\nRetained {len(gedi_df)} shots with valid embeddings")
    print()

    # Save processed data
    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    # Step 3: Spatiotemporal split
    print("Step 3: Creating spatiotemporal split...")
    splitter = SpatiotemporalSplitter(
        gedi_df,
        train_years=args.train_years,
        test_year=args.test_year,
        buffer_size=args.buffer_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()

    # Filter test data to specific months if specified
    if args.test_months:
        test_df['month'] = pd.to_datetime(test_df['time']).dt.month
        original_count = len(test_df)
        test_df = test_df[test_df['month'].isin(args.test_months)].copy()
        test_df = test_df.drop(columns=['month'])
        print(f"Filtered test data to months {args.test_months}: {len(test_df)} shots (from {original_count})")
    print()

    # Compute global bounds from training data
    global_bounds = (
        train_df['longitude'].min(),
        train_df['latitude'].min(),
        train_df['longitude'].max(),
        train_df['latitude'].max()
    )

    # Compute temporal bounds from ALL data (for consistent normalization)
    timestamps = pd.to_datetime(gedi_df['time'])
    unix_time = timestamps.astype(np.int64) / 1e9
    temporal_bounds = (unix_time.min(), unix_time.max())

    config['global_bounds'] = list(global_bounds)
    config['temporal_bounds'] = list(temporal_bounds)
    save_config(config, output_dir / 'config.json')

    print(f"Global spatial bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
          f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")
    print(f"Temporal bounds: {pd.Timestamp(temporal_bounds[0], unit='s')} to "
          f"{pd.Timestamp(temporal_bounds[1], unit='s')}")

    # Step 4: Create datasets (no max_shots_per_tile - using runtime subsampling instead)
    print("\nStep 4: Creating datasets...")
    print(f"  Runtime subsampling: max_context={args.max_context_shots}, max_target={args.max_target_shots}")
    include_temporal = not args.no_temporal_encoding

    # Choose dataset class based on training mode
    if args.cross_year_training:
        print(f"  Training mode: CROSS-YEAR (context from other years, target from held-out year)")
        DatasetClass = CrossYearSpatiotemporalDataset
        extra_args = {'min_years_per_tile': 2}
    else:
        print(f"  Training mode: STANDARD (random context/target split within mixed years)")
        DatasetClass = GEDISpatiotemporalDataset
        extra_args = {}

    train_dataset = DatasetClass(
        train_df,
        min_shots_per_tile=args.min_shots_per_tile,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=args.augment_coords,
        coord_noise_std=args.coord_noise_std,
        global_bounds=global_bounds,
        temporal_bounds=temporal_bounds,
        include_temporal=include_temporal,
        **extra_args
    )

    # Validation and test always use standard dataset (we evaluate on specific years)
    val_dataset = GEDISpatiotemporalDataset(
        val_df,
        min_shots_per_tile=args.min_shots_per_tile,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=False,
        global_bounds=global_bounds,
        temporal_bounds=temporal_bounds,
        include_temporal=include_temporal
    )
    test_dataset = GEDISpatiotemporalDataset(
        test_df,
        min_shots_per_tile=args.min_shots_per_tile,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=False,
        global_bounds=global_bounds,
        temporal_bounds=temporal_bounds,
        include_temporal=include_temporal
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_neural_process, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_neural_process, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_neural_process, num_workers=args.num_workers
    )
    print()

    # Step 5: Initialize model with appropriate coord_dim
    print("Step 5: Initializing model...")
    print(f"Architecture mode: {args.architecture_mode}")
    coord_dim = 2 if args.no_temporal_encoding else 5
    if args.no_temporal_encoding:
        print(f"Coordinate dimension: 2 (lon, lat) - spatial-only baseline")
    else:
        print(f"Coordinate dimension: 5 (lon, lat, sin_doy, cos_doy, norm_time)")

    model = GEDINeuralProcess(
        patch_size=args.patch_size,
        embedding_channels=128,
        embedding_feature_dim=args.embedding_feature_dim,
        context_repr_dim=args.context_repr_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_uncertainty=True,
        architecture_mode=args.architecture_mode,
        num_attention_heads=args.num_attention_heads,
        coord_dim=coord_dim
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )

    # Step 6: Training loop
    print("Step 6: Training...")
    best_val_loss = float('inf')
    best_r2 = float('-inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    training_start_time = time()

    # Extended history tracking
    history = {
        'train_loss': [],
        'train_nll': [],
        'train_kl': [],
        'val_loss': [],
        'val_nll': [],
        'val_kl': [],
        'val_log_r2': [],
        'val_coverage_1sigma': [],
        'val_coverage_2sigma': [],
        'kl_weight': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # KL warmup
        if args.kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / args.kl_warmup_epochs) * args.kl_weight_max
        else:
            kl_weight = args.kl_weight_max

        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, kl_weight,
            max_context_shots=args.max_context_shots,
            max_target_shots=args.max_target_shots,
            target_chunk_size=args.target_chunk_size
        )
        train_losses.append(train_metrics['loss'])

        val_losses_dict, val_metrics = validate(
            model, val_loader, args.device, kl_weight,
            agbd_scale=args.agbd_scale,
            log_transform_agbd=args.log_transform_agbd,
            max_context_shots=args.max_context_shots,
            max_targets_per_chunk=args.max_target_shots
        )
        val_losses.append(val_losses_dict['loss'])

        # Track extended history
        history['train_loss'].append(train_metrics['loss'])
        history['train_nll'].append(train_metrics['nll'])
        history['train_kl'].append(train_metrics['kl'])
        history['val_loss'].append(val_losses_dict['loss'])
        history['val_nll'].append(val_losses_dict['nll'])
        history['val_kl'].append(val_losses_dict['kl'])
        history['val_log_r2'].append(val_metrics.get('log_r2', 0) if val_metrics else 0)
        history['val_coverage_1sigma'].append(val_metrics.get('coverage_1sigma', 0) if val_metrics else 0)
        history['val_coverage_2sigma'].append(val_metrics.get('coverage_2sigma', 0) if val_metrics else 0)
        history['kl_weight'].append(kl_weight)

        # Print training metrics with loss components
        print(f"Train: Loss={train_metrics['loss']:.4e} (NLL={train_metrics['nll']:.4e}, KL={train_metrics['kl']:.4e})")
        print(f"Val:   Loss={val_losses_dict['loss']:.4e} (NLL={val_losses_dict['nll']:.4e}, KL={val_losses_dict['kl']:.4e})")
        if val_metrics:
            log_r2 = val_metrics.get('log_r2', 0)
            cov_1s = val_metrics.get('coverage_1sigma', 0)
            cov_2s = val_metrics.get('coverage_2sigma', 0)
            print(f"Val:   R²={log_r2:.4f}, Coverage: {cov_1s:.1f}% (1σ), {cov_2s:.1f}% (2σ)  [ideal: 68.3%, 95.4%]")

        scheduler.step(val_losses_dict['loss'])

        # Save best model
        if val_losses_dict['loss'] < best_val_loss:
            best_val_loss = val_losses_dict['loss']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses_dict['loss'],
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pt')
            print("Saved best model (lowest val loss)")
        else:
            epochs_without_improvement += 1

        current_r2 = val_metrics.get('log_r2', float('-inf')) if val_metrics else float('-inf')
        if current_r2 > best_r2:
            best_r2 = current_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_r2_model.pt')
            print(f"Saved best R² model (R² = {best_r2:.4f})")

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    training_time = time() - training_start_time

    # Save training history (extended with loss components and calibration)
    # Also include legacy format for backwards compatibility
    history['train_losses'] = train_losses
    history['val_losses'] = val_losses
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(_make_serializable(history), f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6e}")
    print(f"Best R² score: {best_r2:.4f}")
    print("=" * 80)

    # Step 7: Test evaluation
    print("\nStep 7: Evaluating on test set (held-out year)...")
    checkpoint = torch.load(output_dir / 'best_r2_model.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_losses_dict, test_metrics = validate(
        model, test_loader, args.device, kl_weight=1.0,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        max_context_shots=args.max_context_shots,
        max_targets_per_chunk=args.max_target_shots
    )

    print(f"\nTest Results (Year {args.test_year}):")
    print(f"  Loss: {test_losses_dict['loss']:.4e} (NLL={test_losses_dict['nll']:.4e}, KL={test_losses_dict['kl']:.4e})")
    if test_metrics:
        print(f"  Log R²: {test_metrics.get('log_r2', 0):.4f}")
        print(f"  Log RMSE: {test_metrics.get('log_rmse', 0):.4f}")
        print(f"  Linear R²: {test_metrics.get('linear_r2', 0):.4f}")
        print(f"  Linear RMSE: {test_metrics.get('linear_rmse', 0):.2f} Mg/ha")
        print(f"  Linear MAE: {test_metrics.get('linear_mae', 0):.2f} Mg/ha")
        # UQ calibration metrics
        if 'coverage_1sigma' in test_metrics:
            print(f"\n  Uncertainty Calibration:")
            print(f"    Z-score mean: {test_metrics.get('z_mean', 0):.3f} (ideal: 0.0)")
            print(f"    Z-score std:  {test_metrics.get('z_std', 0):.3f} (ideal: 1.0)")
            print(f"    Coverage 1σ:  {test_metrics.get('coverage_1sigma', 0):.1f}% (ideal: 68.3%)")
            print(f"    Coverage 2σ:  {test_metrics.get('coverage_2sigma', 0):.1f}% (ideal: 95.4%)")
            print(f"    Coverage 3σ:  {test_metrics.get('coverage_3sigma', 0):.1f}% (ideal: 99.7%)")

    # Generate predictions with shot-level mapping for stratified analysis
    from utils.normalization import denormalize_agbd

    if args.temporal_context:
        # Use train years as context for true temporal prediction
        # This tests the model's ability to predict held-out year from surrounding years
        context_df = gedi_df[gedi_df['year'].isin(args.train_years)].copy()
        print(f"\n  Generating predictions with TEMPORAL context (train years: {args.train_years})...")
        print(f"    Context pool: {len(context_df)} shots from {args.train_years}")
        print(f"    Targets: {len(test_df)} shots from {args.test_year}")

        # Debug: Check overlap between test tiles and context tiles
        test_tiles = set(test_df['tile_id'].unique())
        context_tiles = set(context_df['tile_id'].unique())
        overlap_tiles = test_tiles & context_tiles
        print(f"    Test tiles: {len(test_tiles)}, Context tiles with train data: {len(overlap_tiles)}")
        if len(overlap_tiles) == 0:
            print("    WARNING: No test tiles have train year observations! Predictions will be NaN.")
    else:
        # Use test_df itself as context (same tile, same year - spatial interpolation)
        context_df = test_df
        print("\n  Generating predictions with SAME-YEAR context (spatial interpolation)...")

    test_preds_log, test_unc_log = predict_on_test_df(
        model, test_df, context_df,
        global_bounds, temporal_bounds, args.device,
        max_context_shots=args.max_context_shots,
        include_temporal=include_temporal
    )
    # Denormalize predictions to linear scale for disturbance analysis
    test_preds_linear = denormalize_agbd(test_preds_log)

    # Split train years into pre/post relative to test year
    pre_years = [y for y in args.train_years if y < args.test_year]
    post_years = [y for y in args.train_years if y > args.test_year]

    # Compute disturbance analysis with predictions for stratified R²
    disturbance_analysis = compute_disturbance_analysis(
        gedi_df, test_df, pre_years, post_years, args.test_year,
        predictions=test_preds_linear
    )

    # Print disturbance analysis using shared utility
    print_disturbance_analysis(disturbance_analysis, indent="    ")

    # Print stratified R² using shared utility
    if 'stratified_r2' in disturbance_analysis:
        print_stratified_r2(disturbance_analysis['stratified_r2'], indent="    ")

    # Save per-tile disturbance analysis
    tile_disturbance_df = pd.DataFrame(disturbance_analysis['per_tile'])
    tile_disturbance_df.to_parquet(output_dir / 'tile_disturbance.parquet')

    # Save predictions for analysis
    test_df_out = test_df[['latitude', 'longitude', 'agbd', 'time', 'tile_id']].copy()
    test_df_out['pred'] = test_preds_linear
    test_df_out['unc'] = denormalize_agbd(test_unc_log) if test_unc_log is not None else np.nan
    test_df_out['residual'] = test_df['agbd'].values - test_preds_linear
    test_df_out.to_parquet(output_dir / 'test_predictions.parquet')

    # Save final results
    checkpoint['test_metrics'] = test_metrics
    checkpoint['test_loss'] = test_losses_dict
    torch.save(checkpoint, output_dir / 'best_r2_model.pt')

    results = {
        'train_years': args.train_years,
        'test_year': args.test_year,
        'train_time': training_time,
        'best_val_loss': best_val_loss,
        'best_r2': best_r2,
        'test_loss': test_losses_dict,
        'test_metrics': test_metrics,
        'disturbance': {
            'pre_years': disturbance_analysis['pre_years'],
            'post_years': disturbance_analysis['post_years'],
            'correlation': disturbance_analysis['correlation'],
            'quartile_rmse': disturbance_analysis['quartile_rmse'],
            'summary': disturbance_analysis['summary']
        }
    }
    if 'stratified_r2' in disturbance_analysis:
        results['stratified_r2'] = disturbance_analysis['stratified_r2']

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
