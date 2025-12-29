"""
Temporal Linear Interpolation Baseline.

This baseline:
1. Trains a spatial-only model on pre-event data (e.g., 2019-2020)
2. Trains a spatial-only model on post-event data (e.g., 2022-2023)
3. For test points in the event year (e.g., 2021), linearly interpolates
   predictions based on timestamp

This encodes the assumption that biomass changes smoothly over time,
which will fail for abrupt disturbance events like fires.
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


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Interpolation Baseline')

    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box')
    parser.add_argument('--pre_years', type=int, nargs='+', required=True,
                        help='Years before event (e.g., 2019 2020)')
    parser.add_argument('--post_years', type=int, nargs='+', required=True,
                        help='Years after event (e.g., 2022 2023)')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Event year to test on (e.g., 2021)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    # Model args
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_context', type=int, default=100,
                        help='Number of nearest context points for prediction')
    parser.add_argument('--max_context_shots', type=int, default=1024,
                        help='Max context shots per tile during training (runtime subsampling)')
    parser.add_argument('--max_target_shots', type=int, default=1024,
                        help='Max target shots per tile during training (runtime subsampling)')
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


def predict_with_model(
    model: GEDINeuralProcess,
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    global_bounds: Tuple[float, float, float, float],
    device: str,
    n_context: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions for test points using k-NN context selection."""
    from scipy.spatial import cKDTree

    model.eval()
    predictions = []
    uncertainties = []

    # Build KD-tree for context point lookup
    context_coords = context_df[['longitude', 'latitude']].values
    context_embeddings = np.stack(context_df['embedding_patch'].values)
    context_agbd = context_df['agbd'].values[:, None]
    context_tree = cKDTree(context_coords)

    lon_min, lat_min, lon_max, lat_max = global_bounds

    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc='Predicting'):
            row = test_df.iloc[idx]
            query_coord = np.array([[row['longitude'], row['latitude']]])

            # Find k nearest context points
            _, nn_indices = context_tree.query(query_coord[0], k=min(n_context, len(context_df)))
            if isinstance(nn_indices, np.integer):
                nn_indices = [nn_indices]

            # Get context subset
            ctx_coords = context_coords[nn_indices]
            ctx_embeddings = context_embeddings[nn_indices]
            ctx_agbd = context_agbd[nn_indices]

            # Normalize using utility functions
            ctx_coords_norm = normalize_coords(ctx_coords, global_bounds)
            ctx_agbd_norm = normalize_agbd(ctx_agbd)
            query_coord_norm = normalize_coords(query_coord, global_bounds)

            query_embedding = row['embedding_patch'][np.newaxis, ...]

            # To tensors
            ctx_coords_t = torch.from_numpy(ctx_coords_norm).float().to(device)
            ctx_embeddings_t = torch.from_numpy(ctx_embeddings).float().to(device)
            ctx_agbd_t = torch.from_numpy(ctx_agbd_norm).float().to(device)
            query_coords_t = torch.from_numpy(query_coord_norm).float().to(device)
            query_embeddings_t = torch.from_numpy(query_embedding).float().to(device)

            pred_mean, pred_std = model.predict(
                ctx_coords_t, ctx_embeddings_t, ctx_agbd_t,
                query_coords_t, query_embeddings_t
            )

            predictions.append(pred_mean.cpu().numpy().item())
            uncertainties.append(pred_std.cpu().numpy().item())

    return np.array(predictions), np.array(uncertainties)


def compute_disturbance_analysis(
    gedi_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    pre_years: list,
    post_years: list,
    test_year: int
) -> dict:
    """
    Compute per-tile disturbance metrics and correlate with prediction errors.

    Disturbance intensity = (expected - observed) / expected
    where expected = (pre_mean + post_mean) / 2

    Returns dict with:
    - per_tile: DataFrame with tile-level metrics
    - correlation: Pearson correlation between disturbance and abs error
    - summary: Aggregate statistics
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

        # Prediction error for this tile
        tile_mask = test_df['tile_id'] == tile_id
        tile_preds = predictions[tile_mask.values]
        tile_targets = test_tile_df['agbd'].values
        tile_rmse = np.sqrt(np.mean((tile_preds - tile_targets) ** 2))
        tile_mae = np.mean(np.abs(tile_preds - tile_targets))

        # Expected value (linear interpolation assumption)
        if not np.isnan(pre_mean) and not np.isnan(post_mean):
            expected = (pre_mean + post_mean) / 2
            # Disturbance intensity: positive = biomass loss
            disturbance = (expected - test_mean) / expected if expected > 0 else 0
            # Relative change from pre
            change_from_pre = (pre_mean - test_mean) / pre_mean if pre_mean > 0 else 0
        else:
            expected = np.nan
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

    # Compute correlation between disturbance and error
    valid_mask = ~(tile_df['disturbance'].isna() | tile_df['tile_rmse'].isna())
    if valid_mask.sum() >= 3:
        corr, p_value = stats.pearsonr(
            tile_df.loc[valid_mask, 'disturbance'],
            tile_df.loc[valid_mask, 'tile_rmse']
        )
    else:
        corr, p_value = np.nan, np.nan

    # Compute quartile breakdown
    valid_tiles = tile_df[valid_mask].copy()
    if len(valid_tiles) >= 4:
        valid_tiles['disturbance_quartile'] = pd.qcut(
            valid_tiles['disturbance'], q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high']
        )
        quartile_rmse = valid_tiles.groupby('disturbance_quartile')['tile_rmse'].mean().to_dict()
    else:
        quartile_rmse = {}

    return {
        'per_tile': tile_df.to_dict('records'),
        'correlation': {
            'pearson_r': float(corr) if not np.isnan(corr) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None
        },
        'quartile_rmse': {k: float(v) for k, v in quartile_rmse.items()},
        'summary': {
            'mean_disturbance': float(tile_df['disturbance'].mean()),
            'std_disturbance': float(tile_df['disturbance'].std()),
            'n_tiles_with_loss': int((tile_df['disturbance'] > 0).sum()),
            'n_tiles_with_gain': int((tile_df['disturbance'] < 0).sum()),
            'pct_tiles_major_loss': float((tile_df['disturbance'] > 0.3).mean() * 100),
            'mean_change_from_pre': float(tile_df['change_from_pre'].mean())
        }
    }


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

    print("=" * 80)
    print("Temporal Linear Interpolation Baseline")
    print("=" * 80)
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

    # Extract embeddings (reuse same extractor to avoid reinitializing GeoTessera)
    print("\nExtracting embeddings...")
    extractor = EmbeddingExtractor(year=all_years[0], embeddings_dir=args.embeddings_dir)

    all_dfs = []
    for year in all_years:
        year_df = gedi_df[gedi_df['year'] == year].copy()
        if len(year_df) == 0:
            continue
        extractor.set_year(year)
        year_df = extractor.extract_patches_batch(year_df, verbose=True)
        all_dfs.append(year_df)

    gedi_df = pd.concat(all_dfs, ignore_index=True)
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]

    # Spatiotemporal split (for test tiles)
    splitter = SpatiotemporalSplitter(
        gedi_df, train_years=train_years, test_year=args.test_year,
        buffer_size=0.1, random_state=args.seed
    )
    train_df_full, val_df_full, test_df = splitter.split()

    # Get test tile IDs
    test_tiles = splitter.test_tiles

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

    # Train pre-event model
    pre_model, pre_metrics = train_spatial_model(
        pre_train_df, pre_val_df, global_bounds, args, "Pre-event model",
        max_context_shots=args.max_context_shots, max_target_shots=args.max_target_shots
    )
    torch.save(pre_model.state_dict(), output_dir / 'pre_model.pt')

    # Train post-event model
    post_model, post_metrics = train_spatial_model(
        post_train_df, post_val_df, global_bounds, args, "Post-event model",
        max_context_shots=args.max_context_shots, max_target_shots=args.max_target_shots
    )
    torch.save(post_model.state_dict(), output_dir / 'post_model.pt')

    # Generate predictions for test set from both models
    print(f"\nGenerating predictions (n_context={args.n_context})...")

    # Use all pre-event data as context for pre-model
    pre_context = gedi_df[gedi_df['year'].isin(args.pre_years)]
    pred_pre, unc_pre = predict_with_model(
        pre_model, pre_context, test_df, global_bounds, args.device, args.n_context
    )

    # Use all post-event data as context for post-model
    post_context = gedi_df[gedi_df['year'].isin(args.post_years)]
    pred_post, unc_post = predict_with_model(
        post_model, post_context, test_df, global_bounds, args.device, args.n_context
    )

    # Temporal interpolation with proper variance blending
    pre_end = pd.Timestamp(f'{max(args.pre_years)}-12-31')
    post_start = pd.Timestamp(f'{min(args.post_years)}-01-01')
    pred_interp, unc_interp, alpha = temporal_interpolation(
        pred_pre, pred_post, unc_pre, unc_post,
        test_df['time'], pre_end, post_start
    )

    # Denormalize predictions using delta method
    pred_pre_linear = denormalize_agbd(pred_pre)
    pred_post_linear = denormalize_agbd(pred_post)
    pred_interp_linear = denormalize_agbd(pred_interp)
    target_linear = test_df['agbd'].values

    # Denormalize uncertainties using delta method (requires corresponding mean)
    unc_pre_linear = denormalize_std(unc_pre, pred_pre)
    unc_post_linear = denormalize_std(unc_post, pred_post)
    unc_interp_linear = denormalize_std(unc_interp, pred_interp)

    # Compute metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(target_linear, pred_interp_linear))
    mae = mean_absolute_error(target_linear, pred_interp_linear)
    r2 = r2_score(target_linear, pred_interp_linear)

    # Log-space metrics
    target_log = normalize_agbd(target_linear)
    log_rmse = np.sqrt(mean_squared_error(target_log, pred_interp))
    log_r2 = r2_score(target_log, pred_interp)

    # UQ calibration metrics (in log space)
    calibration = compute_calibration_metrics(pred_interp, target_log, unc_interp)

    # Disturbance analysis: correlate tile-level disturbance with prediction error
    disturbance_analysis = compute_disturbance_analysis(
        gedi_df, test_df, pred_interp_linear,
        args.pre_years, args.post_years, args.test_year
    )

    print("\n" + "=" * 80)
    print("RESULTS: Temporal Linear Interpolation Baseline")
    print("=" * 80)
    print(f"Test year: {args.test_year}")
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

    print(f"\nDisturbance Analysis:")
    print(f"  Mean disturbance: {disturbance_analysis['summary']['mean_disturbance']:.1%}")
    print(f"  Tiles with biomass loss: {disturbance_analysis['summary']['n_tiles_with_loss']}")
    print(f"  Tiles with major loss (>30%): {disturbance_analysis['summary']['pct_tiles_major_loss']:.1f}%")
    if disturbance_analysis['correlation']['pearson_r'] is not None:
        print(f"  Error-disturbance correlation: r={disturbance_analysis['correlation']['pearson_r']:.3f} (p={disturbance_analysis['correlation']['p_value']:.3f})")
    if disturbance_analysis['quartile_rmse']:
        print(f"  RMSE by disturbance quartile:")
        for q, rmse_val in disturbance_analysis['quartile_rmse'].items():
            print(f"    {q}: {rmse_val:.2f} Mg/ha")

    # Save results
    results = {
        'method': 'temporal_linear_interpolation',
        'pre_years': args.pre_years,
        'post_years': args.post_years,
        'test_year': args.test_year,
        'n_context': args.n_context,
        'metrics': {
            'linear_r2': float(r2),
            'linear_rmse': float(rmse),
            'linear_mae': float(mae),
            'log_r2': float(log_r2),
            'log_rmse': float(log_rmse)
        },
        'calibration': calibration,
        'n_test_shots': len(test_df),
        'alpha_stats': {
            'mean': float(alpha.mean()),
            'std': float(alpha.std()),
            'min': float(alpha.min()),
            'max': float(alpha.max())
        },
        'disturbance': {
            'correlation': disturbance_analysis['correlation'],
            'quartile_rmse': disturbance_analysis['quartile_rmse'],
            'summary': disturbance_analysis['summary']
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions for analysis
    test_df_out = test_df[['latitude', 'longitude', 'agbd', 'time', 'tile_id']].copy()
    test_df_out['pred_pre'] = pred_pre_linear
    test_df_out['pred_post'] = pred_post_linear
    test_df_out['pred_interp'] = pred_interp_linear
    test_df_out['unc_pre'] = unc_pre_linear
    test_df_out['unc_post'] = unc_post_linear
    test_df_out['unc_interp'] = unc_interp_linear
    test_df_out['alpha'] = alpha
    test_df_out['residual'] = target_linear - pred_interp_linear
    test_df_out.to_parquet(output_dir / 'predictions.parquet')

    # Save per-tile disturbance analysis
    tile_disturbance_df = pd.DataFrame(disturbance_analysis['per_tile'])
    tile_disturbance_df.to_parquet(output_dir / 'tile_disturbance.parquet')

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
