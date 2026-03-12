"""
Generate a 5-year biomass progression map for a single tile.

Creates a multi-panel figure showing AGBD predictions for 2019-2023,
with 2021 as the temporally interpolated gap year. Designed for paper figures.

Usage:
    python generate_progression_map.py \
        --checkpoint /path/to/model \
        --tile_lon -74.35 --tile_lat -10.45 \
        --output figures/progression_map.pdf

    # With GEDI footprint overlay:
    python generate_progression_map.py \
        --checkpoint /path/to/model \
        --tile_lon -74.35 --tile_lat -10.45 \
        --show_footprints \
        --output figures/progression_map.pdf

    # With baseline comparison row:
    python generate_progression_map.py \
        --checkpoint /path/to/anp_model \
        --baseline_checkpoint /path/to/xgb_model \
        --tile_lon -74.35 --tile_lat -10.45 \
        --output figures/progression_map.pdf
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import compute_temporal_encoding
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd, denormalize_std
from utils.model import load_model_from_checkpoint
from utils.config import load_config, get_global_bounds


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 5-year biomass progression map for a single tile'
    )

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to ANP model checkpoint directory')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                        help='Path to baseline model checkpoint (adds comparison row)')

    # Tile selection
    parser.add_argument('--tile_lon', type=float, required=True,
                        help='Tile center longitude')
    parser.add_argument('--tile_lat', type=float, required=True,
                        help='Tile center latitude')
    parser.add_argument('--tile_size', type=float, default=0.1,
                        help='Tile size in degrees (default: 0.1)')

    # Years
    parser.add_argument('--years', type=int, nargs='+', default=[2019, 2020, 2021, 2022, 2023],
                        help='Years to show (default: 2019-2023)')
    parser.add_argument('--holdout_year', type=int, default=2021,
                        help='Held-out year (shown as predicted, default: 2021)')
    parser.add_argument('--context_years', type=int, nargs='+', default=None,
                        help='Years to use as context (default: all except holdout)')

    # Prediction grid
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='Prediction resolution in meters (default: 10m)')
    parser.add_argument('--n_context', type=int, default=500,
                        help='Number of context GEDI shots per year (default: 500)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Inference batch size (default: 1024)')

    # Visualization
    parser.add_argument('--show_footprints', action='store_true',
                        help='Overlay GEDI footprint locations on each panel')
    parser.add_argument('--show_uncertainty', action='store_true',
                        help='Show uncertainty panel for holdout year')
    parser.add_argument('--vmin', type=float, default=0,
                        help='Colorbar minimum AGBD (Mg/ha, default: 0)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Colorbar maximum AGBD (Mg/ha, default: auto)')
    parser.add_argument('--cmap', type=str, default='YlGn',
                        help='Colormap for AGBD (default: YlGn)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    parser.add_argument('--figwidth', type=float, default=18,
                        help='Figure width in inches (default: 18)')

    # Output
    parser.add_argument('--output', type=str, default='figures/progression_map.pdf',
                        help='Output file path (default: figures/progression_map.pdf)')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for GEDI queries')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Directory for geotessera embedding tiles')

    return parser.parse_args()


def generate_prediction_grid(
    tile_lon: float,
    tile_lat: float,
    tile_size: float,
    resolution_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a regular lon/lat grid for the tile."""
    half = tile_size / 2
    min_lon = tile_lon - half
    max_lon = tile_lon + half
    min_lat = tile_lat - half
    max_lat = tile_lat + half

    meters_per_degree = 111000.0
    lat_spacing = resolution_m / meters_per_degree
    center_lat = tile_lat
    lon_spacing = lat_spacing / np.cos(np.radians(center_lat))

    lons = np.arange(min_lon, max_lon, lon_spacing)
    lats = np.arange(min_lat, max_lat, lat_spacing)

    return lons, lats



def select_nearest_context(
    gedi_df: pd.DataFrame,
    center_lon: float,
    center_lat: float,
    n_context: int
) -> pd.DataFrame:
    """Select N nearest GEDI shots to tile center."""
    if len(gedi_df) <= n_context:
        return gedi_df

    coords = gedi_df[['longitude', 'latitude']].values
    center = np.array([[center_lon, center_lat]])
    distances = np.sqrt(((coords - center) ** 2).sum(axis=1))
    nearest = np.argsort(distances)[:n_context]
    return gedi_df.iloc[nearest].copy()


def predict_tile_year(
    model: torch.nn.Module,
    config: dict,
    extractor: EmbeddingExtractor,
    context_dfs: Dict[int, pd.DataFrame],
    query_lons: np.ndarray,
    query_lats: np.ndarray,
    target_year: int,
    holdout_year: int,
    temporal_bounds: Tuple[float, float],
    global_bounds: tuple,
    batch_size: int,
    device: str,
    cache_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Run ANP prediction for one year on the tile grid.

    Returns (predictions_grid, uncertainties_grid, n_rows, n_cols).
    """
    n_cols = len(query_lons)
    n_rows = len(query_lats)

    # Set extractor to target year for query embeddings
    extractor.set_year(target_year)

    # Build query grid
    lon_grid, lat_grid = np.meshgrid(query_lons, query_lats)
    query_df = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })

    # Extract query embeddings using batch method with caching
    query_df = extractor.extract_patches_batch(
        query_df,
        verbose=True,
        desc=f"Query embeddings {target_year}",
        cache_dir=cache_dir
    )

    # Filter to valid embeddings
    valid_mask = query_df['embedding_patch'].notna().values
    valid_indices = np.where(valid_mask)[0]
    query_df_valid = query_df[valid_mask]

    if len(query_df_valid) == 0:
        return (np.full(n_rows * n_cols, np.nan),
                np.full(n_rows * n_cols, np.nan),
                n_rows, n_cols)

    query_embeddings = np.stack(query_df_valid['embedding_patch'].values)
    query_coords = query_df_valid[['longitude', 'latitude']].values

    # Build context: use observations from context years
    # For the holdout year, context comes from surrounding years
    # For training years, context comes from all training years
    context_years = [y for y in context_dfs.keys() if y != holdout_year]
    all_context = pd.concat([context_dfs[y] for y in context_years if y in context_dfs],
                            ignore_index=True)

    if len(all_context) == 0:
        return (np.full(n_rows * n_cols, np.nan),
                np.full(n_rows * n_cols, np.nan),
                n_rows, n_cols)

    # Build spatiotemporal coordinates for context
    coord_dim = config.get('coord_dim', 2)

    context_coords = all_context[['longitude', 'latitude']].values
    context_coords_norm = normalize_coords(context_coords, global_bounds)

    if coord_dim == 5 and 'time' in all_context.columns:
        context_temporal = compute_temporal_encoding(
            all_context['time'], temporal_bounds
        )
        context_coords_full = np.concatenate([context_coords_norm, context_temporal], axis=1)
    else:
        context_coords_full = context_coords_norm

    context_embeddings_arr = np.stack(all_context['embedding_patch'].values)
    context_agbd_norm = normalize_agbd(all_context['agbd'].values[:, None])

    # Build spatiotemporal coordinates for queries
    query_coords_norm = normalize_coords(query_coords, global_bounds)

    if coord_dim == 5:
        # Create a synthetic timestamp for the target year (mid-year)
        mid_year_ts = pd.Timestamp(f"{target_year}-07-01")
        query_times = pd.Series([mid_year_ts] * len(query_coords))
        query_temporal = compute_temporal_encoding(query_times, temporal_bounds)
        query_coords_full = np.concatenate([query_coords_norm, query_temporal], axis=1)
    else:
        query_coords_full = query_coords_norm

    # Convert to tensors
    ctx_coords_t = torch.from_numpy(context_coords_full).float().to(device)
    ctx_emb_t = torch.from_numpy(context_embeddings_arr).float().to(device)
    ctx_agbd_t = torch.from_numpy(context_agbd_norm).float().to(device)

    # Pre-encode context once
    with torch.no_grad():
        context_encoded = model.encode_context(ctx_coords_t, ctx_emb_t, ctx_agbd_t)

    # Batch inference over query points
    all_preds = []
    all_stds = []
    n_batches = (len(query_coords) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc=f"  Inference {target_year}", leave=False):
            s = i * batch_size
            e = min((i + 1) * batch_size, len(query_coords))

            q_coords_t = torch.from_numpy(query_coords_full[s:e]).float().to(device)
            q_emb_t = torch.from_numpy(query_embeddings[s:e]).float().to(device)

            pred_mean, pred_log_var, _, _, _, _ = model.forward(
                context_coords=None,
                context_embeddings=None,
                context_agbd=None,
                query_coords=q_coords_t,
                query_embeddings=q_emb_t,
                training=False,
                context_encoded=context_encoded
            )

            pred_mean_np = pred_mean.cpu().numpy().flatten()
            if pred_log_var is not None:
                pred_std_np = torch.exp(0.5 * pred_log_var).cpu().numpy().flatten()
            else:
                pred_std_np = np.zeros_like(pred_mean_np)

            all_preds.append(pred_mean_np)
            all_stds.append(pred_std_np)

    preds_norm = np.concatenate(all_preds)
    stds_norm = np.concatenate(all_stds)

    # Denormalize
    preds_raw = denormalize_agbd(preds_norm)
    stds_raw = denormalize_std(stds_norm, preds_norm, simple_transform=False)

    # Place into full grid
    full_preds = np.full(n_rows * n_cols, np.nan)
    full_stds = np.full(n_rows * n_cols, np.nan)
    full_preds[valid_indices] = preds_raw
    full_stds[valid_indices] = stds_raw

    return full_preds, full_stds, n_rows, n_cols


def create_figure(
    predictions: Dict[int, np.ndarray],
    uncertainties: Dict[int, np.ndarray],
    footprints: Optional[Dict[int, pd.DataFrame]],
    lons: np.ndarray,
    lats: np.ndarray,
    n_rows: int,
    n_cols: int,
    years: List[int],
    holdout_year: int,
    output_path: str,
    vmin: float = 0,
    vmax: Optional[float] = None,
    cmap: str = 'YlGn',
    show_uncertainty: bool = True,
    dpi: int = 300,
    figwidth: float = 18
):
    """Create multi-panel figure: first year absolute AGBD, subsequent years as year-over-year gain/loss."""
    n_years = len(years)
    extent = [lons[0], lons[-1], lats[0], lats[-1]]
    base_year = years[0]

    # Compute base grid
    base_grid = predictions[base_year].reshape(n_rows, n_cols) if base_year in predictions else None

    # Auto vmax for absolute panel
    if vmax is None and base_grid is not None:
        valid = base_grid[~np.isnan(base_grid)]
        vmax = np.percentile(valid, 98) if len(valid) > 0 else 300

    # Compute year-over-year difference grids
    diff_grids = {}
    for j in range(1, len(years)):
        cur_year = years[j]
        prev_year = years[j - 1]
        if cur_year in predictions and prev_year in predictions:
            cur_grid = predictions[cur_year].reshape(n_rows, n_cols)
            prev_grid = predictions[prev_year].reshape(n_rows, n_cols)
            diff_grids[cur_year] = cur_grid - prev_grid

    # Symmetric vmax for diverging colormap
    all_diffs = [d[~np.isnan(d)] for d in diff_grids.values() if d is not None]
    if all_diffs:
        combined = np.concatenate(all_diffs)
        diff_limit = np.percentile(np.abs(combined), 98)
    else:
        diff_limit = 50

    # Figure layout: panels on top, two horizontal colorbars at bottom
    fig = plt.figure(figsize=(figwidth, figwidth / n_years * 1.3))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[1, 0.04],
        hspace=0.25
    )
    gs_panels = gridspec.GridSpecFromSubplotSpec(
        1, n_years, subplot_spec=gs[0, 0], wspace=0.08
    )
    gs_cbars = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], wspace=0.3
    )

    abs_norm = Normalize(vmin=vmin, vmax=vmax)
    diff_norm = Normalize(vmin=-diff_limit, vmax=diff_limit)

    for i, year in enumerate(years):
        ax = fig.add_subplot(gs_panels[0, i])

        is_holdout = (year == holdout_year)
        is_base = (i == 0)
        prev_year = years[i - 1] if i > 0 else None

        if is_base:
            # First panel: absolute AGBD
            if base_grid is not None:
                ax.imshow(
                    base_grid, extent=extent, origin='lower',
                    cmap=cmap, norm=abs_norm, interpolation='nearest'
                )
        else:
            # Subsequent panels: year-over-year gain/loss
            if year in diff_grids:
                ax.imshow(
                    diff_grids[year], extent=extent, origin='lower',
                    cmap='RdBu_r', norm=diff_norm, interpolation='nearest'
                )
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')

        # Overlay GEDI footprints
        if footprints is not None and year in footprints:
            fp = footprints[year]
            if len(fp) > 0:
                ax.scatter(
                    fp['longitude'], fp['latitude'],
                    s=0.3, c='black', alpha=0.4, rasterized=True
                )
                ax.text(
                    0.97, 0.03, f'n={len(fp):,}',
                    ha='right', va='bottom',
                    transform=ax.transAxes,
                    fontsize=7, color='white',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
                )

        # Year label
        if is_base:
            label = f"{year}\n(AGBD)"
        elif is_holdout:
            label = f"{year}\n(Predicted \u0394)"
        else:
            label = f"{year}\n(\u0394 vs {prev_year})"

        ax.set_title(label,
                      fontsize=11,
                      fontweight='bold' if is_holdout else 'normal',
                      color='#c0392b' if is_holdout else 'black')

        if is_holdout:
            for spine in ax.spines.values():
                spine.set_edgecolor('#c0392b')
                spine.set_linewidth(2.5)

        if i == 0:
            ax.set_ylabel('Latitude', fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('Longitude', fontsize=9)
        ax.tick_params(labelsize=7)

    # Horizontal AGBD colorbar (bottom-left)
    cbar_abs_ax = fig.add_subplot(gs_cbars[0, 0])
    cbar_abs = fig.colorbar(
        plt.cm.ScalarMappable(norm=abs_norm, cmap=cmap),
        cax=cbar_abs_ax, orientation='horizontal'
    )
    cbar_abs.set_label('AGBD (Mg/ha)', fontsize=9)
    cbar_abs.ax.tick_params(labelsize=7)

    # Horizontal delta colorbar (bottom-right)
    cbar_diff_ax = fig.add_subplot(gs_cbars[0, 1])
    cbar_diff = fig.colorbar(
        plt.cm.ScalarMappable(norm=diff_norm, cmap='RdBu_r'),
        cax=cbar_diff_ax, orientation='horizontal'
    )
    cbar_diff.set_label('\u0394 AGBD (Mg/ha)', fontsize=9)
    cbar_diff.ax.tick_params(labelsize=7)

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    show_uncertainty = args.show_uncertainty
    years = sorted(args.years)
    holdout_year = args.holdout_year
    context_years = args.context_years or [y for y in years if y != holdout_year]

    print("=" * 70)
    print("BIOMASS PROGRESSION MAP")
    print("=" * 70)
    print(f"Tile: ({args.tile_lon}, {args.tile_lat}), size={args.tile_size} deg")
    print(f"Years: {years}, holdout: {holdout_year}")
    print(f"Context years: {context_years}")
    print(f"Resolution: {args.resolution}m")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model (load_model_from_checkpoint infers coord_dim from weights)
    checkpoint_dir = Path(args.checkpoint)
    model, checkpoint, _ = load_model_from_checkpoint(checkpoint_dir, args.device)
    config = load_config(checkpoint_dir / 'config.json')
    # Infer coord_dim from model weights (matches what load_model_from_checkpoint does)
    from utils.model import infer_coord_dim
    state_dict = checkpoint['model_state_dict']
    config['coord_dim'] = infer_coord_dim(state_dict, config)
    global_bounds = get_global_bounds(config)
    coord_dim = config['coord_dim']

    print(f"Model loaded: {config.get('architecture_mode', 'unknown')}, coord_dim={coord_dim}")

    # Compute temporal bounds for the full study period
    t_min = pd.Timestamp("2019-01-01").timestamp()
    t_max = pd.Timestamp("2023-12-31").timestamp()
    temporal_bounds = (t_min, t_max)

    # Generate prediction grid
    lons, lats = generate_prediction_grid(
        args.tile_lon, args.tile_lat, args.tile_size, args.resolution
    )
    print(f"Grid: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,} pixels")

    # Initialize embedding extractor
    extractor = EmbeddingExtractor(
        year=years[0],
        patch_size=config.get('patch_size', 3),
        embeddings_dir=args.embeddings_dir
    )

    # Query GEDI context shots for each context year
    print("\nQuerying GEDI context data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    context_dfs = {}
    half = args.tile_size / 2
    buffer = 0.3

    for year in context_years:
        print(f"  Year {year}...", end=" ")
        bbox = (
            args.tile_lon - half - buffer,
            args.tile_lat - half - buffer,
            args.tile_lon + half + buffer,
            args.tile_lat + half + buffer
        )
        df = querier.query_bbox(
            bbox,
            start_time=f"{year}-01-01",
            end_time=f"{year}-12-31"
        )
        if len(df) > 0:
            df = select_nearest_context(df, args.tile_lon, args.tile_lat, args.n_context)

            # Extract embeddings using batch method with caching
            extractor.set_year(year)
            df = extractor.extract_patches_batch(
                df,
                verbose=False,
                desc=f"Context embeddings {year}",
                cache_dir=args.cache_dir
            )
            df = df[df['embedding_patch'].notna()].copy()

            # Add time column if needed
            if 'time' not in df.columns:
                df['time'] = pd.Timestamp(f"{year}-07-01")

            context_dfs[year] = df
            print(f"{len(df)} shots with embeddings")
        else:
            print("no data")

    if len(context_dfs) == 0:
        raise ValueError("No GEDI context data found for any year!")

    # Query GEDI footprints within tile (for overlay)
    footprints = None
    if args.show_footprints:
        print("\nQuerying GEDI footprints for overlay...")
        footprints = {}
        tile_bbox = (
            args.tile_lon - half,
            args.tile_lat - half,
            args.tile_lon + half,
            args.tile_lat + half
        )
        for year in years:
            fp = querier.query_bbox(
                tile_bbox,
                start_time=f"{year}-01-01",
                end_time=f"{year}-12-31"
            )
            footprints[year] = fp
            print(f"  {year}: {len(fp)} footprints")

    # Run predictions for each year
    print("\nGenerating predictions...")
    predictions = {}
    uncertainties = {}

    for year in years:
        print(f"\nYear {year}:")
        preds, stds, n_rows, n_cols = predict_tile_year(
            model=model,
            config=config,
            extractor=extractor,
            context_dfs=context_dfs,
            query_lons=lons,
            query_lats=lats,
            target_year=year,
            holdout_year=holdout_year,
            temporal_bounds=temporal_bounds,
            global_bounds=global_bounds,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir
        )

        predictions[year] = preds
        uncertainties[year] = stds

        valid = preds[~np.isnan(preds)]
        if len(valid) > 0:
            print(f"  AGBD: mean={valid.mean():.1f}, "
                  f"std={valid.std():.1f}, "
                  f"range=[{valid.min():.1f}, {valid.max():.1f}] Mg/ha")

    # Create figure
    print("\nCreating figure...")
    create_figure(
        predictions=predictions,
        uncertainties=uncertainties,
        footprints=footprints,
        lons=lons,
        lats=lats,
        n_rows=n_rows,
        n_cols=n_cols,
        years=years,
        holdout_year=holdout_year,
        output_path=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        show_uncertainty=show_uncertainty,
        dpi=args.dpi,
        figwidth=args.figwidth
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
