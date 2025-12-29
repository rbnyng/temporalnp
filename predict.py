import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.spatial import cKDTree
from typing import Optional, Tuple, Union
import pickle
from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd, denormalize_std
from utils.model import load_model_from_checkpoint
from utils.config import load_config, get_global_bounds


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate AGB predictions at 10m resolution'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--region', type=float, nargs=4, required=True,
                        metavar=('min_lon', 'min_lat', 'max_lon', 'max_lat'),
                        help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='Output resolution in meters (default: 10m)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory (default: ./predictions)')
    parser.add_argument('--n_context', type=int, default=100,
                        help='Number of nearest GEDI shots to use as context (default: 100)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Inference batch size (default: 1024)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--no_preview', action='store_true',
                        help='Disable preview generation')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching GEDI query results')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Directory where geotessera stores embedding tiles')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='GEDI query start date (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='GEDI query end date (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='GeoTessera embedding year (default: 2022)')

    return parser.parse_args()


def detect_model_type(checkpoint_dir: Path) -> str:
    checkpoint_dir = Path(checkpoint_dir)

    if (checkpoint_dir / 'xgboost.pkl').exists():
        return 'xgboost'
    elif (checkpoint_dir / 'random_forest.pkl').exists():
        return 'random_forest'
    elif (checkpoint_dir / 'mlp_dropout.pkl').exists():
        return 'mlp'
    elif (checkpoint_dir / 'idw.pkl').exists():
        return 'idw'
    elif (checkpoint_dir / 'best_r2_model.pt').exists() or (checkpoint_dir / 'best_model.pt').exists():
        return 'neural_process'
    else:
        raise ValueError(
            f"Could not detect model type in {checkpoint_dir}. "
            f"Expected either .pt files (Neural Process) or .pkl files (baselines)"
        )


def load_baseline_model(checkpoint_dir: Path, model_type: str) -> Tuple[object, dict]:
    print(f"Loading {model_type.upper()} baseline model...")
    config = load_config(checkpoint_dir / 'config.json')

    model_files = {
        'xgboost': 'xgboost.pkl',
        'random_forest': 'random_forest.pkl',
        'mlp': 'mlp_dropout.pkl',
        'idw': 'idw.pkl'
    }

    model_path = checkpoint_dir / model_files[model_type]

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Loaded {model_type} model from: {model_path}")

    return model, config


def load_model_and_config(checkpoint_dir: Path, device: str):
    checkpoint_dir = Path(checkpoint_dir)
    model_type = detect_model_type(checkpoint_dir)

    print(f"Detected model type: {model_type}")

    if model_type == 'neural_process':
        print("Loading model configuration...")
        config = load_config(checkpoint_dir / 'config.json')

        print(f"Architecture mode: {config.get('architecture_mode', 'deterministic')}")

        print("Initializing model...")
        model, checkpoint, checkpoint_path = load_model_from_checkpoint(
            checkpoint_dir, device
        )

        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_metrics' in checkpoint:
            print("Validation metrics:")
            for k, v in checkpoint['val_metrics'].items():
                print(f"  {k}: {v:.4f}")

        return model, config, model_type
    else:
        model, config = load_baseline_model(checkpoint_dir, model_type)
        return model, config, model_type


def query_context_gedi(
    region_bbox: tuple,
    n_context: int,
    start_time: str,
    end_time: str,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    print(f"\nQuerying GEDI context shots...")
    print(f"Region: {region_bbox}")
    print(f"Requesting {n_context} nearest shots")

    querier = GEDIQuerier(cache_dir=cache_dir)

    # Query a larger region to ensure we get enough shots
    min_lon, min_lat, max_lon, max_lat = region_bbox
    buffer = 0.5  # degrees (~50km)
    buffered_bbox = (
        min_lon - buffer,
        min_lat - buffer,
        max_lon + buffer,
        max_lat + buffer
    )

    gedi_df = querier.query_bbox(
        buffered_bbox,
        start_time=start_time,
        end_time=end_time
    )

    if len(gedi_df) == 0:
        raise ValueError(f"No GEDI data found in region {buffered_bbox}")

    print(f"Found {len(gedi_df)} GEDI shots in buffered region")

    # select N nearest to region center
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # compute distances to center
    gedi_coords = gedi_df[['longitude', 'latitude']].values
    center = np.array([[center_lon, center_lat]])

    # simple euclidean distance (good enough for small regions)
    distances = np.sqrt(
        ((gedi_coords - center) ** 2).sum(axis=1)
    )

    # sort by distance and take top N
    nearest_indices = np.argsort(distances)[:n_context]
    context_df = gedi_df.iloc[nearest_indices].copy()

    print(f"Selected {len(context_df)} nearest context shots")
    print(f"Context AGBD range: [{context_df['agbd'].min():.1f}, {context_df['agbd'].max():.1f}] Mg/ha")

    return context_df


def generate_prediction_grid(
    region_bbox: tuple,
    resolution_m: float
) -> tuple:
    min_lon, min_lat, max_lon, max_lat = region_bbox

    # resolution to degrees (approximate)
    # At equator: 1 degree ≈ 111km
    meters_per_degree = 111000.0
    resolution_deg = resolution_m / meters_per_degree

    # adj for latitude (longitude spacing varies with latitude)
    center_lat = (min_lat + max_lat) / 2
    lon_resolution_deg = resolution_deg / np.cos(np.radians(center_lat))
    lat_resolution_deg = resolution_deg

    lons = np.arange(min_lon, max_lon, lon_resolution_deg)
    lats = np.arange(min_lat, max_lat, lat_resolution_deg)

    n_cols = len(lons)
    n_rows = len(lats)

    print(f"\nGenerated prediction grid:")
    print(f"  Resolution: {resolution_m}m (~{resolution_deg:.6f}°)")
    print(f"  Grid size: {n_rows} x {n_cols} = {n_rows * n_cols:,} pixels")
    print(f"  Lon range: [{lons[0]:.6f}, {lons[-1]:.6f}]")
    print(f"  Lat range: [{lats[0]:.6f}, {lats[-1]:.6f}]")

    return lons, lats, n_rows, n_cols


def extract_embeddings(
    coords_df: pd.DataFrame,
    extractor: EmbeddingExtractor,
    desc: str = "Extracting embeddings"
) -> pd.DataFrame:
    patches = []
    valid_indices = []

    print(f"\n{desc}...")
    for idx, row in tqdm(coords_df.iterrows(), total=len(coords_df), desc=desc):
        patch = extractor.extract_patch(row['longitude'], row['latitude'])
        if patch is not None:
            patches.append(patch)
            valid_indices.append(idx)

    print(f"Successfully extracted {len(patches)}/{len(coords_df)} embeddings "
          f"({100*len(patches)/len(coords_df):.1f}%)")

    result_df = coords_df.loc[valid_indices].copy()
    result_df['embedding_patch'] = patches

    return result_df


def run_inference_neural_process(
    model: torch.nn.Module,
    context_df: pd.DataFrame,
    query_df: pd.DataFrame,
    global_bounds: tuple,
    batch_size: int,
    device: str
) -> tuple:
    print(f"\nRunning Neural Process inference on {len(query_df)} query points...")
    print(f"Using {len(context_df)} context shots")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    context_coords = context_df[['longitude', 'latitude']].values
    context_coords_norm = normalize_coords(context_coords, global_bounds)
    context_embeddings = np.stack(context_df['embedding_patch'].values)
    context_agbd_norm = normalize_agbd(context_df['agbd'].values[:, None])

    context_coords_t = torch.from_numpy(context_coords_norm).float().to(device)
    context_embeddings_t = torch.from_numpy(context_embeddings).float().to(device)
    context_agbd_t = torch.from_numpy(context_agbd_norm).float().to(device)

    query_coords = query_df[['longitude', 'latitude']].values
    query_coords_norm = normalize_coords(query_coords, global_bounds)
    query_embeddings = np.stack(query_df['embedding_patch'].values)

    all_predictions = []
    all_uncertainties = []

    n_batches = (len(query_df) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Inference"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(query_df))

            batch_coords = query_coords_norm[start_idx:end_idx]
            batch_embeddings = query_embeddings[start_idx:end_idx]

            batch_coords_t = torch.from_numpy(batch_coords).float().to(device)
            batch_embeddings_t = torch.from_numpy(batch_embeddings).float().to(device)

            pred_mean, pred_std = model.predict(
                context_coords_t,
                context_embeddings_t,
                context_agbd_t,
                batch_coords_t,
                batch_embeddings_t
            )

            pred_mean_np = pred_mean.cpu().numpy().flatten()
            pred_std_np = pred_std.cpu().numpy().flatten()

            all_predictions.append(pred_mean_np)
            all_uncertainties.append(pred_std_np)

    predictions_norm = np.concatenate(all_predictions)
    uncertainties_norm = np.concatenate(all_uncertainties)

    predictions = denormalize_agbd(predictions_norm)
    uncertainties = denormalize_std(uncertainties_norm, predictions_norm, simple_transform=False)

    print(f"\nPrediction statistics:")
    print(f"  Mean AGB: {predictions.mean():.2f} Mg/ha")
    print(f"  Std AGB: {predictions.std():.2f} Mg/ha")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}] Mg/ha")
    print(f"  Mean uncertainty: {uncertainties.mean():.2f} Mg/ha")

    return predictions, uncertainties


def run_inference_baseline(
    model: object,
    query_df: pd.DataFrame,
    config: dict,
    global_bounds: tuple
) -> tuple:
    print(f"\nRunning baseline inference on {len(query_df)} query points...")

    query_coords = query_df[['longitude', 'latitude']].values
    query_coords_norm = normalize_coords(query_coords, global_bounds)
    query_embeddings = np.stack(query_df['embedding_patch'].values)

    predictions_norm, uncertainties_norm = model.predict(
        query_coords_norm,
        query_embeddings,
        return_std=True
    )

    agbd_scale = config.get('agbd_scale', 200.0)
    log_transform = config.get('log_transform_agbd', True)

    predictions = denormalize_agbd(
        predictions_norm,
        agbd_scale=agbd_scale,
        log_transform=log_transform
    )

    if log_transform:
        predictions_linear_scale = np.exp(predictions_norm) * agbd_scale
        uncertainties = predictions_linear_scale * uncertainties_norm
    else:
        uncertainties = uncertainties_norm * agbd_scale

    print(f"\nPrediction statistics:")
    print(f"  Mean AGB: {predictions.mean():.2f} Mg/ha")
    print(f"  Std AGB: {predictions.std():.2f} Mg/ha")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}] Mg/ha")
    print(f"  Mean uncertainty: {uncertainties.mean():.2f} Mg/ha")

    return predictions, uncertainties


def save_geotiff(
    data: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    output_path: Path,
    description: str = "AGB"
):
    n_rows = len(lats)
    n_cols = len(lons)
    grid = data.reshape(n_rows, n_cols)

    transform = from_bounds(
        lons[0], lats[0],
        lons[-1], lats[-1],
        n_cols, n_rows
    )

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=n_rows,
        width=n_cols,
        count=1,
        dtype=grid.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(grid, 1)
        dst.set_band_description(1, description)

    print(f"Saved {description} to: {output_path}")


def save_context_geojson(
    context_df: pd.DataFrame,
    output_path: Path
):
    import geopandas as gpd
    from shapely.geometry import Point

    geometry = [Point(lon, lat) for lon, lat in
                zip(context_df['longitude'], context_df['latitude'])]

    gdf = gpd.GeoDataFrame(
        context_df[['agbd']],
        geometry=geometry,
        crs='EPSG:4326'
    )

    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Saved context points to: {output_path}")


def create_visualization(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    context_df: Optional[pd.DataFrame],
    output_path: Path,
    region_bbox: tuple
):
    print("\nGenerating visualization...")

    n_rows = len(lats)
    n_cols = len(lons)
    pred_grid = predictions.reshape(n_rows, n_cols)
    std_grid = uncertainties.reshape(n_rows, n_cols)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot mean AGB
    ax = axes[0]
    im1 = ax.imshow(
        pred_grid,
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        origin='lower',
        cmap='YlGn',
        vmin=0,
        vmax=np.nanmax(predictions)
    )

    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_title('Mean AGB Prediction')
    ax.grid(True, alpha=0.3)

    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('AGB (Mg/ha)', fontweight='bold')

    # Plot uncertainty
    ax = axes[1]
    im2 = ax.imshow(
        std_grid,
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        origin='lower',
        cmap='Reds',
        vmin=0,
        vmax=np.nanmax(uncertainties)
    )

    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_title('Prediction Uncertainty (Std)')
    ax.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('Uncertainty (Mg/ha)', fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    print("=" * 80)
    print("GEDI AGB PREDICTION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Region: {args.region}")
    print(f"Resolution: {args.resolution}m")
    print(f"Device: {args.device}")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_lon, min_lat, max_lon, max_lat = args.region
    region_name = f"region_{min_lon:.3f}_{min_lat:.3f}_{max_lon:.3f}_{max_lat:.3f}"

    checkpoint_dir = Path(args.checkpoint)
    model, config, model_type = load_model_and_config(checkpoint_dir, args.device)

    global_bounds = get_global_bounds(config)
    print(f"\nGlobal coordinate bounds from training:")
    print(f"  Lon: [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}]")
    print(f"  Lat: [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")

    # only query context GEDI data for Neural Process models
    context_df = None
    if model_type == 'neural_process':
        print(f"Context shots: {args.n_context}")
        context_df = query_context_gedi(
            tuple(args.region),
            args.n_context,
            args.start_time,
            args.end_time,
            args.cache_dir
        )
    else:
        print("Baseline model detected - no context shots needed")

    print(f"\nInitializing GeoTessera extractor (year={args.embedding_year})...")
    extractor = EmbeddingExtractor(
        year=args.embedding_year,
        patch_size=config.get('patch_size', 3),
        embeddings_dir=args.embeddings_dir
    )

    # embeddings for context (NP only)
    if context_df is not None:
        context_df = extract_embeddings(
            context_df,
            extractor,
            desc="Extracting context embeddings"
        )

        if len(context_df) == 0:
            raise ValueError("No valid context embeddings extracted!")

    lons, lats, n_rows, n_cols = generate_prediction_grid(
        tuple(args.region),
        args.resolution
    )

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    query_df = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })

    print(f"Total query points: {len(query_df):,}")

    query_df = extract_embeddings(
        query_df,
        extractor,
        desc="Extracting query embeddings"
    )

    if len(query_df) == 0:
        raise ValueError("No valid query embeddings extracted!")

    print(f"\nWill predict for {len(query_df):,} valid points "
          f"({100*len(query_df)/(n_rows*n_cols):.1f}% of grid)")

    if model_type == 'neural_process':
        predictions, uncertainties = run_inference_neural_process(
            model,
            context_df,
            query_df,
            global_bounds,
            args.batch_size,
            args.device
        )
    else:
        predictions, uncertainties = run_inference_baseline(
            model,
            query_df,
            config,
            global_bounds
        )

    full_predictions = np.full(n_rows * n_cols, np.nan)
    full_uncertainties = np.full(n_rows * n_cols, np.nan)

    query_lons = query_df['longitude'].values
    query_lats = query_df['latitude'].values

    for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
        lon_idx = np.argmin(np.abs(lons - query_lons[i]))
        lat_idx = np.argmin(np.abs(lats - query_lats[i]))
        grid_idx = lat_idx * n_cols + lon_idx

        full_predictions[grid_idx] = pred
        full_uncertainties[grid_idx] = unc

    print("\nSaving outputs...")

    save_geotiff(
        full_predictions,
        lons, lats,
        output_dir / f"{region_name}_agb_mean.tif",
        "AGB Mean (Mg/ha)"
    )

    save_geotiff(
        full_uncertainties,
        lons, lats,
        output_dir / f"{region_name}_agb_std.tif",
        "AGB Uncertainty (Mg/ha)"
    )

    if context_df is not None:
        save_context_geojson(
            context_df,
            output_dir / f"{region_name}_context.geojson"
        )

    if not args.no_preview:
        create_visualization(
            full_predictions,
            full_uncertainties,
            lons, lats,
            context_df,
            output_dir / f"{region_name}_preview.png",
            tuple(args.region)
        )

    metadata = {
        'region_bbox': args.region,
        'resolution_m': args.resolution,
        'model_type': model_type,
        'n_context': len(context_df) if context_df is not None else 0,
        'n_predictions': int((~np.isnan(full_predictions)).sum()),
        'grid_size': [n_rows, n_cols],
        'checkpoint': str(checkpoint_dir),
        'config': config,
        'statistics': {
            'mean_agb': float(np.nanmean(full_predictions)),
            'std_agb': float(np.nanstd(full_predictions)),
            'min_agb': float(np.nanmin(full_predictions)),
            'max_agb': float(np.nanmax(full_predictions)),
            'mean_uncertainty': float(np.nanmean(full_uncertainties))
        }
    }

    with open(output_dir / f"{region_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - {region_name}_agb_mean.tif")
    print(f"  - {region_name}_agb_std.tif")
    if context_df is not None:
        print(f"  - {region_name}_context.geojson")
    if not args.no_preview:
        print(f"  - {region_name}_preview.png")
    print(f"  - {region_name}_metadata.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
