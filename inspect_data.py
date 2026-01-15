"""
Inspect GEDI data for spatiotemporal experiments.

Checks:
1. Timestamp format and parsing
2. Temporal distribution by year
3. Spatial coverage per year
4. Data availability for fire case study regions
5. Validates temporal encoding computation
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from data.gedi import GEDIQuerier
from data.dataset import compute_temporal_encoding


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect GEDI data')
    parser.add_argument('--region_bbox', type=float, nargs=4,
                        default=[-73, 2, -72, 3],
                        help='Region bounding box')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings')
    parser.add_argument('--start_year', type=int, default=2019)
    parser.add_argument('--end_year', type=int, default=2023)
    parser.add_argument('--check_embeddings', action='store_true',
                        help='Also check embedding extraction for each year')
    return parser.parse_args()


def inspect_timestamps(df: pd.DataFrame) -> None:
    """Inspect timestamp column format and values."""
    print("\n" + "=" * 60)
    print("TIMESTAMP INSPECTION")
    print("=" * 60)

    # Check what columns exist
    print("\nAvailable columns:")
    print(f"  {list(df.columns)}")

    # Find timestamp column
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    print(f"\nPotential time columns: {time_cols}")

    if 'time' in df.columns:
        time_col = 'time'
    elif 'datetime' in df.columns:
        time_col = 'datetime'
    elif 'date_time' in df.columns:
        time_col = 'date_time'
    else:
        print("WARNING: No obvious timestamp column found!")
        return

    print(f"\nUsing column: '{time_col}'")
    print(f"  dtype: {df[time_col].dtype}")
    print(f"  First 5 values:")
    for i, val in enumerate(df[time_col].head(5)):
        print(f"    {i}: {val} (type: {type(val).__name__})")

    # Try parsing
    print("\nParsing timestamps...")
    try:
        timestamps = pd.to_datetime(df[time_col])
        print(f"  Successfully parsed {len(timestamps)} timestamps")
        print(f"  Range: {timestamps.min()} to {timestamps.max()}")
        print(f"  dtype after parsing: {timestamps.dtype}")
    except Exception as e:
        print(f"  ERROR parsing: {e}")
        return

    # Check components
    print("\nTimestamp components:")
    print(f"  Year range: {timestamps.dt.year.min()} - {timestamps.dt.year.max()}")
    print(f"  Day of year range: {timestamps.dt.dayofyear.min()} - {timestamps.dt.dayofyear.max()}")

    # Unix timestamp conversion
    unix_ts = timestamps.astype(np.int64) / 1e9
    print(f"\nUnix timestamp (seconds):")
    print(f"  Range: {unix_ts.min():.0f} to {unix_ts.max():.0f}")
    print(f"  As dates: {pd.Timestamp(unix_ts.min(), unit='s')} to {pd.Timestamp(unix_ts.max(), unit='s')}")


def inspect_temporal_distribution(df: pd.DataFrame) -> None:
    """Inspect temporal distribution of data."""
    print("\n" + "=" * 60)
    print("TEMPORAL DISTRIBUTION")
    print("=" * 60)

    timestamps = pd.to_datetime(df['time'])
    df = df.copy()
    df['year'] = timestamps.dt.year
    df['month'] = timestamps.dt.month
    df['day_of_year'] = timestamps.dt.dayofyear

    # By year
    print("\nShots per year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} shots")

    # By month (aggregated across years)
    print("\nShots per month (all years):")
    month_counts = df['month'].value_counts().sort_index()
    for month, count in month_counts.items():
        print(f"  {month:2d}: {count:,} shots")

    # Tiles per year
    print("\nTiles per year:")
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        n_tiles = year_df['tile_id'].nunique()
        print(f"  {year}: {n_tiles} tiles")


def inspect_spatial_coverage(df: pd.DataFrame, bbox: list) -> None:
    """Inspect spatial coverage by year."""
    print("\n" + "=" * 60)
    print("SPATIAL COVERAGE BY YEAR")
    print("=" * 60)

    timestamps = pd.to_datetime(df['time'])
    df = df.copy()
    df['year'] = timestamps.dt.year

    print(f"\nRegion bbox: {bbox}")
    print(f"  Lon range: [{bbox[0]}, {bbox[2]}]")
    print(f"  Lat range: [{bbox[1]}, {bbox[3]}]")

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        print(f"\n{year}:")
        print(f"  Shots: {len(year_df):,}")
        print(f"  Tiles: {year_df['tile_id'].nunique()}")
        print(f"  Lon: [{year_df['longitude'].min():.3f}, {year_df['longitude'].max():.3f}]")
        print(f"  Lat: [{year_df['latitude'].min():.3f}, {year_df['latitude'].max():.3f}]")
        print(f"  AGBD: mean={year_df['agbd'].mean():.1f}, std={year_df['agbd'].std():.1f}, "
              f"range=[{year_df['agbd'].min():.1f}, {year_df['agbd'].max():.1f}]")


def validate_temporal_encoding(df: pd.DataFrame) -> None:
    """Validate that temporal encoding computation works correctly."""
    print("\n" + "=" * 60)
    print("TEMPORAL ENCODING VALIDATION")
    print("=" * 60)

    timestamps = pd.to_datetime(df['time'])

    # Compute temporal bounds
    unix_time = timestamps.astype(np.int64) / 1e9
    t_min, t_max = unix_time.min(), unix_time.max()
    print(f"\nTemporal bounds:")
    print(f"  t_min: {t_min:.0f} ({pd.Timestamp(t_min, unit='s')})")
    print(f"  t_max: {t_max:.0f} ({pd.Timestamp(t_max, unit='s')})")

    # Compute encoding
    encoding = compute_temporal_encoding(df['time'], (t_min, t_max))
    print(f"\nEncoding shape: {encoding.shape}")
    print(f"  [sin(2π·doy), cos(2π·doy), normalized_time]")

    # Check encoding values
    sin_doy, cos_doy, norm_time = encoding[:, 0], encoding[:, 1], encoding[:, 2]

    print(f"\nsin(2π·doy/365):")
    print(f"  Range: [{sin_doy.min():.4f}, {sin_doy.max():.4f}]")
    print(f"  Should be in [-1, 1]: {sin_doy.min() >= -1 and sin_doy.max() <= 1}")

    print(f"\ncos(2π·doy/365):")
    print(f"  Range: [{cos_doy.min():.4f}, {cos_doy.max():.4f}]")
    print(f"  Should be in [-1, 1]: {cos_doy.min() >= -1 and cos_doy.max() <= 1}")

    print(f"\nnormalized_time:")
    print(f"  Range: [{norm_time.min():.4f}, {norm_time.max():.4f}]")
    print(f"  Should be in [0, 1]: {norm_time.min() >= 0 and norm_time.max() <= 1}")

    # Check a few specific examples
    print("\nExample encodings:")
    sample_idx = np.linspace(0, len(df)-1, 5, dtype=int)
    for idx in sample_idx:
        ts = timestamps.iloc[idx]
        enc = encoding[idx]
        print(f"  {ts} -> sin={enc[0]:.3f}, cos={enc[1]:.3f}, norm_t={enc[2]:.3f}")

    # Verify sin²+cos² ≈ 1
    sin_cos_sum = sin_doy**2 + cos_doy**2
    print(f"\nsin²+cos² check (should be ~1.0):")
    print(f"  Mean: {sin_cos_sum.mean():.6f}")
    print(f"  Std: {sin_cos_sum.std():.6f}")


def check_fire_case_study_data(df: pd.DataFrame) -> None:
    """Check data availability for fire case study."""
    print("\n" + "=" * 60)
    print("FIRE CASE STUDY DATA CHECK")
    print("=" * 60)

    timestamps = pd.to_datetime(df['time'])
    df = df.copy()
    df['year'] = timestamps.dt.year

    train_years = [2019, 2020, 2022, 2023]
    test_year = 2021

    print(f"\nRequired years:")
    print(f"  Train: {train_years}")
    print(f"  Test: {test_year}")

    available_years = set(df['year'].unique())
    missing_train = set(train_years) - available_years
    if missing_train:
        print(f"\n⚠️  Missing train years: {missing_train}")
    else:
        print(f"\n✓ All train years available")

    if test_year not in available_years:
        print(f"⚠️  Missing test year: {test_year}")
    else:
        print(f"✓ Test year {test_year} available")

    # Check tile overlap across years
    print("\nTile overlap analysis:")
    tiles_by_year = {year: set(df[df['year'] == year]['tile_id'].unique())
                     for year in available_years}

    if test_year in tiles_by_year:
        test_tiles = tiles_by_year[test_year]
        train_tiles = set()
        for y in train_years:
            if y in tiles_by_year:
                train_tiles |= tiles_by_year[y]

        overlap = test_tiles & train_tiles
        print(f"  Test year tiles: {len(test_tiles)}")
        print(f"  Train years tiles: {len(train_tiles)}")
        print(f"  Overlapping tiles: {len(overlap)} ({100*len(overlap)/len(test_tiles):.1f}% of test)")

    # Estimate data for experiment
    print("\nEstimated experiment data:")
    train_shots = len(df[df['year'].isin(train_years)])
    test_shots = len(df[df['year'] == test_year])
    print(f"  Train shots (all train years): {train_shots:,}")
    print(f"  Test shots ({test_year}): {test_shots:,}")


def check_embeddings_by_year(df: pd.DataFrame, embeddings_dir: str, n_samples: int = 5) -> None:
    """Check that embeddings can be extracted for each year."""
    print("\n" + "=" * 60)
    print("EMBEDDING EXTRACTION CHECK")
    print("=" * 60)

    try:
        from data.embeddings import EmbeddingExtractor
    except ImportError:
        print("Could not import EmbeddingExtractor, skipping check")
        return

    timestamps = pd.to_datetime(df['time'])
    df = df.copy()
    df['year'] = timestamps.dt.year

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year].head(n_samples).copy()
        print(f"\n{year}: Testing {len(year_df)} samples...")

        try:
            extractor = EmbeddingExtractor(
                year=year,
                patch_size=3,
                embeddings_dir=embeddings_dir
            )
            year_df = extractor.extract_patches_batch(year_df, verbose=False)

            valid = year_df['embedding_patch'].notna().sum()
            if valid == len(year_df):
                print(f"  ✓ All {valid} embeddings extracted successfully")
                # Check shape
                sample_patch = year_df['embedding_patch'].iloc[0]
                print(f"  ✓ Patch shape: {sample_patch.shape}")
            else:
                print(f"  ⚠️  Only {valid}/{len(year_df)} embeddings extracted")

        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    args = parse_args()

    print("=" * 60)
    print("GEDI DATA INSPECTION")
    print("=" * 60)
    print(f"Region: {args.region_bbox}")
    print(f"Years: {args.start_year} - {args.end_year}")

    # Query data
    print("\nQuerying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=f'{args.start_year}-01-01',
        end_time=f'{args.end_year}-12-31',
        max_agbd=500.0
    )
    print(f"Retrieved {len(df):,} shots")

    if len(df) == 0:
        print("No data retrieved. Check region and date range.")
        return

    # Run inspections
    inspect_timestamps(df)
    inspect_temporal_distribution(df)
    inspect_spatial_coverage(df, args.region_bbox)
    validate_temporal_encoding(df)
    check_fire_case_study_data(df)

    if args.check_embeddings:
        check_embeddings_by_year(df, args.embeddings_dir)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
