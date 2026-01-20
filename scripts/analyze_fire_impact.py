"""
Analyze GEDI biomass changes within fire perimeters using MTBS data.

This script:
1. Loads MTBS fire boundary shapefile
2. Queries GEDI shots within/near the fire boundary
3. Compares pre-fire vs post-fire biomass (AGBD)
4. Optionally stratifies by burn severity from dNBR

Usage:
    python scripts/analyze_fire_impact.py \
        --shapefile /path/to/burn_bndy.shp \
        --fire_year 2021 \
        --output_dir ./results/dixie_fire_analysis
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.gedi import GEDIQuerier
from utils.config import _make_serializable


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze GEDI biomass changes within fire perimeters')

    parser.add_argument('--shapefile', type=str, required=True,
                        help='Path to MTBS burn boundary shapefile (.shp)')
    parser.add_argument('--dnbr_raster', type=str, default=None,
                        help='Optional: Path to dNBR raster for severity stratification')
    parser.add_argument('--fire_year', type=int, required=True,
                        help='Year of the fire event')
    parser.add_argument('--pre_years', type=int, nargs='+', default=None,
                        help='Years before fire (default: fire_year-2, fire_year-1)')
    parser.add_argument('--post_years', type=int, nargs='+', default=None,
                        help='Years after fire (default: fire_year+1, fire_year+2)')
    parser.add_argument('--buffer_km', type=float, default=0,
                        help='Buffer around fire boundary in km (for control comparison)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='GEDI cache directory')

    return parser.parse_args()


def load_fire_boundary(shapefile_path: str) -> gpd.GeoDataFrame:
    """Load MTBS fire boundary shapefile."""
    gdf = gpd.read_file(shapefile_path)

    # Ensure CRS is WGS84 for compatibility with GEDI
    if gdf.crs is None:
        print("Warning: No CRS found, assuming EPSG:4326")
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        print(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')

    return gdf


def get_bounding_box(gdf: gpd.GeoDataFrame, buffer_deg: float = 0.1) -> Tuple[float, float, float, float]:
    """Get bounding box with optional buffer."""
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    return (
        bounds[0] - buffer_deg,  # min_lon
        bounds[1] - buffer_deg,  # min_lat
        bounds[2] + buffer_deg,  # max_lon
        bounds[3] + buffer_deg   # max_lat
    )


def filter_shots_by_geometry(df: pd.DataFrame, geometry, inside: bool = True) -> pd.DataFrame:
    """Filter GEDI shots to those inside/outside a geometry."""
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
                          crs='EPSG:4326')

    # Check which points are within the geometry
    within_mask = points.within(geometry.union_all() if hasattr(geometry, 'union_all') else geometry.unary_union)

    if inside:
        return df[within_mask].copy()
    else:
        return df[~within_mask].copy()


def extract_dnbr_values(df: pd.DataFrame, dnbr_path: str) -> pd.Series:
    """Extract dNBR values at GEDI shot locations."""
    with rasterio.open(dnbr_path) as src:
        dnbr_values = []
        for _, row in df.iterrows():
            try:
                # Sample raster at point location
                for val in src.sample([(row['longitude'], row['latitude'])]):
                    dnbr_values.append(val[0] if val[0] != src.nodata else np.nan)
            except Exception:
                dnbr_values.append(np.nan)

    return pd.Series(dnbr_values, index=df.index)


def classify_burn_severity(dnbr: pd.Series) -> pd.Series:
    """
    Classify burn severity based on dNBR values.

    MTBS severity classes (approximate thresholds):
    - Unburned/Low: dNBR < 100
    - Low: 100 <= dNBR < 270
    - Moderate: 270 <= dNBR < 440
    - High: dNBR >= 440
    """
    conditions = [
        dnbr < 100,
        (dnbr >= 100) & (dnbr < 270),
        (dnbr >= 270) & (dnbr < 440),
        dnbr >= 440
    ]
    choices = ['unburned', 'low', 'moderate', 'high']

    return pd.Series(np.select(conditions, choices, default='unknown'), index=dnbr.index)


def compute_biomass_change_stats(pre_df: pd.DataFrame, post_df: pd.DataFrame,
                                  fire_df: Optional[pd.DataFrame] = None) -> dict:
    """Compute statistics on biomass change."""
    stats_dict = {
        'pre_fire': {
            'n_shots': len(pre_df),
            'mean_agbd': float(pre_df['agbd'].mean()),
            'std_agbd': float(pre_df['agbd'].std()),
            'median_agbd': float(pre_df['agbd'].median()),
            'total_shots_per_year': pre_df.groupby('year').size().to_dict()
        },
        'post_fire': {
            'n_shots': len(post_df),
            'mean_agbd': float(post_df['agbd'].mean()),
            'std_agbd': float(post_df['agbd'].std()),
            'median_agbd': float(post_df['agbd'].median()),
            'total_shots_per_year': post_df.groupby('year').size().to_dict()
        }
    }

    # Compute change
    pre_mean = stats_dict['pre_fire']['mean_agbd']
    post_mean = stats_dict['post_fire']['mean_agbd']

    stats_dict['change'] = {
        'absolute_change': float(post_mean - pre_mean),
        'relative_change': float((post_mean - pre_mean) / pre_mean) if pre_mean > 0 else None,
        'percent_change': float((post_mean - pre_mean) / pre_mean * 100) if pre_mean > 0 else None
    }

    # Statistical test (Welch's t-test)
    if len(pre_df) >= 10 and len(post_df) >= 10:
        t_stat, p_value = stats.ttest_ind(pre_df['agbd'], post_df['agbd'], equal_var=False)
        stats_dict['statistical_test'] = {
            'test': 'welch_t_test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }

    # Fire year stats if available
    if fire_df is not None and len(fire_df) > 0:
        stats_dict['fire_year'] = {
            'n_shots': len(fire_df),
            'mean_agbd': float(fire_df['agbd'].mean()),
            'std_agbd': float(fire_df['agbd'].std()),
            'median_agbd': float(fire_df['agbd'].median())
        }

    return stats_dict


def analyze_by_severity(df: pd.DataFrame, severity_col: str = 'severity') -> dict:
    """Analyze biomass by burn severity class."""
    results = {}

    for severity in ['unburned', 'low', 'moderate', 'high']:
        severity_df = df[df[severity_col] == severity]
        if len(severity_df) > 0:
            results[severity] = {
                'n_shots': len(severity_df),
                'mean_agbd': float(severity_df['agbd'].mean()),
                'std_agbd': float(severity_df['agbd'].std()),
                'median_agbd': float(severity_df['agbd'].median())
            }

    return results


def create_visualizations(pre_df: pd.DataFrame, post_df: pd.DataFrame,
                          fire_boundary: gpd.GeoDataFrame,
                          output_dir: Path, fire_year: int,
                          fire_df: Optional[pd.DataFrame] = None):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Spatial distribution of shots
    ax1 = axes[0, 0]
    fire_boundary.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=2, label='Fire Boundary')
    if len(pre_df) > 0:
        ax1.scatter(pre_df['longitude'], pre_df['latitude'], c='blue', s=1, alpha=0.3, label='Pre-fire')
    if len(post_df) > 0:
        ax1.scatter(post_df['longitude'], post_df['latitude'], c='green', s=1, alpha=0.3, label='Post-fire')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GEDI Shot Locations')
    ax1.legend()

    # 2. AGBD distributions
    ax2 = axes[0, 1]
    bins = np.linspace(0, 300, 50)
    if len(pre_df) > 0:
        ax2.hist(pre_df['agbd'], bins=bins, alpha=0.5, label=f'Pre-fire (n={len(pre_df)})', color='blue')
    if len(post_df) > 0:
        ax2.hist(post_df['agbd'], bins=bins, alpha=0.5, label=f'Post-fire (n={len(post_df)})', color='green')
    ax2.axvline(pre_df['agbd'].mean() if len(pre_df) > 0 else 0, color='blue', linestyle='--', label='Pre-fire mean')
    ax2.axvline(post_df['agbd'].mean() if len(post_df) > 0 else 0, color='green', linestyle='--', label='Post-fire mean')
    ax2.set_xlabel('AGBD (Mg/ha)')
    ax2.set_ylabel('Count')
    ax2.set_title('AGBD Distribution: Pre vs Post Fire')
    ax2.legend()

    # 3. Yearly trends
    ax3 = axes[1, 0]
    all_df = pd.concat([pre_df, post_df])
    if fire_df is not None:
        all_df = pd.concat([all_df, fire_df])

    yearly_stats = all_df.groupby('year')['agbd'].agg(['mean', 'std', 'count'])
    years = yearly_stats.index
    ax3.errorbar(years, yearly_stats['mean'], yerr=yearly_stats['std']/np.sqrt(yearly_stats['count']),
                 fmt='o-', capsize=5, capthick=2)
    ax3.axvline(fire_year, color='red', linestyle='--', alpha=0.7, label=f'Fire Year ({fire_year})')
    ax3.fill_betweenx([0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 200],
                       fire_year - 0.5, fire_year + 0.5, alpha=0.2, color='red')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Mean AGBD (Mg/ha)')
    ax3.set_title('Yearly Mean AGBD Trend')
    ax3.legend()

    # 4. Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = []
    labels = []
    if len(pre_df) > 0:
        data_to_plot.append(pre_df['agbd'].values)
        labels.append(f'Pre-fire\n(n={len(pre_df)})')
    if fire_df is not None and len(fire_df) > 0:
        data_to_plot.append(fire_df['agbd'].values)
        labels.append(f'Fire Year\n(n={len(fire_df)})')
    if len(post_df) > 0:
        data_to_plot.append(post_df['agbd'].values)
        labels.append(f'Post-fire\n(n={len(post_df)})')

    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightyellow', 'lightgreen'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    ax4.set_ylabel('AGBD (Mg/ha)')
    ax4.set_title('AGBD Distribution by Period')

    plt.tight_layout()
    plt.savefig(output_dir / 'biomass_change_analysis.png', dpi=150)
    plt.close()

    print(f"Saved visualization to {output_dir / 'biomass_change_analysis.png'}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default years if not provided
    if args.pre_years is None:
        args.pre_years = [args.fire_year - 2, args.fire_year - 1]
    if args.post_years is None:
        args.post_years = [args.fire_year + 1, args.fire_year + 2]

    print("=" * 80)
    print("Fire Impact Analysis using GEDI and MTBS Data")
    print("=" * 80)
    print(f"Fire boundary: {args.shapefile}")
    print(f"Fire year: {args.fire_year}")
    print(f"Pre-fire years: {args.pre_years}")
    print(f"Post-fire years: {args.post_years}")
    print()

    # Load fire boundary
    print("Loading fire boundary...")
    fire_boundary = load_fire_boundary(args.shapefile)
    fire_area_km2 = fire_boundary.to_crs('EPSG:32610').area.sum() / 1e6  # Convert to km²
    print(f"Fire perimeter area: {fire_area_km2:.1f} km²")

    # Get bounding box for GEDI query
    bbox = get_bounding_box(fire_boundary, buffer_deg=0.05)
    print(f"Query bounding box: {bbox}")

    # Query GEDI data
    print("\nQuerying GEDI data...")
    all_years = args.pre_years + [args.fire_year] + args.post_years
    start_year = min(all_years)
    end_year = max(all_years)

    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=bbox,
        tile_size=0.1,
        start_time=f'{start_year}-01-01',
        end_time=f'{end_year}-12-31',
        max_agbd=500.0
    )

    if len(gedi_df) == 0:
        print("No GEDI data found in region!")
        return

    gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
    print(f"Retrieved {len(gedi_df)} GEDI shots")
    print(f"Shots per year: {dict(gedi_df['year'].value_counts().sort_index())}")

    # Filter shots within fire boundary
    print("\nFiltering shots within fire boundary...")
    inside_fire = filter_shots_by_geometry(gedi_df, fire_boundary.geometry, inside=True)
    print(f"Shots inside fire boundary: {len(inside_fire)}")

    if len(inside_fire) == 0:
        print("No GEDI shots found within fire boundary!")
        return

    # Split by time period
    pre_fire_df = inside_fire[inside_fire['year'].isin(args.pre_years)]
    fire_year_df = inside_fire[inside_fire['year'] == args.fire_year]
    post_fire_df = inside_fire[inside_fire['year'].isin(args.post_years)]

    print(f"\nPre-fire shots ({args.pre_years}): {len(pre_fire_df)}")
    print(f"Fire year shots ({args.fire_year}): {len(fire_year_df)}")
    print(f"Post-fire shots ({args.post_years}): {len(post_fire_df)}")

    # Compute biomass change statistics
    print("\n" + "=" * 80)
    print("BIOMASS CHANGE ANALYSIS")
    print("=" * 80)

    change_stats = compute_biomass_change_stats(pre_fire_df, post_fire_df, fire_year_df)

    print(f"\nPre-fire AGBD:")
    print(f"  Mean: {change_stats['pre_fire']['mean_agbd']:.1f} Mg/ha")
    print(f"  Std:  {change_stats['pre_fire']['std_agbd']:.1f} Mg/ha")
    print(f"  Median: {change_stats['pre_fire']['median_agbd']:.1f} Mg/ha")

    if 'fire_year' in change_stats:
        print(f"\nFire year AGBD ({args.fire_year}):")
        print(f"  Mean: {change_stats['fire_year']['mean_agbd']:.1f} Mg/ha")
        print(f"  Std:  {change_stats['fire_year']['std_agbd']:.1f} Mg/ha")

    print(f"\nPost-fire AGBD:")
    print(f"  Mean: {change_stats['post_fire']['mean_agbd']:.1f} Mg/ha")
    print(f"  Std:  {change_stats['post_fire']['std_agbd']:.1f} Mg/ha")
    print(f"  Median: {change_stats['post_fire']['median_agbd']:.1f} Mg/ha")

    print(f"\nBiomass Change:")
    print(f"  Absolute: {change_stats['change']['absolute_change']:.1f} Mg/ha")
    if change_stats['change']['percent_change'] is not None:
        print(f"  Relative: {change_stats['change']['percent_change']:.1f}%")

    if 'statistical_test' in change_stats:
        print(f"\nStatistical Significance (Welch's t-test):")
        print(f"  t-statistic: {change_stats['statistical_test']['t_statistic']:.2f}")
        print(f"  p-value: {change_stats['statistical_test']['p_value']:.2e}")
        print(f"  Significant at α=0.05: {change_stats['statistical_test']['significant_at_0.05']}")

    # Monthly breakdown for fire year (to see pre vs during/post fire within that year)
    if len(fire_year_df) > 0:
        print(f"\n" + "=" * 80)
        print(f"MONTHLY BREAKDOWN FOR FIRE YEAR ({args.fire_year})")
        print("=" * 80)
        fire_year_df = fire_year_df.copy()
        fire_year_df['month'] = pd.to_datetime(fire_year_df['time']).dt.month
        fire_year_df['month_name'] = pd.to_datetime(fire_year_df['time']).dt.strftime('%b')

        monthly_stats = fire_year_df.groupby(['month', 'month_name'])['agbd'].agg(['mean', 'std', 'count'])
        monthly_stats = monthly_stats.reset_index().sort_values('month')

        print(f"\nNote: Dixie Fire started July 14, 2021")
        print(f"\n{'Month':<10} {'Mean AGBD':>12} {'Std':>10} {'N Shots':>10}")
        print("-" * 45)
        for _, row in monthly_stats.iterrows():
            fire_marker = " <-- fire started" if row['month'] == 7 else ""
            print(f"{row['month_name']:<10} {row['mean']:>10.1f} Mg/ha {row['std']:>8.1f} {int(row['count']):>10}{fire_marker}")

        # Compare pre-fire months (Jan-Jun) vs fire/post-fire months (Jul-Dec)
        pre_fire_months = fire_year_df[fire_year_df['month'] <= 6]
        post_fire_months = fire_year_df[fire_year_df['month'] >= 7]

        if len(pre_fire_months) > 0 and len(post_fire_months) > 0:
            print(f"\nWithin-year comparison:")
            print(f"  Jan-Jun (pre-fire):  {pre_fire_months['agbd'].mean():.1f} Mg/ha (n={len(pre_fire_months)})")
            print(f"  Jul-Dec (fire/post): {post_fire_months['agbd'].mean():.1f} Mg/ha (n={len(post_fire_months)})")
            within_year_change = post_fire_months['agbd'].mean() - pre_fire_months['agbd'].mean()
            print(f"  Within-year change:  {within_year_change:.1f} Mg/ha")

        change_stats['fire_year_monthly'] = {
            'monthly': {row['month_name']: {'mean': float(row['mean']), 'std': float(row['std']), 'count': int(row['count'])}
                        for _, row in monthly_stats.iterrows()},
            'pre_fire_months_mean': float(pre_fire_months['agbd'].mean()) if len(pre_fire_months) > 0 else None,
            'post_fire_months_mean': float(post_fire_months['agbd'].mean()) if len(post_fire_months) > 0 else None
        }

    # Paired tile-level analysis (controls for GEDI track sampling bias)
    print(f"\n" + "=" * 80)
    print("PAIRED TILE-LEVEL ANALYSIS (Controls for Sampling Bias)")
    print("=" * 80)

    # Compute tile-level means for pre and post periods
    pre_tile_means = pre_fire_df.groupby('tile_id')['agbd'].agg(['mean', 'count']).rename(
        columns={'mean': 'pre_mean', 'count': 'pre_count'})
    post_tile_means = post_fire_df.groupby('tile_id')['agbd'].agg(['mean', 'count']).rename(
        columns={'mean': 'post_mean', 'count': 'post_count'})

    # Find tiles with shots in BOTH periods
    paired_tiles = pre_tile_means.join(post_tile_means, how='inner')
    paired_tiles['change'] = paired_tiles['post_mean'] - paired_tiles['pre_mean']

    print(f"\nTiles with shots in both pre ({args.pre_years}) and post ({args.post_years}): {len(paired_tiles)}")
    print(f"Total tiles in fire area: {inside_fire['tile_id'].nunique()}")

    if len(paired_tiles) >= 5:
        # Paired t-test (same tiles, before vs after)
        paired_t, paired_p = stats.ttest_rel(paired_tiles['pre_mean'], paired_tiles['post_mean'])

        print(f"\nPaired tile-level comparison:")
        print(f"  Pre-fire mean (across tiles):  {paired_tiles['pre_mean'].mean():.1f} Mg/ha")
        print(f"  Post-fire mean (across tiles): {paired_tiles['post_mean'].mean():.1f} Mg/ha")
        print(f"  Mean change per tile:          {paired_tiles['change'].mean():.1f} Mg/ha")
        print(f"  Median change per tile:        {paired_tiles['change'].median():.1f} Mg/ha")
        print(f"  Tiles with biomass loss:       {(paired_tiles['change'] < 0).sum()} / {len(paired_tiles)} ({(paired_tiles['change'] < 0).mean()*100:.1f}%)")
        print(f"\n  Paired t-test: t={paired_t:.2f}, p={paired_p:.2e}")
        print(f"  Significant at α=0.05: {paired_p < 0.05}")

        # Breakdown by magnitude of change
        print(f"\n  Distribution of tile-level changes:")
        print(f"    Large loss (< -50 Mg/ha):    {(paired_tiles['change'] < -50).sum()} tiles")
        print(f"    Moderate loss (-50 to -20):  {((paired_tiles['change'] >= -50) & (paired_tiles['change'] < -20)).sum()} tiles")
        print(f"    Small loss (-20 to 0):       {((paired_tiles['change'] >= -20) & (paired_tiles['change'] < 0)).sum()} tiles")
        print(f"    Stable/gain (>= 0):          {(paired_tiles['change'] >= 0).sum()} tiles")

        change_stats['paired_tile_analysis'] = {
            'n_paired_tiles': len(paired_tiles),
            'pre_mean': float(paired_tiles['pre_mean'].mean()),
            'post_mean': float(paired_tiles['post_mean'].mean()),
            'mean_change': float(paired_tiles['change'].mean()),
            'median_change': float(paired_tiles['change'].median()),
            'pct_tiles_with_loss': float((paired_tiles['change'] < 0).mean() * 100),
            'paired_t_stat': float(paired_t),
            'paired_p_value': float(paired_p)
        }

        # Save paired tile data
        paired_tiles.to_parquet(output_dir / 'paired_tile_analysis.parquet')
    else:
        print("  Not enough paired tiles for analysis")

    # Optional: Analyze by burn severity if dNBR provided
    if args.dnbr_raster and Path(args.dnbr_raster).exists():
        print("\n" + "=" * 80)
        print("SEVERITY-STRATIFIED ANALYSIS")
        print("=" * 80)

        # Extract dNBR for post-fire shots
        print("Extracting dNBR values...")
        post_fire_df = post_fire_df.copy()
        post_fire_df['dnbr'] = extract_dnbr_values(post_fire_df, args.dnbr_raster)
        post_fire_df['severity'] = classify_burn_severity(post_fire_df['dnbr'])

        severity_stats = analyze_by_severity(post_fire_df)
        change_stats['severity_analysis'] = severity_stats

        print("\nPost-fire AGBD by Burn Severity:")
        for severity, sev_stats in severity_stats.items():
            print(f"  {severity.capitalize()}: {sev_stats['mean_agbd']:.1f} ± {sev_stats['std_agbd']:.1f} Mg/ha "
                  f"(n={sev_stats['n_shots']})")

    # Compute control area statistics (outside fire but in bbox)
    outside_fire = filter_shots_by_geometry(gedi_df, fire_boundary.geometry, inside=False)
    if len(outside_fire) > 100:
        print("\n" + "=" * 80)
        print("CONTROL AREA COMPARISON (Outside Fire Boundary)")
        print("=" * 80)

        control_pre = outside_fire[outside_fire['year'].isin(args.pre_years)]
        control_post = outside_fire[outside_fire['year'].isin(args.post_years)]

        control_stats = compute_biomass_change_stats(control_pre, control_post)
        change_stats['control_area'] = control_stats

        print(f"\nControl Pre-fire AGBD: {control_stats['pre_fire']['mean_agbd']:.1f} Mg/ha (n={control_stats['pre_fire']['n_shots']})")
        print(f"Control Post-fire AGBD: {control_stats['post_fire']['mean_agbd']:.1f} Mg/ha (n={control_stats['post_fire']['n_shots']})")
        print(f"Control Change: {control_stats['change']['absolute_change']:.1f} Mg/ha ({control_stats['change']['percent_change']:.1f}%)")

        # Difference-in-differences
        fire_change = change_stats['change']['absolute_change']
        control_change = control_stats['change']['absolute_change']
        did_effect = fire_change - control_change
        print(f"\nDifference-in-Differences Effect: {did_effect:.1f} Mg/ha")
        change_stats['difference_in_differences'] = {
            'fire_change': fire_change,
            'control_change': control_change,
            'did_effect': did_effect
        }

    # Save results
    results = {
        'fire_boundary': str(args.shapefile),
        'fire_year': args.fire_year,
        'fire_area_km2': fire_area_km2,
        'pre_years': args.pre_years,
        'post_years': args.post_years,
        'bbox': list(bbox),
        'total_gedi_shots': len(gedi_df),
        'shots_inside_fire': len(inside_fire),
        'biomass_change': change_stats
    }

    with open(output_dir / 'fire_impact_analysis.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)

    # Save shot-level data
    inside_fire.to_parquet(output_dir / 'gedi_shots_inside_fire.parquet')

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(pre_fire_df, post_fire_df, fire_boundary, output_dir, args.fire_year, fire_year_df)

    print(f"\nResults saved to {output_dir}")
    print("\nSummary:")
    print(f"  Fire area: {fire_area_km2:.1f} km²")
    print(f"  Pre-fire mean AGBD: {change_stats['pre_fire']['mean_agbd']:.1f} Mg/ha")
    print(f"  Post-fire mean AGBD: {change_stats['post_fire']['mean_agbd']:.1f} Mg/ha")
    print(f"  Biomass loss: {abs(change_stats['change']['absolute_change']):.1f} Mg/ha ({abs(change_stats['change']['percent_change']):.1f}%)")


if __name__ == '__main__':
    main()
