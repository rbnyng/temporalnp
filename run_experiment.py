"""
Run spatiotemporal fire detection experiment with multiple seeds.

This script runs the full experiment:
1. Trains on years surrounding a fire event (e.g., 2019-2020, 2022-2023)
2. Tests on the fire year (e.g., 2021)
3. Runs multiple seeds for statistical robustness

Usage:
    python run_experiment.py \
        --region_bbox -122.5 40.5 -121.5 41.5 \
        --train_years 2019 2020 2022 2023 \
        --test_year 2021 \
        --n_seeds 10 \
        --output_dir ./results/mcfarland
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

from utils.disturbance import (
    aggregate_stratified_r2,
    print_aggregated_stratified_r2,
    compute_pooled_stratified_r2
)

# Get project root directory for PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description='Run Spatiotemporal Experiment')

    # Region and temporal arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--fire_shapefile', type=str, default=None,
                        help='Optional: Path to fire boundary shapefile (.shp) to filter GEDI shots')
    parser.add_argument('--train_years', type=int, nargs='+', required=True,
                        help='Years to use for training')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Year to hold out for testing')
    parser.add_argument('--test_months', type=int, nargs='+', default=None,
                        help='Optional: Filter test year to specific months (e.g., 8 9 10 11 12 for Aug-Dec)')

    # Experiment arguments
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of random seeds to run')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting seed value')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for experiment results')

    # Model arguments (passed through to train_spatiotemporal.py)
    parser.add_argument('--architecture_mode', type=str, default='anp',
                        choices=['deterministic', 'latent', 'anp', 'cnp'],
                        help='Architecture mode')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_context_shots', type=int, default=1024,
                        help='Maximum context shots per tile (runtime subsampling)')
    parser.add_argument('--max_target_shots', type=int, default=1024,
                        help='Maximum target shots per tile (runtime subsampling)')
    parser.add_argument('--no_temporal_encoding', action='store_true',
                        help='Disable temporal encoding (spatial-only baseline)')

    # Infrastructure
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='GEDI cache directory')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Embeddings directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def run_single_seed(seed: int, args, seed_output_dir: Path) -> dict:
    """Run training for a single seed."""
    cmd = [
        sys.executable, 'train_spatiotemporal.py',
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--train_years', *[str(y) for y in args.train_years],
        '--test_year', str(args.test_year),
        '--output_dir', str(seed_output_dir),
        '--seed', str(seed),
        '--architecture_mode', args.architecture_mode,
        '--hidden_dim', str(args.hidden_dim),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--max_context_shots', str(args.max_context_shots),
        '--max_target_shots', str(args.max_target_shots),
        '--cache_dir', args.cache_dir,
        '--embeddings_dir', args.embeddings_dir,
        '--device', args.device,
    ]

    # Add optional fire shapefile filter
    if args.fire_shapefile:
        cmd.extend(['--fire_shapefile', args.fire_shapefile])

    # Add optional test months filter
    if args.test_months:
        cmd.extend(['--test_months', *[str(m) for m in args.test_months]])

    # Add spatial-only baseline flag
    if args.no_temporal_encoding:
        cmd.append('--no_temporal_encoding')

    print(f"\n{'='*80}")
    print(f"Running seed {seed}")
    print(f"Output: {seed_output_dir}")
    print(f"{'='*80}")

    # Set PYTHONPATH to project root so imports work
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(cmd, capture_output=False, env=env, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"Warning: Seed {seed} failed with return code {result.returncode}")
        return {'seed': seed, 'status': 'failed'}

    # Load results
    results_path = seed_output_dir / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            seed_results = json.load(f)
        seed_results['seed'] = seed
        seed_results['status'] = 'success'
        return seed_results
    else:
        return {'seed': seed, 'status': 'no_results'}


def aggregate_results(all_results: list) -> dict:
    """Aggregate results across seeds."""
    successful = [r for r in all_results if r.get('status') == 'success']

    if not successful:
        return {
            'error': 'No successful runs',
            'n_seeds': len(all_results),
            'n_successful': 0
        }

    # Extract metrics (matching baseline format)
    log_r2 = [r['test_metrics'].get('log_r2', 0) for r in successful if r.get('test_metrics')]
    log_rmse = [r['test_metrics'].get('log_rmse', 0) for r in successful if r.get('test_metrics')]
    linear_r2 = [r['test_metrics'].get('linear_r2', 0) for r in successful if r.get('test_metrics')]
    linear_rmse = [r['test_metrics'].get('linear_rmse', 0) for r in successful if r.get('test_metrics')]
    linear_mae = [r['test_metrics'].get('linear_mae', 0) for r in successful if r.get('test_metrics')]
    train_times = [r.get('train_time', 0) for r in successful]

    # Extract calibration metrics
    z_mean = [r['test_metrics'].get('z_mean') for r in successful if r.get('test_metrics') and r['test_metrics'].get('z_mean') is not None]
    z_std = [r['test_metrics'].get('z_std') for r in successful if r.get('test_metrics') and r['test_metrics'].get('z_std') is not None]
    coverage_1sigma = [r['test_metrics'].get('coverage_1sigma') for r in successful if r.get('test_metrics') and r['test_metrics'].get('coverage_1sigma') is not None]
    coverage_2sigma = [r['test_metrics'].get('coverage_2sigma') for r in successful if r.get('test_metrics') and r['test_metrics'].get('coverage_2sigma') is not None]

    # Extract disturbance metrics
    mean_disturbance = [r['disturbance']['summary']['mean_disturbance']
                        for r in successful if r.get('disturbance') and r['disturbance']['summary'].get('mean_disturbance') is not None]
    pct_major_loss = [r['disturbance']['summary']['pct_tiles_major_loss']
                      for r in successful if r.get('disturbance') and r['disturbance']['summary'].get('pct_tiles_major_loss') is not None]
    error_disturbance_corr = [r['disturbance']['correlation']['pearson_r']
                               for r in successful if r.get('disturbance') and r['disturbance'].get('correlation') and r['disturbance']['correlation'].get('pearson_r') is not None]

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
                'values': z_mean
            },
            'z_std': {
                'mean': float(np.mean(z_std)) if z_std else None,
                'std': float(np.std(z_std)) if z_std else None,
                'values': z_std
            },
            'coverage_1sigma': {
                'mean': float(np.mean(coverage_1sigma)) if coverage_1sigma else None,
                'std': float(np.std(coverage_1sigma)) if coverage_1sigma else None,
                'values': coverage_1sigma
            },
            'coverage_2sigma': {
                'mean': float(np.mean(coverage_2sigma)) if coverage_2sigma else None,
                'std': float(np.std(coverage_2sigma)) if coverage_2sigma else None,
                'values': coverage_2sigma
            }
        },
        'disturbance': {
            'mean_disturbance': {
                'mean': float(np.mean(mean_disturbance)) if mean_disturbance else None,
                'std': float(np.std(mean_disturbance)) if mean_disturbance else None,
                'values': mean_disturbance
            },
            'pct_tiles_major_loss': {
                'mean': float(np.mean(pct_major_loss)) if pct_major_loss else None,
                'std': float(np.std(pct_major_loss)) if pct_major_loss else None,
                'values': pct_major_loss
            },
            'error_disturbance_correlation': {
                'mean': float(np.mean(error_disturbance_corr)) if error_disturbance_corr else None,
                'std': float(np.std(error_disturbance_corr)) if error_disturbance_corr else None,
                'values': error_disturbance_corr
            }
        },
        'stratified_r2': stratified_r2_agg
    }

    return aggregated


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    experiment_config = {
        'region_bbox': args.region_bbox,
        'fire_shapefile': args.fire_shapefile,
        'train_years': args.train_years,
        'test_year': args.test_year,
        'test_months': args.test_months,
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'architecture_mode': args.architecture_mode,
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'max_context_shots': args.max_context_shots,
        'max_target_shots': args.max_target_shots,
        'started_at': datetime.now().isoformat()
    }

    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)

    print("=" * 80)
    print("Spatiotemporal Fire Detection Experiment")
    print("=" * 80)
    print(f"Region: {args.region_bbox}")
    print(f"Train years: {args.train_years}")
    print(f"Test year: {args.test_year}")
    print(f"Seeds: {args.n_seeds} (starting from {args.start_seed})")
    print(f"Output: {output_dir}")
    print()

    # Run all seeds
    all_results = []
    seeds = [args.start_seed + i for i in range(args.n_seeds)]

    for seed in seeds:
        seed_output_dir = output_dir / f'seed_{seed}'
        result = run_single_seed(seed, args, seed_output_dir)
        all_results.append(result)

        # Save intermediate results
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Aggregate results
    aggregated = aggregate_results(all_results)
    aggregated['experiment_config'] = experiment_config
    aggregated['completed_at'] = datetime.now().isoformat()

    # Compute pooled stratified R² (more robust than averaging across seeds)
    try:
        import pickle
        import pandas as pd
        # Load GEDI data from first successful seed
        gedi_df = None
        for seed_dir in sorted(output_dir.glob('seed_*')):
            pkl_file = seed_dir / 'processed_data.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    gedi_df = pickle.load(f)
                # Ensure year column exists
                if 'year' not in gedi_df.columns:
                    gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
                break

        if gedi_df is not None:
            # Determine pre/post years from train_years relative to test_year
            pre_years = [y for y in args.train_years if y < args.test_year]
            post_years = [y for y in args.train_years if y > args.test_year]

            pooled_stratified = compute_pooled_stratified_r2(
                output_dir, gedi_df, pre_years, post_years
            )
            aggregated['pooled_stratified_r2'] = pooled_stratified
            print("\nComputed pooled stratified R² across all seeds")
    except Exception as e:
        print(f"\nWarning: Could not compute pooled stratified R²: {e}")

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {aggregated['n_successful']}/{aggregated['n_seeds']}")

    if aggregated.get('log_r2', {}).get('mean') is not None:
        print(f"\nLog-space metrics:")
        print(f"  R²:   {aggregated['log_r2']['mean']:.4f} ± {aggregated['log_r2']['std']:.4f}  (range: [{aggregated['log_r2']['min']:.4f}, {aggregated['log_r2']['max']:.4f}])")
        print(f"  RMSE: {aggregated['log_rmse']['mean']:.4f} ± {aggregated['log_rmse']['std']:.4f}")

    if aggregated.get('linear_r2', {}).get('mean') is not None:
        print(f"\nLinear-space metrics:")
        print(f"  R²:   {aggregated['linear_r2']['mean']:.4f} ± {aggregated['linear_r2']['std']:.4f}")
        print(f"  RMSE: {aggregated['linear_rmse']['mean']:.2f} ± {aggregated['linear_rmse']['std']:.2f} Mg/ha")
        print(f"  MAE:  {aggregated['linear_mae']['mean']:.2f} ± {aggregated['linear_mae']['std']:.2f} Mg/ha")

    if aggregated.get('train_time', {}).get('total') is not None:
        print(f"\nTotal training time: {aggregated['train_time']['total']/3600:.2f} hours")

    if aggregated.get('calibration', {}).get('coverage_1sigma', {}).get('mean') is not None:
        print(f"\nUncertainty Calibration:")
        print(f"  Z-score mean: {aggregated['calibration']['z_mean']['mean']:.3f} ± {aggregated['calibration']['z_mean']['std']:.3f} (ideal: 0.0)")
        print(f"  Z-score std:  {aggregated['calibration']['z_std']['mean']:.3f} ± {aggregated['calibration']['z_std']['std']:.3f} (ideal: 1.0)")
        print(f"  Coverage 1σ:  {aggregated['calibration']['coverage_1sigma']['mean']:.1f}% ± {aggregated['calibration']['coverage_1sigma']['std']:.1f}% (ideal: 68.3%)")
        print(f"  Coverage 2σ:  {aggregated['calibration']['coverage_2sigma']['mean']:.1f}% ± {aggregated['calibration']['coverage_2sigma']['std']:.1f}% (ideal: 95.4%)")

    if aggregated.get('disturbance', {}).get('mean_disturbance', {}).get('mean') is not None:
        print(f"\nDisturbance Analysis:")
        print(f"  Mean disturbance: {aggregated['disturbance']['mean_disturbance']['mean']:.1%} ± {aggregated['disturbance']['mean_disturbance']['std']:.1%}")
        print(f"  Tiles with major loss (>30%): {aggregated['disturbance']['pct_tiles_major_loss']['mean']:.1f}% ± {aggregated['disturbance']['pct_tiles_major_loss']['std']:.1f}%")
        if aggregated['disturbance'].get('error_disturbance_correlation', {}).get('mean') is not None:
            print(f"  Error-disturbance correlation: r={aggregated['disturbance']['error_disturbance_correlation']['mean']:.3f} ± {aggregated['disturbance']['error_disturbance_correlation']['std']:.3f}")

    # Print stratified R² using shared utility (per-seed averaging)
    if aggregated.get('stratified_r2'):
        print("\n(Per-seed averaged - see pooled_stratified_r2 for more robust analysis)")
        print_aggregated_stratified_r2(aggregated['stratified_r2'])

    # Print pooled stratified R² (more robust)
    if aggregated.get('pooled_stratified_r2') and not aggregated['pooled_stratified_r2'].get('error'):
        pooled = aggregated['pooled_stratified_r2']
        thresholds = pooled.get('thresholds', {'stable_max': 0.1, 'disturbed_min': 0.3})
        stable_max = int(thresholds['stable_max'] * 100)
        disturbed_min = int(thresholds['disturbed_min'] * 100)

        print(f"\nPooled Stratified R² (all seeds combined, {pooled['total_tiles']} tiles, {pooled['total_shots']} shots):")
        for stratum, label in [('stable', f'Stable (<{stable_max}% change)'),
                               ('moderate', f'Moderate ({stable_max}-{disturbed_min}%)'),
                               ('disturbed', f'Disturbed (>{disturbed_min}% loss)')]:
            s = pooled.get(stratum, {})
            if s.get('r2') is not None:
                print(f"  {label:24s} R²={s['r2']:.4f}, RMSE={s['rmse']:.2f} Mg/ha ({s['n_shots']} shots, {s['n_tiles']} tiles)")
            else:
                print(f"  {label:24s} Not enough data")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
