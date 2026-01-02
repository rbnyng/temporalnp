"""
Run temporal baseline experiment with multiple seeds.

This script runs baseline experiments for comparison with the spatiotemporal model:
1. Supports multiple baseline modes (interpolation, pre_only, post_only, mean)
2. Uses tile-based context selection (same as main model)
3. Runs multiple seeds for statistical robustness

Usage:
    python run_baseline_experiment.py \
        --region_bbox -122.5 40.5 -121.5 41.5 \
        --pre_years 2019 2020 \
        --post_years 2022 2023 \
        --test_year 2021 \
        --mode interpolation \
        --n_seeds 10 \
        --output_dir ./results/mcfarland_baseline
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

from utils.disturbance import aggregate_stratified_r2, print_aggregated_stratified_r2

# Get project root directory for PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description='Run Temporal Baseline Experiment')

    # Baseline mode
    parser.add_argument('--mode', type=str, default='interpolation',
                        choices=['interpolation', 'pre_only', 'post_only', 'mean'],
                        help='Baseline mode: interpolation (linear blend), pre_only (historical), '
                             'post_only (oracle), mean (simple average)')

    # Region and temporal arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--pre_years', type=int, nargs='+', required=True,
                        help='Years before event (e.g., 2019 2020)')
    parser.add_argument('--post_years', type=int, nargs='+', required=True,
                        help='Years after event (e.g., 2022 2023)')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Event year to test on (e.g., 2021)')

    # Experiment arguments
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of random seeds to run')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting seed value')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for experiment results')

    # Model arguments (passed through to temporal_interpolation.py)
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--max_context_shots', type=int, default=1024,
                        help='Max context shots per tile during training/inference')
    parser.add_argument('--max_target_shots', type=int, default=1024,
                        help='Max target shots per tile during training/inference')

    # Infrastructure
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='GEDI cache directory')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Embeddings directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def run_single_seed(seed: int, args, seed_output_dir: Path) -> dict:
    """Run baseline for a single seed."""
    cmd = [
        sys.executable, 'baselines/temporal_interpolation.py',
        '--mode', args.mode,
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--pre_years', *[str(y) for y in args.pre_years],
        '--post_years', *[str(y) for y in args.post_years],
        '--test_year', str(args.test_year),
        '--output_dir', str(seed_output_dir),
        '--seed', str(seed),
        '--hidden_dim', str(args.hidden_dim),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--max_context_shots', str(args.max_context_shots),
        '--max_target_shots', str(args.max_target_shots),
        '--cache_dir', args.cache_dir,
        '--embeddings_dir', args.embeddings_dir,
        '--device', args.device,
    ]

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

    # Extract metrics
    log_r2 = [r['metrics']['log_r2'] for r in successful if r.get('metrics')]
    log_rmse = [r['metrics']['log_rmse'] for r in successful if r.get('metrics')]
    linear_r2 = [r['metrics']['linear_r2'] for r in successful if r.get('metrics')]
    linear_rmse = [r['metrics']['linear_rmse'] for r in successful if r.get('metrics')]
    linear_mae = [r['metrics']['linear_mae'] for r in successful if r.get('metrics')]

    # Extract calibration metrics
    z_mean = [r['calibration']['z_mean'] for r in successful if r.get('calibration')]
    z_std = [r['calibration']['z_std'] for r in successful if r.get('calibration')]
    coverage_1sigma = [r['calibration']['coverage_1sigma'] for r in successful if r.get('calibration')]
    coverage_2sigma = [r['calibration']['coverage_2sigma'] for r in successful if r.get('calibration')]

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

    mode_names = {
        'interpolation': 'Temporal Linear Interpolation',
        'pre_only': 'Pre-Event Only (Historical)',
        'post_only': 'Post-Event Only (Oracle)',
        'mean': 'Mean of Pre/Post'
    }

    # Save experiment config
    experiment_config = {
        'method': args.mode,
        'method_name': mode_names[args.mode],
        'region_bbox': args.region_bbox,
        'pre_years': args.pre_years,
        'post_years': args.post_years,
        'test_year': args.test_year,
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_context_shots': args.max_context_shots,
        'context_selection': 'tile_based',
        'started_at': datetime.now().isoformat()
    }

    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)

    print("=" * 80)
    print(f"Baseline Experiment: {mode_names[args.mode]}")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Region: {args.region_bbox}")
    print(f"Pre-event years: {args.pre_years}")
    print(f"Post-event years: {args.post_years}")
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

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {aggregated['n_successful']}/{aggregated['n_seeds']}")

    if aggregated.get('log_r2', {}).get('mean') is not None:
        print(f"\nTest R² (log-space):")
        print(f"  Mean: {aggregated['log_r2']['mean']:.4f} ± {aggregated['log_r2']['std']:.4f}")
        print(f"  Range: [{aggregated['log_r2']['min']:.4f}, {aggregated['log_r2']['max']:.4f}]")

    if aggregated.get('linear_rmse', {}).get('mean') is not None:
        print(f"\nTest RMSE (Mg/ha):")
        print(f"  Mean: {aggregated['linear_rmse']['mean']:.2f} ± {aggregated['linear_rmse']['std']:.2f}")

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

    # Print stratified R² using shared utility
    if aggregated.get('stratified_r2'):
        print_aggregated_stratified_r2(aggregated['stratified_r2'])

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
