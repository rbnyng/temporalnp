"""
Benchmark the effect of embedding source on uncertainty quantification.

Runs the same experiment pipeline with different embedding sources
(GeoTessera vs AlphaEarth) and compares UQ metrics:
- Calibration (z-score mean/std, coverage at 1σ/2σ/3σ)
- Accuracy (R², RMSE, MAE in log and linear space)
- Disturbance detection (stratified R² by disturbance level)

Usage:
    python run_embedding_benchmark.py \
        --region_bbox -122.5 40.5 -121.5 41.5 \
        --train_years 2019 2020 2022 2023 \
        --test_year 2021 \
        --n_seeds 5 \
        --output_dir ./results/embedding_benchmark

    # AlphaEarth requires Earth Engine auth:
    #   pip install earthengine-api
    #   earthengine authenticate
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


PROJECT_ROOT = Path(__file__).parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark embedding sources for UQ'
    )

    # Region and temporal
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--fire_shapefile', type=str, default=None,
                        help='Optional fire boundary shapefile')
    parser.add_argument('--train_years', type=int, nargs='+', required=True,
                        help='Years for training')
    parser.add_argument('--test_year', type=int, required=True,
                        help='Year to hold out for testing')
    parser.add_argument('--test_months', type=int, nargs='+', default=None,
                        help='Optional: filter test year to specific months')

    # Experiment
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds per condition')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting seed value')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for benchmark results')

    # Embedding sources to benchmark
    parser.add_argument('--sources', type=str, nargs='+',
                        default=['geotessera', 'alphaearth'],
                        choices=['geotessera', 'alphaearth'],
                        help='Embedding sources to benchmark')
    parser.add_argument('--include_no_embedding', action='store_true',
                        help='Include a no-embedding baseline (coords + AGBD only)')

    # Model configuration (shared across conditions)
    parser.add_argument('--architecture_mode', type=str, default='anp',
                        choices=['deterministic', 'latent', 'anp', 'cnp'],
                        help='Architecture mode')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_context_shots', type=int, default=10000,
                        help='Maximum context shots per tile')
    parser.add_argument('--max_target_shots', type=int, default=1024,
                        help='Maximum target shots per tile')
    parser.add_argument('--temporal_context', action='store_true',
                        help='Use train years as context for test prediction')
    parser.add_argument('--cross_year_training', action='store_true',
                        help='Train with cross-year context/target splits')
    parser.add_argument('--no_temporal_encoding', action='store_true',
                        help='Disable temporal encoding')

    # Also run RF baselines
    parser.add_argument('--include_rf', action='store_true',
                        help='Also run Quantile RF baselines for each embedding source')
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                        help='Number of trees for RF baseline')

    # Infrastructure
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='GEDI cache directory')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Embeddings directory')
    parser.add_argument('--ee_project', type=str, default=None,
                        help='Google Cloud project for Earth Engine')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def run_np_experiment(args, embedding_source: str, condition_dir: Path) -> dict:
    """Run Neural Process experiment for a given embedding source."""
    cmd = [
        sys.executable, 'run_experiment.py',
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--train_years', *[str(y) for y in args.train_years],
        '--test_year', str(args.test_year),
        '--n_seeds', str(args.n_seeds),
        '--start_seed', str(args.start_seed),
        '--output_dir', str(condition_dir),
        '--architecture_mode', args.architecture_mode,
        '--hidden_dim', str(args.hidden_dim),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--max_context_shots', str(args.max_context_shots),
        '--max_target_shots', str(args.max_target_shots),
        '--embedding_source', embedding_source,
        '--cache_dir', args.cache_dir,
        '--embeddings_dir', args.embeddings_dir,
        '--device', args.device,
    ]

    if args.fire_shapefile:
        cmd.extend(['--fire_shapefile', args.fire_shapefile])
    if args.test_months:
        cmd.extend(['--test_months', *[str(m) for m in args.test_months]])
    if args.no_temporal_encoding:
        cmd.append('--no_temporal_encoding')
    if args.temporal_context:
        cmd.append('--temporal_context')
    if args.cross_year_training:
        cmd.append('--cross_year_training')
    if args.ee_project:
        cmd.extend(['--ee_project', args.ee_project])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    print(f"\n{'='*80}")
    print(f"Running NP experiment: embedding_source={embedding_source}")
    print(f"Output: {condition_dir}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False, env=env, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"Warning: NP experiment with {embedding_source} failed")
        return {'status': 'failed', 'embedding_source': embedding_source}

    results_path = condition_dir / 'aggregated_results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {'status': 'no_results', 'embedding_source': embedding_source}


def run_rf_experiment(args, embedding_source: str, condition_dir: Path) -> dict:
    """Run Quantile RF experiment for a given embedding source."""
    cmd = [
        sys.executable, 'run_rf_experiment.py',
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--train_years', *[str(y) for y in args.train_years],
        '--test_year', str(args.test_year),
        '--model', 'quantile_rf',
        '--n_estimators', str(args.rf_n_estimators),
        '--n_seeds', str(args.n_seeds),
        '--start_seed', str(args.start_seed),
        '--output_dir', str(condition_dir),
        '--embedding_source', embedding_source,
        '--cache_dir', args.cache_dir,
        '--embeddings_dir', args.embeddings_dir,
    ]

    if args.fire_shapefile:
        cmd.extend(['--fire_shapefile', args.fire_shapefile])
    if args.test_months:
        cmd.extend(['--test_months', *[str(m) for m in args.test_months]])
    if args.ee_project:
        cmd.extend(['--ee_project', args.ee_project])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    print(f"\n{'='*80}")
    print(f"Running QRF experiment: embedding_source={embedding_source}")
    print(f"Output: {condition_dir}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False, env=env, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"Warning: RF experiment with {embedding_source} failed")
        return {'status': 'failed', 'embedding_source': embedding_source}

    results_path = condition_dir / 'aggregated_results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {'status': 'no_results', 'embedding_source': embedding_source}


def extract_comparison_metrics(results: dict) -> dict:
    """Extract key metrics for comparison from aggregated results."""
    metrics = {}

    # Accuracy
    for key in ['log_r2', 'log_rmse', 'linear_r2', 'linear_rmse', 'linear_mae']:
        if results.get(key, {}).get('mean') is not None:
            metrics[key] = {
                'mean': results[key]['mean'],
                'std': results[key]['std'],
            }

    # Calibration
    cal = results.get('calibration', {})
    for key in ['z_mean', 'z_std', 'coverage_1sigma', 'coverage_2sigma']:
        if cal.get(key, {}).get('mean') is not None:
            metrics[f'cal_{key}'] = {
                'mean': cal[key]['mean'],
                'std': cal[key].get('std', 0),
            }

    # Disturbance
    dist = results.get('disturbance', {})
    if dist.get('error_disturbance_correlation', {}).get('mean') is not None:
        metrics['error_dist_corr'] = {
            'mean': dist['error_disturbance_correlation']['mean'],
            'std': dist['error_disturbance_correlation'].get('std', 0),
        }

    # Stratified R²
    pooled = results.get('pooled_stratified_r2', {})
    for stratum in ['stable', 'moderate', 'disturbed']:
        if pooled.get(stratum, {}).get('r2') is not None:
            metrics[f'stratified_r2_{stratum}'] = pooled[stratum]['r2']

    return metrics


def print_comparison_table(all_conditions: dict):
    """Print a formatted comparison table of all conditions."""

    print("\n" + "=" * 100)
    print("EMBEDDING BENCHMARK COMPARISON")
    print("=" * 100)

    # Collect all condition names and metrics
    conditions = list(all_conditions.keys())
    if not conditions:
        print("No successful conditions to compare.")
        return

    # Header
    header = f"{'Metric':<25}"
    for cond in conditions:
        header += f"  {cond:>20}"
    print(header)
    print("-" * (25 + 22 * len(conditions)))

    # Define metrics to display with ideal values
    display_metrics = [
        ('linear_r2', 'Linear R²', '↑', None),
        ('linear_rmse', 'Linear RMSE (Mg/ha)', '↓', None),
        ('linear_mae', 'Linear MAE (Mg/ha)', '↓', None),
        ('log_r2', 'Log R²', '↑', None),
        ('log_rmse', 'Log RMSE', '↓', None),
        ('cal_z_mean', 'Z-score mean', '→0', 0.0),
        ('cal_z_std', 'Z-score std', '→1', 1.0),
        ('cal_coverage_1sigma', 'Coverage 1σ (%)', '→68.3', 68.3),
        ('cal_coverage_2sigma', 'Coverage 2σ (%)', '→95.4', 95.4),
        ('error_dist_corr', 'Error-Dist. corr.', '↓', None),
        ('stratified_r2_stable', 'R² (stable)', '↑', None),
        ('stratified_r2_moderate', 'R² (moderate)', '↑', None),
        ('stratified_r2_disturbed', 'R² (disturbed)', '↑', None),
    ]

    for metric_key, display_name, direction, ideal in display_metrics:
        row = f"{display_name} {direction:<3}  "

        has_any = False
        for cond in conditions:
            metrics = all_conditions[cond]
            val = metrics.get(metric_key)

            if val is None:
                row += f"  {'---':>20}"
            elif isinstance(val, dict):
                mean = val.get('mean')
                std = val.get('std', 0)
                if mean is not None:
                    has_any = True
                    row += f"  {mean:>10.4f} ± {std:<7.4f}"
                else:
                    row += f"  {'---':>20}"
            else:
                has_any = True
                row += f"  {val:>20.4f}"

        if has_any:
            print(row)

    print("=" * 100)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark config
    benchmark_config = {
        'region_bbox': args.region_bbox,
        'train_years': args.train_years,
        'test_year': args.test_year,
        'sources': args.sources,
        'include_rf': args.include_rf,
        'n_seeds': args.n_seeds,
        'architecture_mode': args.architecture_mode,
        'started_at': datetime.now().isoformat(),
    }
    with open(output_dir / 'benchmark_config.json', 'w') as f:
        json.dump(benchmark_config, f, indent=2)

    print("=" * 80)
    print("Embedding Source UQ Benchmark")
    print("=" * 80)
    print(f"Region: {args.region_bbox}")
    print(f"Train years: {args.train_years}")
    print(f"Test year: {args.test_year}")
    print(f"Embedding sources: {args.sources}")
    print(f"Seeds per condition: {args.n_seeds}")
    print(f"Architecture: {args.architecture_mode}")
    if args.include_rf:
        print(f"RF baselines: quantile_rf (n_estimators={args.rf_n_estimators})")
    print()

    all_results = {}
    all_metrics = {}

    # Run NP experiments for each embedding source
    for source in args.sources:
        condition_name = f'np_{source}'
        condition_dir = output_dir / condition_name

        results = run_np_experiment(args, source, condition_dir)
        all_results[condition_name] = results

        if results.get('n_successful', 0) > 0:
            all_metrics[condition_name] = extract_comparison_metrics(results)

    # Run RF baselines if requested
    if args.include_rf:
        for source in args.sources:
            condition_name = f'qrf_{source}'
            condition_dir = output_dir / condition_name

            results = run_rf_experiment(args, source, condition_dir)
            all_results[condition_name] = results

            if results.get('n_successful', 0) > 0:
                all_metrics[condition_name] = extract_comparison_metrics(results)

    # Save all results
    benchmark_results = {
        'config': benchmark_config,
        'completed_at': datetime.now().isoformat(),
        'conditions': {k: v for k, v in all_results.items()},
        'comparison_metrics': all_metrics,
    }
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)

    # Print comparison
    print_comparison_table(all_metrics)

    print(f"\nBenchmark results saved to: {output_dir}")


if __name__ == '__main__':
    main()
