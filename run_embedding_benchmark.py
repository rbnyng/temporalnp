"""
Benchmark the effect of embedding source on uncertainty quantification.

Runs the static (single-year) training pipeline with different embedding
sources (GeoTessera vs AlphaEarth) and compares UQ metrics:
- Calibration (z-score mean/std, coverage at 1σ/2σ/3σ)
- Accuracy (R², RMSE, MAE in log and linear space)

Uses BufferedSpatialSplitter + GEDINeuralProcessDataset for consistent
spatial-only evaluation.

Usage:
    python run_embedding_benchmark.py \
        --region_bbox -73 2 -72 3 \
        --year 2022 \
        --n_seeds 5 \
        --output_dir ./results/guaviare_2022_bench

    # Multiple years (runs each independently):
    python run_embedding_benchmark.py \
        --region_bbox -73 2 -72 3 \
        --year 2019 2020 2022 \
        --n_seeds 5 \
        --output_dir ./results/guaviare_multi_bench

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
        description='Benchmark embedding sources for UQ (static spatial evaluation)'
    )

    # Region and temporal
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--year', type=int, nargs='+', required=True,
                        help='Year(s) to evaluate (each year is run independently)')

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


def run_single_seed(args, seed: int, year: int, embedding_source: str,
                    seed_dir: Path) -> dict:
    """Run train_static.py for a single seed."""
    cmd = [
        sys.executable, 'train_static.py',
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--start_time', f'{year}-01-01',
        '--end_time', f'{year}-12-31',
        '--embedding_year', str(year),
        '--embedding_source', embedding_source,
        '--output_dir', str(seed_dir),
        '--seed', str(seed),
        '--architecture_mode', args.architecture_mode,
        '--hidden_dim', str(args.hidden_dim),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--cache_dir', args.cache_dir,
        '--embeddings_dir', args.embeddings_dir,
        '--device', args.device,
    ]

    if args.ee_project:
        cmd.extend(['--ee_project', args.ee_project])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    print(f"\n  Seed {seed} | {embedding_source} | {year}")
    result = subprocess.run(cmd, capture_output=False, env=env, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"  Warning: Seed {seed} failed")
        return {'seed': seed, 'status': 'failed'}

    results_path = seed_dir / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            seed_results = json.load(f)
        seed_results['seed'] = seed
        seed_results['status'] = 'success'
        return seed_results

    return {'seed': seed, 'status': 'no_results'}


def run_rf_seed(args, seed: int, year: int, embedding_source: str,
                seed_dir: Path) -> dict:
    """Run Quantile RF experiment for a single seed."""
    cmd = [
        sys.executable, 'run_rf_experiment.py',
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--train_years', str(year),
        '--test_year', str(year),
        '--model', 'quantile_rf',
        '--n_estimators', str(args.rf_n_estimators),
        '--n_seeds', '1',
        '--start_seed', str(seed),
        '--output_dir', str(seed_dir),
        '--embedding_source', embedding_source,
        '--cache_dir', args.cache_dir,
        '--embeddings_dir', args.embeddings_dir,
    ]

    if args.ee_project:
        cmd.extend(['--ee_project', args.ee_project])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    print(f"\n  QRF Seed {seed} | {embedding_source} | {year}")
    result = subprocess.run(cmd, capture_output=False, env=env, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"  Warning: QRF seed {seed} failed")
        return {'seed': seed, 'status': 'failed'}

    # RF experiment writes aggregated_results.json (even for single seed)
    results_path = seed_dir / 'aggregated_results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    return {'seed': seed, 'status': 'no_results'}


def aggregate_seed_results(all_results: list) -> dict:
    """Aggregate results across seeds."""
    successful = [r for r in all_results if r.get('status') == 'success']

    if not successful:
        return {'n_seeds': len(all_results), 'n_successful': 0}

    def collect(key):
        return [r['test_metrics'][key] for r in successful
                if r.get('test_metrics') and r['test_metrics'].get(key) is not None]

    def stat(values):
        if not values:
            return {'mean': None, 'std': None, 'values': values}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values,
        }

    train_times = [r.get('train_time', 0) for r in successful]

    return {
        'n_seeds': len(all_results),
        'n_successful': len(successful),
        'log_r2': stat(collect('log_r2')),
        'log_rmse': stat(collect('log_rmse')),
        'linear_r2': stat(collect('linear_r2')),
        'linear_rmse': stat(collect('linear_rmse')),
        'linear_mae': stat(collect('linear_mae')),
        'train_time': {
            'mean': float(np.mean(train_times)) if train_times else None,
            'total': float(np.sum(train_times)) if train_times else None,
        },
        'calibration': {
            'z_mean': stat(collect('z_mean')),
            'z_std': stat(collect('z_std')),
            'coverage_1sigma': stat(collect('coverage_1sigma')),
            'coverage_2sigma': stat(collect('coverage_2sigma')),
            'coverage_3sigma': stat(collect('coverage_3sigma')),
        },
    }


def extract_comparison_metrics(results: dict) -> dict:
    """Extract key metrics for comparison from aggregated results."""
    metrics = {}

    for key in ['log_r2', 'log_rmse', 'linear_r2', 'linear_rmse', 'linear_mae']:
        if results.get(key, {}).get('mean') is not None:
            metrics[key] = {
                'mean': results[key]['mean'],
                'std': results[key]['std'],
            }

    cal = results.get('calibration', {})
    for key in ['z_mean', 'z_std', 'coverage_1sigma', 'coverage_2sigma']:
        if cal.get(key, {}).get('mean') is not None:
            metrics[f'cal_{key}'] = {
                'mean': cal[key]['mean'],
                'std': cal[key].get('std', 0),
            }

    return metrics


def print_comparison_table(all_conditions: dict):
    """Print a formatted comparison table of all conditions."""

    print("\n" + "=" * 100)
    print("EMBEDDING BENCHMARK COMPARISON")
    print("=" * 100)

    conditions = list(all_conditions.keys())
    if not conditions:
        print("No successful conditions to compare.")
        return

    header = f"{'Metric':<25}"
    for cond in conditions:
        header += f"  {cond:>20}"
    print(header)
    print("-" * (25 + 22 * len(conditions)))

    display_metrics = [
        ('linear_r2', 'Linear R²', '↑'),
        ('linear_rmse', 'Linear RMSE (Mg/ha)', '↓'),
        ('linear_mae', 'Linear MAE (Mg/ha)', '↓'),
        ('log_r2', 'Log R²', '↑'),
        ('log_rmse', 'Log RMSE', '↓'),
        ('cal_z_mean', 'Z-score mean', '→0'),
        ('cal_z_std', 'Z-score std', '→1'),
        ('cal_coverage_1sigma', 'Coverage 1σ (%)', '→68.3'),
        ('cal_coverage_2sigma', 'Coverage 2σ (%)', '→95.4'),
    ]

    for metric_key, display_name, direction in display_metrics:
        row = f"{display_name} {direction:<5}"

        has_any = False
        for cond in conditions:
            val = all_conditions[cond].get(metric_key)

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

    benchmark_config = {
        'region_bbox': args.region_bbox,
        'years': args.year,
        'sources': args.sources,
        'include_rf': args.include_rf,
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'architecture_mode': args.architecture_mode,
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'started_at': datetime.now().isoformat(),
    }
    with open(output_dir / 'benchmark_config.json', 'w') as f:
        json.dump(benchmark_config, f, indent=2)

    print("=" * 80)
    print("Embedding Source UQ Benchmark (Static)")
    print("=" * 80)
    print(f"Region: {args.region_bbox}")
    print(f"Years: {args.year}")
    print(f"Embedding sources: {args.sources}")
    print(f"Seeds per condition: {args.n_seeds}")
    print(f"Architecture: {args.architecture_mode}")
    if args.include_rf:
        print(f"RF baselines: quantile_rf (n_estimators={args.rf_n_estimators})")
    print()

    seeds = [args.start_seed + i for i in range(args.n_seeds)]
    all_results = {}
    all_metrics = {}

    for year in args.year:
        for source in args.sources:
            condition_name = f'np_{source}_{year}'
            condition_dir = output_dir / condition_name
            condition_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"NP | {source} | {year}")
            print(f"{'='*80}")

            seed_results = []
            for seed in seeds:
                seed_dir = condition_dir / f'seed_{seed}'
                result = run_single_seed(args, seed, year, source, seed_dir)
                seed_results.append(result)

                # Save intermediate
                with open(condition_dir / 'all_results.json', 'w') as f:
                    json.dump(seed_results, f, indent=2)

            aggregated = aggregate_seed_results(seed_results)
            aggregated['year'] = year
            aggregated['embedding_source'] = source
            with open(condition_dir / 'aggregated_results.json', 'w') as f:
                json.dump(aggregated, f, indent=2)

            all_results[condition_name] = aggregated
            if aggregated.get('n_successful', 0) > 0:
                all_metrics[condition_name] = extract_comparison_metrics(aggregated)

        # RF baselines
        if args.include_rf:
            for source in args.sources:
                condition_name = f'qrf_{source}_{year}'
                condition_dir = output_dir / condition_name
                condition_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*80}")
                print(f"QRF | {source} | {year}")
                print(f"{'='*80}")

                seed_results = []
                for seed in seeds:
                    seed_dir = condition_dir / f'seed_{seed}'
                    result = run_rf_seed(args, seed, year, source, seed_dir)
                    seed_results.append(result)

                # RF aggregation: collect from sub-runs
                rf_successful = [r for r in seed_results
                                 if r.get('n_successful', 0) > 0 or r.get('status') == 'success']
                if rf_successful:
                    # Use last successful aggregated result (RF runner handles its own aggregation)
                    aggregated = rf_successful[-1]
                    all_results[condition_name] = aggregated
                    if aggregated.get('n_successful', 0) > 0:
                        all_metrics[condition_name] = extract_comparison_metrics(aggregated)

    # Save all results
    benchmark_results = {
        'config': benchmark_config,
        'completed_at': datetime.now().isoformat(),
        'conditions': all_results,
        'comparison_metrics': all_metrics,
    }
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)

    print_comparison_table(all_metrics)

    print(f"\nBenchmark results saved to: {output_dir}")


if __name__ == '__main__':
    main()
