#!/usr/bin/env python
"""
Collect and compare results across benchmark output directories.

Usage:
    python collect_benchmark_results.py ./results/*_bench
    python collect_benchmark_results.py ./results/guaviare_2019_bench ./results/guaviare_2022_bench
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_benchmark(bench_dir: Path) -> dict:
    """Load benchmark results from a directory."""
    results_path = bench_dir / 'benchmark_results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    # Fall back to aggregated_results.json (single-condition runs)
    agg_path = bench_dir / 'aggregated_results.json'
    if agg_path.exists():
        with open(agg_path) as f:
            return {'conditions': {bench_dir.name: json.load(f)}}

    return None


def fmt(val, key):
    """Format a metric value."""
    if val is None:
        return '---'
    if isinstance(val, dict):
        mean = val.get('mean')
        std = val.get('std', 0)
        if mean is None:
            return '---'
        return f'{mean:.4f} ± {std:.4f}'
    return f'{val:.4f}'


def main():
    parser = argparse.ArgumentParser(description='Collect benchmark results')
    parser.add_argument('dirs', nargs='+', help='Benchmark output directories')
    parser.add_argument('--csv', type=str, default=None,
                        help='Write results to CSV file')
    args = parser.parse_args()

    metrics_keys = [
        ('log_r2', 'Log R²'),
        ('log_rmse', 'Log RMSE'),
        ('linear_r2', 'Linear R²'),
        ('linear_rmse', 'Linear RMSE'),
        ('linear_mae', 'Linear MAE'),
        ('cal_z_mean', 'Z-score mean'),
        ('cal_z_std', 'Z-score std'),
        ('cal_coverage_1sigma', 'Coverage 1σ'),
        ('cal_coverage_2sigma', 'Coverage 2σ'),
    ]

    rows = []

    for dir_path in args.dirs:
        bench_dir = Path(dir_path)
        if not bench_dir.is_dir():
            print(f"Skipping {dir_path}: not a directory", file=sys.stderr)
            continue

        data = load_benchmark(bench_dir)
        if data is None:
            print(f"Skipping {bench_dir.name}: no results found", file=sys.stderr)
            continue

        comparison = data.get('comparison_metrics', {})
        config = data.get('config', {})

        # If no comparison_metrics, build from conditions directly
        if not comparison:
            for cond_name, cond_data in data.get('conditions', {}).items():
                row = {
                    'benchmark': bench_dir.name,
                    'condition': cond_name,
                    'n_successful': cond_data.get('n_successful', '?'),
                    'region': config.get('region_bbox', '?'),
                }
                for key, _ in metrics_keys:
                    # Try to extract from aggregated results
                    if key.startswith('cal_'):
                        cal_key = key[4:]
                        val = cond_data.get('calibration', {}).get(cal_key, {})
                    else:
                        val = cond_data.get(key, {})
                    row[key] = val.get('mean') if isinstance(val, dict) else val
                    row[f'{key}_std'] = val.get('std', 0) if isinstance(val, dict) else 0
                rows.append(row)
        else:
            for cond_name, cond_metrics in comparison.items():
                cond_data = data.get('conditions', {}).get(cond_name, {})
                row = {
                    'benchmark': bench_dir.name,
                    'condition': cond_name,
                    'n_successful': cond_data.get('n_successful', '?'),
                    'region': config.get('region_bbox', '?'),
                }
                for key, _ in metrics_keys:
                    val = cond_metrics.get(key)
                    row[key] = val.get('mean') if isinstance(val, dict) else val
                    row[f'{key}_std'] = val.get('std', 0) if isinstance(val, dict) else 0
                rows.append(row)

    if not rows:
        print("No results found.")
        return

    # Print table
    print(f"\n{'Benchmark':<30} {'Condition':<25} {'N':>3}", end='')
    for _, label in metrics_keys:
        print(f"  {label:>20}", end='')
    print()
    print("-" * (60 + 22 * len(metrics_keys)))

    for row in rows:
        print(f"{row['benchmark']:<30} {row['condition']:<25} {row['n_successful']:>3}", end='')
        for key, _ in metrics_keys:
            mean = row.get(key)
            std = row.get(f'{key}_std', 0)
            if mean is None:
                print(f"  {'---':>20}", end='')
            else:
                print(f"  {mean:>10.4f} ± {std:<7.4f}", end='')
        print()

    print()

    # Write CSV if requested
    if args.csv:
        import csv
        fieldnames = ['benchmark', 'condition', 'n_successful', 'region']
        for key, label in metrics_keys:
            fieldnames.extend([key, f'{key}_std'])

        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {args.csv}")


if __name__ == '__main__':
    main()
