import json
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Default path
RESULTS_DIR = Path("/maps-priv/maps/ray25/temporalanp/results")

def get_region_from_name(name):
    """Group experiments by region based on folder name."""
    name = name.lower()
    if 'guaviare' in name: return 'Guaviare'
    if 'maine' in name: return 'Maine'
    if 'tolima' in name: return 'Tolima'
    if 'tyrol' in name: return 'Tyrol'
    return 'Other'

def detect_model_details(folder_name, config):
    """Detailed model type detection including ablation variants."""
    name = folder_name.lower()
    
    # Base model type
    if 'xgb' in name: base = 'XGB'
    elif 'qrf' in name or ('rf' in name and 'perf' not in name): base = 'QRF'
    else: base = 'ANP'

    # Check for specific variants
    variant = []
    
    # Check config for temporal flag (baselines)
    if config.get('include_temporal') is True:
        variant.append('Temporal')
    elif config.get('include_temporal') is False:
        variant.append('Spatial-Only')
    
    # ANP variants
    elif base == 'ANP':
        if config.get('no_temporal_encoding'):
            variant.append('Spatial-Only')
        elif 'context' in name:
            variant.append('Temporal Context')
        else:
            variant.append('Standard')
            
    return base, " + ".join(variant) if variant else "Standard"

def safe_get(d, path, default=np.nan):
    """Helper to safely retrieve nested dictionary keys."""
    try:
        keys = path.split('.')
        val = d
        for k in keys:
            val = val[k]
        return val if val is not None else default
    except (KeyError, TypeError, AttributeError):
        return default

def load_metrics(folder_path):
    json_path = folder_path / 'aggregated_results.json'
    
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        config = data.get('experiment_config', {})
        pooled = data.get('pooled_stratified_r2', {})
        dist = data.get('disturbance', {})
        cal = data.get('calibration', {})
        
        # Handle legacy key name for disturbed class in stratified results
        disturbed_stats = pooled.get('disturbed', pooled.get('fire', {}))
        
        base_model, variant = detect_model_details(folder_path.name, config)

        row = {
            # --- Meta ---
            'Region': get_region_from_name(folder_path.name),
            'Model': base_model,
            'Variant': variant,
            'Run Name': folder_path.name,
            'Seeds': data.get('n_successful', 0),
            
            # --- Log Space (Mean ± Std) ---
            'Log R2 (Mean)': safe_get(data, 'log_r2.mean'),
            'Log R2 (Std)': safe_get(data, 'log_r2.std'),
            'Log RMSE (Mean)': safe_get(data, 'log_rmse.mean'),
            'Log RMSE (Std)': safe_get(data, 'log_rmse.std'),
            
            # --- Linear Space (Mean ± Std) ---
            'Lin R2 (Mean)': safe_get(data, 'linear_r2.mean'),
            'Lin R2 (Std)': safe_get(data, 'linear_r2.std'),
            'Lin RMSE (Mean)': safe_get(data, 'linear_rmse.mean'),
            'Lin RMSE (Std)': safe_get(data, 'linear_rmse.std'),
            'Lin MAE (Mean)': safe_get(data, 'linear_mae.mean'),
            'Lin MAE (Std)': safe_get(data, 'linear_mae.std'),

            # --- Uncertainty Calibration (Mean ± Std) ---
            'Cov 1s (Mean)': safe_get(cal, 'coverage_1sigma.mean'),
            'Cov 1s (Std)': safe_get(cal, 'coverage_1sigma.std'),
            'Cov 2s (Mean)': safe_get(cal, 'coverage_2sigma.mean'),
            'Cov 2s (Std)': safe_get(cal, 'coverage_2sigma.std'),
            'Z-Score Mean (Mean)': safe_get(cal, 'z_mean.mean'),
            'Z-Score Mean (Std)': safe_get(cal, 'z_mean.std'),
            'Z-Score Std (Mean)': safe_get(cal, 'z_std.mean'),
            'Z-Score Std (Std)': safe_get(cal, 'z_std.std'),

            # --- Disturbance Stats (Mean ± Std) ---
            'Err-Dist Corr (Mean)': safe_get(dist, 'error_disturbance_correlation.mean'),
            'Err-Dist Corr (Std)': safe_get(dist, 'error_disturbance_correlation.std'),
            '% Major Loss (Mean)': safe_get(dist, 'pct_tiles_major_loss.mean'),
            '% Major Loss (Std)': safe_get(dist, 'pct_tiles_major_loss.std'),

            # --- POOLED Stratified Metrics (Single Values) ---
            # These are computed across all seeds, so they have no Std
            'Pooled Stable R2': safe_get(pooled, 'stable.r2'),
            'Pooled Stable RMSE': safe_get(pooled, 'stable.rmse'),
            'Pooled Mod R2': safe_get(pooled, 'moderate.r2'),
            'Pooled Mod RMSE': safe_get(pooled, 'moderate.rmse'),
            'Pooled Disturbed R2': safe_get(disturbed_stats, 'r2'),
            'Pooled Disturbed RMSE': safe_get(disturbed_stats, 'rmse'),
            
            # --- Timing ---
            'Train Time (s)': safe_get(data, 'train_time.mean')
        }
        return row
    except Exception as e:
        print(f"Error reading {folder_path.name}: {e}")
        return None

def main():
    target_dir = RESULTS_DIR
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])

    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return

    results = []
    folders = sorted([f for f in target_dir.iterdir() if f.is_dir()])
    
    print(f"Scanning {len(folders)} experiments...")
    for folder in folders:
        row = load_metrics(folder)
        if row:
            results.append(row)

    if not results:
        print("No valid results found.")
        return

    df = pd.DataFrame(results)
    
    # Sort for readability
    df = df.sort_values(['Region', 'Model', 'Variant'])

    # 1. Print a concise summary to console
    console_cols = [
        'Region', 'Model', 'Variant', 
        'Log R2 (Mean)', 'Log R2 (Std)',
        'Pooled Disturbed R2',
        'Cov 1s (Mean)', 'Cov 1s (Std)'
    ]
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print("\n" + "="*100)
    print("CONCISE SUMMARY (Key Means & Stds)")
    print("="*100)
    print(df[console_cols].to_string(index=False))
    print("="*100)

    # 2. Save the FULL monster table to CSV
    csv_path = target_dir / 'experiment_comprehensive_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nComprehensive metrics (ALL Means & Stds) saved to:\n{csv_path}")

if __name__ == "__main__":
    main()