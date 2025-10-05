#!/usr/bin/env python3
"""
SRP Parameter Optimization Script

Two-phase parameter optimization:
1. Screen all parameter combinations on 100 test samples
2. Run top 10 combinations on full dataset

Usage: python optimize_srp_parameters.py
"""

import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime

# Parameter ranges to test
PARAMETER_RANGES = {
    'n_dft_bins': [256, 512, 1024, 2048, 4096, 8192],  # 6 values, up to 2^13
    'n_avg_samples': [1, 5, 10, 50],                   # 4 values
    'srp_grid_cells': [360, 720],                      # 2 values
    'freq_min': [200, 300],                            # 2 values
    'freq_max': [3000, 4000],                          # 2 values
}

# Fixed parameters
FIXED_PARAMS = {
    'csv': "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv",
    'base_dir': "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted",
    'array_diameter': "6cm",  # Test on 6cm array only
    'srp_mode': "gcc_phat_freq",
    'enable_bandpass': True,
    'filter_order': 6,
}

# Phase settings
PHASE1_N_SAMPLES = 100  # Number of samples for initial screening
TOP_K = 10              # Number of top combinations to test in phase 2

def generate_parameter_combinations():
    """Generate all parameter combinations to test."""
    param_names = list(PARAMETER_RANGES.keys())
    param_values = list(PARAMETER_RANGES.values())

    combinations = []
    for combo in product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)

    print(f"Generated {len(combinations)} parameter combinations")
    return combinations

def run_srp_with_params(params, n_samples=None, quiet=True):
    """Run SRP with given parameters and return results."""

    # Build command
    cmd = [
        'python', '-m', 'xsrpMain.xsrp.run_SRP_parametric',
        '--csv', FIXED_PARAMS['csv'],
        '--base-dir', FIXED_PARAMS['base_dir'],
        '--array_diameter', FIXED_PARAMS['array_diameter'],
        '--srp_mode', FIXED_PARAMS['srp_mode'],
        '--filter_order', str(FIXED_PARAMS['filter_order']),
    ]

    if FIXED_PARAMS['enable_bandpass']:
        cmd.append('--enable_bandpass')

    # Add variable parameters
    for param, value in params.items():
        cmd.extend([f'--{param}', str(value)])

    # Add sample limit and randomization for phase 1
    if n_samples is not None:
        cmd.extend(['--n', str(n_samples), '--random', '--seed', '42'])

    try:
        # Run the command
        if quiet:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        else:
            result = subprocess.run(cmd, timeout=600)

        if result.returncode != 0:
            print(f"ERROR: SRP run failed with params {params}")
            if quiet:
                print(f"STDERR: {result.stderr}")
            return None

        # Parse results from CSV file
        csv_path = Path(FIXED_PARAMS['csv']).with_name(
            Path(FIXED_PARAMS['csv']).stem + "_srp_results.csv"
        )

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df
        else:
            print(f"WARNING: Results CSV not found at {csv_path}")
            return None

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: SRP run timed out with params {params}")
        return None
    except Exception as e:
        print(f"ERROR: Exception during SRP run: {e}")
        return None

def calculate_metrics(results_df):
    """Calculate performance metrics from results dataframe."""
    if results_df is None or results_df.empty:
        return {
            'mae': float('inf'),
            'median_error': float('inf'),
            'p95_error': float('inf'),
            'n_samples': 0,
            'success_rate': 0.0
        }

    # Filter successful results
    success_mask = results_df['status'] == 'ok'
    successful_results = results_df[success_mask]

    if successful_results.empty:
        return {
            'mae': float('inf'),
            'median_error': float('inf'),
            'p95_error': float('inf'),
            'n_samples': len(results_df),
            'success_rate': 0.0
        }

    errors = successful_results['abs_err_deg'].values

    return {
        'mae': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'p95_error': float(np.percentile(errors, 95)),
        'n_samples': len(results_df),
        'success_rate': float(len(successful_results) / len(results_df))
    }

def phase1_screening(combinations):
    """Phase 1: Screen all combinations on limited samples."""
    print(f"\n=== PHASE 1: Screening {len(combinations)} combinations on {PHASE1_N_SAMPLES} samples ===")

    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing: {params}")

        # Run SRP with this parameter combination
        run_start = time.time()
        df_results = run_srp_with_params(params, n_samples=PHASE1_N_SAMPLES)
        run_time = time.time() - run_start

        # Calculate metrics
        metrics = calculate_metrics(df_results)

        # Store results
        result_entry = {
            'combination_id': i,
            'run_time_sec': run_time,
            **params,  # Unpack parameter values
            **metrics  # Unpack metrics
        }
        results.append(result_entry)

        print(f"    MAE: {metrics['mae']:.2f}°, Time: {run_time:.1f}s")

        # Save intermediate results every 10 combinations
        if (i + 1) % 10 == 0:
            df_temp = pd.DataFrame(results)
            df_temp.to_csv('srp_phase1_results_temp.csv', index=False)
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (len(combinations) - i - 1)
            print(f"    Progress: {i+1}/{len(combinations)}, Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")

    # Save final phase 1 results
    df_phase1 = pd.DataFrame(results)
    df_phase1.to_csv('srp_phase1_results.csv', index=False)

    # Sort by MAE and get top combinations
    df_sorted = df_phase1.sort_values('mae')
    top_combinations = df_sorted.head(TOP_K)

    print(f"\n=== Phase 1 Complete ===")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best MAE: {df_sorted.iloc[0]['mae']:.2f}°")
    print(f"Top {TOP_K} combinations selected for Phase 2")

    return top_combinations

def phase2_validation(top_combinations):
    """Phase 2: Test top combinations on full dataset."""
    print(f"\n=== PHASE 2: Testing top {len(top_combinations)} combinations on full dataset ===")

    results = []
    start_time = time.time()

    for i, (_, row) in enumerate(top_combinations.iterrows()):
        # Extract parameters
        params = {key: row[key] for key in PARAMETER_RANGES.keys()}
        print(f"[{i+1}/{len(top_combinations)}] Full test: {params}")

        # Run SRP on full dataset
        run_start = time.time()
        df_results = run_srp_with_params(params, n_samples=None)  # Full dataset
        run_time = time.time() - run_start

        # Calculate metrics
        metrics = calculate_metrics(df_results)

        # Store results
        result_entry = {
            'combination_id': row['combination_id'],
            'phase1_mae': row['mae'],  # Original phase 1 MAE
            'run_time_sec': run_time,
            **params,  # Unpack parameter values
            **metrics  # Unpack metrics (phase 2)
        }
        results.append(result_entry)

        print(f"    MAE: {metrics['mae']:.2f}° (Phase 1: {row['mae']:.2f}°), Time: {run_time:.1f}s")

    # Save phase 2 results
    df_phase2 = pd.DataFrame(results)
    df_phase2.to_csv('srp_phase2_results.csv', index=False)

    # Find best combination
    best_combo = df_phase2.loc[df_phase2['mae'].idxmin()]

    print(f"\n=== Phase 2 Complete ===")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best final MAE: {best_combo['mae']:.2f}°")

    return df_phase2, best_combo

def generate_report(df_phase1, df_phase2, best_combo):
    """Generate optimization report."""

    report_lines = [
        "# SRP Parameter Optimization Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- Total combinations tested: {len(df_phase1)}",
        f"- Phase 1 samples per combination: {PHASE1_N_SAMPLES}",
        f"- Phase 2 combinations tested: {len(df_phase2)}",
        f"- Best MAE achieved: {best_combo['mae']:.2f}°",
        "",
        "## Optimal Parameters",
        f"```",
    ]

    # Add optimal parameters
    for param in PARAMETER_RANGES.keys():
        report_lines.append(f"{param.upper()} = {best_combo[param]}")

    report_lines.extend([
        "```",
        "",
        "## Top 5 Combinations (Phase 2 Results)",
        "",
    ])

    # Add top 5 table
    top5 = df_phase2.nsmallest(5, 'mae')
    for i, (_, row) in enumerate(top5.iterrows()):
        report_lines.append(f"{i+1}. MAE: {row['mae']:.2f}°")
        for param in PARAMETER_RANGES.keys():
            report_lines.append(f"   {param}: {row[param]}")
        report_lines.append("")

    report_lines.extend([
        "## Performance Analysis",
        "",
        f"- Mean MAE (all Phase 2): {df_phase2['mae'].mean():.2f}°",
        f"- Std MAE (all Phase 2): {df_phase2['mae'].std():.2f}°",
        f"- Best vs Worst: {df_phase2['mae'].min():.2f}° vs {df_phase2['mae'].max():.2f}°",
        "",
        "## Files Generated",
        "- srp_phase1_results.csv - All combinations screening results",
        "- srp_phase2_results.csv - Top combinations full validation",
        "- srp_optimization_report.txt - This report"
    ])

    # Write report
    with open('srp_optimization_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nReport saved to: srp_optimization_report.txt")

def main():
    """Main optimization routine."""
    print("SRP Parameter Optimization")
    print("=" * 50)
    print(f"Total parameter combinations: {np.prod([len(v) for v in PARAMETER_RANGES.values()])}")
    print(f"Phase 1: {PHASE1_N_SAMPLES} samples per combination")
    print(f"Phase 2: Full dataset for top {TOP_K} combinations")

    # Generate all parameter combinations
    combinations = generate_parameter_combinations()

    # Phase 1: Screen all combinations
    top_combinations = phase1_screening(combinations)

    # Phase 2: Validate top combinations
    df_phase2, best_combo = phase2_validation(top_combinations)

    # Generate report
    df_phase1 = pd.read_csv('srp_phase1_results.csv')
    generate_report(df_phase1, df_phase2, best_combo)

    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Parameters:")
    for param in PARAMETER_RANGES.keys():
        print(f"  {param.upper()}: {best_combo[param]}")
    print(f"Best MAE: {best_combo['mae']:.2f}°")
    print("="*50)

if __name__ == "__main__":
    main()