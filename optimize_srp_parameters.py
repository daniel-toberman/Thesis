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
    'n_dft_bins': [256, 512, 1024, 2048, 4096, 8192, 2**14, 2**15],  # 6 values, up to 2^13
    'n_avg_samples': [1, 5, 10, 50],                   # 4 values
    'srp_grid_cells': [360],                      # 2 values
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
RESULTS_CSV = 'srp_optimization_results.csv'  # Unified results file

def generate_parameter_hash(params):
    """Generate a unique hash for parameter combination."""
    # Sort parameters by key to ensure consistent hashing
    sorted_params = tuple(sorted(params.items()))
    return hash(sorted_params)

def load_existing_results():
    """Load existing results if they exist."""
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            print(f"Loaded {len(df)} existing results from {RESULTS_CSV}")
            return df
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            return pd.DataFrame()
    else:
        print(f"No existing results found at {RESULTS_CSV}")
        return pd.DataFrame()

def get_existing_parameter_hashes(df_existing):
    """Get hashes of existing parameter combinations."""
    if df_existing.empty:
        return set()

    existing_hashes = set()
    param_cols = list(PARAMETER_RANGES.keys())

    for _, row in df_existing.iterrows():
        params = {col: int(row[col]) for col in param_cols if col in row}
        param_hash = generate_parameter_hash(params)
        existing_hashes.add(param_hash)

    return existing_hashes

def generate_parameter_combinations():
    """Generate all parameter combinations to test, excluding existing ones."""
    param_names = list(PARAMETER_RANGES.keys())
    param_values = list(PARAMETER_RANGES.values())

    # Load existing results and get parameter hashes
    df_existing = load_existing_results()
    existing_hashes = get_existing_parameter_hashes(df_existing)

    all_combinations = []
    new_combinations = []

    for combo in product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_hash = generate_parameter_hash(param_dict)

        all_combinations.append(param_dict)

        if param_hash not in existing_hashes:
            new_combinations.append(param_dict)

    print(f"Total parameter combinations: {len(all_combinations)}")
    print(f"Existing combinations: {len(existing_hashes)}")
    print(f"New combinations to test: {len(new_combinations)}")

    return new_combinations, df_existing

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

    # Add variable parameters (ensure integers are passed as integers)
    for param, value in params.items():
        cmd.extend([f'--{param}', str(int(value)) if isinstance(value, (int, float)) else str(value)])

    # Add sample limit and randomization for phase 1
    if n_samples is not None:
        cmd.extend(['--n', str(n_samples), '--random', '--seed', '42'])

    try:
        # Run the command without timeout
        # For phase 2 (full dataset), show output in real-time
        if quiet and n_samples is not None:
            # Phase 1: capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            # Phase 2: show output in real-time (no capture)
            result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"ERROR: SRP run failed with params {params}")
            if quiet and n_samples is not None:
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

def phase1_screening(combinations, df_existing):
    """Phase 1: Screen all combinations on limited samples."""
    if not combinations:
        print("\n=== PHASE 1: No new combinations to test ===\n")
        # If no new combinations, return top combinations from existing results
        if not df_existing.empty and 'phase' in df_existing.columns:
            existing_phase1 = df_existing[df_existing['phase'] == 1]
            if not existing_phase1.empty:
                df_sorted = existing_phase1.sort_values('mae')
                top_combinations = df_sorted.head(TOP_K)
                print(f"Using top {len(top_combinations)} combinations from existing Phase 1 results")
                print(f"Best existing MAE: {df_sorted.iloc[0]['mae']:.2f}°")
                return top_combinations
        return pd.DataFrame()

    print(f"\n=== PHASE 1: Screening {len(combinations)} NEW combinations on {PHASE1_N_SAMPLES} samples ===")

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
            'phase': 1,
            'is_full_dataset': False,
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

    # Save phase 1 results and append to unified results
    df_phase1 = pd.DataFrame(results)
    if not df_phase1.empty:
        # Append to unified results file
        if not os.path.exists(RESULTS_CSV):
            df_phase1.to_csv(RESULTS_CSV, index=False)
        else:
            df_phase1.to_csv(RESULTS_CSV, mode='a', header=False, index=False)

    # Save separate phase 1 file for backwards compatibility
    df_phase1.to_csv('srp_phase1_results.csv', index=False)

    # Combine with existing results and get top combinations
    # Filter existing results for phase 1 (handle missing 'phase' column)
    if not df_existing.empty and 'phase' in df_existing.columns:
        existing_phase1 = df_existing[df_existing['phase'] == 1]
    else:
        existing_phase1 = pd.DataFrame()
    df_all_phase1 = pd.concat([existing_phase1, df_phase1], ignore_index=True)
    if df_all_phase1.empty:
        print("No phase 1 results available for phase 2")
        return pd.DataFrame()

    df_sorted = df_all_phase1.sort_values('mae')
    top_combinations = df_sorted.head(TOP_K)

    print(f"\n=== Phase 1 Complete ===")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best MAE: {df_sorted.iloc[0]['mae']:.2f}°")
    print(f"Top {TOP_K} combinations selected for Phase 2")

    return top_combinations

def phase2_validation(top_combinations, df_existing):
    """Phase 2: Test top combinations on full dataset."""
    if top_combinations.empty:
        print("\n=== PHASE 2: No combinations available for full testing ===\n")
        return pd.DataFrame(), pd.Series(dtype=object)

    # Check which combinations already have full dataset results
    existing_phase2_hashes = set()
    if not df_existing.empty and 'is_full_dataset' in df_existing.columns:
        phase2_existing = df_existing[df_existing['is_full_dataset'] == True]
        existing_phase2_hashes = get_existing_parameter_hashes(phase2_existing)

    # Filter out combinations that already have phase 2 results
    combinations_to_test = []
    for _, row in top_combinations.iterrows():
        params = {key: int(row[key]) for key in PARAMETER_RANGES.keys()}
        param_hash = generate_parameter_hash(params)
        if param_hash not in existing_phase2_hashes:
            combinations_to_test.append((_, row))

    if not combinations_to_test:
        print(f"\n=== PHASE 2: All top {len(top_combinations)} combinations already tested on full dataset ===\n")
        # Return existing phase 2 results
        if 'is_full_dataset' in df_existing.columns:
            phase2_results = df_existing[df_existing['is_full_dataset'] == True]
        else:
            phase2_results = pd.DataFrame()
        if not phase2_results.empty:
            best_combo = phase2_results.loc[phase2_results['mae'].idxmin()]
            return phase2_results, best_combo
        else:
            return pd.DataFrame(), pd.Series(dtype=object)

    print(f"\n=== PHASE 2: Testing {len(combinations_to_test)} NEW combinations on full dataset ({len(top_combinations) - len(combinations_to_test)} already completed) ===")

    results = []
    start_time = time.time()

    for i, (_, row) in enumerate(combinations_to_test):
        # Extract parameters and ensure they are integers
        params = {key: int(row[key]) for key in PARAMETER_RANGES.keys()}
        print(f"[{i+1}/{len(combinations_to_test)}] Full test: {params}")

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
            'phase': 2,
            'is_full_dataset': True,
            **params,  # Unpack parameter values
            **metrics  # Unpack metrics (phase 2)
        }
        results.append(result_entry)

        print(f"    MAE: {metrics['mae']:.2f}° (Phase 1: {row['mae']:.2f}°), Time: {run_time:.1f}s")

    # Save phase 2 results
    df_phase2_new = pd.DataFrame(results)
    if not df_phase2_new.empty:
        # Add phase identifier
        df_phase2_new['phase'] = 2
        df_phase2_new['is_full_dataset'] = True

        # Append to unified results file
        df_phase2_new.to_csv(RESULTS_CSV, mode='a', header=False, index=False)

    # Combine with existing phase 2 results
    if not df_existing.empty and 'is_full_dataset' in df_existing.columns:
        existing_phase2 = df_existing[df_existing['is_full_dataset'] == True]
    else:
        existing_phase2 = pd.DataFrame()
    df_phase2_all = pd.concat([existing_phase2, df_phase2_new], ignore_index=True)

    # Save separate phase 2 file for backwards compatibility
    df_phase2_all.to_csv('srp_phase2_results.csv', index=False)

    # Find best combination
    if df_phase2_all.empty:
        return pd.DataFrame(), pd.Series(dtype=object)

    best_combo = df_phase2_all.loc[df_phase2_all['mae'].idxmin()]

    print(f"\n=== Phase 2 Complete ===")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best final MAE: {best_combo['mae']:.2f}°")

    return df_phase2_all, best_combo

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
        f"- {RESULTS_CSV} - Unified optimization results",
        "- srp_phase1_results.csv - All combinations screening results (legacy)",
        "- srp_phase2_results.csv - Top combinations full validation (legacy)",
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

    # Generate all parameter combinations (excluding existing ones)
    combinations, df_existing = generate_parameter_combinations()

    # Phase 1: Screen new combinations
    top_combinations = phase1_screening(combinations, df_existing)

    # Phase 2: Validate top combinations
    df_phase2, best_combo = phase2_validation(top_combinations, df_existing)

    # Generate report
    if os.path.exists(RESULTS_CSV):
        df_all_results = pd.read_csv(RESULTS_CSV)
        if 'phase' in df_all_results.columns:
            df_phase1_all = df_all_results[df_all_results['phase'] == 1]
        else:
            df_phase1_all = df_all_results  # Assume all results are phase 1 if column missing
        generate_report(df_phase1_all, df_phase2, best_combo)
    else:
        print("No results available for report generation")
        return

    if not best_combo.empty and 'mae' in best_combo:
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE")
        print(f"Optimal Parameters:")
        for param in PARAMETER_RANGES.keys():
            if param in best_combo:
                print(f"  {param.upper()}: {int(best_combo[param])}")
        print(f"Best MAE: {best_combo['mae']:.2f}°")
        print("="*50)
    else:
        print("\nOptimization completed but no valid results found.")

if __name__ == "__main__":
    main()