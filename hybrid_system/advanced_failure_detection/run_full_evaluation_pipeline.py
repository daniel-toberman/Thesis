#!/usr/bin/env python3
"""
Master Runner for Full Hybrid System Evaluation

This script orchestrates the evaluation of all 19+ OOD (Out-of-Distribution)
methods across all 30 specified microphone configurations.

It iterates through each microphone setup, runs each OOD method using the
parameterized `evaluate_ood_hybrid.py` script, aggregates the results,
and saves a comprehensive CSV report.

The process is as follows:
1.  **Discover Data:** It automatically finds all CRNN feature files (`.npz`)
in the `crnn_features/` directory and all SRP cache files (`.pkl`) in
the `hybrid_system/advanced_failure_detection/features/` directory.
2.  **Define Methods:** A comprehensive dictionary defines all OOD methods
to be tested, along with their optimal routing thresholds (derived from
the research summary).
3.  **Execute in Nested Loops:**
    - The outer loop iterates through each microphone configuration.
    - The inner loop iterates through each OOD method.
4.  **Invoke Worker Script:** For each combination, it calls
    `evaluate_ood_hybrid.py` with the correct paths for the CRNN features
    and SRP cache, method name, and threshold.
5.  **Aggregate Results:** It reads the `hybrid_summary.csv` file produced
    by each run, adds metadata for the microphone configuration, and appends
    it to a master list.
6.  **Generate Final Report:** After all evaluations are complete, it saves
    the aggregated results into a single `full_hybrid_evaluation_results.csv`
    file for easy analysis.
7.  **Final Analysis:** The script concludes by calculating the mean
    performance metrics for each method across all 30 microphone
    configurations, printing a summary table that ranks the methods by
    average hybrid MAE.
"""

import os
import subprocess
import pandas as pd
import glob
from pathlib import Path

# --- Configuration ---

# Define all methods and their optimal thresholds for ~30% routing
# Thresholds are taken from research_summary.md and other scripts
METHODS_AND_THRESHOLDS = {
    # Supervised
    'confidnet': 0.3,          # ConfidNet 20° for ~30% routing

    # Top-tier Post-hoc (from 30% routing MAE table)
    'vim': 2.5,                # Placeholder, needs calibration via run_new_ood_methods.py
    'she': 0.9,                # Placeholder, needs calibration
    'gradnorm': 1.0373,        # Confirmed
    'max_prob': 0.95,          # Placeholder, needs calibration
    'dice': 0.8,               # Corresponds to DICE 80%

    # Other Post-hoc
    'knn': 3.0428,             # k=10, Confirmed
    'mc_dropout': 4.12,        # Confirmed
    'energy': -0.98,           # Confirmed
    'react': 82.0905,          # p85, Confirmed
    'mahalanobis': 12.9890,    # Confirmed

    # Density-based
    'llr': 5.0,                # Placeholder for GMM-5, needs calibration
}

# --- Paths ---
BASE_DIR = Path(__file__).parent
CRNN_FEATURES_DIR = BASE_DIR.parent.parent / 'crnn features'
SRP_CACHE_DIR = BASE_DIR / 'features'
RESULTS_DIR = BASE_DIR / 'full_evaluation_results'

def run_single_evaluation(crnn_path, srp_path, method, threshold, mic_config):
    """
    Runs a single evaluation instance of evaluate_ood_hybrid.py.
    """
    output_dir = RESULTS_DIR / mic_config / method
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python',
        str(BASE_DIR / 'evaluate_ood_hybrid.py'),
        '--method', method,
        '--threshold', str(threshold),
        '--features_path', str(crnn_path),
        '--srp_cache_path', str(srp_path),
        '--output_dir', str(output_dir)
    ]

    print(f"\n--- Running: {mic_config} / {method} ---")
    print(f"CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        print(f"STDOUT: {result.stdout}")
        summary_path = output_dir / 'hybrid_summary.csv'
        if summary_path.exists():
            return pd.read_csv(summary_path)
        else:
            print(f"ERROR: Summary file not found for {mic_config}/{method}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {mic_config}/{method}:")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None
    except subprocess.TimeoutExpired as e:
        print(f"TIMEOUT ERROR running {mic_config}/{method}:")
        print(f"Process took too long and was killed.")
        return None


def main():
    """
    Main function to run the full evaluation pipeline.
    """
    print("="*80)
    print("Starting Full Hybrid System Evaluation Pipeline")
    print("="*80)

    # Discover data files
    crnn_files = sorted(glob.glob(str(CRNN_FEATURES_DIR / '*.pkl')))
    srp_files = sorted(glob.glob(str(SRP_CACHE_DIR / 'srp_results_mics_*.pkl')))

    print(f"Found {len(crnn_files)} CRNN feature files.")
    print(f"Found {len(srp_files)} SRP cache files.")

    if not crnn_files or not srp_files or len(crnn_files) != len(srp_files):
        print("Error: Mismatch in number of CRNN and SRP files or no files found. Exiting.")
        return

    # Match files
    file_pairs = []
    for crnn_file in crnn_files:
        mic_config_name = Path(crnn_file).stem.replace('crnn_results_', '')
        matching_srp = [srp for srp in srp_files if mic_config_name in Path(srp).name]
        if matching_srp:
            file_pairs.append((crnn_file, matching_srp[0], mic_config_name))

    print(f"Successfully matched {len(file_pairs)} pairs of CRNN and SRP files.")

    all_results = []

    # Outer loop: Iterate over microphone configurations
    for crnn_path, srp_path, mic_config in file_pairs:
        # Inner loop: Iterate over OOD methods
        for method, threshold in METHODS_AND_THRESHOLDS.items():
            summary_df = run_single_evaluation(crnn_path, srp_path, method, threshold, mic_config)
            if summary_df is not None:
                summary_df['mic_config'] = mic_config
                all_results.append(summary_df)

    # Aggregate and save final report
    if not all_results:
        print("\nNo results were generated. Exiting.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_report_path = RESULTS_DIR / 'full_hybrid_evaluation_results.csv'
    final_df.to_csv(final_report_path, index=False)

    print("\n" + "="*80)
    print(f"Full evaluation complete! Master report saved to: {final_report_path}")
    print("="*80)

    # Final analysis: Calculate and display mean performance
    mean_performance = final_df.groupby('method').agg({
        'hybrid_mae': 'mean',
        'hybrid_median': 'mean',
        'hybrid_success': 'mean',
        'routing_rate': 'mean',
        'f1_score': 'mean'
    }).sort_values(by='hybrid_mae').reset_index()

    print("\n--- Mean Performance Across All Configurations ---")
    print(mean_performance.to_string(index=False))

if __name__ == '__main__':
    main()
