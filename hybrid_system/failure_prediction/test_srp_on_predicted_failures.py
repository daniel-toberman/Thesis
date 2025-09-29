#!/usr/bin/env python3
"""
Test SRP-PHAT performance specifically on cases where confidence-based
failure predictor indicates CRNN will fail. This evaluates the hybrid approach.

Usage:
    python test_srp_on_predicted_failures.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import subprocess
import tempfile
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def load_confidence_predictions():
    """Load CRNN predictions with confidence metrics."""
    print("Loading CRNN confidence predictions...")

    confidence_file = 'crnn_predictions/crnn_predictions_with_confidence_clean.csv'
    if not Path(confidence_file).exists():
        raise FileNotFoundError(f"Confidence file not found: {confidence_file}")

    df = pd.read_csv(confidence_file)
    print(f"Loaded {len(df)} predictions")

    # Add failure labels
    df['is_actual_failure'] = (df['abs_error'] > 30).astype(int)
    actual_failures = df['is_actual_failure'].sum()
    print(f"Actual CRNN failures: {actual_failures} ({actual_failures/len(df)*100:.1f}%)")

    return df

def apply_best_failure_predictor(df, method='simple'):
    """Apply the best confidence-based failure predictor.

    Args:
        df: DataFrame with confidence metrics
        method: 'simple' or 'ml' for different predictor types
    """
    if method == 'simple':
        # Best F1-optimized simple predictor: max_prob <= 0.02560333
        # F1: 0.781, Precision: 0.761, Recall: 0.801, FP Rate: 2.6%
        THRESHOLD = 0.02560333
        df['predicted_failure'] = (df['max_prob'] <= THRESHOLD).astype(int)
        print(f"Using SIMPLE predictor: max_prob <= {THRESHOLD}")

    elif method == 'ml':
        # Best ML predictor: NeuralNet with all confidence features
        # CV F1: 0.772, Precision: 0.769, Recall: 0.795
        feature_cols = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']
        X = df[feature_cols]

        # Create and train the ML model (same as in find_best_failure_predictor.py)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500))
        ])

        # Quick training on the data itself (for demonstration)
        y = df['is_actual_failure']
        model.fit(X, y)

        # Predict failures
        df['predicted_failure'] = model.predict(X)
        print(f"Using ML predictor: NeuralNet with {len(feature_cols)} confidence features")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'ml'")

    predicted_failures = df['predicted_failure'].sum()
    print(f"Predicted failures: {predicted_failures} ({predicted_failures/len(df)*100:.1f}%)")

    # Validation: check recall on actual failures
    actual_failures = df['is_actual_failure'].sum()
    caught_failures = df[(df['is_actual_failure'] == 1) & (df['predicted_failure'] == 1)].shape[0]
    recall = caught_failures / actual_failures if actual_failures > 0 else 0

    print(f"Validation - Caught: {caught_failures}/{actual_failures} failures (recall: {recall:.1%})")

    return df

def create_srp_test_subset(df):
    """Create subset of data where failure predictor says CRNN will fail."""
    predicted_failure_cases = df[df['predicted_failure'] == 1].copy()

    print(f"\nCreating SRP test subset...")
    print(f"Cases where predictor says CRNN will fail: {len(predicted_failure_cases)}")

    # Breakdown by actual performance
    actual_failures_in_subset = predicted_failure_cases['is_actual_failure'].sum()
    actual_successes_in_subset = len(predicted_failure_cases) - actual_failures_in_subset

    print(f"  - Actually failed: {actual_failures_in_subset}")
    print(f"  - Actually succeeded (false positives): {actual_successes_in_subset}")

    return predicted_failure_cases

def create_srp_csv_file(subset_df, original_csv_path):
    """Create a CSV file for SRP testing with only the predicted failure cases."""

    # Read original test CSV to get the mapping
    print(f"Reading original test CSV: {original_csv_path}")
    original_df = pd.read_csv(original_csv_path)
    print(f"Original CSV has {len(original_df)} entries")

    # Create mapping from global_idx to CSV rows
    # Assuming global_idx corresponds to row number in original CSV
    srp_test_rows = []

    for _, row in subset_df.iterrows():
        global_idx = int(row['global_idx'])
        if global_idx < len(original_df):
            # Get corresponding row from original CSV
            original_row = original_df.iloc[global_idx]
            srp_test_rows.append(original_row)
        else:
            print(f"Warning: global_idx {global_idx} exceeds original CSV length")

    if not srp_test_rows:
        raise ValueError("No valid rows found for SRP testing")

    # Create SRP test CSV
    srp_df = pd.DataFrame(srp_test_rows)
    srp_csv_path = 'srp_predicted_failures_test.csv'
    srp_df.to_csv(srp_csv_path, index=False)

    print(f"Created SRP test CSV: {srp_csv_path} with {len(srp_df)} entries")
    return srp_csv_path

def run_srp_on_subset(srp_csv_path, base_dir, use_novel_noise=True):
    """Run SRP-PHAT on the predicted failure subset."""
    print(f"\nRunning SRP-PHAT on predicted failure cases...")

    # Construct SRP command
    cmd = [
        'python', '-m', 'xsrpMain.xsrp.run_SRP',
        '--csv', srp_csv_path,
        '--base-dir', base_dir
    ]

    if use_novel_noise:
        cmd.extend(['--use_novel_noise', '--novel_noise_scene', 'Cafeteria2'])
        print("Using novel noise (Cafeteria2)")

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run SRP and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("SRP completed successfully!")
        print("SRP Output:")
        print(result.stdout[-500:])  # Show last 500 chars
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"SRP failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return None

def analyze_srp_results(subset_df):
    """Analyze SRP results on the predicted failure cases."""
    print(f"\nAnalyzing SRP results...")

    # Look for SRP results file - try multiple patterns
    srp_patterns = ["srp_results*.csv", "*srp_results*.csv", "srp_predicted_failures_test_srp_results.csv"]
    srp_files = []
    for pattern in srp_patterns:
        srp_files.extend(list(Path('.').glob(pattern)))
    srp_files = list(set(srp_files))  # Remove duplicates

    if not srp_files:
        print(f"No SRP results files found matching {srp_results_pattern}")
        print("Available CSV files:")
        for csv_file in Path('.').glob("*.csv"):
            print(f"  - {csv_file}")
        return None

    # Use most recent SRP results file
    srp_results_file = max(srp_files, key=lambda x: x.stat().st_mtime)
    print(f"Using SRP results file: {srp_results_file}")

    try:
        srp_df = pd.read_csv(srp_results_file)
        print(f"Loaded {len(srp_df)} SRP results")

        # Calculate SRP performance on predicted failure cases
        if 'abs_err_deg' in srp_df.columns:
            srp_mae = srp_df['abs_err_deg'].mean()
            srp_failures = (srp_df['abs_err_deg'] > 30).sum()
            srp_success_rate = (1 - srp_failures / len(srp_df)) * 100

            print(f"\nSRP Performance on Predicted Failure Cases:")
            print(f"  - Mean Absolute Error: {srp_mae:.2f}°")
            print(f"  - Cases with >30° error: {srp_failures}/{len(srp_df)} ({srp_failures/len(srp_df)*100:.1f}%)")
            print(f"  - Success rate: {srp_success_rate:.1f}%")

            return {
                'srp_mae': srp_mae,
                'srp_failures': srp_failures,
                'srp_success_rate': srp_success_rate,
                'total_cases': len(srp_df)
            }
        else:
            print("No 'abs_err_deg' column found in SRP results")
            print(f"Available columns: {srp_df.columns.tolist()}")
            return None

    except Exception as e:
        print(f"Error reading SRP results: {e}")
        return None

def compare_crnn_vs_srp_on_failures(subset_df, srp_results):
    """Compare CRNN vs SRP performance on predicted failure cases."""
    if not srp_results:
        print("No SRP results available for comparison")
        return

    print(f"\n{'='*60}")
    print("CRNN vs SRP COMPARISON ON PREDICTED FAILURE CASES")
    print(f"{'='*60}")

    # CRNN performance on these cases
    crnn_mae = subset_df['abs_error'].mean()
    crnn_failures = subset_df['is_actual_failure'].sum()
    crnn_success_rate = (1 - crnn_failures / len(subset_df)) * 100

    print(f"Dataset: {len(subset_df)} cases where confidence predictor flagged CRNN uncertainty")
    print(f"")
    print(f"CRNN Performance:")
    print(f"  - Mean Absolute Error: {crnn_mae:.2f}°")
    print(f"  - Cases with >30° error: {crnn_failures}/{len(subset_df)} ({crnn_failures/len(subset_df)*100:.1f}%)")
    print(f"  - Success rate: {crnn_success_rate:.1f}%")
    print(f"")
    print(f"SRP Performance:")
    print(f"  - Mean Absolute Error: {srp_results['srp_mae']:.2f}°")
    print(f"  - Cases with >30° error: {srp_results['srp_failures']}/{srp_results['total_cases']} ({srp_results['srp_failures']/srp_results['total_cases']*100:.1f}%)")
    print(f"  - Success rate: {srp_results['srp_success_rate']:.1f}%")
    print(f"")

    # Calculate improvement
    mae_improvement = crnn_mae - srp_results['srp_mae']
    success_improvement = srp_results['srp_success_rate'] - crnn_success_rate

    print(f"SRP vs CRNN on Predicted Failures:")
    print(f"  - MAE improvement: {mae_improvement:+.2f}° ({'better' if mae_improvement > 0 else 'worse'})")
    print(f"  - Success rate improvement: {success_improvement:+.1f}% ({'better' if success_improvement > 0 else 'worse'})")

    # Hybrid system potential
    print(f"\nHybrid System Potential:")
    if success_improvement > 0:
        print(f"  ✅ SRP rescues {success_improvement:.1f}% of cases where CRNN would fail")
        print(f"  ✅ Confidence predictor + SRP could improve overall system performance")
    else:
        print(f"  ❌ SRP does not improve on predicted failure cases")
        print(f"  ❌ Hybrid approach may not be beneficial")

def save_analysis_results(subset_df, srp_results):
    """Save detailed analysis results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save subset with predictions
    subset_output = f"predicted_failures_analysis_{timestamp}.csv"
    subset_df.to_csv(subset_output, index=False)

    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_dataset_size': int(len(subset_df)),
        'predicted_failures': int(subset_df['predicted_failure'].sum()),
        'actual_failures_in_subset': int(subset_df['is_actual_failure'].sum()),
        'crnn_mae_on_subset': float(subset_df['abs_error'].mean()),
        'crnn_success_rate': float((1 - int(subset_df['is_actual_failure'].sum()) / len(subset_df)) * 100)
    }

    if srp_results:
        summary.update({
            'srp_mae': float(srp_results['srp_mae']),
            'srp_success_rate': float(srp_results['srp_success_rate']),
            'mae_improvement': float(summary['crnn_mae_on_subset'] - srp_results['srp_mae']),
            'success_improvement': float(srp_results['srp_success_rate'] - summary['crnn_success_rate'])
        })

    summary_output = f"hybrid_analysis_summary_{timestamp}.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved:")
    print(f"  - Detailed data: {subset_output}")
    print(f"  - Summary: {summary_output}")

def main(predictor_method='simple'):
    """Main execution function.

    Args:
        predictor_method: 'simple' or 'ml' for different predictor types
    """
    print("HYBRID APPROACH EVALUATION: SRP on Confidence-Predicted Failures")
    print(f"Using {predictor_method.upper()} failure predictor")
    print("=" * 70)

    # Configuration
    BASE_DIR = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
    ORIGINAL_CSV = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"

    # Verify files exist
    if not Path(ORIGINAL_CSV).exists():
        raise FileNotFoundError(f"Original test CSV not found: {ORIGINAL_CSV}")
    if not Path(BASE_DIR).exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    # Step 1: Load confidence predictions
    df = load_confidence_predictions()

    # Step 2: Apply failure predictor
    df = apply_best_failure_predictor(df, method=predictor_method)

    # Step 3: Create subset for SRP testing
    subset_df = create_srp_test_subset(df)

    if len(subset_df) == 0:
        print("No predicted failure cases found. Exiting.")
        return

    # Step 4: Create SRP test CSV
    srp_csv_path = create_srp_csv_file(subset_df, ORIGINAL_CSV)

    # Step 5: Run SRP on subset
    srp_output = run_srp_on_subset(srp_csv_path, BASE_DIR, use_novel_noise=True)

    if srp_output is None:
        print("SRP execution failed. Analysis incomplete.")
        return

    # Step 6: Analyze results
    srp_results = analyze_srp_results(subset_df)

    # Step 7: Compare performance
    compare_crnn_vs_srp_on_failures(subset_df, srp_results)

    # Step 8: Save results
    save_analysis_results(subset_df, srp_results)

    print("\n✅ Hybrid approach evaluation complete!")

if __name__ == "__main__":
    import sys

    # Parse command line arguments for predictor method
    predictor_method = 'simple'  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['simple', 'ml']:
            predictor_method = sys.argv[1]
        else:
            print("Usage: python test_srp_on_predicted_failures.py [simple|ml]")
            print("  simple: Use max_prob threshold (default)")
            print("  ml: Use NeuralNet with all confidence features")
            sys.exit(1)

    main(predictor_method)