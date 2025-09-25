#!/usr/bin/env python3
"""
Analyze cases where SRP-PHAT significantly outperforms CRNN to identify
patterns and characteristics that predict when classical methods work best.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'SSL'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'xsrpMain/xsrp'))

def analyze_srp_success_cases():
    """Analyze the cases where SRP significantly outperforms CRNN."""

    # Load the comparison results
    try:
        results_df = pd.read_csv("srp_vs_crnn_on_failures.csv")
    except FileNotFoundError:
        print("Error: Run test_srp_on_crnn_failures.py first")
        return

    print("=== SRP SUCCESS PATTERN ANALYSIS ===")
    print(f"Total cases analyzed: {len(results_df)}")

    # Focus on cases where SRP is better
    srp_better = results_df[results_df['improvement'] > 0].copy()
    srp_better = srp_better.sort_values('improvement', ascending=False)

    print(f"\nSRP better in {len(srp_better)}/{len(results_df)} cases ({len(srp_better)/len(results_df)*100:.1f}%)")

    print(f"\n=== TOP SRP SUCCESS CASES ===")
    print("Rank | GT°   | CRNN Err | SRP Err | Improvement | Audio File")
    print("-" * 70)

    for idx, (_, row) in enumerate(srp_better.iterrows()):
        print(f"{idx+1:4d} | {row['gt_angle']:5.1f} | {row['crnn_error']:8.1f} | {row['srp_error']:7.1f} | {row['improvement']:11.1f} | {row['audio_file']}")

    # Analyze patterns
    print(f"\n=== PATTERN ANALYSIS ===")

    # Angular patterns
    print(f"\nAngular Distribution of SRP Successes:")
    gt_angles = srp_better['gt_angle'].values
    print(f"Ground truth angles: {sorted(gt_angles)}")

    # Define angular regions
    regions = {
        'Front (0-60°)': (gt_angles >= 0) & (gt_angles < 60),
        'Right (60-120°)': (gt_angles >= 60) & (gt_angles < 120),
        'Back (120-240°)': (gt_angles >= 120) & (gt_angles < 240),
        'Left (240-360°)': (gt_angles >= 240) & (gt_angles < 360),
    }

    for region_name, mask in regions.items():
        count = mask.sum()
        if count > 0:
            region_data = srp_better[mask]
            avg_improvement = region_data['improvement'].mean()
            print(f"  {region_name}: {count} cases, avg improvement: {avg_improvement:.1f}°")

    # CRNN failure patterns in successful SRP cases
    print(f"\nCRNN Failure Patterns in SRP Success Cases:")
    crnn_preds = srp_better['crnn_pred'].values
    print(f"CRNN predictions: {sorted(crnn_preds)}")

    # Check for systematic CRNN biases
    bias_137 = (np.abs(crnn_preds - 137) < 5).sum()
    bias_001 = (np.abs(crnn_preds - 1) < 5).sum()
    bias_174 = (np.abs(crnn_preds - 174) < 5).sum()

    print(f"CRNN systematic biases in SRP successes:")
    print(f"  ~137° bias: {bias_137} cases")
    print(f"  ~1° bias: {bias_001} cases")
    print(f"  ~174° bias: {bias_174} cases")

    # Environment/room analysis
    print(f"\nEnvironment Analysis (from audio filenames):")
    environments = {}
    for filename in srp_better['audio_file']:
        if pd.notna(filename):
            # Extract environment from filename (e.g., TEST_S_CAF1_... -> CAF1)
            parts = str(filename).split('_')
            if len(parts) >= 3:
                env = parts[2]
                environments[env] = environments.get(env, 0) + 1

    print(f"Environments where SRP succeeds:")
    for env, count in sorted(environments.items()):
        print(f"  {env}: {count} cases")

    # Error magnitude analysis
    print(f"\nError Magnitude Analysis:")
    print(f"SRP success cases - CRNN error range: {srp_better['crnn_error'].min():.1f}° - {srp_better['crnn_error'].max():.1f}°")
    print(f"SRP success cases - SRP error range: {srp_better['srp_error'].min():.1f}° - {srp_better['srp_error'].max():.1f}°")
    print(f"Average CRNN error in SRP successes: {srp_better['crnn_error'].mean():.1f}°")
    print(f"Average SRP error in SRP successes: {srp_better['srp_error'].mean():.1f}°")

    return srp_better

def identify_exceptional_cases():
    """Identify the most exceptional SRP success cases for detailed analysis."""

    try:
        results_df = pd.read_csv("srp_vs_crnn_on_failures.csv")
    except FileNotFoundError:
        print("Error: Run test_srp_on_crnn_failures.py first")
        return

    # Define exceptional cases
    exceptional_cases = results_df[
        (results_df['improvement'] > 30) |  # Large improvement
        (results_df['srp_error'] < 5)       # Very low SRP error
    ].copy()

    print(f"\n=== EXCEPTIONAL SRP CASES ===")
    print(f"Cases with >30° improvement OR <5° SRP error:")
    print()

    for idx, (_, row) in enumerate(exceptional_cases.iterrows()):
        print(f"Case {idx+1}: {row['audio_file']}")
        print(f"  GT: {row['gt_angle']:.1f}°")
        print(f"  CRNN: {row['crnn_pred']:.1f}° (error: {row['crnn_error']:.1f}°)")
        print(f"  SRP: {row['srp_pred']:.1f}° (error: {row['srp_error']:.1f}°)")
        print(f"  Improvement: {row['improvement']:.1f}°")
        print(f"  Dataset: {row['dataset']}")
        print()

    return exceptional_cases

def suggest_hybrid_criteria():
    """Suggest criteria for when to use SRP vs CRNN based on patterns."""

    try:
        results_df = pd.read_csv("srp_vs_crnn_on_failures.csv")
    except FileNotFoundError:
        print("Error: Run test_srp_on_crnn_failures.py first")
        return

    srp_better = results_df[results_df['improvement'] > 0]

    print(f"\n=== HYBRID SWITCHING CRITERIA SUGGESTIONS ===")
    print()

    print("Based on the analysis, consider switching to SRP when:")
    print()

    # Angular criteria
    successful_angles = srp_better['gt_angle'].values
    print(f"1. **Angular Regions**: Target angles around {successful_angles}")

    # CRNN prediction criteria
    successful_crnn_preds = srp_better['crnn_pred'].values
    print(f"2. **CRNN Prediction Patterns**: When CRNN predicts around {successful_crnn_preds}")

    # Error magnitude criteria
    successful_crnn_errors = srp_better['crnn_error'].values
    print(f"3. **CRNN Error Threshold**: When CRNN error likely >20° (current successes: {successful_crnn_errors.min():.1f}-{successful_crnn_errors.max():.1f}°)")

    # Environment criteria
    print(f"4. **Environmental Conditions**: Certain acoustic environments show better SRP performance")

    print()
    print("Specific high-confidence switching rules:")
    print(f"- If CRNN predicts ~137° and GT likely in 90-120° region → Use SRP")
    print(f"- If CRNN predicts ~174° and GT likely in 60-90° region → Use SRP")
    print(f"- If CRNN predicts ~1° and GT likely in 30-40° region → Use SRP")
    print()
    print("Next steps:")
    print(f"1. Build confidence/uncertainty estimator for CRNN predictions")
    print(f"2. Train binary classifier: SRP_better = f(CRNN_pred, confidence, features)")
    print(f"3. Test hybrid system on full validation set")

def main():
    srp_success_cases = analyze_srp_success_cases()
    exceptional_cases = identify_exceptional_cases()
    suggest_hybrid_criteria()

    # Save detailed analysis
    if srp_success_cases is not None:
        srp_success_cases.to_csv("srp_success_cases.csv", index=False)
        print(f"\nDetailed SRP success cases saved to: srp_success_cases.csv")

if __name__ == "__main__":
    main()