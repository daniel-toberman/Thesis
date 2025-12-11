#!/usr/bin/env python3
"""
Test script to verify distribution analysis on a single method (VIM).
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_ood_distributions import (
    load_data_splits,
    compute_method_scores,
    find_optimal_f1_threshold,
    compute_overlap_metrics,
    plot_three_histogram
)

def main():
    print("Testing distribution analysis on VIM method...")

    # Create output directory
    output_dir = 'results/ood_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n1. Loading data splits...")
    val_features, test_success_features, test_fail_features = load_data_splits()

    # Test on VIM method
    method = 'vim'
    print(f"\n2. Computing scores for {method}...")
    val_scores = compute_method_scores(method, val_features, val_features)
    success_scores = compute_method_scores(method, test_success_features, val_features)
    fail_scores = compute_method_scores(method, test_fail_features, val_features)

    print(f"\n3. Score statistics:")
    print(f"  Validation: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}")
    print(f"  Success: mean={success_scores.mean():.4f}, std={success_scores.std():.4f}")
    print(f"  Fail: mean={fail_scores.mean():.4f}, std={fail_scores.std():.4f}")

    # Find thresholds
    print(f"\n4. Computing thresholds...")
    import numpy as np
    test_scores_combined = np.concatenate([success_scores, fail_scores])
    threshold_30pct = np.percentile(test_scores_combined, 70)
    optimal_threshold, optimal_f1 = find_optimal_f1_threshold(success_scores, fail_scores, ascending=True)

    print(f"  30% threshold: {threshold_30pct:.4f}")
    print(f"  Optimal F1 threshold: {optimal_threshold:.4f} (F1={optimal_f1:.3f})")

    # Compute metrics
    print(f"\n5. Computing overlap metrics...")
    metrics = compute_overlap_metrics(val_scores, success_scores, fail_scores)
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  Success-Fail separation: {metrics['success_fail_separation']:.4f}")
    print(f"  Val-Success overlap: {metrics['val_success_overlap']:.4f}")

    # Plot histogram
    print(f"\n6. Generating histogram...")
    plot_three_histogram(method, val_scores, success_scores, fail_scores,
                        threshold_30pct, optimal_threshold, output_dir)

    print(f"\n✅ Test complete! Check: {output_dir}/{method}_histogram.png")

if __name__ == '__main__':
    main()
