#!/usr/bin/env python3
"""
Analyze CRNN's worst-performing cases to identify specific scenarios
where the network fails and classical methods might help.
"""

import pandas as pd
import numpy as np

def analyze_worst_cases():
    """Load and analyze the worst CRNN failures."""

    # Load the detailed results we already generated
    try:
        val_df = pd.read_csv("validation_detailed_results.csv")
        test_df = pd.read_csv("test_detailed_results.csv")
        combined_df = pd.concat([val_df, test_df], ignore_index=True)
        print(f"Loaded {len(combined_df)} total examples")
    except FileNotFoundError:
        print("Error: Run analyze_180_degree_cases.py first to generate the data files")
        return

    # Find worst cases
    print("\n=== WORST CRNN FAILURES ===")
    worst_cases = combined_df.nlargest(20, 'error_degrees')

    print("Top 20 worst cases:")
    print(worst_cases[['dataset', 'example_idx', 'pred_angle', 'gt_angle', 'error_degrees']].to_string(index=False))

    # Analyze patterns in failures
    print(f"\n=== FAILURE PATTERN ANALYSIS ===")

    # Define thresholds for analysis
    thresholds = [10, 15, 20, 30, 45]

    print("Failure counts by threshold:")
    for thresh in thresholds:
        failures = combined_df[combined_df['error_degrees'] > thresh]
        print(f"  >{thresh}°: {len(failures)} cases ({len(failures)/len(combined_df)*100:.2f}%)")

    # Focus on significant failures (>20° error)
    major_failures = combined_df[combined_df['error_degrees'] > 20]
    print(f"\nAnalyzing {len(major_failures)} major failures (>20° error):")

    if len(major_failures) > 0:
        # Angular distribution of failures
        print(f"\nGround truth angles of major failures:")
        failure_angles = major_failures['gt_angle'].values
        print(f"  Angles: {sorted(failure_angles)}")

        # Check if failures cluster around specific angles
        angle_bins = np.histogram(failure_angles, bins=8, range=(0, 360))[0]
        print(f"  Distribution across 8 bins (0-360°): {angle_bins}")

        # Predicted vs actual for major failures
        print(f"\nPredicted vs Ground Truth for major failures:")
        for idx, row in major_failures.iterrows():
            pred_error = row['pred_angle'] - row['gt_angle']
            print(f"  GT: {row['gt_angle']:6.1f}° → Pred: {row['pred_angle']:6.1f}° (error: {row['error_degrees']:6.1f}°, bias: {pred_error:+6.1f}°)")

        # Check for systematic biases
        pred_errors = major_failures['pred_angle'] - major_failures['gt_angle']
        # Handle circular nature
        pred_errors = np.array([(e + 180) % 360 - 180 for e in pred_errors])

        print(f"\nSystematic bias analysis for major failures:")
        print(f"  Mean bias: {pred_errors.mean():+.1f}°")
        print(f"  Std bias: {pred_errors.std():.1f}°")

        # Check if front-back confusion is common
        front_back_confusion = 0
        for idx, row in major_failures.iterrows():
            gt, pred = row['gt_angle'], row['pred_angle']
            # Check if ~180° apart (front-back confusion)
            diff = abs((gt - pred + 180) % 360 - 180)
            opposite_diff = abs(diff - 180)
            if opposite_diff < 30:  # Within 30° of being opposite
                front_back_confusion += 1

        print(f"  Front-back confusion cases: {front_back_confusion}/{len(major_failures)} ({front_back_confusion/len(major_failures)*100:.1f}%)")

    # Dataset comparison
    print(f"\n=== DATASET COMPARISON ===")
    for dataset in ['validation', 'test']:
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        if len(dataset_df) > 0:
            major_fails = (dataset_df['error_degrees'] > 20).sum()
            print(f"{dataset.capitalize()} set:")
            print(f"  Total examples: {len(dataset_df)}")
            print(f"  Major failures (>20°): {major_fails} ({major_fails/len(dataset_df)*100:.2f}%)")
            print(f"  Mean MAE: {dataset_df['error_degrees'].mean():.2f}°")
            print(f"  Worst case: {dataset_df['error_degrees'].max():.1f}°")

def analyze_performance_distribution():
    """Analyze the distribution of CRNN performance."""

    try:
        val_df = pd.read_csv("validation_detailed_results.csv")
        test_df = pd.read_csv("test_detailed_results.csv")
        combined_df = pd.concat([val_df, test_df], ignore_index=True)
    except FileNotFoundError:
        print("Error: Run analyze_180_degree_cases.py first to generate the data files")
        return

    print(f"\n=== PERFORMANCE DISTRIBUTION ANALYSIS ===")

    errors = combined_df['error_degrees']

    print(f"Statistical summary:")
    print(f"  Count: {len(errors)}")
    print(f"  Mean: {errors.mean():.2f}°")
    print(f"  Median: {errors.median():.2f}°")
    print(f"  Std: {errors.std():.2f}°")
    print(f"  Min: {errors.min():.2f}°")
    print(f"  Max: {errors.max():.2f}°")

    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(errors, p):.2f}°")

    # Error bins
    bins = [0, 1, 2, 5, 10, 15, 20, 30, 45, 90, 180]
    bin_counts = []
    for i in range(len(bins)-1):
        count = ((errors >= bins[i]) & (errors < bins[i+1])).sum()
        bin_counts.append(count)

    print(f"\nError distribution:")
    for i in range(len(bins)-1):
        percentage = bin_counts[i] / len(errors) * 100
        print(f"  {bins[i]:3.0f}°-{bins[i+1]:3.0f}°: {bin_counts[i]:4d} cases ({percentage:5.2f}%)")

def suggest_hybrid_strategies():
    """Suggest hybrid strategies based on failure analysis."""

    print(f"\n=== HYBRID STRATEGY SUGGESTIONS ===")

    try:
        val_df = pd.read_csv("validation_detailed_results.csv")
        test_df = pd.read_csv("test_detailed_results.csv")
        combined_df = pd.concat([val_df, test_df], ignore_index=True)
    except FileNotFoundError:
        print("Error: Run analyze_180_degree_cases.py first")
        return

    major_failures = combined_df[combined_df['error_degrees'] > 20]
    moderate_failures = combined_df[combined_df['error_degrees'] > 10]

    print(f"Based on the failure analysis:")
    print(f"1. **Target Population**: {len(major_failures)} major failures (>20°) + {len(moderate_failures)} moderate failures (>10°)")
    print(f"2. **Opportunity Size**: {len(major_failures)/len(combined_df)*100:.2f}% of cases have >20° error")
    print(f"3. **Maximum Benefit**: Could improve worst-case performance from {combined_df['error_degrees'].max():.1f}° to ~10-30° range")

    print(f"\nRecommended hybrid approaches:")
    print(f"1. **Outlier Detection + Fallback**: Train a classifier to detect likely CRNN failures")
    print(f"2. **Uncertainty-Based Switching**: Use prediction confidence/entropy as switching criterion")
    print(f"3. **Angular Region Specialization**: Use SRP-PHAT for specific angular regions with higher failure rates")
    print(f"4. **Ensemble Voting**: Combine both methods when they disagree significantly")
    print(f"5. **Multi-Stage Processing**: CRNN for coarse estimate, classical methods for refinement")

def main():
    analyze_worst_cases()
    analyze_performance_distribution()
    suggest_hybrid_strategies()

if __name__ == "__main__":
    main()