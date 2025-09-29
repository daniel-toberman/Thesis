#!/usr/bin/env python3
"""
Analyze CRNN confidence metrics to identify patterns between successful and failed predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_predictions_with_confidence(filepath):
    """Load CRNN predictions with confidence metrics."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} predictions from {filepath}")
    print(f"Columns: {list(df.columns)}")
    return df

def add_failure_categories(df):
    """Add failure category columns for analysis."""
    # Add failure flag
    df['is_failure'] = df['abs_error'] > 30

    # Add error categories
    df['error_category'] = pd.cut(df['abs_error'],
                                  bins=[0, 5, 10, 30, 180],
                                  labels=['Excellent (≤5°)', 'Good (5-10°)', 'Fair (10-30°)', 'Failure (>30°)'])

    # Add environment info if we can match with test CSV
    try:
        test_csv_path = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
        test_df = pd.read_csv(test_csv_path)

        # Assuming the order matches (which it should for test runs)
        if len(df) == len(test_df):
            df['room'] = test_df['room'].values
            df['filename'] = test_df['filename'].values
            print("Successfully matched with room information")
        else:
            print(f"Warning: Length mismatch - predictions: {len(df)}, test csv: {len(test_df)}")

    except Exception as e:
        print(f"Could not load room information: {e}")

    return df

def analyze_confidence_distributions(df):
    """Analyze confidence metric distributions for failures vs successes."""
    print("\n" + "="*50)
    print("CONFIDENCE METRICS ANALYSIS")
    print("="*50)

    failures = df[df['is_failure']]
    successes = df[~df['is_failure']]

    print(f"Total predictions: {len(df)}")
    print(f"Failures (>30°): {len(failures)} ({len(failures)/len(df)*100:.1f}%)")
    print(f"Successes (≤30°): {len(successes)} ({len(successes)/len(df)*100:.1f}%)")

    confidence_metrics = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']

    print(f"\nConfidence Metrics Comparison:")
    print("="*80)
    print(f"{'Metric':<20} {'Success Mean':<15} {'Failure Mean':<15} {'Difference':<15} {'P-value':<10}")
    print("-"*80)

    results = []

    for metric in confidence_metrics:
        if metric in df.columns:
            success_mean = successes[metric].mean()
            failure_mean = failures[metric].mean()
            difference = failure_mean - success_mean

            # Simple t-test approximation
            from scipy import stats
            try:
                t_stat, p_value = stats.ttest_ind(failures[metric], successes[metric])
            except:
                p_value = "N/A"

            print(f"{metric:<20} {success_mean:<15.4f} {failure_mean:<15.4f} {difference:<15.4f} {p_value:<10}")

            results.append({
                'metric': metric,
                'success_mean': success_mean,
                'failure_mean': failure_mean,
                'difference': difference,
                'p_value': p_value if isinstance(p_value, str) else p_value
            })

    return results

def create_confidence_plots(df, output_dir="confidence_analysis"):
    """Create visualization plots for confidence metrics."""
    Path(output_dir).mkdir(exist_ok=True)

    confidence_metrics = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']
    available_metrics = [m for m in confidence_metrics if m in df.columns]

    if not available_metrics:
        print("No confidence metrics found in data")
        return

    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(available_metrics):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot distributions
        failures = df[df['is_failure']][metric]
        successes = df[~df['is_failure']][metric]

        ax.hist(successes, bins=30, alpha=0.7, label=f'Success (n={len(successes)})', color='green')
        ax.hist(failures, bins=30, alpha=0.7, label=f'Failure (n={len(failures)})', color='red')

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create box plots
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(available_metrics):
        box_data = [df[~df['is_failure']][metric], df[df['is_failure']][metric]]
        box = axes[i].boxplot(box_data, labels=['Success', 'Failure'], patch_artist=True)
        box['boxes'][0].set_facecolor('green')
        box['boxes'][1].set_facecolor('red')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()}\nBox Plot')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_boxplots.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_by_environment(df):
    """Analyze confidence patterns by environment if room info available."""
    if 'room' not in df.columns:
        print("\nRoom information not available for environment analysis")
        return

    print("\n" + "="*50)
    print("ENVIRONMENT-SPECIFIC ANALYSIS")
    print("="*50)

    room_analysis = df.groupby('room').agg({
        'abs_error': ['count', 'mean', lambda x: (x > 30).sum()],
        'max_prob': 'mean',
        'entropy': 'mean',
        'local_concentration': 'mean'
    }).round(3)

    room_analysis.columns = ['Count', 'Mean_Error', 'Failures', 'Mean_MaxProb', 'Mean_Entropy', 'Mean_LocalConc']
    room_analysis['Failure_Rate'] = (room_analysis['Failures'] / room_analysis['Count'] * 100).round(1)

    print("\nRoom-wise Performance:")
    print(room_analysis.sort_values('Failure_Rate', ascending=False))

    # Focus on automotive vs non-automotive
    automotive_rooms = ['Car-Gasoline', 'Car-Electric']
    df['is_automotive'] = df['room'].isin(automotive_rooms)

    print(f"\nAutomotive vs Non-Automotive Comparison:")
    comparison = df.groupby('is_automotive').agg({
        'abs_error': ['count', 'mean', lambda x: (x > 30).sum()],
        'max_prob': 'mean',
        'entropy': 'mean',
        'prediction_variance': 'mean',
        'peak_sharpness': 'mean',
        'local_concentration': 'mean'
    }).round(4)

    print(comparison)

def find_confidence_thresholds(df):
    """Find optimal confidence thresholds for failure detection."""
    print("\n" + "="*50)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("="*50)

    confidence_metrics = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']
    available_metrics = [m for m in confidence_metrics if m in df.columns]

    best_thresholds = {}

    for metric in available_metrics:
        print(f"\nAnalyzing {metric}:")

        # Try different thresholds
        metric_values = df[metric].values
        thresholds = np.percentile(metric_values, [10, 20, 30, 40, 50, 60, 70, 80, 90])

        best_threshold = None
        best_f1 = 0

        for threshold in thresholds:
            if metric in ['entropy']:  # Higher values indicate uncertainty
                predicted_failures = df[metric] > threshold
            else:  # Lower values indicate uncertainty
                predicted_failures = df[metric] < threshold

            actual_failures = df['is_failure']

            tp = (predicted_failures & actual_failures).sum()
            fp = (predicted_failures & ~actual_failures).sum()
            fn = (~predicted_failures & actual_failures).sum()
            tn = (~predicted_failures & ~actual_failures).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds[metric] = {'threshold': best_threshold, 'f1_score': best_f1}
        print(f"  Best threshold: {best_threshold:.4f}, F1-score: {best_f1:.3f}")

    return best_thresholds

def main():
    """Main analysis function."""
    print("CRNN Confidence Metrics Analysis")
    print("="*50)

    # Check if confidence file exists
    confidence_file = "crnn_predictions/crnn_predictions_with_confidence_Cafeteria2.csv"
    fallback_file = "crnn_predictions/crnn_predictions_with_confidence_clean.csv"

    if Path(confidence_file).exists():
        df = load_predictions_with_confidence(confidence_file)
    elif Path(fallback_file).exists():
        print(f"Primary file not found, using fallback: {fallback_file}")
        df = load_predictions_with_confidence(fallback_file)
    else:
        print("No confidence prediction file found!")
        print("Please run CRNN with confidence extraction first.")
        return

    # Add failure categories and room info
    df = add_failure_categories(df)

    # Perform analyses
    confidence_results = analyze_confidence_distributions(df)
    analyze_by_environment(df)
    threshold_results = find_confidence_thresholds(df)

    # Create visualizations
    create_confidence_plots(df)

    print(f"\nAnalysis complete! Check the confidence_analysis/ folder for plots.")

    return df, confidence_results, threshold_results

if __name__ == "__main__":
    try:
        import scipy.stats
    except ImportError:
        print("Installing scipy for statistical tests...")
        import subprocess
        subprocess.run(["pip", "install", "scipy"])
        import scipy.stats

    df, confidence_results, threshold_results = main()