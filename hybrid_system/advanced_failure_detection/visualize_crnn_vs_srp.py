#!/usr/bin/env python3
"""
Visualize CRNN vs SRP Performance Complementarity

Creates multiple plots showing where CRNN and SRP perform well/poorly
to understand routing opportunities.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data():
    """Load CRNN and SRP results."""
    # Load test features
    test_data = np.load('features/test_3x12cm_consecutive_features.npz', allow_pickle=True)
    crnn_preds = test_data['predicted_angles']
    gt = test_data['gt_angles']

    # Load SRP results
    with open('features/test_3x12cm_srp_results.pkl', 'rb') as f:
        srp_results = pickle.load(f)
    srp_preds = srp_results['srp_pred'].values

    # Compute errors with wraparound
    crnn_errors = np.abs(crnn_preds - gt)
    crnn_errors = np.minimum(crnn_errors, 360 - crnn_errors)

    srp_errors = np.abs(srp_preds - gt)
    srp_errors = np.minimum(srp_errors, 360 - srp_errors)

    return {
        'crnn_preds': crnn_preds,
        'srp_preds': srp_preds,
        'gt': gt,
        'crnn_errors': crnn_errors,
        'srp_errors': srp_errors
    }


def plot_scatter_with_quadrants(data, output_dir):
    """
    Scatter plot: CRNN error vs SRP error with quadrants.

    Quadrants:
    - Q1 (bottom-left): Both good (≤5°) - no routing needed
    - Q2 (bottom-right): CRNN bad, SRP good - IDEAL routing target!
    - Q3 (top-right): Both bad (>5°) - routing helps but both struggle
    - Q4 (top-left): CRNN good, SRP bad - avoid routing
    """
    crnn_errors = data['crnn_errors']
    srp_errors = data['srp_errors']

    # Classify into quadrants
    threshold = 5.0
    q1 = (crnn_errors <= threshold) & (srp_errors <= threshold)  # Both good
    q2 = (crnn_errors > threshold) & (srp_errors <= threshold)   # CRNN bad, SRP good (ROUTING TARGET!)
    q3 = (crnn_errors > threshold) & (srp_errors > threshold)    # Both bad
    q4 = (crnn_errors <= threshold) & (srp_errors > threshold)   # CRNN good, SRP bad

    # Count quadrants
    counts = {
        'Q1: Both Good': q1.sum(),
        'Q2: Route Here!': q2.sum(),
        'Q3: Both Struggle': q3.sum(),
        'Q4: CRNN Better': q4.sum()
    }

    print("\n" + "="*80)
    print("QUADRANT ANALYSIS")
    print("="*80)
    for label, count in counts.items():
        pct = count / len(crnn_errors) * 100
        print(f"{label:20s}: {count:4d} samples ({pct:5.1f}%)")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot quadrants with different colors
    alpha = 0.6
    size = 20

    ax.scatter(crnn_errors[q1], srp_errors[q1], c='green', alpha=alpha, s=size,
               label=f'Q1: Both Good ({counts["Q1: Both Good"]})', marker='o')
    ax.scatter(crnn_errors[q2], srp_errors[q2], c='blue', alpha=alpha, s=size,
               label=f'Q2: Route Here! ({counts["Q2: Route Here!"]})', marker='^')
    ax.scatter(crnn_errors[q3], srp_errors[q3], c='red', alpha=alpha, s=size,
               label=f'Q3: Both Struggle ({counts["Q3: Both Struggle"]})', marker='x')
    ax.scatter(crnn_errors[q4], srp_errors[q4], c='orange', alpha=alpha, s=size,
               label=f'Q4: CRNN Better ({counts["Q4: CRNN Better"]})', marker='s')

    # Add threshold lines
    ax.axhline(threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='5° Threshold')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add diagonal (where errors are equal)
    max_val = max(crnn_errors.max(), srp_errors.max())
    ax.plot([0, max_val], [0, max_val], 'k:', linewidth=2, alpha=0.5, label='Equal Error')

    # Labels and formatting
    ax.set_xlabel('CRNN Error (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('SRP Error (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('CRNN vs SRP Error Complementarity\n(Quadrant Analysis)',
                 fontsize=16, fontweight='bold')

    # Add quadrant labels
    max_plot = min(max_val, 100)  # Limit axis for readability
    ax.text(threshold/2, threshold/2, 'Q1\nBoth\nGood',
            fontsize=12, ha='center', va='center', alpha=0.5, fontweight='bold')
    ax.text(max_plot*0.7, threshold/2, 'Q2\nRoute\nHere!',
            fontsize=12, ha='center', va='center', alpha=0.5, fontweight='bold', color='blue')
    ax.text(max_plot*0.7, max_plot*0.7, 'Q3\nBoth\nStruggle',
            fontsize=12, ha='center', va='center', alpha=0.5, fontweight='bold')
    ax.text(threshold/2, max_plot*0.7, 'Q4\nCRNN\nBetter',
            fontsize=12, ha='center', va='center', alpha=0.5, fontweight='bold')

    ax.set_xlim(0, max_plot)
    ax.set_ylim(0, max_plot)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'crnn_vs_srp_quadrants.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {output_path}")

    return counts


def plot_error_difference_histogram(data, output_dir):
    """
    Histogram of error differences: (SRP error - CRNN error)

    Negative = SRP better (routing helps)
    Positive = CRNN better (avoid routing)
    """
    crnn_errors = data['crnn_errors']
    srp_errors = data['srp_errors']

    error_diff = srp_errors - crnn_errors

    fig, ax = plt.subplots(figsize=(14, 6))

    # Histogram
    bins = np.linspace(-80, 80, 81)
    counts, bins, patches = ax.hist(error_diff, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Color bars based on sign
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('blue')  # SRP better
        else:
            patch.set_facecolor('red')   # CRNN better

    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Equal Error')

    # Add mean and median
    mean_diff = error_diff.mean()
    median_diff = np.median(error_diff)
    ax.axvline(mean_diff, color='darkblue', linestyle='--', linewidth=2,
               label=f'Mean: {mean_diff:.2f}°')
    ax.axvline(median_diff, color='darkgreen', linestyle='--', linewidth=2,
               label=f'Median: {median_diff:.2f}°')

    # Statistics
    srp_better = (error_diff < 0).sum()
    crnn_better = (error_diff > 0).sum()
    equal = (error_diff == 0).sum()

    stats_text = (
        f"SRP Better: {srp_better} ({srp_better/len(error_diff)*100:.1f}%)\n"
        f"CRNN Better: {crnn_better} ({crnn_better/len(error_diff)*100:.1f}%)\n"
        f"Equal: {equal} ({equal/len(error_diff)*100:.1f}%)\n"
        f"Mean Δ: {mean_diff:.2f}°\n"
        f"Median Δ: {median_diff:.2f}°"
    )

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Error Difference: SRP - CRNN (degrees)\n← SRP Better | CRNN Better →',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('When Does Routing to SRP Help?\n(Negative = SRP Better, Positive = CRNN Better)',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'error_difference_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

    print("\n" + "="*80)
    print("ERROR DIFFERENCE ANALYSIS")
    print("="*80)
    print(f"SRP performs better: {srp_better} samples ({srp_better/len(error_diff)*100:.1f}%)")
    print(f"CRNN performs better: {crnn_better} samples ({crnn_better/len(error_diff)*100:.1f}%)")
    print(f"Mean difference: {mean_diff:.2f}° (positive = CRNN better)")
    print(f"Median difference: {median_diff:.2f}°")


def plot_performance_by_angle(data, output_dir):
    """
    Compare CRNN vs SRP performance across different ground truth angles.
    Shows if certain angles benefit more from routing.
    """
    gt = data['gt']
    crnn_errors = data['crnn_errors']
    srp_errors = data['srp_errors']

    # Create angle bins (every 30 degrees)
    angle_bins = np.arange(0, 361, 30)
    angle_labels = [f"{i}-{i+30}°" for i in angle_bins[:-1]]

    # Digitize angles
    angle_indices = np.digitize(gt, angle_bins) - 1

    # Compute mean errors per bin
    crnn_means = []
    srp_means = []
    counts = []

    for i in range(len(angle_bins) - 1):
        mask = angle_indices == i
        if mask.sum() > 0:
            crnn_means.append(crnn_errors[mask].mean())
            srp_means.append(srp_errors[mask].mean())
            counts.append(mask.sum())
        else:
            crnn_means.append(0)
            srp_means.append(0)
            counts.append(0)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    x = np.arange(len(angle_labels))
    width = 0.35

    # Bar plot
    bars1 = ax1.bar(x - width/2, crnn_means, width, label='CRNN', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, srp_means, width, label='SRP', alpha=0.8, color='coral')

    # Add 5° threshold line
    ax1.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5° Threshold')

    ax1.set_xlabel('Ground Truth Angle Range', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error (degrees)', fontsize=14, fontweight='bold')
    ax1.set_title('CRNN vs SRP Performance by Angle Range', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(angle_labels, rotation=45, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Sample count subplot
    ax2.bar(x, counts, alpha=0.6, color='gray')
    ax2.set_xlabel('Ground Truth Angle Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(angle_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'performance_by_angle.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_2d_density_heatmap(data, output_dir):
    """
    2D histogram/heatmap showing density of samples in CRNN vs SRP error space.
    """
    crnn_errors = data['crnn_errors']
    srp_errors = data['srp_errors']

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create 2D histogram
    bins = 50
    max_val = min(max(crnn_errors.max(), srp_errors.max()), 100)

    h = ax.hist2d(crnn_errors, srp_errors, bins=bins,
                  range=[[0, max_val], [0, max_val]],
                  cmap='YlOrRd', cmin=1)

    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Number of Samples', fontsize=12, fontweight='bold')

    # Add threshold lines
    threshold = 5.0
    ax.axhline(threshold, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, alpha=0.7)

    # Add diagonal
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='Equal Error')

    ax.set_xlabel('CRNN Error (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('SRP Error (degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Sample Density: CRNN vs SRP Error\n(Darker = More Samples)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'error_density_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def generate_summary_stats(data, quadrant_counts):
    """Generate summary statistics."""
    crnn_errors = data['crnn_errors']
    srp_errors = data['srp_errors']

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nCRNN Performance:")
    print(f"  Mean Error: {crnn_errors.mean():.2f}°")
    print(f"  Median Error: {np.median(crnn_errors):.2f}°")
    print(f"  Success Rate (≤5°): {(crnn_errors <= 5).sum()} / {len(crnn_errors)} ({(crnn_errors <= 5).mean()*100:.1f}%)")

    print("\nSRP Performance:")
    print(f"  Mean Error: {srp_errors.mean():.2f}°")
    print(f"  Median Error: {np.median(srp_errors):.2f}°")
    print(f"  Success Rate (≤5°): {(srp_errors <= 5).sum()} / {len(srp_errors)} ({(srp_errors <= 5).mean()*100:.1f}%)")

    print("\nRouting Opportunities:")
    print(f"  Ideal routing targets (Q2): {quadrant_counts['Q2: Route Here!']} ({quadrant_counts['Q2: Route Here!']/len(crnn_errors)*100:.1f}%)")
    print(f"  Cases where routing helps (SRP < CRNN): {(srp_errors < crnn_errors).sum()} ({(srp_errors < crnn_errors).mean()*100:.1f}%)")
    print(f"  Cases where routing hurts (SRP > CRNN): {(srp_errors > crnn_errors).sum()} ({(srp_errors > crnn_errors).mean()*100:.1f}%)")


def main():
    """Main execution."""
    print("="*80)
    print("CRNN vs SRP PERFORMANCE VISUALIZATION")
    print("="*80)

    # Create output directory
    output_dir = Path('results/crnn_vs_srp_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = load_data()
    print(f"  Total samples: {len(data['crnn_errors'])}")

    # Generate plots
    print("\nGenerating visualizations...")
    quadrant_counts = plot_scatter_with_quadrants(data, output_dir)
    plot_error_difference_histogram(data, output_dir)
    plot_performance_by_angle(data, output_dir)
    plot_2d_density_heatmap(data, output_dir)

    # Generate summary
    generate_summary_stats(data, quadrant_counts)

    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("\nGenerated plots:")
    print("  1. crnn_vs_srp_quadrants.png - Scatter plot with quadrant analysis")
    print("  2. error_difference_histogram.png - When routing helps/hurts")
    print("  3. performance_by_angle.png - Angle-specific performance comparison")
    print("  4. error_density_heatmap.png - 2D density visualization")


if __name__ == '__main__':
    main()
