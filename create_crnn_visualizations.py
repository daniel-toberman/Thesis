#!/usr/bin/env python3
"""
Create visualizations for CRNN confidence metrics and failure analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def create_confidence_visualizations():
    """Create comprehensive visualizations for CRNN confidence analysis."""

    # Load data
    df = pd.read_csv('crnn_predictions/crnn_predictions_with_confidence_clean.csv')
    df['is_failure'] = df['abs_error'] > 30
    failures = df[df['is_failure']]
    successes = df[~df['is_failure']]

    print(f'Creating visualizations for {len(df)} predictions...')
    print(f'Failures: {len(failures)} ({len(failures)/len(df)*100:.1f}%)')
    print(f'Successes: {len(successes)} ({len(successes)/len(df)*100:.1f}%)')

    # Create output directory
    output_dir = Path('confidence_analysis')
    output_dir.mkdir(exist_ok=True)

    confidence_metrics = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']

    # 1. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(confidence_metrics):
        ax = axes[i]

        # Plot histograms
        ax.hist(successes[metric], bins=40, alpha=0.7, label=f'Success (n={len(successes)})',
               color='green', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(failures[metric], bins=40, alpha=0.7, label=f'Failure (n={len(failures)})',
               color='red', density=True, edgecolor='black', linewidth=0.5)

        # Add statistics
        success_mean = successes[metric].mean()
        failure_mean = failures[metric].mean()
        ax.axvline(success_mean, color='darkgreen', linestyle='--', alpha=0.9, linewidth=2,
                  label=f'Success μ={success_mean:.3f}')
        ax.axvline(failure_mean, color='darkred', linestyle='--', alpha=0.9, linewidth=2,
                  label=f'Failure μ={failure_mean:.3f}')

        # Statistical test
        t_stat, p_value = stats.ttest_ind(failures[metric], successes[metric])
        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()}\n(p={p_value:.2e} {significance})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved: confidence_distributions.png')

    # 2. Box plots
    fig, axes = plt.subplots(1, len(confidence_metrics), figsize=(20, 6))

    for i, metric in enumerate(confidence_metrics):
        ax = axes[i]

        # Create box plot data
        box_data = [successes[metric], failures[metric]]
        box_labels = ['Success', 'Failure']

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                       showmeans=True, meanline=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        bp['boxes'][0].set_edgecolor('green')
        bp['boxes'][1].set_edgecolor('red')

        # Add statistical info
        success_mean = successes[metric].mean()
        failure_mean = failures[metric].mean()
        difference = failure_mean - success_mean

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()}\nΔ = {difference:.3f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved: confidence_boxplots.png')

    # 3. Error vs Confidence Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(confidence_metrics):
        ax = axes[i]

        # Create scatter plot
        scatter = ax.scatter(df[metric], df['abs_error'], c=df['is_failure'],
                           cmap='RdYlGn_r', alpha=0.6, s=20, edgecolors='black', linewidth=0.3)

        # Add failure threshold line
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Failure threshold (30°)')

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Absolute Error (degrees)', fontsize=12)
        ax.set_title(f'Error vs {metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Failure (1) / Success (0)')

    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved: error_vs_confidence.png')

    # 4. Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_cols = confidence_metrics + ['abs_error', 'is_failure']
    corr_matrix = df[correlation_cols].corr()

    mask = np.triu(np.ones_like(corr_matrix))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                mask=mask, square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix: Confidence Metrics vs Failures', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved: correlation_matrix.png')

    # 5. Summary statistics table visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create summary table
    summary_data = []
    for metric in confidence_metrics:
        success_mean = successes[metric].mean()
        failure_mean = failures[metric].mean()
        success_std = successes[metric].std()
        failure_std = failures[metric].std()
        difference = failure_mean - success_mean
        t_stat, p_value = stats.ttest_ind(failures[metric], successes[metric])

        summary_data.append([
            metric.replace('_', ' ').title(),
            f'{success_mean:.4f} ± {success_std:.4f}',
            f'{failure_mean:.4f} ± {failure_std:.4f}',
            f'{difference:.4f}',
            f'{p_value:.2e}'
        ])

    columns = ['Metric', 'Success (μ ± σ)', 'Failure (μ ± σ)', 'Difference', 'P-value']
    table = ax.table(cellText=summary_data, colLabels=columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('CRNN Confidence Metrics: Statistical Comparison\nSuccess (≤30°) vs Failure (>30°)',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved: summary_statistics.png')

    # 6. ROC-like analysis for failure detection
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Best thresholds analysis
    ax1 = axes[0]
    best_metrics = {}

    for metric in confidence_metrics:
        metric_values = df[metric].values
        thresholds = np.percentile(metric_values, np.arange(10, 91, 5))

        f1_scores = []
        precisions = []
        recalls = []

        for threshold in thresholds:
            if metric in ['entropy']:  # Higher entropy = more uncertainty
                predicted_failures = df[metric] > threshold
            else:  # Lower values = more uncertainty
                predicted_failures = df[metric] < threshold

            actual_failures = df['is_failure']

            tp = (predicted_failures & actual_failures).sum()
            fp = (predicted_failures & ~actual_failures).sum()
            fn = (~predicted_failures & actual_failures).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        # Plot F1 scores
        ax1.plot(thresholds, f1_scores, marker='o', label=f'{metric.replace("_", " ").title()}')

        # Store best performance
        best_idx = np.argmax(f1_scores)
        best_metrics[metric] = {
            'threshold': thresholds[best_idx],
            'f1': f1_scores[best_idx],
            'precision': precisions[best_idx],
            'recall': recalls[best_idx]
        }

    ax1.set_xlabel('Threshold Percentile', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score vs Threshold for Failure Detection', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Performance comparison bar chart
    ax2 = axes[1]
    metrics_list = list(best_metrics.keys())
    f1_scores = [best_metrics[m]['f1'] for m in metrics_list]
    precisions = [best_metrics[m]['precision'] for m in metrics_list]
    recalls = [best_metrics[m]['recall'] for m in metrics_list]

    x = np.arange(len(metrics_list))
    width = 0.25

    ax2.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8)
    ax2.bar(x, precisions, width, label='Precision', alpha=0.8)
    ax2.bar(x + width, recalls, width, label='Recall', alpha=0.8)

    ax2.set_xlabel('Confidence Metrics', fontsize=12)
    ax2.set_ylabel('Performance Score', fontsize=12)
    ax2.set_title('Best Performance by Metric', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics_list], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Saved: threshold_analysis.png')

    # Print summary
    print(f'\n{"="*60}')
    print("CRNN CONFIDENCE ANALYSIS SUMMARY")
    print(f'{"="*60}')
    print(f'Total predictions: {len(df)}')
    print(f'Failures (>30°): {len(failures)} ({len(failures)/len(df)*100:.1f}%)')
    print(f'Successes (≤30°): {len(successes)} ({len(successes)/len(df)*100:.1f}%)')
    print()

    print("BEST FAILURE DETECTION PERFORMANCE:")
    for metric in sorted(best_metrics.keys(), key=lambda x: best_metrics[x]['f1'], reverse=True):
        info = best_metrics[metric]
        print(f'{metric.upper():<20}: F1={info["f1"]:.3f}, P={info["precision"]:.3f}, R={info["recall"]:.3f}')

    print(f'\nAll visualizations saved to: {output_dir.absolute()}')

    return best_metrics

if __name__ == "__main__":
    best_metrics = create_confidence_visualizations()