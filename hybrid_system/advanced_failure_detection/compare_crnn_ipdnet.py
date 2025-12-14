"""
Comparative Analysis: CRNN vs IPDnet Generalization

This script compares OOD routing performance across two deep learning architectures
to validate that routing strategies generalize beyond CRNN.

Analysis includes:
1. Baseline performance comparison (CRNN-only, IPDnet-only, SRP-only)
2. OOD method F1 score comparison across architectures
3. Routing quality metrics (precision, recall, routing rate)
4. Hybrid system performance comparison
5. Generalization analysis (which methods transfer)

Usage:
    python compare_crnn_ipdnet.py \
        --crnn_results results/optimal_thresholds \
        --ipdnet_results results/optimal_thresholds_ipdnet \
        --output_dir results/model_comparison
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results(results_dir):
    """Load OOD evaluation results from directory."""
    results_dir = Path(results_dir)

    # Load CSVs
    try:
        routing_stats = pd.read_csv(results_dir / 'routing_statistics.csv')
        hybrid_results = pd.read_csv(results_dir / 'hybrid_results.csv')
        thresholds = pd.read_csv(results_dir / 'all_methods_optimal_thresholds.csv')

        return {
            'routing': routing_stats,
            'hybrid': hybrid_results,
            'thresholds': thresholds
        }
    except FileNotFoundError as e:
        print(f"Warning: Could not load all result files from {results_dir}: {e}")
        return None


def compare_baseline_performance(crnn_results, ipdnet_results, output_dir):
    """Compare baseline performance of CRNN, IPDnet, and SRP."""
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE COMPARISON")
    print("="*80)

    # Extract baseline MAE from hybrid results (these contain baseline info)
    crnn_hybrid = crnn_results['hybrid']
    ipdnet_hybrid = ipdnet_results['hybrid']

    # Get baseline values (should be consistent across methods)
    # Look for a row to extract baseline values
    crnn_baseline_mae = crnn_hybrid['dl_only_mae'].iloc[0] if 'dl_only_mae' in crnn_hybrid.columns else None
    ipdnet_baseline_mae = ipdnet_hybrid['dl_only_mae'].iloc[0] if 'dl_only_mae' in ipdnet_hybrid.columns else None
    srp_baseline_mae = crnn_hybrid['srp_only_mae'].iloc[0] if 'srp_only_mae' in crnn_hybrid.columns else None

    # Create comparison table
    baseline_data = {
        'Model': ['CRNN-only', 'IPDnet-only', 'SRP-only'],
        'MAE (degrees)': [crnn_baseline_mae, ipdnet_baseline_mae, srp_baseline_mae],
    }

    # If we have success rate info
    if 'dl_only_success' in crnn_hybrid.columns:
        crnn_success = crnn_hybrid['dl_only_success'].iloc[0]
        ipdnet_success = ipdnet_hybrid['dl_only_success'].iloc[0]
        srp_success = crnn_hybrid['srp_only_success'].iloc[0]
        baseline_data['Success Rate (%)'] = [crnn_success, ipdnet_success, srp_success]

    baseline_df = pd.DataFrame(baseline_data)

    print("\nBaseline Performance:")
    print(baseline_df.to_string(index=False))

    # Save
    baseline_df.to_csv(output_dir / 'baseline_comparison.csv', index=False)

    return baseline_df


def compare_f1_scores(crnn_results, ipdnet_results, output_dir):
    """Compare F1 scores across all OOD methods for both models."""
    print("\n" + "="*80)
    print("F1 SCORE COMPARISON")
    print("="*80)

    crnn_routing = crnn_results['routing']
    ipdnet_routing = ipdnet_results['routing']

    # Merge on method name
    comparison = pd.merge(
        crnn_routing[['method', 'f1', 'precision', 'recall', 'routing_rate']],
        ipdnet_routing[['method', 'f1', 'precision', 'recall', 'routing_rate']],
        on='method',
        suffixes=('_crnn', '_ipdnet')
    )

    # Calculate differences
    comparison['f1_diff'] = comparison['f1_ipdnet'] - comparison['f1_crnn']
    comparison['generalization_quality'] = comparison['f1_diff'].abs()

    # Sort by CRNN F1 (descending)
    comparison = comparison.sort_values('f1_crnn', ascending=False)

    print("\nF1 Score Comparison (sorted by CRNN F1):")
    print(comparison[['method', 'f1_crnn', 'f1_ipdnet', 'f1_diff']].to_string(index=False))

    # Statistics
    corr_pearson, p_pearson = pearsonr(comparison['f1_crnn'], comparison['f1_ipdnet'])
    corr_spearman, p_spearman = spearmanr(comparison['f1_crnn'], comparison['f1_ipdnet'])

    print(f"\nCorrelation Statistics:")
    print(f"  Pearson r: {corr_pearson:.3f} (p={p_pearson:.4f})")
    print(f"  Spearman ρ: {corr_spearman:.3f} (p={p_spearman:.4f})")

    # Identify generalization categories
    strong_generalization = comparison[comparison['f1_diff'].abs() <= 0.05]
    weak_generalization = comparison[comparison['f1_diff'].abs() > 0.1]

    print(f"\nGeneralization Categories:")
    print(f"  Strong (|ΔF1| ≤ 0.05): {len(strong_generalization)} methods")
    if len(strong_generalization) > 0:
        print(f"    {', '.join(strong_generalization['method'].tolist())}")

    print(f"  Weak (|ΔF1| > 0.10): {len(weak_generalization)} methods")
    if len(weak_generalization) > 0:
        print(f"    {', '.join(weak_generalization['method'].tolist())}")

    # Save
    comparison.to_csv(output_dir / 'f1_comparison.csv', index=False)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter with method labels
    scatter = ax.scatter(comparison['f1_crnn'], comparison['f1_ipdnet'],
                        s=100, alpha=0.6, c=comparison['generalization_quality'],
                        cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)

    # Add diagonal line (perfect generalization)
    min_f1 = min(comparison['f1_crnn'].min(), comparison['f1_ipdnet'].min())
    max_f1 = max(comparison['f1_crnn'].max(), comparison['f1_ipdnet'].max())
    ax.plot([min_f1, max_f1], [min_f1, max_f1], 'k--', alpha=0.3, label='Perfect Generalization')

    # Add ±0.05 bands
    ax.plot([min_f1, max_f1], [min_f1+0.05, max_f1+0.05], 'gray', alpha=0.2, linestyle=':')
    ax.plot([min_f1, max_f1], [min_f1-0.05, max_f1-0.05], 'gray', alpha=0.2, linestyle=':')

    # Label points
    for idx, row in comparison.iterrows():
        ax.annotate(row['method'],
                   (row['f1_crnn'], row['f1_ipdnet']),
                   fontsize=8, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('CRNN F1 Score', fontsize=12)
    ax.set_ylabel('IPDnet F1 Score', fontsize=12)
    ax.set_title('OOD Method Generalization: CRNN vs IPDnet\n(Points near diagonal generalize well)',
                fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|ΔF1| (lower = better generalization)', fontsize=10)

    # Add correlation text
    ax.text(0.05, 0.95, f'Pearson r = {corr_pearson:.3f}\nSpearman ρ = {corr_spearman:.3f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'f1_score_comparison.pdf', bbox_inches='tight')
    plt.close()

    return comparison


def compare_routing_quality(crnn_results, ipdnet_results, output_dir):
    """Compare routing quality metrics (precision, recall, routing rate)."""
    print("\n" + "="*80)
    print("ROUTING QUALITY COMPARISON")
    print("="*80)

    crnn_routing = crnn_results['routing']
    ipdnet_routing = ipdnet_results['routing']

    # Merge
    comparison = pd.merge(
        crnn_routing[['method', 'precision', 'recall', 'routing_rate']],
        ipdnet_routing[['method', 'precision', 'recall', 'routing_rate']],
        on='method',
        suffixes=('_crnn', '_ipdnet')
    )

    # Calculate differences
    comparison['precision_diff'] = comparison['precision_ipdnet'] - comparison['precision_crnn']
    comparison['recall_diff'] = comparison['recall_ipdnet'] - comparison['recall_crnn']
    comparison['routing_rate_diff'] = comparison['routing_rate_ipdnet'] - comparison['routing_rate_crnn']

    print("\nRouting Quality Metrics:")
    print(comparison[['method', 'precision_crnn', 'precision_ipdnet',
                     'recall_crnn', 'recall_ipdnet',
                     'routing_rate_crnn', 'routing_rate_ipdnet']].to_string(index=False))

    # Save
    comparison.to_csv(output_dir / 'routing_quality_comparison.csv', index=False)

    # Create multi-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['precision', 'recall', 'routing_rate']
    titles = ['Precision', 'Recall', 'Routing Rate']

    for ax, metric, title in zip(axes, metrics, titles):
        crnn_col = f'{metric}_crnn'
        ipdnet_col = f'{metric}_ipdnet'

        ax.scatter(comparison[crnn_col], comparison[ipdnet_col],
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Diagonal
        min_val = min(comparison[crnn_col].min(), comparison[ipdnet_col].min())
        max_val = max(comparison[crnn_col].max(), comparison[ipdnet_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

        # Correlation
        corr, _ = pearsonr(comparison[crnn_col], comparison[ipdnet_col])
        ax.text(0.05, 0.95, f'r = {corr:.3f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'CRNN {title}', fontsize=11)
        ax.set_ylabel(f'IPDnet {title}', fontsize=11)
        ax.set_title(f'{title} Comparison', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'routing_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'routing_quality_comparison.pdf', bbox_inches='tight')
    plt.close()

    return comparison


def compare_hybrid_performance(crnn_results, ipdnet_results, output_dir):
    """Compare hybrid system performance (MAE improvement)."""
    print("\n" + "="*80)
    print("HYBRID PERFORMANCE COMPARISON")
    print("="*80)

    crnn_hybrid = crnn_results['hybrid']
    ipdnet_hybrid = ipdnet_results['hybrid']

    # Merge
    comparison = pd.merge(
        crnn_hybrid[['method', 'hybrid_mae', 'mae_improvement', 'hybrid_success']],
        ipdnet_hybrid[['method', 'hybrid_mae', 'mae_improvement', 'hybrid_success']],
        on='method',
        suffixes=('_crnn', '_ipdnet')
    )

    # Sort by CRNN improvement
    comparison = comparison.sort_values('mae_improvement_crnn', ascending=False)

    print("\nHybrid MAE Improvement:")
    print(comparison[['method', 'hybrid_mae_crnn', 'mae_improvement_crnn',
                     'hybrid_mae_ipdnet', 'mae_improvement_ipdnet']].to_string(index=False))

    # Identify best methods for each model
    crnn_best = comparison.nlargest(3, 'mae_improvement_crnn')[['method', 'mae_improvement_crnn']]
    ipdnet_best = comparison.nlargest(3, 'mae_improvement_ipdnet')[['method', 'mae_improvement_ipdnet']]

    print(f"\nTop 3 Methods:")
    print(f"\nCRNN Best:")
    print(crnn_best.to_string(index=False))
    print(f"\nIPDnet Best:")
    print(ipdnet_best.to_string(index=False))

    # Check if same methods win
    crnn_top_methods = set(crnn_best['method'])
    ipdnet_top_methods = set(ipdnet_best['method'])
    overlap = crnn_top_methods.intersection(ipdnet_top_methods)

    print(f"\nTop-3 Overlap: {len(overlap)} methods")
    if len(overlap) > 0:
        print(f"  {', '.join(overlap)}")

    # Save
    comparison.to_csv(output_dir / 'hybrid_performance_comparison.csv', index=False)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(comparison['mae_improvement_crnn'], comparison['mae_improvement_ipdnet'],
              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Diagonal
    min_imp = min(comparison['mae_improvement_crnn'].min(), comparison['mae_improvement_ipdnet'].min())
    max_imp = max(comparison['mae_improvement_crnn'].max(), comparison['mae_improvement_ipdnet'].max())
    ax.plot([min_imp, max_imp], [min_imp, max_imp], 'k--', alpha=0.3, label='Equal Improvement')

    # Label best methods
    top_methods = crnn_top_methods.union(ipdnet_top_methods)
    for idx, row in comparison.iterrows():
        if row['method'] in top_methods:
            ax.annotate(row['method'],
                       (row['mae_improvement_crnn'], row['mae_improvement_ipdnet']),
                       fontsize=9, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')

    # Correlation
    corr, _ = pearsonr(comparison['mae_improvement_crnn'], comparison['mae_improvement_ipdnet'])
    ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('CRNN MAE Improvement (°)', fontsize=12)
    ax.set_ylabel('IPDnet MAE Improvement (°)', fontsize=12)
    ax.set_title('Hybrid System Performance: MAE Improvement Comparison', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hybrid_mae_comparison.pdf', bbox_inches='tight')
    plt.close()

    return comparison


def generate_report(baseline_df, f1_comparison, routing_comparison, hybrid_comparison,
                   crnn_results, ipdnet_results, output_dir):
    """Generate comprehensive generalization report."""
    print("\n" + "="*80)
    print("GENERATING GENERALIZATION REPORT")
    print("="*80)

    report_path = output_dir / 'generalization_report.md'

    with open(report_path, 'w') as f:
        f.write("# OOD Routing Generalization: CRNN vs IPDnet\n\n")
        f.write("## Executive Summary\n\n")

        # Correlation summary
        f1_corr, _ = pearsonr(f1_comparison['f1_crnn'], f1_comparison['f1_ipdnet'])
        hybrid_corr, _ = pearsonr(hybrid_comparison['mae_improvement_crnn'],
                                  hybrid_comparison['mae_improvement_ipdnet'])

        f.write(f"This analysis validates whether OOD routing strategies generalize ")
        f.write(f"beyond CRNN by comparing performance across two architectures.\n\n")

        f.write(f"**Key Findings:**\n")
        f.write(f"- F1 score correlation: r = {f1_corr:.3f}\n")
        f.write(f"- Hybrid improvement correlation: r = {hybrid_corr:.3f}\n")

        # Generalization quality
        strong_gen = len(f1_comparison[f1_comparison['f1_diff'].abs() <= 0.05])
        weak_gen = len(f1_comparison[f1_comparison['f1_diff'].abs() > 0.1])
        total_methods = len(f1_comparison)

        f.write(f"- Strong generalization (|ΔF1| ≤ 0.05): {strong_gen}/{total_methods} methods\n")
        f.write(f"- Weak generalization (|ΔF1| > 0.10): {weak_gen}/{total_methods} methods\n\n")

        # Interpretation
        if f1_corr > 0.7:
            f.write("**Interpretation:** Strong positive correlation suggests OOD routing ")
            f.write("strategies generalize well across architectures.\n\n")
        elif f1_corr > 0.4:
            f.write("**Interpretation:** Moderate correlation suggests partial generalization. ")
            f.write("Some methods are architecture-specific.\n\n")
        else:
            f.write("**Interpretation:** Weak correlation suggests routing strategies may be ")
            f.write("architecture-dependent.\n\n")

        # Baseline comparison
        f.write("## 1. Baseline Performance\n\n")
        f.write("Comparison of stand-alone model performance:\n\n")
        f.write(baseline_df.to_markdown(index=False))
        f.write("\n\n")

        # F1 comparison
        f.write("## 2. OOD Method F1 Scores\n\n")
        f.write("F1 score comparison for all 19 OOD methods:\n\n")

        f1_table = f1_comparison[['method', 'f1_crnn', 'f1_ipdnet', 'f1_diff']].copy()
        f1_table = f1_table.round(3)
        f.write(f1_table.to_markdown(index=False))
        f.write("\n\n")

        f.write(f"**Statistics:**\n")
        f.write(f"- Pearson correlation: r = {f1_corr:.3f}\n")
        f.write(f"- Mean |ΔF1|: {f1_comparison['f1_diff'].abs().mean():.3f}\n")
        f.write(f"- Median |ΔF1|: {f1_comparison['f1_diff'].abs().median():.3f}\n\n")

        # Best generalizing methods
        best_generalizing = f1_comparison.nsmallest(5, 'generalization_quality')
        f.write("**Best Generalizing Methods (smallest |ΔF1|):**\n")
        for idx, row in best_generalizing.iterrows():
            f.write(f"- {row['method']}: |ΔF1| = {row['generalization_quality']:.3f}\n")
        f.write("\n")

        # Hybrid performance
        f.write("## 3. Hybrid System Performance\n\n")
        f.write("MAE improvement from hybrid routing (DL model + SRP fallback):\n\n")

        hybrid_table = hybrid_comparison[['method', 'hybrid_mae_crnn', 'mae_improvement_crnn',
                                         'hybrid_mae_ipdnet', 'mae_improvement_ipdnet']].copy()
        hybrid_table = hybrid_table.round(2)
        f.write(hybrid_table.to_markdown(index=False))
        f.write("\n\n")

        # Top methods
        f.write("**Top 3 Methods by MAE Improvement:**\n\n")

        crnn_best = hybrid_comparison.nlargest(3, 'mae_improvement_crnn')
        f.write("CRNN:\n")
        for idx, row in crnn_best.iterrows():
            f.write(f"1. {row['method']}: {row['mae_improvement_crnn']:.2f}° improvement "
                   f"(MAE: {row['hybrid_mae_crnn']:.2f}°)\n")
        f.write("\n")

        ipdnet_best = hybrid_comparison.nlargest(3, 'mae_improvement_ipdnet')
        f.write("IPDnet:\n")
        for idx, row in ipdnet_best.iterrows():
            f.write(f"1. {row['method']}: {row['mae_improvement_ipdnet']:.2f}° improvement "
                   f"(MAE: {row['hybrid_mae_ipdnet']:.2f}°)\n")
        f.write("\n")

        # Check overlap
        crnn_top = set(crnn_best['method'])
        ipdnet_top = set(ipdnet_best['method'])
        overlap = crnn_top.intersection(ipdnet_top)

        f.write(f"**Top-3 Overlap:** {len(overlap)} methods")
        if len(overlap) > 0:
            f.write(f" ({', '.join(overlap)})")
        f.write("\n\n")

        # Routing quality
        f.write("## 4. Routing Quality Analysis\n\n")

        routing_table = routing_comparison[['method', 'precision_crnn', 'precision_ipdnet',
                                           'recall_crnn', 'recall_ipdnet',
                                           'routing_rate_crnn', 'routing_rate_ipdnet']].copy()
        routing_table = routing_table.round(3)
        f.write(routing_table.to_markdown(index=False))
        f.write("\n\n")

        # Correlations
        prec_corr, _ = pearsonr(routing_comparison['precision_crnn'],
                               routing_comparison['precision_ipdnet'])
        recall_corr, _ = pearsonr(routing_comparison['recall_crnn'],
                                 routing_comparison['recall_ipdnet'])
        rate_corr, _ = pearsonr(routing_comparison['routing_rate_crnn'],
                               routing_comparison['routing_rate_ipdnet'])

        f.write(f"**Correlations:**\n")
        f.write(f"- Precision: r = {prec_corr:.3f}\n")
        f.write(f"- Recall: r = {recall_corr:.3f}\n")
        f.write(f"- Routing Rate: r = {rate_corr:.3f}\n\n")

        # Conclusions
        f.write("## 5. Conclusions\n\n")

        if f1_corr > 0.7 and hybrid_corr > 0.6:
            f.write("### Strong Generalization ✓\n\n")
            f.write("The OOD routing methods demonstrate strong generalization across ")
            f.write("architectures. Both F1 scores and hybrid improvements are highly correlated ")
            f.write("between CRNN and IPDnet, suggesting that routing strategies are ")
            f.write("**model-agnostic** and transfer effectively.\n\n")
            f.write("**Implication:** The routing framework can be reliably applied to other ")
            f.write("deep learning models for DOA estimation.\n\n")

        elif f1_corr > 0.4 and hybrid_corr > 0.3:
            f.write("### Partial Generalization ~\n\n")
            f.write("The OOD routing methods show moderate generalization. While some correlation ")
            f.write("exists, there is evidence that certain methods are architecture-specific.\n\n")

            # Identify architecture-specific methods
            arch_specific = f1_comparison[f1_comparison['f1_diff'].abs() > 0.1]
            if len(arch_specific) > 0:
                f.write("**Architecture-Specific Methods:**\n")
                for idx, row in arch_specific.iterrows():
                    f.write(f"- {row['method']}: ΔF1 = {row['f1_diff']:+.3f}\n")
                f.write("\n")

            f.write("**Implication:** Distance-based methods (KNN, Mahalanobis) may generalize ")
            f.write("better than representation-based methods (VIM, SHE).\n\n")

        else:
            f.write("### Poor Generalization ✗\n\n")
            f.write("The OOD routing methods do not generalize well across architectures. ")
            f.write("F1 scores and hybrid improvements differ significantly between CRNN and IPDnet.\n\n")
            f.write("**Implication:** Architecture-aware routing strategies may be needed. ")
            f.write("The routing methods may be overfitted to CRNN's specific failure patterns.\n\n")

        # Recommendations
        f.write("## 6. Recommendations\n\n")

        # Find consistently good methods
        good_methods = f1_comparison[
            (f1_comparison['f1_crnn'] > f1_comparison['f1_crnn'].median()) &
            (f1_comparison['f1_ipdnet'] > f1_comparison['f1_ipdnet'].median()) &
            (f1_comparison['f1_diff'].abs() < 0.08)
        ]

        if len(good_methods) > 0:
            f.write("**Recommended Methods (good performance on both models):**\n")
            for idx, row in good_methods.iterrows():
                f.write(f"- **{row['method']}**: CRNN F1={row['f1_crnn']:.3f}, ")
                f.write(f"IPDnet F1={row['f1_ipdnet']:.3f}\n")
            f.write("\n")

        f.write("**Future Work:**\n")
        f.write("- Test on additional architectures (Transformers, GNNs)\n")
        f.write("- Investigate why certain methods don't generalize\n")
        f.write("- Develop architecture-agnostic OOD scores\n")
        f.write("- Analyze feature space differences between CRNN and IPDnet\n\n")

        # Visualization references
        f.write("## 7. Visualizations\n\n")
        f.write("See generated plots:\n")
        f.write("- `f1_score_comparison.png` - F1 score scatter plot\n")
        f.write("- `routing_quality_comparison.png` - Precision/recall/rate comparison\n")
        f.write("- `hybrid_mae_comparison.png` - Hybrid MAE improvement scatter\n")

    print(f"\n✓ Generalization report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Compare CRNN and IPDnet OOD routing results')
    parser.add_argument('--crnn_results', type=str,
                       default='results/optimal_thresholds',
                       help='Directory with CRNN OOD evaluation results')
    parser.add_argument('--ipdnet_results', type=str,
                       default='results/optimal_thresholds_ipdnet',
                       help='Directory with IPDnet OOD evaluation results')
    parser.add_argument('--output_dir', type=str,
                       default='results/model_comparison',
                       help='Output directory for comparison results')
    args = parser.parse_args()

    print("="*80)
    print("CRNN vs IPDnet Generalization Analysis")
    print("="*80)
    print(f"CRNN results: {args.crnn_results}")
    print(f"IPDnet results: {args.ipdnet_results}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\nLoading results...")
    crnn_results = load_results(args.crnn_results)
    ipdnet_results = load_results(args.ipdnet_results)

    if crnn_results is None or ipdnet_results is None:
        print("ERROR: Could not load results from one or both directories")
        print("Make sure you have run the full OOD evaluation pipeline for both models.")
        return

    # Run comparisons
    baseline_df = compare_baseline_performance(crnn_results, ipdnet_results, output_dir)
    f1_comparison = compare_f1_scores(crnn_results, ipdnet_results, output_dir)
    routing_comparison = compare_routing_quality(crnn_results, ipdnet_results, output_dir)
    hybrid_comparison = compare_hybrid_performance(crnn_results, ipdnet_results, output_dir)

    # Generate report
    report_path = generate_report(
        baseline_df, f1_comparison, routing_comparison, hybrid_comparison,
        crnn_results, ipdnet_results, output_dir
    )

    print("\n" + "="*80)
    print("✅ COMPARISON ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - baseline_comparison.csv")
    print(f"  - f1_comparison.csv")
    print(f"  - routing_quality_comparison.csv")
    print(f"  - hybrid_performance_comparison.csv")
    print(f"  - generalization_report.md")
    print(f"  - f1_score_comparison.png/pdf")
    print(f"  - routing_quality_comparison.png/pdf")
    print(f"  - hybrid_mae_comparison.png/pdf")
    print(f"\nRead the report: {report_path}")


if __name__ == "__main__":
    main()
