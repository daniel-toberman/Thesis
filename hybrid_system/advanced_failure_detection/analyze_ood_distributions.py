#!/usr/bin/env python3
"""
Histogram-Based Threshold Analysis for OOD Methods

Generates three-histogram visualizations for each OOD method to analyze score
distributions across validation (6cm), test success (error ≤5°), and test fail (error >5°).
Helps identify methods with good separation and inform principled threshold selection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, f1_score

# Import all OOD routing classes
from energy_ood_routing import EnergyOODRouter
from mc_dropout_routing import MCDropoutRouter
from knn_ood_routing import KNNOODRouter
from mahalanobis_ood_routing import MahalanobisOODRouter
from gradnorm_ood_routing import GradNormOODRouter
from react_ood_routing import ReActOODRouter
from vim_ood_routing import VIMOODRouter
from she_ood_routing import SHEOODRouter
from dice_ood_routing import DICEOODRouter
from llr_ood_routing import LLROODRouter
from max_prob_routing import MaxProbRouter


def load_data_splits():
    """
    Load and split data into three datasets:
    - Validation (6cm only)
    - Test success (3x12cm, error ≤ 5°)
    - Test fail (3x12cm, error > 5°)
    """
    print("\nLoading data splits...")

    # Load validation (6cm only)
    val_path = 'features/train_6cm_features.npz'
    val_data = np.load(val_path, allow_pickle=True)
    val_features = {key: val_data[key] for key in val_data.files}
    print(f"  Loaded validation: {len(val_features['abs_errors'])} samples")

    # Load test (3x12cm)
    test_path = 'features/test_3x12cm_consecutive_features.npz'
    test_data = np.load(test_path, allow_pickle=True)
    test_features_full = {key: test_data[key] for key in test_data.files}

    # Split test by success/fail (5° threshold)
    abs_errors = test_features_full['abs_errors']
    success_mask = abs_errors <= 5.0
    fail_mask = abs_errors > 5.0

    # Create success features dict
    test_success_features = {}
    for key in test_features_full.keys():
        test_success_features[key] = test_features_full[key][success_mask]

    # Create fail features dict
    test_fail_features = {}
    for key in test_features_full.keys():
        test_fail_features[key] = test_features_full[key][fail_mask]

    print(f"  Test success: {len(test_success_features['abs_errors'])} samples ({success_mask.sum()/len(abs_errors)*100:.1f}%)")
    print(f"  Test fail: {len(test_fail_features['abs_errors'])} samples ({fail_mask.sum()/len(abs_errors)*100:.1f}%)")

    return val_features, test_success_features, test_fail_features


def compute_method_scores(method_name, features, val_features=None):
    """
    Compute OOD scores for a given method.

    Args:
        method_name: Name of the method (e.g., 'energy', 'vim', 'knn_k10')
        features: Feature dictionary to score
        val_features: Validation features for training (if needed)

    Returns:
        scores: Array of OOD scores
    """
    # Energy OOD
    if method_name == 'energy':
        router = EnergyOODRouter()
        router.temperature = 1.58  # Set temperature directly
        scores = router.compute_energy_scores(features['logits_pre_sig'])

    # MC Dropout Entropy
    elif method_name == 'mc_dropout_entropy':
        router = MCDropoutRouter()
        scores = router.compute_entropy_from_logits(features['logits_pre_sig'])

    # MC Dropout Variance
    elif method_name == 'mc_dropout_variance':
        router = MCDropoutRouter()
        scores = router.compute_variance_from_logits(features['logits_pre_sig'])

    # KNN Distance
    elif method_name.startswith('knn_k'):
        k = int(method_name.split('_k')[1])
        router = KNNOODRouter(k=k)
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_knn_distances(features)

    # Mahalanobis Distance
    elif method_name == 'mahalanobis':
        router = MahalanobisOODRouter()
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_mahalanobis_distances(features)

    # GradNorm
    elif method_name == 'gradnorm':
        router = GradNormOODRouter()
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_gradnorm_scores(features)

    # ReAct
    elif method_name.startswith('react_p'):
        percentile = int(method_name.split('_p')[1])
        router = ReActOODRouter(clip_percentile=percentile)
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_react_scores(features['features'], features['logits_pre_sig'])

    # VIM
    elif method_name == 'vim':
        router = VIMOODRouter()
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_vim_scores(features)

    # SHE
    elif method_name == 'she':
        router = SHEOODRouter()
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_she_scores(features)

    # DICE
    elif method_name.startswith('dice_'):
        sparsity = int(method_name.split('_')[1])
        router = DICEOODRouter(sparsity_percentile=sparsity)
        if val_features is not None:
            router.train(val_features)
        scores = router.compute_dice_scores(features)

    # LLR
    elif method_name.startswith('llr_gmm'):
        n_components = int(method_name.split('gmm')[1])
        router = LLROODRouter(n_components=n_components)
        if val_features is not None:
            # LLR needs combined training features (not just 6cm)
            # Load the combined training features
            train_combined_path = 'features/train_combined_features.npz'
            train_combined_data = np.load(train_combined_path, allow_pickle=True)
            train_combined = {key: train_combined_data[key] for key in train_combined_data.files}
            router.train(train_combined)
        scores = router.compute_llr_scores(features)

    # Max Probability
    elif method_name == 'max_prob':
        router = MaxProbRouter()
        scores = router.compute_max_prob_scores(features)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    return scores


def find_optimal_f1_threshold(success_scores, fail_scores, ascending=True):
    """
    Find threshold that maximizes F1 score for separating success from fail.

    Args:
        success_scores: Scores for successful predictions
        fail_scores: Scores for failed predictions
        ascending: If True, higher score means route (fail). If False, lower score means route.

    Returns:
        best_threshold: Threshold that maximizes F1
        best_f1: Maximum F1 score achieved
    """
    # Combine scores and create labels (0=success, 1=fail)
    all_scores = np.concatenate([success_scores, fail_scores])
    labels = np.concatenate([np.zeros(len(success_scores)), np.ones(len(fail_scores))])

    # Test thresholds at percentiles
    percentiles = np.linspace(1, 99, 99)
    thresholds = [np.percentile(all_scores, p) for p in percentiles]

    best_f1 = 0
    best_threshold = None

    for threshold in thresholds:
        if ascending:
            predictions = (all_scores > threshold).astype(int)  # Higher = fail
        else:
            predictions = (all_scores < threshold).astype(int)  # Lower = fail

        f1 = f1_score(labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def compute_overlap_metrics(val_scores, success_scores, fail_scores):
    """
    Compute distribution overlap metrics.

    Returns:
        dict with:
        - success_fail_separation: Wasserstein distance between success and fail
        - val_success_overlap: Wasserstein distance between validation and success
        - roc_auc: ROC-AUC for discriminating success vs fail
    """
    # Measure separation using Wasserstein distance (Earth Mover's Distance)
    success_fail_separation = wasserstein_distance(success_scores, fail_scores)
    val_success_overlap = wasserstein_distance(val_scores, success_scores)

    # ROC-AUC: Can scores distinguish success from fail?
    labels = np.concatenate([np.zeros(len(success_scores)), np.ones(len(fail_scores))])
    scores_combined = np.concatenate([success_scores, fail_scores])

    try:
        roc_auc = roc_auc_score(labels, scores_combined)
        # If ROC-AUC < 0.5, scores are inverted (lower=fail instead of higher=fail)
        # Take complement to report discrimination ability
        if roc_auc < 0.5:
            roc_auc = 1.0 - roc_auc
    except:
        roc_auc = 0.5  # Default if computation fails

    return {
        'success_fail_separation': success_fail_separation,
        'val_success_overlap': val_success_overlap,
        'roc_auc': roc_auc
    }


def plot_three_histogram(method_name, val_scores, success_scores, fail_scores,
                        threshold_30pct, optimal_threshold, output_dir='results/ood_distributions'):
    """
    Plot three overlapping histograms with thresholds and percentile lines.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Determine bin range to include all distributions
    all_scores = np.concatenate([val_scores, success_scores, fail_scores])
    bins = 50

    # Plot histograms with transparency
    ax.hist(val_scores, bins=bins, alpha=0.5, label='Validation (6cm)', color='blue', density=True)
    ax.hist(success_scores, bins=bins, alpha=0.5, label='Test Success (≤5°)', color='green', density=True)
    ax.hist(fail_scores, bins=bins, alpha=0.5, label='Test Fail (>5°)', color='red', density=True)

    # Add 30% threshold line (from test data)
    ax.axvline(threshold_30pct, color='black', linestyle='--', linewidth=2,
               label=f'30% Threshold (Test): {threshold_30pct:.3f}')

    # Add optimal F1 threshold
    ax.axvline(optimal_threshold, color='purple', linestyle='-', linewidth=2,
               label=f'Optimal F1 Threshold: {optimal_threshold:.3f}')

    # Add percentile lines (from test data combined)
    test_scores_combined = np.concatenate([success_scores, fail_scores])
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(test_scores_combined, p)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.text(val, ax.get_ylim()[1] * 0.95, f'p{p}', fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel(f'{method_name} Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{method_name} Score Distributions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{method_name}_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved histogram: {output_dir}/{method_name}_histogram.png")


def generate_summary_report(all_method_results, output_dir='results/ood_distributions'):
    """
    Generate summary CSV and markdown report ranking methods by discriminative power.
    """
    # Create DataFrame
    df = pd.DataFrame(all_method_results)

    # Sort by ROC-AUC (primary), then by success-fail separation (secondary)
    df = df.sort_values(['roc_auc', 'success_fail_separation'], ascending=[False, False])

    # Save CSV
    csv_path = f'{output_dir}/distribution_analysis_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Summary CSV saved: {csv_path}")

    # Generate markdown report
    md_path = f'{output_dir}/ANALYSIS_REPORT.md'
    with open(md_path, 'w') as f:
        f.write("# OOD Method Distribution Analysis\n\n")
        f.write("## Summary\n\n")
        f.write("This report analyzes score distributions for post-hoc OOD methods across three datasets:\n")
        f.write("- **Validation (6cm)**: In-distribution training data\n")
        f.write("- **Test Success (≤5°)**: Test samples where CRNN performed well\n")
        f.write("- **Test Fail (>5°)**: Test samples where CRNN failed\n\n")
        f.write("Methods are ranked by discriminative power (ROC-AUC and distribution separation).\n\n")

        f.write("## Methods Ranked by Discriminative Power\n\n")
        f.write("**Key Metrics**:\n")
        f.write("- **ROC-AUC**: How well scores discriminate success vs fail (higher is better, >0.7 is good)\n")
        f.write("- **Success-Fail Separation**: Wasserstein distance between distributions (larger is better)\n")
        f.write("- **Val-Success Overlap**: Wasserstein distance between validation and success (smaller is better)\n")
        f.write("- **Optimal F1**: Best F1 score achievable by thresholding\n\n")

        # Create markdown table
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        # Add interpretation
        f.write("## Interpretation\n\n")
        f.write("### Good Methods (Clear Separation)\n")
        good_methods = df[df['roc_auc'] >= 0.7]
        if len(good_methods) > 0:
            for _, row in good_methods.iterrows():
                f.write(f"- **{row['method']}**: ROC-AUC={row['roc_auc']:.3f}, Optimal F1={row['optimal_f1']:.3f}\n")
        else:
            f.write("- None found (all ROC-AUC < 0.7)\n")
        f.write("\n")

        f.write("### Poor Methods (Overlapping Distributions)\n")
        poor_methods = df[df['roc_auc'] < 0.6]
        if len(poor_methods) > 0:
            for _, row in poor_methods.iterrows():
                f.write(f"- **{row['method']}**: ROC-AUC={row['roc_auc']:.3f}, Low discrimination\n")
        else:
            f.write("- None found (all ROC-AUC >= 0.6)\n")
        f.write("\n")

    print(f"✅ Markdown report saved: {md_path}")


def main():
    """Main execution: analyze all post-hoc OOD methods."""

    print("="*80)
    print("HISTOGRAM-BASED THRESHOLD ANALYSIS FOR OOD METHODS")
    print("="*80)

    # ALL post-hoc OOD methods (16 methods total)
    methods = [
        # Best performing methods
        'vim',                       # Virtual-logit matching (13.00° MAE - best post-hoc)
        'she',                       # Stored patterns (13.24° MAE - 2nd best)
        'gradnorm',                  # Gradient norm (13.86° MAE - 3rd best)
        'max_prob',                  # Max probability baseline (13.90° MAE - simple baseline)

        # Good performing methods
        'dice_80',                   # DICE 80% sparsity (14.46° MAE)
        'knn_k10',                   # KNN k=10 (14.73° MAE)

        # Moderate performing methods
        'mc_dropout_entropy',        # MC Dropout Entropy (15.16° MAE)
        'energy',                    # Energy OOD (15.27° MAE)

        # Poor performing methods
        'llr_gmm5',                  # Likelihood ratio (15.34° MAE - shows why density fails)
        'dice_90',                   # DICE 90% sparsity (15.54° MAE - worse than baseline)
        'mahalanobis',               # Mahalanobis distance (17.16° MAE)
        'react_p85',                 # ReAct p85 (17.32° MAE)

        # Additional variants for completeness
        'knn_k5',                    # KNN k=5
        'knn_k20',                   # KNN k=20
        'mc_dropout_variance',       # MC Dropout Variance
        'react_p90',                 # ReAct p90
    ]

    # Create output directory
    output_dir = 'results/ood_distributions'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data splits
    val_features, test_success_features, test_fail_features = load_data_splits()

    results = []

    for i, method in enumerate(methods, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(methods)}] Analyzing: {method}")
        print(f"{'='*80}")

        try:
            # Compute scores on all three datasets
            print("  Computing scores...")
            val_scores = compute_method_scores(method, val_features, val_features)
            success_scores = compute_method_scores(method, test_success_features, val_features)
            fail_scores = compute_method_scores(method, test_fail_features, val_features)

            # Find 30% threshold on TEST data (combined)
            test_scores_combined = np.concatenate([success_scores, fail_scores])
            threshold_30pct = np.percentile(test_scores_combined, 70)

            # Find optimal F1 threshold
            # Auto-detect score direction: if fail scores are higher on average, use ascending
            ascending = fail_scores.mean() > success_scores.mean()
            optimal_threshold, optimal_f1 = find_optimal_f1_threshold(
                success_scores, fail_scores, ascending=ascending
            )

            print(f"  30% threshold: {threshold_30pct:.4f}")
            print(f"  Optimal F1 threshold: {optimal_threshold:.4f} (F1={optimal_f1:.3f})")

            # Plot histograms
            plot_three_histogram(method, val_scores, success_scores, fail_scores,
                                threshold_30pct, optimal_threshold, output_dir)

            # Compute overlap metrics
            metrics = compute_overlap_metrics(val_scores, success_scores, fail_scores)
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"  Success-Fail separation: {metrics['success_fail_separation']:.4f}")
            print(f"  Val-Success overlap: {metrics['val_success_overlap']:.4f}")

            # Store results
            results.append({
                'method': method,
                'val_mean': val_scores.mean(),
                'val_std': val_scores.std(),
                'success_mean': success_scores.mean(),
                'success_std': success_scores.std(),
                'fail_mean': fail_scores.mean(),
                'fail_std': fail_scores.std(),
                'threshold_30pct': threshold_30pct,
                'optimal_threshold': optimal_threshold,
                'optimal_f1': optimal_f1,
                'roc_auc': metrics['roc_auc'],
                'success_fail_separation': metrics['success_fail_separation'],
                'val_success_overlap': metrics['val_success_overlap']
            })

        except Exception as e:
            print(f"  ❌ Error analyzing {method}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    if results:
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}")
        generate_summary_report(results, output_dir)

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Histograms saved to: {output_dir}/")
    print(f"Summary CSV: {output_dir}/distribution_analysis_summary.csv")
    print(f"Summary report: {output_dir}/ANALYSIS_REPORT.md")
    print(f"Total methods analyzed: {len(results)}/{len(methods)}")


if __name__ == '__main__':
    main()
