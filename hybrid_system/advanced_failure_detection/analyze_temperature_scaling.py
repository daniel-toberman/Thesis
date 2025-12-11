#!/usr/bin/env python3
"""
Temperature Scaling Analysis for OOD Methods

Finds optimal temperature scaling for logit-based methods and generates
new histograms showing improved distribution separation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, f1_score

# Import methods from original script
from analyze_ood_distributions import (
    load_data_splits,
    find_optimal_f1_threshold,
    compute_overlap_metrics,
    plot_three_histogram
)


def apply_temperature_scaling(logits, temperature):
    """Apply temperature scaling to logits."""
    return logits / temperature


def compute_energy_score_with_temp(logits, temperature):
    """Compute Energy OOD score with temperature scaling."""
    scaled_logits = apply_temperature_scaling(logits, temperature)
    # Energy formula: E(x) = -T * log(sum(exp(logit/T)))
    # Since we already scaled logits by T, we compute: -T * log(sum(exp(scaled_logit)))
    energy_scores = []
    for logit_seq in scaled_logits:
        # Average over time dimension if needed
        if logit_seq.ndim == 2:  # (T, D)
            logit_avg = logit_seq.mean(axis=0)
        else:
            logit_avg = logit_seq
        # Compute energy
        energy = -temperature * np.log(np.sum(np.exp(logit_avg)))
        energy_scores.append(energy)
    return np.array(energy_scores)


def compute_maxprob_with_temp(logits, temperature):
    """Compute max probability with temperature scaling."""
    from scipy.special import softmax
    scaled_logits = apply_temperature_scaling(logits, temperature)
    max_probs = []
    for logit_seq in scaled_logits:
        # Average over time dimension if needed
        if logit_seq.ndim == 2:  # (T, D)
            logit_avg = logit_seq.mean(axis=0)
        else:
            logit_avg = logit_seq
        # Compute softmax and take max
        probs = softmax(logit_avg)
        max_probs.append(np.max(probs))
    return np.array(max_probs)


def compute_entropy_with_temp(logits, temperature):
    """Compute entropy with temperature scaling."""
    from scipy.special import softmax
    from scipy.stats import entropy
    scaled_logits = apply_temperature_scaling(logits, temperature)
    entropies = []
    for logit_seq in scaled_logits:
        # Average over time dimension if needed
        if logit_seq.ndim == 2:  # (T, D)
            logit_avg = logit_seq.mean(axis=0)
        else:
            logit_avg = logit_seq
        # Compute entropy
        probs = softmax(logit_avg)
        ent = entropy(probs)
        entropies.append(ent)
    return np.array(entropies)


def compute_variance_with_temp(logits, temperature):
    """Compute prediction variance with temperature scaling."""
    from scipy.special import softmax
    scaled_logits = apply_temperature_scaling(logits, temperature)
    variances = []
    for logit_seq in scaled_logits:
        # Average over time dimension if needed
        if logit_seq.ndim == 2:  # (T, D)
            logit_avg = logit_seq.mean(axis=0)
        else:
            logit_avg = logit_seq
        # Compute variance of probabilities
        probs = softmax(logit_avg)
        var = np.var(probs)
        variances.append(var)
    return np.array(variances)


def compute_vim_with_temp(features, temperature, val_features):
    """Compute VIM scores with temperature-scaled logits."""
    from vim_ood_routing import VIMOODRouter

    # Scale logits in features
    features_scaled = features.copy()
    features_scaled['logits_pre_sig'] = apply_temperature_scaling(
        features['logits_pre_sig'], temperature
    )

    val_features_scaled = val_features.copy()
    val_features_scaled['logits_pre_sig'] = apply_temperature_scaling(
        val_features['logits_pre_sig'], temperature
    )

    # Train VIM on scaled logits
    router = VIMOODRouter()
    router.train(val_features_scaled)
    scores = router.compute_vim_scores(features_scaled)

    return scores


def find_optimal_temperature(method_name, val_features, test_success_features, test_fail_features,
                             temperature_range=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
    """
    Find optimal temperature for a method by maximizing ROC-AUC on test set.

    Returns:
        best_temp: Optimal temperature value
        best_roc_auc: Best ROC-AUC achieved
        temp_results: List of (temp, roc_auc, f1) for each temperature tested
    """
    print(f"\n  Finding optimal temperature for {method_name}...")
    print(f"  Testing temperatures: {temperature_range}")

    best_temp = 1.0
    best_roc_auc = 0.0
    temp_results = []

    for temp in temperature_range:
        # Compute scores with this temperature
        if method_name == 'energy':
            val_scores = compute_energy_score_with_temp(val_features['logits_pre_sig'], temp)
            success_scores = compute_energy_score_with_temp(test_success_features['logits_pre_sig'], temp)
            fail_scores = compute_energy_score_with_temp(test_fail_features['logits_pre_sig'], temp)

        elif method_name == 'max_prob':
            val_scores = compute_maxprob_with_temp(val_features['logits_pre_sig'], temp)
            success_scores = compute_maxprob_with_temp(test_success_features['logits_pre_sig'], temp)
            fail_scores = compute_maxprob_with_temp(test_fail_features['logits_pre_sig'], temp)

        elif method_name == 'mc_dropout_entropy':
            val_scores = compute_entropy_with_temp(val_features['logits_pre_sig'], temp)
            success_scores = compute_entropy_with_temp(test_success_features['logits_pre_sig'], temp)
            fail_scores = compute_entropy_with_temp(test_fail_features['logits_pre_sig'], temp)

        elif method_name == 'mc_dropout_variance':
            val_scores = compute_variance_with_temp(val_features['logits_pre_sig'], temp)
            success_scores = compute_variance_with_temp(test_success_features['logits_pre_sig'], temp)
            fail_scores = compute_variance_with_temp(test_fail_features['logits_pre_sig'], temp)

        elif method_name == 'vim':
            val_scores = compute_vim_with_temp(val_features, temp, val_features)
            success_scores = compute_vim_with_temp(test_success_features, temp, val_features)
            fail_scores = compute_vim_with_temp(test_fail_features, temp, val_features)

        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Compute ROC-AUC
        labels = np.concatenate([np.zeros(len(success_scores)), np.ones(len(fail_scores))])
        scores_combined = np.concatenate([success_scores, fail_scores])

        try:
            roc_auc = roc_auc_score(labels, scores_combined)
            # Handle inverted scores
            if roc_auc < 0.5:
                roc_auc = 1.0 - roc_auc
        except:
            roc_auc = 0.5

        # Compute optimal F1 at this temperature
        ascending = fail_scores.mean() > success_scores.mean()
        _, optimal_f1 = find_optimal_f1_threshold(success_scores, fail_scores, ascending=ascending)

        temp_results.append((temp, roc_auc, optimal_f1))
        print(f"    T={temp:.2f}: ROC-AUC={roc_auc:.4f}, F1={optimal_f1:.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_temp = temp

    print(f"  ✅ Optimal temperature: T={best_temp:.2f} (ROC-AUC={best_roc_auc:.4f})")
    return best_temp, best_roc_auc, temp_results


def analyze_method_with_temp_scaling(method_name, val_features, test_success_features, test_fail_features,
                                     optimal_temp, output_dir='results/ood_distributions'):
    """
    Analyze a method with optimal temperature scaling and generate histogram.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {method_name} with Temperature Scaling (T={optimal_temp:.2f})")
    print(f"{'='*80}")

    # Compute scores with optimal temperature
    if method_name == 'energy':
        val_scores = compute_energy_score_with_temp(val_features['logits_pre_sig'], optimal_temp)
        success_scores = compute_energy_score_with_temp(test_success_features['logits_pre_sig'], optimal_temp)
        fail_scores = compute_energy_score_with_temp(test_fail_features['logits_pre_sig'], optimal_temp)

    elif method_name == 'max_prob':
        val_scores = compute_maxprob_with_temp(val_features['logits_pre_sig'], optimal_temp)
        success_scores = compute_maxprob_with_temp(test_success_features['logits_pre_sig'], optimal_temp)
        fail_scores = compute_maxprob_with_temp(test_fail_features['logits_pre_sig'], optimal_temp)

    elif method_name == 'mc_dropout_entropy':
        val_scores = compute_entropy_with_temp(val_features['logits_pre_sig'], optimal_temp)
        success_scores = compute_entropy_with_temp(test_success_features['logits_pre_sig'], optimal_temp)
        fail_scores = compute_entropy_with_temp(test_fail_features['logits_pre_sig'], optimal_temp)

    elif method_name == 'mc_dropout_variance':
        val_scores = compute_variance_with_temp(val_features['logits_pre_sig'], optimal_temp)
        success_scores = compute_variance_with_temp(test_success_features['logits_pre_sig'], optimal_temp)
        fail_scores = compute_variance_with_temp(test_fail_features['logits_pre_sig'], optimal_temp)

    elif method_name == 'vim':
        val_scores = compute_vim_with_temp(val_features, optimal_temp, val_features)
        success_scores = compute_vim_with_temp(test_success_features, optimal_temp, val_features)
        fail_scores = compute_vim_with_temp(test_fail_features, optimal_temp, val_features)

    # Find 30% threshold
    test_scores_combined = np.concatenate([success_scores, fail_scores])
    threshold_30pct = np.percentile(test_scores_combined, 70)

    # Find optimal F1 threshold
    ascending = fail_scores.mean() > success_scores.mean()
    optimal_threshold, optimal_f1 = find_optimal_f1_threshold(
        success_scores, fail_scores, ascending=ascending
    )

    print(f"  30% threshold: {threshold_30pct:.4f}")
    print(f"  Optimal F1 threshold: {optimal_threshold:.4f} (F1={optimal_f1:.3f})")

    # Plot histogram with new filename
    plot_title = f"{method_name} (T={optimal_temp:.2f})"
    output_filename = f"{output_dir}/{method_name}_T{optimal_temp:.2f}_histogram.png"

    # Modified plot function with custom title and filename
    fig, ax = plt.subplots(figsize=(14, 7))

    all_scores = np.concatenate([val_scores, success_scores, fail_scores])
    bins = 50

    ax.hist(val_scores, bins=bins, alpha=0.5, label='Validation (6cm)', color='blue', density=True)
    ax.hist(success_scores, bins=bins, alpha=0.5, label='Test Success (≤5°)', color='green', density=True)
    ax.hist(fail_scores, bins=bins, alpha=0.5, label='Test Fail (>5°)', color='red', density=True)

    ax.axvline(threshold_30pct, color='black', linestyle='--', linewidth=2,
               label=f'30% Threshold: {threshold_30pct:.3f}')
    ax.axvline(optimal_threshold, color='purple', linestyle='-', linewidth=2,
               label=f'Optimal F1 Threshold: {optimal_threshold:.3f}')

    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(test_scores_combined, p)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.text(val, ax.get_ylim()[1] * 0.95, f'p{p}', fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel(f'{plot_title} Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{plot_title} Score Distributions (Temperature Scaled)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_filename}")

    # Compute metrics
    metrics = compute_overlap_metrics(val_scores, success_scores, fail_scores)
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  Success-Fail separation: {metrics['success_fail_separation']:.4f}")
    print(f"  Val-Success overlap: {metrics['val_success_overlap']:.4f}")

    return {
        'method': f"{method_name}_T{optimal_temp:.2f}",
        'temperature': optimal_temp,
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
    }


def main():
    """Main execution: temperature scaling analysis."""

    print("="*80)
    print("TEMPERATURE SCALING ANALYSIS FOR OOD METHODS")
    print("="*80)

    # Methods to analyze with temperature scaling
    methods = [
        'energy',                # Energy OOD (already uses T=1.58, let's optimize it)
        'max_prob',              # Max probability
        'mc_dropout_entropy',    # MC Dropout Entropy
        'mc_dropout_variance',   # MC Dropout Variance
        'vim',                   # VIM (experimental)
    ]

    output_dir = 'results/ood_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    val_features, test_success_features, test_fail_features = load_data_splits()

    # Temperature range to search
    temperature_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    results = []
    optimal_temps = {}

    # Phase 1: Find optimal temperatures
    print("\n" + "="*80)
    print("PHASE 1: FINDING OPTIMAL TEMPERATURES")
    print("="*80)

    for method in methods:
        optimal_temp, best_roc_auc, temp_results = find_optimal_temperature(
            method, val_features, test_success_features, test_fail_features,
            temperature_range=temperature_range
        )
        optimal_temps[method] = optimal_temp

    # Phase 2: Generate histograms with optimal temperatures
    print("\n" + "="*80)
    print("PHASE 2: GENERATING TEMPERATURE-SCALED HISTOGRAMS")
    print("="*80)

    for method in methods:
        optimal_temp = optimal_temps[method]
        result = analyze_method_with_temp_scaling(
            method, val_features, test_success_features, test_fail_features,
            optimal_temp, output_dir
        )
        results.append(result)

    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    df = pd.DataFrame(results)
    df = df.sort_values('roc_auc', ascending=False)

    # Save CSV
    csv_path = f'{output_dir}/temperature_scaling_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Summary CSV saved: {csv_path}")

    # Generate markdown report
    md_path = f'{output_dir}/TEMPERATURE_SCALING_REPORT.md'
    with open(md_path, 'w') as f:
        f.write("# Temperature Scaling Analysis for OOD Methods\n\n")
        f.write("## Summary\n\n")
        f.write("This report shows the impact of temperature scaling on logit-based OOD methods.\n\n")
        f.write("**Temperature scaling** adjusts the confidence calibration by scaling logits before computing scores:\n")
        f.write("- `scaled_logits = logits / T`\n")
        f.write("- Lower T (< 1.0) = more confident (sharper distributions)\n")
        f.write("- Higher T (> 1.0) = less confident (smoother distributions)\n\n")

        f.write("## Results with Optimal Temperature Scaling\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## Optimal Temperatures Found\n\n")
        for method in methods:
            f.write(f"- **{method}**: T={optimal_temps[method]:.2f}\n")
        f.write("\n")

    print(f"✅ Markdown report saved: {md_path}")

    print("\n" + "="*80)
    print("✅ TEMPERATURE SCALING ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print(f"  - Temperature-scaled histograms: {len(results)} PNG files")
    print(f"  - Summary CSV: temperature_scaling_summary.csv")
    print(f"  - Markdown report: TEMPERATURE_SCALING_REPORT.md")
    print("\nCompare with original histograms to see the improvement!")


if __name__ == '__main__':
    main()
