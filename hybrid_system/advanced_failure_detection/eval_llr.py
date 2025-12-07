#!/usr/bin/env python3
"""
Evaluate LLR (Likelihood Ratio) OOD routing.
"""

import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from llr_ood_routing import LLROODRouter

# Paths
TRAIN_FEATURES_PATH = Path("features/train_combined_features.npz")
TEST_FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")
CACHED_SRP_PATH = Path("features/test_3x12cm_srp_results.pkl")
OUTPUT_BASE = Path("results/ood_methods")

# Target routing rate
TARGET_ROUTING = 0.30  # 30%


def load_train_features():
    """Load training features."""
    print(f"Loading training features from {TRAIN_FEATURES_PATH}")
    data = np.load(TRAIN_FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"Loaded {len(features['gt_angles'])} training samples")
    return features


def load_test_features():
    """Load test features."""
    print(f"Loading test features from {TEST_FEATURES_PATH}")
    data = np.load(TEST_FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"Loaded {len(features['gt_angles'])} test samples")
    return features


def load_cached_srp():
    """Load cached SRP results."""
    print(f"Loading cached SRP results from {CACHED_SRP_PATH}")
    with open(CACHED_SRP_PATH, 'rb') as f:
        cached_df = pickle.load(f)
    print(f"Loaded SRP results for {len(cached_df)} samples")
    return cached_df


def find_threshold_for_routing(llr_scores, target_rate=TARGET_ROUTING):
    """
    Find threshold that achieves target routing rate.

    Higher LLR score = route to SRP
    So we want top X% of scores
    """
    percentile = (1 - target_rate) * 100
    threshold = np.percentile(llr_scores, percentile)
    return threshold


def evaluate_routing_quality(route_to_srp, abs_errors, threshold_deg=5.0):
    """Evaluate routing decision quality."""
    failures = abs_errors > threshold_deg

    n_routed = route_to_srp.sum()
    routing_rate = n_routed / len(route_to_srp)

    # Routing metrics
    true_positives = (route_to_srp & failures).sum()
    false_positives = (route_to_srp & ~failures).sum()
    false_negatives = (~route_to_srp & failures).sum()

    precision = true_positives / (true_positives + false_positives) if n_routed > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if failures.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'n_routed': n_routed,
        'routing_rate': routing_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_hybrid_performance(route_to_srp, features, cached_srp_df):
    """Evaluate hybrid CRNN-SRP performance using cached SRP results."""
    abs_errors = features['abs_errors']
    n_samples = len(abs_errors)

    # Initialize with CRNN errors
    hybrid_errors = abs_errors.copy()

    # Get routed indices
    routed_indices = np.where(route_to_srp)[0]

    # Replace with cached SRP results
    n_replaced = 0
    for idx in routed_indices:
        cached_row = cached_srp_df[cached_srp_df['sample_idx'] == idx]
        if len(cached_row) > 0 and pd.notna(cached_row.iloc[0]['srp_pred']):
            hybrid_errors[idx] = cached_row.iloc[0]['srp_error']
            n_replaced += 1

    # Compute metrics
    hybrid_mae = hybrid_errors.mean()
    hybrid_median = np.median(hybrid_errors)
    hybrid_success = (hybrid_errors <= 5).sum() / n_samples * 100

    crnn_mae = abs_errors.mean()

    return {
        'hybrid_mae': hybrid_mae,
        'hybrid_median': hybrid_median,
        'hybrid_success': hybrid_success,
        'mae_improvement': crnn_mae - hybrid_mae,
        'srp_used': n_replaced
    }


def evaluate_llr_config(n_components, train_features, test_features, cached_srp_df, output_dir):
    """Evaluate LLR with a specific n_components configuration."""
    print("\n" + "="*100)
    print(f"EVALUATING LLR (n_components={n_components})")
    print("="*100)

    # Train LLR
    print(f"\nTraining LLR with {n_components} components...")
    router = LLROODRouter(n_components=n_components)
    router.train(train_features)

    # Save trained model
    model_path = output_dir / f"llr_gmm_{n_components}.pkl"
    router.save(model_path)

    # Compute LLR scores on test set
    print(f"\nComputing LLR scores on test set...")
    llr_scores = router.compute_llr_scores(test_features)

    print(f"LLR score stats:")
    print(f"  Range: [{llr_scores.min():.2f}, {llr_scores.max():.2f}]")
    print(f"  Mean: {llr_scores.mean():.2f}")
    print(f"  Median: {np.median(llr_scores):.2f}")
    print(f"  Std: {llr_scores.std():.2f}")

    # Find threshold for 30% routing
    threshold = find_threshold_for_routing(llr_scores, TARGET_ROUTING)
    route_to_srp = llr_scores > threshold

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Routing rate: {route_to_srp.sum() / len(route_to_srp) * 100:.1f}%")

    # Evaluate routing quality
    routing_metrics = evaluate_routing_quality(route_to_srp, test_features['abs_errors'])
    print(f"\nRouting Quality:")
    print(f"  Precision: {routing_metrics['precision']:.3f}")
    print(f"  Recall: {routing_metrics['recall']:.3f}")
    print(f"  F1 Score: {routing_metrics['f1_score']:.3f}")

    # Evaluate hybrid performance
    print(f"\nEvaluating hybrid performance with cached SRP...")
    hybrid_metrics = evaluate_hybrid_performance(route_to_srp, test_features, cached_srp_df)
    print(f"  Hybrid MAE: {hybrid_metrics['hybrid_mae']:.2f}°")
    print(f"  Hybrid Median: {hybrid_metrics['hybrid_median']:.2f}°")
    print(f"  Hybrid Success: {hybrid_metrics['hybrid_success']:.1f}%")
    print(f"  MAE Improvement: {hybrid_metrics['mae_improvement']:+.2f}°")

    # Save results
    summary = {
        'method': f'LLR_GMM{n_components}',
        'n_components': n_components,
        'threshold': threshold,
        'n_samples': len(test_features['gt_angles']),
        **routing_metrics,
        **hybrid_metrics
    }

    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"llr_gmm{n_components}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Results saved to: {summary_path}")

    return summary


def main():
    print("="*100)
    print("LLR (LIKELIHOOD RATIO) - COMPLETE EVALUATION")
    print("="*100)

    # Load data
    train_features = load_train_features()
    test_features = load_test_features()
    cached_srp_df = load_cached_srp()

    # Output directory
    output_dir = OUTPUT_BASE / "llr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test different n_components configurations
    configs = [5, 10, 15, 20]

    results = []
    for n_comp in configs:
        try:
            summary = evaluate_llr_config(n_comp, train_features, test_features,
                                         cached_srp_df, output_dir)
            results.append(summary)
        except Exception as e:
            print(f"❌ n_components={n_comp} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    if results:
        print("\n" + "="*100)
        print("LLR RESULTS SUMMARY")
        print("="*100)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('hybrid_mae', ascending=True)

        print("\n" + results_df.to_string(index=False))

        # Save combined results
        combined_path = output_dir / "llr_comparison.csv"
        results_df.to_csv(combined_path, index=False)
        print(f"\n✅ Combined results saved to: {combined_path}")

        # Print best configuration
        best = results_df.iloc[0]
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   n_components: {best['n_components']}")
        print(f"   Hybrid MAE: {best['hybrid_mae']:.2f}°")
        print(f"   F1 Score: {best['f1_score']:.3f}")
        print(f"   Success Rate: {best['hybrid_success']:.1f}%")


if __name__ == "__main__":
    main()
