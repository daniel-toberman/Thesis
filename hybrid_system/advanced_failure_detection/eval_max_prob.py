#!/usr/bin/env python3
"""
Evaluate Max Probability threshold routing.
"""

import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from max_prob_routing import MaxProbRouter

# Paths
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")
CACHED_SRP_PATH = Path("features/test_3x12cm_srp_results.pkl")
OUTPUT_DIR = Path("results/ood_methods/max_prob")

# Target routing rate
TARGET_ROUTING = 0.30  # 30%


def load_features():
    """Load test features."""
    print(f"Loading features from {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"Loaded {len(features['gt_angles'])} samples")
    return features


def load_cached_srp():
    """Load cached SRP results."""
    print(f"Loading cached SRP results from {CACHED_SRP_PATH}")
    with open(CACHED_SRP_PATH, 'rb') as f:
        cached_df = pickle.load(f)
    print(f"Loaded SRP results for {len(cached_df)} samples")
    return cached_df


def find_threshold_for_routing(max_probs, target_rate=TARGET_ROUTING):
    """
    Find threshold that achieves target routing rate.

    Lower max_prob = route to SRP
    So we want bottom X% of max_probs
    """
    percentile = target_rate * 100
    threshold = np.percentile(max_probs, percentile)
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


def main():
    print("="*100)
    print("MAX PROBABILITY THRESHOLD - EVALUATION")
    print("="*100)

    # Load data
    features = load_features()
    cached_srp_df = load_cached_srp()

    # Create router
    print("\nInitializing MaxProb router...")
    router = MaxProbRouter()
    router.train(features)

    # Compute max probabilities
    print("\nComputing maximum probabilities...")
    max_probs = router.compute_max_prob_scores(features)

    print(f"Max probability stats:")
    print(f"  Range: [{max_probs.min():.4f}, {max_probs.max():.4f}]")
    print(f"  Mean: {max_probs.mean():.4f}")
    print(f"  Median: {np.median(max_probs):.4f}")

    # Find threshold for 30% routing
    threshold = find_threshold_for_routing(max_probs, TARGET_ROUTING)
    route_to_srp, _ = router.predict_routing(features, threshold)

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Routing rate: {route_to_srp.sum() / len(route_to_srp) * 100:.1f}%")

    # Evaluate routing quality
    routing_metrics = evaluate_routing_quality(route_to_srp, features['abs_errors'])
    print(f"\nRouting Quality:")
    print(f"  Precision: {routing_metrics['precision']:.3f}")
    print(f"  Recall: {routing_metrics['recall']:.3f}")
    print(f"  F1 Score: {routing_metrics['f1_score']:.3f}")

    # Evaluate hybrid performance
    print(f"\nEvaluating hybrid performance with cached SRP...")
    hybrid_metrics = evaluate_hybrid_performance(route_to_srp, features, cached_srp_df)
    print(f"  Hybrid MAE: {hybrid_metrics['hybrid_mae']:.2f}°")
    print(f"  Hybrid Median: {hybrid_metrics['hybrid_median']:.2f}°")
    print(f"  Hybrid Success: {hybrid_metrics['hybrid_success']:.1f}%")
    print(f"  MAE Improvement: {hybrid_metrics['mae_improvement']:+.2f}°")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        'method': 'MaxProb',
        'threshold': threshold,
        'n_samples': len(features['gt_angles']),
        **routing_metrics,
        **hybrid_metrics
    }

    summary_df = pd.DataFrame([summary])
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
