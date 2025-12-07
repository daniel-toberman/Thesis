#!/usr/bin/env python3
"""
Complete evaluation pipeline for VIM, SHE, and DICE methods.

Runs threshold optimization AND hybrid evaluation with cached SRP results.
"""

import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# Import the new routers
from vim_ood_routing import VIMOODRouter
from she_ood_routing import SHEOODRouter
from dice_ood_routing import DICEOODRouter

# Paths
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")
CACHED_SRP_PATH = Path("features/test_3x12cm_srp_results.pkl")
OUTPUT_BASE = Path("results/ood_methods")

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


def find_threshold_for_routing(scores, target_rate=TARGET_ROUTING):
    """Find threshold that achieves target routing rate."""
    # Higher scores = route to SRP
    # So we want top X% of scores
    percentile = (1 - target_rate) * 100
    threshold = np.percentile(scores, percentile)
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


def evaluate_method(method_name, router, features, cached_srp_df, output_dir):
    """Evaluate a single method."""
    print("\n" + "="*100)
    print(f"EVALUATING {method_name}")
    print("="*100)

    # Train router
    print(f"\nTraining {method_name}...")
    router.train(features)

    # Compute OOD scores
    print(f"Computing OOD scores...")
    if method_name == "VIM":
        scores = router.compute_vim_scores(features)
    elif method_name == "SHE":
        scores = router.compute_she_scores(features)
    elif method_name.startswith("DICE"):
        scores = router.compute_dice_scores(features)

    # Find threshold for 30% routing
    threshold = find_threshold_for_routing(scores, TARGET_ROUTING)
    route_to_srp = scores > threshold

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        'method': method_name,
        'threshold': threshold,
        'n_samples': len(features['gt_angles']),
        **routing_metrics,
        **hybrid_metrics
    }

    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Results saved to: {output_dir}")

    return summary


def main():
    print("="*100)
    print("VIM, SHE, DICE - COMPLETE EVALUATION")
    print("="*100)

    # Load data
    features = load_features()
    cached_srp_df = load_cached_srp()

    # Methods to evaluate
    methods = [
        ('VIM', VIMOODRouter(alpha=1.0), 'vim'),
        ('SHE', SHEOODRouter(), 'she'),
        ('DICE (90%)', DICEOODRouter(sparsity_percentile=90), 'dice_s90'),
        ('DICE (80%)', DICEOODRouter(sparsity_percentile=80), 'dice_s80'),
    ]

    # Evaluate all methods
    results = []
    for method_name, router, output_name in methods:
        output_dir = OUTPUT_BASE / output_name
        try:
            summary = evaluate_method(method_name, router, features, cached_srp_df, output_dir)
            results.append(summary)
        except Exception as e:
            print(f"❌ {method_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    if results:
        print("\n" + "="*100)
        print("RESULTS SUMMARY")
        print("="*100)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)

        print("\n" + results_df.to_string(index=False))

        # Save combined results
        combined_path = OUTPUT_BASE / "vim_she_dice_results.csv"
        results_df.to_csv(combined_path, index=False)
        print(f"\n✅ Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
