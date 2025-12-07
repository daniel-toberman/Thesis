#!/usr/bin/env python3
"""
ConfidNet routing analysis with routing rate constraints.
Finds optimal confidence threshold while constraining routing rate to target range.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from confidnet_routing import ConfidNetRouter

# Paths
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")

def find_constrained_threshold(features, router, error_threshold, target_routing_pct=30):
    """Find confidence threshold that achieves target routing rate with best F1."""

    router.error_threshold = error_threshold
    errors = features['abs_errors']
    should_route = (errors > error_threshold)

    # Predict confidences
    confidences = router.predict_confidences(features)

    # Try different confidence thresholds
    conf_thresholds = np.linspace(0.05, 0.95, 100)

    best_result = None
    best_f1 = 0
    target_n_routed = int(len(errors) * target_routing_pct / 100)

    for conf_thresh in conf_thresholds:
        route_to_srp = confidences < conf_thresh
        routing_rate = route_to_srp.sum() / len(route_to_srp) * 100

        # Skip if routing rate is too far from target (allow ±5%)
        if abs(routing_rate - target_routing_pct) > 5:
            continue

        # Compute F1
        tp = (route_to_srp & should_route).sum()
        fp = (route_to_srp & ~should_route).sum()
        fn = (~route_to_srp & should_route).sum()

        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate FP rate
        correct_predictions = errors <= error_threshold
        false_positives = route_to_srp & correct_predictions
        fp_rate = false_positives.sum() / correct_predictions.sum() if correct_predictions.sum() > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                'confidence_threshold': conf_thresh,
                'routing_rate': routing_rate,
                'n_routed': route_to_srp.sum(),
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'false_positive_rate': fp_rate,
            }

    return best_result

def main():
    print("="*80)
    print("CONFIDNET ROUTING ANALYSIS WITH RATE CONSTRAINTS")
    print("="*80)

    # Load features
    print(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"  Total samples: {len(features['predicted_angles'])}")

    # Load router
    router = ConfidNetRouter(
        model_path="models/confidnet_combined/best_model.ckpt",
        device='mps'
    )

    # Test different configurations
    error_thresholds = [15, 20, 25, 30]
    target_routing_rates = [25, 30, 35]

    results = []

    for error_thresh in error_thresholds:
        print(f"\n{'='*80}")
        print(f"Error Threshold: {error_thresh}°")
        print(f"{'='*80}")

        for target_rate in target_routing_rates:
            result = find_constrained_threshold(features, router, error_thresh, target_rate)

            if result is not None:
                results.append({
                    'error_threshold': error_thresh,
                    'target_routing_rate': target_rate,
                    **result
                })

                print(f"\nTarget routing: {target_rate}%")
                print(f"  Confidence threshold: {result['confidence_threshold']:.4f}")
                print(f"  Actual routing rate: {result['routing_rate']:.1f}% ({result['n_routed']} cases)")
                print(f"  F1 Score: {result['f1_score']:.4f}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  False Positive Rate: {result['false_positive_rate']*100:.1f}%")
            else:
                print(f"\nTarget routing: {target_rate}%")
                print(f"  No valid configuration found")

    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR 25-35% ROUTING RATE")
    print("="*80)

    if results:
        # Filter to target range
        good_results = [r for r in results if 25 <= r['routing_rate'] <= 35]

        if good_results:
            # Sort by F1 score
            good_results.sort(key=lambda x: x['f1_score'], reverse=True)

            print("\nTop 3 configurations:")
            for i, r in enumerate(good_results[:3], 1):
                print(f"\n{i}. Error threshold {r['error_threshold']}°, "
                      f"Confidence threshold {r['confidence_threshold']:.4f}")
                print(f"   Routing: {r['routing_rate']:.1f}% ({r['n_routed']} cases)")
                print(f"   F1: {r['f1_score']:.4f}, Precision: {r['precision']:.4f}, "
                      f"Recall: {r['recall']:.4f}, FP Rate: {r['false_positive_rate']*100:.1f}%")
        else:
            print("\nNo configurations found in 25-35% routing range!")

    # Save results
    import pandas as pd
    output_dir = Path("results/routing_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / "confidnet_constrained_routing.csv", index=False)
        print(f"\n\n✅ Results saved to: {output_dir}/confidnet_constrained_routing.csv")

if __name__ == "__main__":
    main()
