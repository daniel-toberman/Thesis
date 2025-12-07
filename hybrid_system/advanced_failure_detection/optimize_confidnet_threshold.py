#!/usr/bin/env python3
"""
Optimize ConfidNet confidence threshold via grid search.
Tests multiple thresholds to find optimal routing strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from confidnet_routing import ConfidNetRouter

# Paths
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")


def evaluate_threshold(confidences, abs_errors, threshold):
    """Evaluate routing performance at a given confidence threshold."""

    # Route cases with confidence < threshold
    route_to_srp = confidences < threshold
    n_routed = route_to_srp.sum()
    routing_rate = n_routed / len(confidences) * 100

    # Routing quality metrics (using 5° as "should route")
    should_route = abs_errors > 5
    catastrophic = abs_errors > 30

    tp = (route_to_srp & should_route).sum()
    fp = (route_to_srp & ~should_route).sum()
    fn = (~route_to_srp & should_route).sum()
    tn = (~route_to_srp & ~should_route).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    catastrophic_capture = (route_to_srp & catastrophic).sum() / catastrophic.sum() if catastrophic.sum() > 0 else 0

    # Analyze routed cases
    routed_errors = abs_errors[route_to_srp]
    routed_mae = routed_errors.mean() if len(routed_errors) > 0 else 0

    return {
        'threshold': threshold,
        'routing_rate': routing_rate,
        'n_routed': n_routed,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fp_rate': fp_rate,
        'catastrophic_capture': catastrophic_capture,
        'routed_mae': routed_mae,
    }


def grid_search_thresholds(model_path, device, output_dir):
    """Grid search over confidence thresholds."""

    print("="*100)
    print(f"CONFIDNET THRESHOLD OPTIMIZATION")
    print("="*100)

    # Load model and generate predictions
    router = ConfidNetRouter(model_path=model_path, device=device)

    # Load test features
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}

    print(f"\nGenerating confidence predictions for {len(features['gt_angles'])} samples...")
    confidences = router.predict_confidences(features)
    abs_errors = features['abs_errors']

    print(f"\nConfidence distribution:")
    print(f"  Min: {confidences.min():.4f}")
    print(f"  Max: {confidences.max():.4f}")
    print(f"  Mean: {confidences.mean():.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    print(f"  Q1: {np.percentile(confidences, 25):.4f}")
    print(f"  Q3: {np.percentile(confidences, 75):.4f}")

    # Test thresholds
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n{'='*100}")
    print(f"TESTING {len(thresholds)} THRESHOLDS")
    print("="*100)

    results = []
    for threshold in thresholds:
        result = evaluate_threshold(confidences, abs_errors, threshold)
        results.append(result)

        print(f"\nThreshold < {threshold:.2f}:")
        print(f"  Routing: {result['routing_rate']:.1f}% ({result['n_routed']} cases)")
        print(f"  F1 Score: {result['f1_score']:.3f}")
        print(f"  Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}")
        print(f"  FP Rate: {result['fp_rate']*100:.1f}%")
        print(f"  Catastrophic Capture: {result['catastrophic_capture']*100:.1f}%")
        print(f"  Routed MAE: {result['routed_mae']:.2f}°")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Find best thresholds
    print(f"\n{'='*100}")
    print("BEST THRESHOLDS")
    print("="*100)

    # Best F1 score
    best_f1_idx = results_df['f1_score'].idxmax()
    print(f"\nBest F1 Score: {results_df.loc[best_f1_idx, 'f1_score']:.3f}")
    print(f"  Threshold: {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    print(f"  Routing: {results_df.loc[best_f1_idx, 'routing_rate']:.1f}%")
    print(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.3f}")
    print(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.3f}")

    # Best recall (within reasonable routing range 20-35%)
    reasonable_routing = results_df[(results_df['routing_rate'] >= 20) & (results_df['routing_rate'] <= 35)]
    if len(reasonable_routing) > 0:
        best_recall_idx = reasonable_routing['recall'].idxmax()
        print(f"\nBest Recall (20-35% routing): {results_df.loc[best_recall_idx, 'recall']:.3f}")
        print(f"  Threshold: {results_df.loc[best_recall_idx, 'threshold']:.2f}")
        print(f"  Routing: {results_df.loc[best_recall_idx, 'routing_rate']:.1f}%")
        print(f"  F1: {results_df.loc[best_recall_idx, 'f1_score']:.3f}")

    # Lowest false positive rate (within reasonable routing)
    if len(reasonable_routing) > 0:
        best_fp_idx = reasonable_routing['fp_rate'].idxmin()
        print(f"\nLowest FP Rate (20-35% routing): {results_df.loc[best_fp_idx, 'fp_rate']*100:.1f}%")
        print(f"  Threshold: {results_df.loc[best_fp_idx, 'threshold']:.2f}")
        print(f"  Routing: {results_df.loc[best_fp_idx, 'routing_rate']:.1f}%")
        print(f"  F1: {results_df.loc[best_fp_idx, 'f1_score']:.3f}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Optimize ConfidNet confidence threshold')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained ConfidNet model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or mps)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    args = parser.parse_args()

    grid_search_thresholds(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
