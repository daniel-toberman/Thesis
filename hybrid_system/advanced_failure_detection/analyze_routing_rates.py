#!/usr/bin/env python3
"""
Quick analysis of routing rates for different error thresholds.
This helps select optimal configurations before running expensive SRP evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from confidnet_routing import ConfidNetRouter
from temperature_scaling import apply_temperature_scaling
from mahalanobis_ood import MahalanobisOODDetector

# Paths
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")

def analyze_error_distribution(features):
    """Analyze CRNN error distribution to understand routing targets."""
    errors = features['abs_errors']

    print("\n" + "="*80)
    print("CRNN ERROR DISTRIBUTION ON TEST SET")
    print("="*80)
    print(f"Total samples: {len(errors)}")
    print(f"MAE: {errors.mean():.2f}°")
    print(f"Median: {np.median(errors):.2f}°")
    print(f"Success (≤5°): {(errors <= 5).sum()} ({(errors <= 5).sum()/len(errors)*100:.1f}%)")

    print(f"\nError breakdown:")
    thresholds = [5, 10, 15, 20, 25, 30]
    for thresh in thresholds:
        count = (errors > thresh).sum()
        pct = count / len(errors) * 100
        print(f"  Errors > {thresh:2d}°: {count:4d} ({pct:5.1f}%) <- Potential routing targets")

    return errors


def analyze_confidnet_routing(features, error_thresholds=[15, 20, 25, 30]):
    """Analyze ConfidNet routing rates for different error thresholds."""
    print("\n" + "="*80)
    print("CONFIDNET ROUTING ANALYSIS")
    print("="*80)

    # Load router
    router = ConfidNetRouter(
        model_path="models/confidnet_combined/best_model.ckpt",
        device='mps'
    )

    # Predict confidences
    confidences = router.predict_confidences(features)

    results = []
    for error_thresh in error_thresholds:
        # Override error threshold
        router.error_threshold = error_thresh

        # Find optimal confidence threshold
        optimal_results = router.find_optimal_threshold(features)
        conf_thresh = optimal_results['best_threshold']
        route_to_srp = optimal_results['best_results']['route_to_srp']

        routing_rate = route_to_srp.sum() / len(route_to_srp) * 100

        results.append({
            'error_threshold': error_thresh,
            'confidence_threshold': conf_thresh,
            'routing_rate': routing_rate,
            'n_routed': route_to_srp.sum(),
            'f1_score': optimal_results['best_results']['f1_score'],
            'precision': optimal_results['best_results']['precision'],
            'recall': optimal_results['best_results']['recall'],
            'false_positive_rate': optimal_results['best_results']['false_positive_rate'],
        })

        print(f"\nError threshold: {error_thresh}°")
        print(f"  Optimal confidence threshold: {conf_thresh:.4f}")
        print(f"  Routing rate: {routing_rate:.1f}% ({route_to_srp.sum()}/{len(route_to_srp)} cases)")
        print(f"  F1 Score: {optimal_results['best_results']['f1_score']:.4f}")
        print(f"  Precision: {optimal_results['best_results']['precision']:.4f}")
        print(f"  Recall: {optimal_results['best_results']['recall']:.4f}")
        print(f"  False Positive Rate: {optimal_results['best_results']['false_positive_rate']*100:.1f}%")

    return pd.DataFrame(results)


def analyze_temp_mahal_routing(features, error_thresholds=[15, 20, 25, 30]):
    """Analyze Temperature + Mahalanobis routing rates for different error thresholds."""
    print("\n" + "="*80)
    print("TEMPERATURE + MAHALANOBIS ROUTING ANALYSIS")
    print("="*80)

    # Load models
    temperature = float(np.load("models/temperature_combined.npy"))
    with open("models/mahalanobis_combined.pkl", 'rb') as f:
        mahal_detector = pickle.load(f)

    # Compute calibrated confidences
    logits = []
    for logit in features['logits_pre_sig']:
        logits.append(logit.mean(axis=0))
    logits = np.array(logits)

    calibrated_probs = apply_temperature_scaling(logits, temperature)
    confidences = calibrated_probs.max(axis=1)

    # Compute Mahalanobis distances
    penult_features = []
    for feat in features['penultimate_features']:
        penult_features.append(feat.mean(axis=0))
    penult_features = np.array(penult_features)

    predicted_angles = features['predicted_angles']
    distances = mahal_detector.compute_mahalanobis_distance(penult_features, predicted_angles)

    errors = features['abs_errors']

    results = []
    for error_thresh in error_thresholds:
        # Define ground truth
        should_route = (errors > error_thresh)

        # Grid search for optimal thresholds
        conf_thresholds = np.linspace(0.1, 0.9, 50)
        dist_thresholds = np.linspace(5, 50, 50)

        best_f1 = 0
        best_conf_thresh = None
        best_dist_thresh = None
        best_route = None

        for conf_thresh in conf_thresholds:
            for dist_thresh in dist_thresholds:
                # Route if EITHER low confidence OR high distance
                route = (confidences < conf_thresh) | (distances > dist_thresh)

                # Compute metrics
                tp = (route & should_route).sum()
                fp = (route & ~should_route).sum()
                fn = (~route & should_route).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_conf_thresh = conf_thresh
                    best_dist_thresh = dist_thresh
                    best_route = route
                    best_precision = precision
                    best_recall = recall

        routing_rate = best_route.sum() / len(best_route) * 100

        # Calculate false positive rate
        correct_predictions = errors <= error_thresh
        false_positives = best_route & correct_predictions
        fp_rate = false_positives.sum() / correct_predictions.sum() if correct_predictions.sum() > 0 else 0

        results.append({
            'error_threshold': error_thresh,
            'confidence_threshold': best_conf_thresh,
            'distance_threshold': best_dist_thresh,
            'routing_rate': routing_rate,
            'n_routed': best_route.sum(),
            'f1_score': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'false_positive_rate': fp_rate,
        })

        print(f"\nError threshold: {error_thresh}°")
        print(f"  Optimal confidence threshold: {best_conf_thresh:.4f}")
        print(f"  Optimal distance threshold: {best_dist_thresh:.2f}")
        print(f"  Routing rate: {routing_rate:.1f}% ({best_route.sum()}/{len(best_route)} cases)")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  Precision: {best_precision:.4f}")
        print(f"  Recall: {best_recall:.4f}")
        print(f"  False Positive Rate: {fp_rate*100:.1f}%")

    return pd.DataFrame(results)


def main():
    print("="*80)
    print("ROUTING RATE ANALYSIS FOR HYBRID CRNN-SRP SYSTEM")
    print("="*80)

    # Load features
    print(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"  Total samples: {len(features['predicted_angles'])}")

    # Analyze error distribution
    analyze_error_distribution(features)

    # Analyze ConfidNet routing
    confidnet_df = analyze_confidnet_routing(features)

    # Analyze Temperature + Mahalanobis routing
    temp_mahal_df = analyze_temp_mahal_routing(features)

    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print("\nConfidNet:")
    print(confidnet_df.to_string(index=False))

    print("\n\nTemperature + Mahalanobis:")
    print(temp_mahal_df.to_string(index=False))

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find configs with 25-35% routing rate
    target_min, target_max = 25, 35

    print(f"\nConfigurations with {target_min}-{target_max}% routing rate:")

    print("\n  ConfidNet:")
    cn_good = confidnet_df[(confidnet_df['routing_rate'] >= target_min) &
                            (confidnet_df['routing_rate'] <= target_max)]
    if len(cn_good) > 0:
        best_cn = cn_good.loc[cn_good['f1_score'].idxmax()]
        print(f"    Best: Error threshold {best_cn['error_threshold']:.0f}° "
              f"-> {best_cn['routing_rate']:.1f}% routing, F1={best_cn['f1_score']:.4f}")
    else:
        print("    None in target range")

    print("\n  Temperature + Mahalanobis:")
    tm_good = temp_mahal_df[(temp_mahal_df['routing_rate'] >= target_min) &
                             (temp_mahal_df['routing_rate'] <= target_max)]
    if len(tm_good) > 0:
        best_tm = tm_good.loc[tm_good['f1_score'].idxmax()]
        print(f"    Best: Error threshold {best_tm['error_threshold']:.0f}° "
              f"-> {best_tm['routing_rate']:.1f}% routing, F1={best_tm['f1_score']:.4f}")
    else:
        print("    None in target range")

    # Save results
    output_dir = Path("results/routing_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    confidnet_df.to_csv(output_dir / "confidnet_routing_analysis.csv", index=False)
    temp_mahal_df.to_csv(output_dir / "temp_mahal_routing_analysis.csv", index=False)

    print(f"\n\n✅ Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
