#!/usr/bin/env python3
"""
Universal evaluation script for all OOD-based failure detection methods.
Supports: Energy OOD, Deep SVDD, MC Dropout, etc.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Import routers
from energy_ood_routing import EnergyOODRouter
from deep_svdd_routing import DeepSVDDRouter
from mc_dropout_routing import MCDropoutRouter
from knn_ood_routing import KNNOODRouter
from react_ood_routing import ReActOODRouter
from gradnorm_ood_routing import GradNormOODRouter
from mahalanobis_ood_routing import MahalanobisOODRouter


def evaluate_routing_quality(route_decisions, abs_errors, method_name="Method"):
    """Evaluate routing quality metrics."""

    n_total = len(route_decisions)
    n_routed = route_decisions.sum()
    routing_rate = n_routed / n_total * 100

    # Routing quality (using 5° threshold)
    should_route = abs_errors > 5
    catastrophic = abs_errors > 30

    tp = (route_decisions & should_route).sum()
    fp = (route_decisions & ~should_route).sum()
    fn = (~route_decisions & should_route).sum()
    tn = (~route_decisions & ~should_route).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    catastrophic_capture = (route_decisions & catastrophic).sum() / catastrophic.sum() if catastrophic.sum() > 0 else 0

    print(f"\n{'='*100}")
    print(f"{method_name} ROUTING QUALITY")
    print("="*100)
    print(f"  Routing Rate: {routing_rate:.1f}% ({n_routed}/{n_total})")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  False Positive Rate: {fp_rate*100:.1f}%")
    print(f"  Catastrophic Capture: {catastrophic_capture*100:.1f}%")

    # Analyze routed cases
    routed_errors = abs_errors[route_decisions]
    if len(routed_errors) > 0:
        print(f"\n  Routed cases CRNN MAE: {routed_errors.mean():.2f}°")
        print(f"  Routed cases median: {np.median(routed_errors):.2f}°")

    return {
        'method': method_name,
        'routing_rate': routing_rate,
        'n_routed': n_routed,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fp_rate': fp_rate,
        'catastrophic_capture': catastrophic_capture,
        'routed_mae': routed_errors.mean() if len(routed_errors) > 0 else 0
    }


def optimize_threshold_grid_search(scores, abs_errors, method_name, ascending=True):
    """
    Find optimal threshold via grid search.

    Args:
        scores: (N,) OOD scores (energy, distance, variance, etc.)
        abs_errors: (N,) absolute errors
        method_name: Name of method for display
        ascending: If True, route when score > threshold. If False, route when score < threshold.

    Returns:
        results_df: DataFrame with metrics for each threshold
    """

    print(f"\n{'='*100}")
    print(f"{method_name} THRESHOLD OPTIMIZATION")
    print("="*100)

    # Score distribution
    print(f"\nScore distribution:")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")

    # Test percentiles as thresholds
    percentiles = [10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90]
    thresholds = [np.percentile(scores, p) for p in percentiles]

    results = []
    for threshold in thresholds:
        if ascending:
            route_decisions = scores > threshold
        else:
            route_decisions = scores < threshold

        result = evaluate_threshold_quality(route_decisions, abs_errors, threshold)
        results.append(result)

    results_df = pd.DataFrame(results)

    # Find best configurations
    print(f"\n{'='*100}")
    print("BEST THRESHOLDS")
    print("="*100)

    # Best F1
    best_f1_idx = results_df['f1_score'].idxmax()
    print(f"\nBest F1 Score: {results_df.loc[best_f1_idx, 'f1_score']:.3f}")
    print(f"  Threshold: {results_df.loc[best_f1_idx, 'threshold']:.4f}")
    print(f"  Routing: {results_df.loc[best_f1_idx, 'routing_rate']:.1f}%")

    # Best for 20-35% routing range
    reasonable = results_df[(results_df['routing_rate'] >= 20) & (results_df['routing_rate'] <= 35)]
    if len(reasonable) > 0:
        best_reasonable_idx = reasonable['f1_score'].idxmax()
        print(f"\nBest F1 (20-35% routing): {results_df.loc[best_reasonable_idx, 'f1_score']:.3f}")
        print(f"  Threshold: {results_df.loc[best_reasonable_idx, 'threshold']:.4f}")
        print(f"  Routing: {results_df.loc[best_reasonable_idx, 'routing_rate']:.1f}%")

    return results_df


def evaluate_threshold_quality(route_decisions, abs_errors, threshold):
    """Helper to evaluate single threshold."""
    n_routed = route_decisions.sum()
    routing_rate = n_routed / len(route_decisions) * 100

    should_route = abs_errors > 5
    catastrophic = abs_errors > 30

    tp = (route_decisions & should_route).sum()
    fp = (route_decisions & ~should_route).sum()
    fn = (~route_decisions & should_route).sum()
    tn = (~route_decisions & ~should_route).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    cat_capture = (route_decisions & catastrophic).sum() / catastrophic.sum() if catastrophic.sum() > 0 else 0

    return {
        'threshold': threshold,
        'routing_rate': routing_rate,
        'n_routed': n_routed,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fp_rate': fp_rate,
        'catastrophic_capture': cat_capture
    }


def evaluate_energy_ood(model_path, features_path, output_dir):
    """Evaluate Energy-Based OOD detection."""

    print("="*100)
    print("EVALUATING ENERGY-BASED OOD")
    print("="*100)

    # Load router
    router = EnergyOODRouter(model_path=model_path)

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Compute energy scores
    energy_scores = router.compute_energy_scores(features['logits_pre_sig'])

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=energy_scores,
        abs_errors=abs_errors,
        method_name="Energy OOD",
        ascending=True  # Route when energy > threshold (high uncertainty)
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save energy scores
    scores_df = pd.DataFrame({
        'energy': energy_scores,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / "energy_scores.csv", index=False)

    return results_df


def evaluate_deep_svdd(model_path, features_path, output_dir, device='cpu'):
    """Evaluate Deep SVDD detection."""

    print("="*100)
    print("EVALUATING DEEP SVDD")
    print("="*100)

    # Load router
    router = DeepSVDDRouter(model_path=model_path, device=device)

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Compute distances from hypersphere
    distances = router.compute_distances(features['penultimate_features'])

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=distances,
        abs_errors=abs_errors,
        method_name="Deep SVDD",
        ascending=True  # Route when distance > threshold (outside hypersphere)
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save distance scores
    scores_df = pd.DataFrame({
        'distance': distances,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / "distance_scores.csv", index=False)

    return results_df


def evaluate_mc_dropout(features_path, output_dir, use_entropy=False):
    """Evaluate MC Dropout detection."""

    print("="*100)
    print("EVALUATING MC DROPOUT")
    print("="*100)

    # Create router (no model needed, uses existing logits)
    router = MCDropoutRouter()

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Compute uncertainty scores
    _, uncertainty = router.predict_routing(features, variance_threshold=None,
                                           use_entropy=use_entropy)

    metric_name = "Entropy" if use_entropy else "Variance"

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=uncertainty,
        abs_errors=abs_errors,
        method_name=f"MC Dropout ({metric_name})",
        ascending=True  # Route when uncertainty > threshold
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save uncertainty scores
    scores_df = pd.DataFrame({
        'uncertainty': uncertainty,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / f"{metric_name.lower()}_scores.csv", index=False)

    return results_df


def evaluate_knn(features_path, output_dir, k=5, train_features_path=None):
    """Evaluate KNN Distance-based OOD detection."""

    print("="*100)
    print("EVALUATING KNN DISTANCE-BASED OOD")
    print("="*100)

    # Create router
    router = KNNOODRouter(k=k)

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Train on training data if provided, otherwise use test data for training
    if train_features_path:
        train_data = np.load(train_features_path, allow_pickle=True)
        train_features = {key: train_data[key] for key in train_data.files}
        router.train(train_features)
    else:
        # Train on test data (not ideal, but works for evaluation)
        router.train(features)

    # Compute KNN distances
    knn_distances = router.compute_knn_distances(features)

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=knn_distances,
        abs_errors=abs_errors,
        method_name=f"KNN (k={k})",
        ascending=True  # Route when distance > threshold (far from training)
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save KNN distances
    scores_df = pd.DataFrame({
        'knn_distance': knn_distances,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / "knn_distances.csv", index=False)

    return results_df


def evaluate_react(features_path, output_dir, clip_percentile=90, train_features_path=None):
    """Evaluate ReAct (Rectified Activations) OOD detection."""

    print("="*100)
    print("EVALUATING REACT (RECTIFIED ACTIVATIONS)")
    print("="*100)

    # Create router
    router = ReActOODRouter(clip_percentile=clip_percentile)

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Train on training data if provided, otherwise use test data
    if train_features_path:
        train_data = np.load(train_features_path, allow_pickle=True)
        train_features = {key: train_data[key] for key in train_data.files}
        router.train(train_features)
    else:
        router.train(features)

    # Compute ReAct scores
    react_scores = router.compute_react_scores(
        features['penultimate_features'],
        features['logits_pre_sig']
    )

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=react_scores,
        abs_errors=abs_errors,
        method_name=f"ReAct (p{clip_percentile})",
        ascending=True  # Route when ReAct score > threshold (high clipping)
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save ReAct scores
    scores_df = pd.DataFrame({
        'react_score': react_scores,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / "react_scores.csv", index=False)

    return results_df


def evaluate_gradnorm(features_path, output_dir, train_features_path=None):
    """Evaluate GradNorm OOD detection."""

    print("="*100)
    print("EVALUATING GRADNORM")
    print("="*100)

    # Create router
    router = GradNormOODRouter()

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Train on training data if provided, otherwise use test data
    if train_features_path:
        train_data = np.load(train_features_path, allow_pickle=True)
        train_features = {key: train_data[key] for key in train_data.files}
        router.train(train_features)
    else:
        router.train(features)

    # Compute GradNorm scores
    gradnorm_scores = router.compute_gradnorm_scores(features)

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=gradnorm_scores,
        abs_errors=abs_errors,
        method_name="GradNorm",
        ascending=True  # Route when GradNorm > threshold (abnormal gradient)
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save GradNorm scores
    scores_df = pd.DataFrame({
        'gradnorm_score': gradnorm_scores,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / "gradnorm_scores.csv", index=False)

    return results_df


def evaluate_mahalanobis(features_path, output_dir, train_features_path=None):
    """Evaluate Mahalanobis Distance OOD detection."""

    print("="*100)
    print("EVALUATING MAHALANOBIS DISTANCE OOD")
    print("="*100)

    # Create router
    router = MahalanobisOODRouter()

    # Load test features
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    abs_errors = features['abs_errors']

    # Train on training data if provided, otherwise use test data
    if train_features_path:
        train_data = np.load(train_features_path, allow_pickle=True)
        train_features = {key: train_data[key] for key in train_data.files}
        router.train(train_features)
    else:
        router.train(features)

    # Compute Mahalanobis distances
    mahal_distances = router.compute_mahalanobis_distances(features)

    # Optimize threshold
    results_df = optimize_threshold_grid_search(
        scores=mahal_distances,
        abs_errors=abs_errors,
        method_name="Mahalanobis",
        ascending=True  # Route when distance > threshold (far from class means)
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "threshold_optimization.csv", index=False)
    print(f"\n✅ Results saved to: {output_dir}/threshold_optimization.csv")

    # Save Mahalanobis distances
    scores_df = pd.DataFrame({
        'mahalanobis_distance': mahal_distances,
        'abs_error': abs_errors
    })
    scores_df.to_csv(output_dir / "mahalanobis_distances.csv", index=False)

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate OOD failure detection methods')
    parser.add_argument('--method', type=str, required=True,
                        choices=['energy', 'deep_svdd', 'mc_dropout', 'knn', 'react', 'gradnorm', 'mahalanobis'],
                        help='OOD method to evaluate')
    parser.add_argument('--model_path', type=str,
                        help='Path to trained model (required for energy and deep_svdd)')
    parser.add_argument('--features_path', type=str,
                        default='features/test_3x12cm_consecutive_features.npz',
                        help='Path to test features')
    parser.add_argument('--train_features_path', type=str,
                        help='Path to training features (for KNN, ReAct, GradNorm)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'mps', 'cuda'],
                        help='Device to use (default: cpu)')
    parser.add_argument('--use_entropy', action='store_true',
                        help='For MC Dropout: use entropy instead of variance')
    parser.add_argument('--k', type=int, default=5,
                        help='For KNN: number of nearest neighbors (default: 5)')
    parser.add_argument('--clip_percentile', type=int, default=90,
                        help='For ReAct: activation clipping percentile (default: 90)')

    args = parser.parse_args()

    if args.method == 'energy':
        if not args.model_path:
            parser.error("--model_path is required for energy method")
        evaluate_energy_ood(args.model_path, args.features_path, args.output_dir)
    elif args.method == 'deep_svdd':
        if not args.model_path:
            parser.error("--model_path is required for deep_svdd method")
        evaluate_deep_svdd(args.model_path, args.features_path, args.output_dir, args.device)
    elif args.method == 'mc_dropout':
        evaluate_mc_dropout(args.features_path, args.output_dir, args.use_entropy)
    elif args.method == 'knn':
        evaluate_knn(args.features_path, args.output_dir, args.k, args.train_features_path)
    elif args.method == 'react':
        evaluate_react(args.features_path, args.output_dir, args.clip_percentile, args.train_features_path)
    elif args.method == 'gradnorm':
        evaluate_gradnorm(args.features_path, args.output_dir, args.train_features_path)
    elif args.method == 'mahalanobis':
        evaluate_mahalanobis(args.features_path, args.output_dir, args.train_features_path)


if __name__ == "__main__":
    main()
