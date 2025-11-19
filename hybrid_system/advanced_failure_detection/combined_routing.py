"""
Combined Routing Strategy using Temperature Scaling + Mahalanobis Distance

Combines calibrated confidence scores with Mahalanobis-based OOD detection
for optimal failure detection and routing decisions.

Strategy:
    Route to SRP if EITHER:
    1. Calibrated confidence < confidence_threshold (low confidence)
    2. Mahalanobis distance > distance_threshold (OOD/geometric mismatch)
"""

import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from temperature_scaling import (
    optimize_temperature,
    apply_temperature_scaling,
    compute_ece
)
from mahalanobis_ood import MahalanobisOODDetector


class CombinedFailureDetector:
    """Combined failure detector using multiple methods."""

    def __init__(self):
        self.temperature = 1.0
        self.mahalanobis_detector = None
        self.confidence_threshold = 0.04
        self.distance_threshold = None
        self.fitted = False

    def fit(
        self,
        train_features: Dict,
        val_features: Dict = None,
        n_pca_components: int = 64
    ):
        """
        Fit both temperature scaling and Mahalanobis detector.

        Args:
            train_features: Training features dict
            val_features: Validation features (if None, use train for temp scaling)
            n_pca_components: Number of PCA components for Mahalanobis
        """
        print("\n" + "="*80)
        print("Fitting Combined Failure Detector")
        print("="*80)

        if val_features is None:
            val_features = train_features

        # 1. Optimize temperature on validation set
        print("\n[1/2] Optimizing Temperature Scaling...")
        if val_features['logits_pre_sig'][0].ndim == 2:
            val_logits = np.array([logits.mean(axis=0) for logits in val_features['logits_pre_sig']])
        else:
            val_logits = val_features['logits_pre_sig']

        self.temperature, temp_results = optimize_temperature(
            val_logits,
            val_features['gt_angles']
        )

        # 2. Fit Mahalanobis detector on training set
        print("\n[2/2] Fitting Mahalanobis OOD Detector...")
        if train_features['penultimate_features'][0].ndim == 2:
            train_feat = np.array([f.mean(axis=0) for f in train_features['penultimate_features']])
        else:
            train_feat = train_features['penultimate_features']

        self.mahalanobis_detector = MahalanobisOODDetector(
            n_components=n_pca_components,
            use_pca=True
        )
        self.mahalanobis_detector.fit(train_feat, train_features['gt_angles'])

        self.fitted = True
        print("\n" + "="*80)
        print("Fitting Complete!")
        print("="*80)

    def evaluate_combined_routing(
        self,
        test_features: Dict,
        confidence_threshold: float,
        distance_threshold: float,
        strategy: str = 'or'
    ) -> Dict:
        """
        Evaluate combined routing strategy.

        Args:
            test_features: Test features dict
            confidence_threshold: Threshold for calibrated confidence
            distance_threshold: Threshold for Mahalanobis distance
            strategy: 'or' (route if EITHER triggers) or 'and' (route if BOTH trigger)

        Returns:
            results: Dictionary with metrics
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted first")

        # Get calibrated confidences
        if test_features['logits_pre_sig'][0].ndim == 2:
            test_logits = np.array([logits.mean(axis=0) for logits in test_features['logits_pre_sig']])
            test_feat = np.array([f.mean(axis=0) for f in test_features['penultimate_features']])
        else:
            test_logits = test_features['logits_pre_sig']
            test_feat = test_features['penultimate_features']

        calibrated_probs = apply_temperature_scaling(test_logits, self.temperature)
        confidences = calibrated_probs.max(axis=1)

        # Get Mahalanobis distances
        predicted_angles = test_features['predicted_angles']
        mahalanobis_distances = self.mahalanobis_detector.compute_mahalanobis_distance(
            test_feat,
            predicted_angles
        )

        # Routing decisions
        low_confidence = confidences < confidence_threshold
        high_distance = mahalanobis_distances > distance_threshold

        if strategy == 'or':
            route_to_srp = low_confidence | high_distance
        elif strategy == 'and':
            route_to_srp = low_confidence & high_distance
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute metrics
        gt_angles = test_features['gt_angles']
        errors = np.abs((predicted_angles - gt_angles + 180) % 360 - 180)
        actual_failures = errors > 5.0

        # Confusion matrix
        tp = np.sum(actual_failures & route_to_srp)
        tn = np.sum(~actual_failures & ~route_to_srp)
        fp = np.sum(~actual_failures & route_to_srp)
        fn = np.sum(actual_failures & ~route_to_srp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(errors)

        results = {
            'strategy': strategy,
            'confidence_threshold': confidence_threshold,
            'distance_threshold': distance_threshold,
            'temperature': self.temperature,
            'n_samples': len(errors),
            'n_routed_to_srp': route_to_srp.sum(),
            'routing_rate': route_to_srp.mean(),
            'n_low_confidence': low_confidence.sum(),
            'n_high_distance': high_distance.sum(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'confidences': confidences,
            'mahalanobis_distances': mahalanobis_distances,
            'errors': errors,
            'route_to_srp': route_to_srp,
        }

        return results

    def grid_search_combined(
        self,
        test_features: Dict,
        confidence_range: np.ndarray = None,
        distance_range: np.ndarray = None,
        strategy: str = 'or'
    ) -> Tuple[float, float, List[Dict]]:
        """
        Grid search for optimal thresholds.

        Args:
            test_features: Test features dict
            confidence_range: Range of confidence thresholds
            distance_range: Range of distance thresholds
            strategy: 'or' or 'and'

        Returns:
            best_conf_threshold: Optimal confidence threshold
            best_dist_threshold: Optimal distance threshold
            all_results: List of all results
        """
        if confidence_range is None:
            confidence_range = np.linspace(0.01, 0.15, 15)

        if distance_range is None:
            # Auto-determine range based on typical distances
            if test_features['penultimate_features'][0].ndim == 2:
                test_feat = np.array([f.mean(axis=0) for f in test_features['penultimate_features']])
            else:
                test_feat = test_features['penultimate_features']

            predicted_angles = test_features['predicted_angles']
            distances = self.mahalanobis_detector.compute_mahalanobis_distance(
                test_feat, predicted_angles
            )
            distance_range = np.percentile(distances, np.linspace(50, 95, 10))

        print("\n" + "="*80)
        print(f"Grid Search: {strategy.upper()} Strategy")
        print("="*80)
        print(f"Confidence thresholds: {len(confidence_range)}")
        print(f"Distance thresholds: {len(distance_range)}")
        print(f"Total combinations: {len(confidence_range) * len(distance_range)}")

        all_results = []
        best_f1 = 0.0
        best_conf = confidence_range[0]
        best_dist = distance_range[0]

        for conf_thresh in confidence_range:
            for dist_thresh in distance_range:
                results = self.evaluate_combined_routing(
                    test_features,
                    conf_thresh,
                    dist_thresh,
                    strategy=strategy
                )
                all_results.append(results)

                if results['f1_score'] > best_f1:
                    best_f1 = results['f1_score']
                    best_conf = conf_thresh
                    best_dist = dist_thresh

        print("\n" + "="*80)
        print(f"Best Configuration (F1={best_f1:.3f}):")
        print(f"  Confidence threshold: {best_conf:.4f}")
        print(f"  Distance threshold: {best_dist:.2f}")
        print("="*80)

        # Print best results
        best_results = [r for r in all_results if r['f1_score'] == best_f1][0]
        print(f"\nMetrics:")
        print(f"  Precision: {best_results['precision']:.3f}")
        print(f"  Recall: {best_results['recall']:.3f}")
        print(f"  F1: {best_results['f1_score']:.3f}")
        print(f"  Accuracy: {best_results['accuracy']:.3f}")
        print(f"  Routing rate: {best_results['routing_rate']*100:.1f}%")

        return best_conf, best_dist, all_results


def visualize_combined_results(results: Dict, output_dir: str):
    """
    Create visualizations for combined routing results.

    Args:
        results: Results dict from evaluate_combined_routing
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter plot: Confidence vs Mahalanobis Distance
    fig, ax = plt.subplots(figsize=(10, 8))

    errors = results['errors']
    confidences = results['confidences']
    distances = results['mahalanobis_distances']

    # Color by success/failure
    successes = errors <= 5.0
    failures = errors > 5.0

    ax.scatter(
        confidences[successes],
        distances[successes],
        c='green',
        alpha=0.5,
        s=10,
        label='Success (error ≤ 5°)'
    )
    ax.scatter(
        confidences[failures],
        distances[failures],
        c='red',
        alpha=0.5,
        s=10,
        label='Failure (error > 5°)'
    )

    # Draw threshold lines
    ax.axvline(
        results['confidence_threshold'],
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Conf. threshold = {results["confidence_threshold"]:.3f}'
    )
    ax.axhline(
        results['distance_threshold'],
        color='purple',
        linestyle='--',
        linewidth=2,
        label=f'Dist. threshold = {results["distance_threshold"]:.1f}'
    )

    ax.set_xlabel('Calibrated Confidence (max prob)')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_title(f'Combined Failure Detection ({results["strategy"].upper()} strategy)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_dir / 'combined_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confidence distribution
    axes[0, 0].hist(confidences[successes], bins=50, alpha=0.5, color='green', label='Success')
    axes[0, 0].hist(confidences[failures], bins=50, alpha=0.5, color='red', label='Failure')
    axes[0, 0].axvline(results['confidence_threshold'], color='blue', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Calibrated Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Confidence Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Distance distribution
    axes[0, 1].hist(distances[successes], bins=50, alpha=0.5, color='green', label='Success')
    axes[0, 1].hist(distances[failures], bins=50, alpha=0.5, color='red', label='Failure')
    axes[0, 1].axvline(results['distance_threshold'], color='purple', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Mahalanobis Distance')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distance Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Error distribution by routing decision
    routed = results['route_to_srp']
    axes[1, 0].hist(errors[~routed], bins=50, alpha=0.5, color='blue', label='Kept in CRNN')
    axes[1, 0].hist(errors[routed], bins=50, alpha=0.5, color='orange', label='Routed to SRP')
    axes[1, 0].axvline(5.0, color='red', linestyle='--', linewidth=2, label='Failure threshold')
    axes[1, 0].set_xlabel('Absolute Error (degrees)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Error Distribution by Routing Decision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confusion matrix as bar chart
    categories = ['TP', 'TN', 'FP', 'FN']
    values = [
        results['true_positives'],
        results['true_negatives'],
        results['false_positives'],
        results['false_negatives']
    ]
    colors_bar = ['green', 'lightgreen', 'orange', 'red']
    axes[1, 1].bar(categories, values, color=colors_bar)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (cat, val) in enumerate(zip(categories, values)):
        axes[1, 1].text(i, val, str(val), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'combined_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to: {output_dir}")


def compare_methods(
    simple_threshold_results: Dict,
    temp_scaling_results: Dict,
    mahalanobis_results: Dict,
    combined_results: Dict,
    output_path: str
):
    """
    Create comparison table of all methods.

    Args:
        simple_threshold_results: Results from simple max_prob threshold
        temp_scaling_results: Results from temperature scaling only
        mahalanobis_results: Results from Mahalanobis only
        combined_results: Results from combined method
        output_path: Path to save comparison table
    """
    import pandas as pd

    methods = ['Simple Threshold', 'Temperature Scaling', 'Mahalanobis', 'Combined (OR)']
    all_results = [simple_threshold_results, temp_scaling_results, mahalanobis_results, combined_results]

    comparison = {
        'Method': methods,
        'Precision': [r['precision'] for r in all_results],
        'Recall': [r['recall'] for r in all_results],
        'F1 Score': [r['f1_score'] for r in all_results],
        'Accuracy': [r['accuracy'] for r in all_results],
        'Routing Rate': [r['routing_rate'] for r in all_results],
    }

    df = pd.DataFrame(comparison)

    print("\n" + "="*80)
    print("Method Comparison")
    print("="*80)
    print(df.to_string(index=False))

    # Save to CSV
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")

    return df


if __name__ == "__main__":
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, required=True)
    parser.add_argument('--test_features', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--n_pca_components', type=int, default=64)
    parser.add_argument('--strategy', type=str, default='or', choices=['or', 'and'])

    args = parser.parse_args()

    # Load features
    print("Loading features...")
    train_data = np.load(args.train_features, allow_pickle=True)
    test_data = np.load(args.test_features, allow_pickle=True)

    train_features = {key: train_data[key] for key in train_data.files}
    test_features = {key: test_data[key] for key in test_data.files}

    # Initialize and fit detector
    detector = CombinedFailureDetector()
    detector.fit(train_features, n_pca_components=args.n_pca_components)

    # Grid search for optimal thresholds
    best_conf, best_dist, all_results = detector.grid_search_combined(
        test_features,
        strategy=args.strategy
    )

    # Get best results
    best_results = [r for r in all_results if
                    r['confidence_threshold'] == best_conf and
                    r['distance_threshold'] == best_dist][0]

    # Visualize
    visualize_combined_results(best_results, args.output_dir)

    # Save results
    output_path = Path(args.output_dir) / 'combined_routing_results.npz'
    np.savez(
        output_path,
        best_confidence_threshold=best_conf,
        best_distance_threshold=best_dist,
        temperature=detector.temperature,
        best_results=best_results,
        all_results=all_results
    )

    print(f"\nResults saved to: {output_path}")
