"""
Mahalanobis Distance for Out-of-Distribution Detection

Implements Mahalanobis distance-based OOD detection in feature space (Lee et al., 2018).
Uses class-conditional Gaussian distributions fitted on training features to detect
when test samples fall far from the training distribution (e.g., geometric mismatch).

Reference:
    Lee, K., Lee, K., Lee, H., & Shin, J. (2018).
    A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.
    NeurIPS 2018.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
from pathlib import Path


class MahalanobisOODDetector:
    """Mahalanobis distance-based OOD detector."""

    def __init__(self, n_components: Optional[int] = None, use_pca: bool = True):
        """
        Initialize detector.

        Args:
            n_components: Number of PCA components (None = no reduction)
            use_pca: Whether to use PCA for dimensionality reduction
        """
        self.n_components = n_components
        self.use_pca = use_pca
        self.pca = None
        self.class_means = {}
        self.shared_covariance = None
        self.fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray, angle_resolution: int = 10):
        """
        Fit class-conditional Gaussians on training features.

        Args:
            features: (N, D) penultimate layer features
            labels: (N,) ground truth angles in degrees
            angle_resolution: Group angles into bins of this size (degrees)
        """
        print("\nFitting Mahalanobis OOD detector...")
        print(f"  Features shape: {features.shape}")
        print(f"  Using PCA: {self.use_pca}")
        if self.use_pca and self.n_components:
            print(f"  PCA components: {self.n_components}")

        # Apply PCA if requested
        if self.use_pca and self.n_components:
            print("  Fitting PCA...")
            self.pca = PCA(n_components=self.n_components)
            features = self.pca.fit_transform(features)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"  PCA variance explained: {explained_var*100:.1f}%")

        # Group angles into bins for class-conditional modeling
        angle_bins = (labels // angle_resolution) * angle_resolution
        unique_bins = np.unique(angle_bins)
        print(f"  Number of angle bins: {len(unique_bins)}")

        # Compute class means
        for bin_angle in unique_bins:
            mask = angle_bins == bin_angle
            self.class_means[bin_angle] = features[mask].mean(axis=0)

        # Compute shared covariance matrix (pooled covariance)
        print("  Computing shared covariance matrix...")
        all_centered = []
        for bin_angle in unique_bins:
            mask = angle_bins == bin_angle
            centered = features[mask] - self.class_means[bin_angle]
            all_centered.append(centered)

        all_centered = np.vstack(all_centered)

        # Use empirical covariance with regularization
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(all_centered)
        self.shared_covariance = cov_estimator.covariance_

        # Add small regularization for numerical stability
        eye = np.eye(self.shared_covariance.shape[0])
        self.shared_covariance += 1e-6 * eye

        self.fitted = True
        print("  Fitting complete!")

    def compute_mahalanobis_distance(
        self,
        features: np.ndarray,
        predicted_angles: np.ndarray
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance for test features.

        Args:
            features: (N, D) test features
            predicted_angles: (N,) predicted angles (used to select class mean)

        Returns:
            distances: (N,) Mahalanobis distances
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before computing distances")

        # Apply PCA transform if used
        if self.pca is not None:
            features = self.pca.transform(features)

        # Compute precision matrix (inverse covariance)
        precision = np.linalg.inv(self.shared_covariance)

        distances = np.zeros(len(features))

        # Find nearest class mean for each sample
        angle_resolution = 10
        predicted_bins = (predicted_angles // angle_resolution) * angle_resolution

        for i, (feat, pred_bin) in enumerate(zip(features, predicted_bins)):
            # Get closest class mean
            if pred_bin in self.class_means:
                class_mean = self.class_means[pred_bin]
            else:
                # Fall back to nearest available bin
                available_bins = np.array(list(self.class_means.keys()))
                nearest_bin = available_bins[np.argmin(np.abs(available_bins - pred_bin))]
                class_mean = self.class_means[nearest_bin]

            # Compute Mahalanobis distance
            diff = feat - class_mean
            distances[i] = np.sqrt(diff @ precision @ diff)

        return distances

    def evaluate_ood_detection(
        self,
        in_dist_features: Dict,
        out_dist_features: Dict,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Evaluate OOD detection performance.

        Args:
            in_dist_features: In-distribution test features (e.g., 6cm array)
            out_dist_features: Out-of-distribution test features (e.g., 3x12cm array)
            threshold: Distance threshold for OOD detection (None = auto-select)

        Returns:
            results: Dictionary with metrics
        """
        print("\n" + "="*80)
        print("Evaluating OOD Detection")
        print("="*80)

        # Get features (average over time if needed)
        if in_dist_features['penultimate_features'][0].ndim == 2:
            in_features = np.array([f.mean(axis=0) for f in in_dist_features['penultimate_features']])
            out_features = np.array([f.mean(axis=0) for f in out_dist_features['penultimate_features']])
        else:
            in_features = in_dist_features['penultimate_features']
            out_features = out_dist_features['penultimate_features']

        # Get predicted angles
        in_pred_angles = in_dist_features['predicted_angles']
        out_pred_angles = out_dist_features['predicted_angles']

        # Compute Mahalanobis distances
        print("Computing Mahalanobis distances...")
        in_distances = self.compute_mahalanobis_distance(in_features, in_pred_angles)
        out_distances = self.compute_mahalanobis_distance(out_features, out_pred_angles)

        print(f"  In-dist distances - Mean: {in_distances.mean():.2f}, Std: {in_distances.std():.2f}")
        print(f"  Out-dist distances - Mean: {out_distances.mean():.2f}, Std: {out_distances.std():.2f}")

        # Auto-select threshold if not provided (using median of out-dist)
        if threshold is None:
            threshold = np.percentile(in_distances, 95)  # 95th percentile of in-dist
            print(f"  Auto-selected threshold: {threshold:.2f}")

        # Classify as OOD if distance > threshold
        in_pred_ood = in_distances > threshold
        out_pred_ood = out_distances > threshold

        # Metrics
        # True labels: 0 = in-dist, 1 = OOD
        tp = out_pred_ood.sum()  # Correctly detected OOD
        tn = (~in_pred_ood).sum()  # Correctly identified in-dist
        fp = in_pred_ood.sum()  # False OOD alarms
        fn = (~out_pred_ood).sum()  # Missed OOD

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (len(in_distances) + len(out_distances))

        # AUROC (using sklearn if available)
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            labels = np.concatenate([np.zeros(len(in_distances)), np.ones(len(out_distances))])
            scores = np.concatenate([in_distances, out_distances])
            auroc = roc_auc_score(labels, scores)
            print(f"  AUROC: {auroc:.3f}")
        except ImportError:
            auroc = None

        results = {
            'threshold': threshold,
            'n_in_dist': len(in_distances),
            'n_out_dist': len(out_distances),
            'in_dist_distances': in_distances,
            'out_dist_distances': out_distances,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'auroc': auroc,
        }

        print(f"\nMetrics:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")

        return results


def visualize_feature_space(
    train_features: Dict,
    in_dist_features: Dict,
    out_dist_features: Dict,
    output_path: str,
    use_pca: bool = True,
    n_samples: int = 2000
):
    """
    Create t-SNE visualization of feature space.

    Args:
        train_features: Training features dict
        in_dist_features: In-distribution test features
        out_dist_features: Out-of-distribution test features
        output_path: Path to save plot
        use_pca: Whether to apply PCA before t-SNE (recommended for speed)
        n_samples: Maximum samples per dataset (for speed)
    """
    print("\nGenerating t-SNE visualization...")

    from sklearn.manifold import TSNE

    # Get features (average over time if needed)
    def get_features(data, max_samples):
        if data['penultimate_features'][0].ndim == 2:
            features = np.array([f.mean(axis=0) for f in data['penultimate_features']])
        else:
            features = data['penultimate_features']

        # Subsample if needed
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]

        return features

    train_feat = get_features(train_features, n_samples)
    in_feat = get_features(in_dist_features, n_samples)
    out_feat = get_features(out_dist_features, n_samples)

    # Combine all features
    all_features = np.vstack([train_feat, in_feat, out_feat])
    labels = np.array(
        ['Train'] * len(train_feat) +
        ['Test (6cm)'] * len(in_feat) +
        ['Test (3x12cm)'] * len(out_feat)
    )

    # Apply PCA for speed
    if use_pca:
        print("  Applying PCA (256 -> 50 dims)...")
        pca = PCA(n_components=50)
        all_features = pca.fit_transform(all_features)

    # Apply t-SNE
    print("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features)

    # Plot
    print("  Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'Train': 'blue', 'Test (6cm)': 'green', 'Test (3x12cm)': 'red'}
    for label in colors.keys():
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[label],
            label=label,
            alpha=0.5,
            s=10
        )

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('Feature Space Visualization: Geometric Mismatch as OOD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")

    plt.close()


def evaluate_routing_with_mahalanobis(
    test_features: Dict,
    mahalanobis_distances: np.ndarray,
    distance_threshold: float
) -> Dict:
    """
    Evaluate routing using Mahalanobis distance.

    Route to SRP if Mahalanobis distance > threshold (OOD).

    Args:
        test_features: Test features dict
        mahalanobis_distances: Precomputed Mahalanobis distances
        distance_threshold: Distance threshold for routing

    Returns:
        results: Dictionary with routing metrics
    """
    print(f"\nEvaluating routing with Mahalanobis threshold={distance_threshold:.2f}")

    # Routing decision
    route_to_srp = mahalanobis_distances > distance_threshold

    # Get predicted angles and errors
    predicted_angles = test_features['predicted_angles']
    gt_angles = test_features['gt_angles']
    errors = np.abs((predicted_angles - gt_angles + 180) % 360 - 180)

    # Success = correctly identify failures (error > 5 degrees)
    actual_failures = errors > 5.0
    detected_failures = route_to_srp

    # Confusion matrix
    tp = np.sum(actual_failures & detected_failures)
    tn = np.sum(~actual_failures & ~detected_failures)
    fp = np.sum(~actual_failures & detected_failures)
    fn = np.sum(actual_failures & ~detected_failures)

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(errors)

    results = {
        'distance_threshold': distance_threshold,
        'n_samples': len(errors),
        'n_routed_to_srp': route_to_srp.sum(),
        'routing_rate': route_to_srp.mean(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
    }

    print(f"  Routing rate: {results['routing_rate']*100:.1f}% ({results['n_routed_to_srp']}/{results['n_samples']})")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")

    return results


if __name__ == "__main__":
    """Example usage for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, required=True,
                        help='Path to training features .npz file (6cm)')
    parser.add_argument('--test_in_dist', type=str, required=True,
                        help='Path to in-distribution test features (6cm)')
    parser.add_argument('--test_out_dist', type=str, required=True,
                        help='Path to out-of-distribution test features (3x12cm)')
    parser.add_argument('--n_components', type=int, default=64,
                        help='Number of PCA components')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results')

    args = parser.parse_args()

    # Load features
    print("Loading features...")
    train_data = np.load(args.train_features, allow_pickle=True)
    test_in_data = np.load(args.test_in_dist, allow_pickle=True)
    test_out_data = np.load(args.test_out_dist, allow_pickle=True)

    # Convert to dict format
    train_features = {key: train_data[key] for key in train_data.files}
    in_dist_features = {key: test_in_data[key] for key in test_in_data.files}
    out_dist_features = {key: test_out_data[key] for key in test_out_data.files}

    # Get average features if needed
    if train_features['penultimate_features'][0].ndim == 2:
        train_feat = np.array([f.mean(axis=0) for f in train_features['penultimate_features']])
    else:
        train_feat = train_features['penultimate_features']

    train_angles = train_features['gt_angles']

    # Initialize and fit detector
    detector = MahalanobisOODDetector(n_components=args.n_components, use_pca=True)
    detector.fit(train_feat, train_angles)

    # Evaluate OOD detection
    results = detector.evaluate_ood_detection(
        in_dist_features,
        out_dist_features
    )

    # Visualize feature space
    visualize_feature_space(
        train_features,
        in_dist_features,
        out_dist_features,
        output_path=f"{args.output_dir}/feature_space_tsne.png"
    )

    # Save results
    output_path = Path(args.output_dir) / 'mahalanobis_results.npz'
    np.savez(
        output_path,
        **results
    )

    print(f"\nResults saved to: {output_path}")
