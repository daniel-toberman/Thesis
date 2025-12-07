#!/usr/bin/env python3
"""
SHE (Spectral-normalized Embedding) OOD Detection for Routing

Based on "SHE: A Fast and Accurate Deep Neural Network for Encrypted Traffic Classification"
Uses stored patterns representing each class to measure discrepancy.

Paper claims: "hyperparameter-free and computationally efficient"
"""

import numpy as np
import pickle
from pathlib import Path


class SHEOODRouter:
    """SHE OOD detection using stored pattern matching."""

    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to saved SHE model
        """
        self.class_patterns = None  # Representative patterns for each class
        self.class_stds = None  # Standard deviations for each class
        self.n_classes = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, features):
        """
        Train SHE by storing representative patterns for each class.

        Args:
            features: Dict with 'penultimate_features', 'predicted_angles', 'gt_angles'
        """
        # Average over MC forward passes if needed
        penult_features = features['penultimate_features']
        if penult_features.dtype == object:
            # Features stored as array of arrays - convert and average
            features_avg = np.array([f.mean(axis=0) if hasattr(f, 'mean') else f for f in penult_features])
        elif len(penult_features.shape) == 3:
            features_avg = np.array([f.mean(axis=0) for f in penult_features])
        else:
            features_avg = penult_features

        predicted_angles = features['predicted_angles']
        gt_angles = features['gt_angles']

        # Bin angles into classes (e.g., 36 classes for 10° bins)
        n_bins = 36  # 360° / 10° bins
        predicted_classes = (predicted_angles // 10).astype(int) % n_bins
        gt_classes = (gt_angles // 10).astype(int) % n_bins

        self.n_classes = n_bins

        # For each class, compute representative pattern (mean) and std
        self.class_patterns = []
        self.class_stds = []

        for class_id in range(self.n_classes):
            # Use samples where prediction matches this class
            class_mask = predicted_classes == class_id

            if class_mask.sum() > 0:
                class_features = features_avg[class_mask]

                # Representative pattern = mean of class features
                class_pattern = class_features.mean(axis=0)

                # Standard deviation for normalization
                class_std = class_features.std(axis=0)
                class_std = np.where(class_std > 1e-8, class_std, 1.0)  # Avoid division by zero

                self.class_patterns.append(class_pattern)
                self.class_stds.append(class_std)
            else:
                # No samples for this class, use zero pattern
                feature_dim = features_avg.shape[1]
                self.class_patterns.append(np.zeros(feature_dim))
                self.class_stds.append(np.ones(feature_dim))

        self.class_patterns = np.array(self.class_patterns)  # (n_classes, feature_dim)
        self.class_stds = np.array(self.class_stds)  # (n_classes, feature_dim)

        print(f"SHE trained:")
        print(f"  Number of classes: {self.n_classes}")
        print(f"  Feature dimension: {self.class_patterns.shape[1]}")
        print(f"  Classes with samples: {(np.linalg.norm(self.class_patterns, axis=1) > 0).sum()}")

    def compute_she_scores(self, features):
        """
        Compute SHE OOD scores.

        Higher score = more likely OOD (higher discrepancy from stored patterns)

        Args:
            features: Dict with 'penultimate_features', 'predicted_angles'

        Returns:
            she_scores: Array of OOD scores (higher = more OOD)
        """
        # Average over MC forward passes if needed
        penult_features = features['penultimate_features']
        if penult_features.dtype == object:
            # Features stored as array of arrays - convert and average
            features_avg = np.array([f.mean(axis=0) if hasattr(f, 'mean') else f for f in penult_features])
        elif len(penult_features.shape) == 3:
            features_avg = np.array([f.mean(axis=0) for f in penult_features])
        else:
            features_avg = penult_features

        predicted_angles = features['predicted_angles']

        # Bin predictions into classes
        predicted_classes = (predicted_angles // 10).astype(int) % self.n_classes

        # Compute discrepancy from stored pattern for each sample
        she_scores = []
        for i, (feature, pred_class) in enumerate(zip(features_avg, predicted_classes)):
            # Get the stored pattern for this predicted class
            class_pattern = self.class_patterns[pred_class]
            class_std = self.class_stds[pred_class]

            # Compute normalized distance to class pattern
            # Distance = ||feature - pattern|| / std
            diff = feature - class_pattern
            normalized_diff = diff / class_std
            distance = np.linalg.norm(normalized_diff)

            she_scores.append(distance)

        return np.array(she_scores)

    def predict_routing(self, features, threshold):
        """
        Predict which samples to route to SRP based on SHE scores.

        Args:
            features: Dict with CRNN features
            threshold: SHE score threshold

        Returns:
            route_to_srp: Boolean array (True = route to SRP)
            she_scores: Array of SHE scores
        """
        she_scores = self.compute_she_scores(features)
        route_to_srp = she_scores > threshold
        return route_to_srp, she_scores

    def save(self, path):
        """Save trained SHE model."""
        model = {
            'class_patterns': self.class_patterns,
            'class_stds': self.class_stds,
            'n_classes': self.n_classes
        }
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"SHE model saved to {path}")

    def load(self, path):
        """Load trained SHE model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.class_patterns = model['class_patterns']
        self.class_stds = model['class_stds']
        self.n_classes = model['n_classes']
        print(f"SHE model loaded from {path}")


if __name__ == "__main__":
    # Test SHE routing
    import sys
    sys.path.append("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection")

    # Load test features
    features_path = "features/test_3x12cm_consecutive_features.npz"
    print(f"Loading features from {features_path}")
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}

    # Train SHE
    print("\nTraining SHE...")
    router = SHEOODRouter()
    router.train(features)

    # Test different thresholds
    print("\nTesting thresholds:")
    she_scores = router.compute_she_scores(features)

    for percentile in [70, 75, 80, 85, 90]:
        threshold = np.percentile(she_scores, percentile)
        route_to_srp, _ = router.predict_routing(features, threshold)
        routing_rate = route_to_srp.sum() / len(route_to_srp) * 100
        print(f"  p{percentile} (threshold={threshold:.4f}): {routing_rate:.1f}% routing ({route_to_srp.sum()} samples)")
