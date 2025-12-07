#!/usr/bin/env python3
"""
Mahalanobis Distance OOD Router for Failure Detection

Based on "Generalized Out-of-Distribution Detection" survey paper.
Computes Mahalanobis distance to class centroids in feature space.

This is the classic Mahalanobis distance method from the paper,
separate from the Temp+Mahal combined method used previously.
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import mahalanobis


class MahalanobisOODRouter:
    """
    Failure detector using Mahalanobis distance-based OOD detection.

    Computes distance to class mean in feature space using covariance matrix.
    High Mahalanobis distance indicates OOD.
    """

    def __init__(self, model_path=None):
        """
        Initialize router.

        Args:
            model_path: Path to trained Mahalanobis model pickle file
        """
        self.class_means = None
        self.precision_matrix = None
        self.error_threshold = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained Mahalanobis model from pickle."""
        print(f"Loading Mahalanobis model from: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.class_means = model['class_means']
        self.precision_matrix = model['precision_matrix']
        self.error_threshold = model.get('error_threshold', 5.0)

        print(f"  Number of classes: {len(self.class_means)}")
        print(f"  Feature dim: {self.class_means[0].shape[0]}")
        print(f"  Error threshold used in training: {self.error_threshold}°")

    def train(self, train_features, train_labels=None, error_threshold=5.0):
        """
        Train Mahalanobis model on in-distribution data.

        Args:
            train_features: Dict with 'penultimate_features' key
            train_labels: (N,) array of class labels (if available)
                          If None, treats all data as single class
            error_threshold: Error threshold defining "failure"
        """
        self.error_threshold = error_threshold

        # Extract and average penultimate features
        features = train_features['penultimate_features']
        features_avg = np.array([f.mean(axis=0) for f in features])

        if train_labels is not None:
            # Compute per-class means
            unique_labels = np.unique(train_labels)
            self.class_means = []

            for label in unique_labels:
                class_features = features_avg[train_labels == label]
                class_mean = class_features.mean(axis=0)
                self.class_means.append(class_mean)

            print(f"Computed means for {len(self.class_means)} classes")
        else:
            # Treat all data as single class (use global mean)
            global_mean = features_avg.mean(axis=0)
            self.class_means = [global_mean]
            print(f"Using single class mean (no labels provided)")

        # Compute tied covariance matrix (shared across classes)
        # Center data around global mean
        global_mean = features_avg.mean(axis=0)
        centered_features = features_avg - global_mean

        # Covariance matrix
        cov_matrix = np.cov(centered_features, rowvar=False)

        # Add regularization to avoid singular matrix
        epsilon = 1e-6
        cov_matrix += epsilon * np.eye(cov_matrix.shape[0])

        # Compute precision matrix (inverse of covariance)
        try:
            self.precision_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print("⚠️  Covariance matrix is singular, using pseudo-inverse")
            self.precision_matrix = np.linalg.pinv(cov_matrix)

        print(f"Mahalanobis trained on {len(features)} samples")
        print(f"  Feature dim: {features_avg.shape[1]}")
        print(f"  Covariance matrix condition number: {np.linalg.cond(cov_matrix):.2e}")

    def compute_mahalanobis_distances(self, features):
        """
        Compute Mahalanobis distance to nearest class mean.

        Args:
            features: Dict with 'penultimate_features' key

        Returns:
            mahal_distances: (N,) array of Mahalanobis distances
        """
        if self.class_means is None or self.precision_matrix is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        # Process features
        features_avg = np.array([f.mean(axis=0) for f in features['penultimate_features']])

        mahal_distances = []

        for feature in features_avg:
            # Compute distance to each class mean
            distances_to_classes = []

            for class_mean in self.class_means:
                # Mahalanobis distance: sqrt((x-μ)^T Σ^-1 (x-μ))
                diff = feature - class_mean
                distance = np.sqrt(diff @ self.precision_matrix @ diff)
                distances_to_classes.append(distance)

            # Use minimum distance (distance to nearest class)
            min_distance = min(distances_to_classes)
            mahal_distances.append(min_distance)

        return np.array(mahal_distances)

    def predict_routing(self, features, mahalanobis_threshold):
        """
        Predict routing decisions.

        Args:
            features: Dict with 'penultimate_features' key
            mahalanobis_threshold: Distance threshold (route if distance > threshold)

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            mahal_distances: (N,) Mahalanobis distances
        """
        # Compute Mahalanobis distances
        mahal_distances = self.compute_mahalanobis_distances(features)

        # Route if distance > threshold (far from class means = OOD = route)
        route_to_srp = mahal_distances > mahalanobis_threshold

        return route_to_srp, mahal_distances

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25):
        """
        Find Mahalanobis distance threshold that achieves target routing rate.

        Args:
            features: Dict with 'penultimate_features' key
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)

        Returns:
            optimal_threshold: Mahalanobis distance threshold
        """
        mahal_distances = self.compute_mahalanobis_distances(features)

        # Find threshold at target percentile
        threshold = np.percentile(mahal_distances, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = mahal_distances > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"Mahalanobis distance threshold: {threshold:.6f}")

        return threshold

    def save_model(self, save_path):
        """Save trained Mahalanobis model to pickle file."""
        model = {
            'class_means': self.class_means,
            'precision_matrix': self.precision_matrix,
            'error_threshold': self.error_threshold
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Mahalanobis model saved to: {save_path}")
