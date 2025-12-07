#!/usr/bin/env python3
"""
KNN Distance-Based OOD Router for Failure Detection

Based on "Generalized Out-of-Distribution Detection" survey paper.
Uses non-parametric nearest-neighbor distance in feature space.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


class KNNOODRouter:
    """
    Failure detector using KNN distance-based OOD detection.

    Routes to SRP when distance to K nearest neighbors is high (OOD).
    Paper: "recent work [117] shows strong promise of non-parametric
    nearest-neighbor distance for OOD detection"
    """

    def __init__(self, model_path=None, k=5):
        """
        Initialize router.

        Args:
            model_path: Path to trained KNN model pickle file
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.knn = None
        self.train_features = None
        self.error_threshold = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained KNN model from pickle."""
        print(f"Loading KNN model from: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.train_features = model['train_features']
        self.k = model.get('k', 5)
        self.error_threshold = model.get('error_threshold', 5.0)

        # Build KNN index
        self.knn = NearestNeighbors(n_neighbors=self.k, metric='euclidean', n_jobs=-1)
        self.knn.fit(self.train_features)

        print(f"  K: {self.k}")
        print(f"  Train samples: {len(self.train_features)}")
        print(f"  Feature dim: {self.train_features.shape[1]}")
        print(f"  Error threshold used in training: {self.error_threshold}°")

    def train(self, train_features, error_threshold=5.0):
        """
        Train KNN model on in-distribution data.

        Args:
            train_features: Dict with 'penultimate_features' key
            error_threshold: Error threshold defining "failure"
        """
        self.error_threshold = error_threshold

        # Extract and average penultimate features
        features = train_features['penultimate_features']
        self.train_features = self._process_features(features)

        # Build KNN index
        self.knn = NearestNeighbors(n_neighbors=self.k, metric='euclidean', n_jobs=-1)
        self.knn.fit(self.train_features)

        print(f"KNN trained on {len(self.train_features)} samples")
        print(f"  K: {self.k}")
        print(f"  Feature dim: {self.train_features.shape[1]}")

    def _process_features(self, features):
        """
        Process penultimate features by averaging over temporal dimension.

        Args:
            features: (N,) array where each element is (T, D) temporal features

        Returns:
            processed: (N, D) array of averaged features
        """
        return np.array([f.mean(axis=0) for f in features])

    def compute_knn_distances(self, features):
        """
        Compute distance to K nearest neighbors in feature space.

        Args:
            features: Dict with 'penultimate_features' key

        Returns:
            knn_distances: (N,) array of average distances to K nearest neighbors
        """
        if self.knn is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        # Process features
        test_features = self._process_features(features['penultimate_features'])

        # Compute distances to K nearest neighbors
        distances, indices = self.knn.kneighbors(test_features)

        # Average distance to K neighbors (higher = more OOD)
        knn_distances = distances.mean(axis=1)

        return knn_distances

    def predict_routing(self, features, distance_threshold):
        """
        Predict routing decisions.

        Args:
            features: Dict with 'penultimate_features' key
            distance_threshold: KNN distance threshold (route if distance > threshold)

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            knn_distances: (N,) KNN distances
        """
        # Compute KNN distances
        knn_distances = self.compute_knn_distances(features)

        # Route if distance > threshold (far from training data = OOD)
        route_to_srp = knn_distances > distance_threshold

        return route_to_srp, knn_distances

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25):
        """
        Find KNN distance threshold that achieves target routing rate.

        Args:
            features: Dict with 'penultimate_features' key
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)

        Returns:
            optimal_threshold: KNN distance threshold
        """
        knn_distances = self.compute_knn_distances(features)

        # Find threshold at target percentile
        threshold = np.percentile(knn_distances, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = knn_distances > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"KNN distance threshold: {threshold:.6f}")

        return threshold

    def save_model(self, save_path):
        """Save trained KNN model to pickle file."""
        model = {
            'train_features': self.train_features,
            'k': self.k,
            'error_threshold': self.error_threshold
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"KNN model saved to: {save_path}")
