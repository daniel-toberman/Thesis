#!/usr/bin/env python3
"""
Deep SVDD Router for Failure Detection
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path


class SVDDNetwork(nn.Module):
    """Deep SVDD network architecture (must match training)."""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(SVDDNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepSVDDRouter:
    """
    Failure detector using Deep SVDD (Support Vector Data Description).

    Routes to SRP when distance from hypersphere center is above threshold.
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize router.

        Args:
            model_path: Path to trained model pickle file
            device: Device to use ('cpu', 'mps', 'cuda')
        """
        self.network = None
        self.center = None
        self.feature_dim = None
        self.hidden_dims = None
        self.error_threshold = None
        self.device = device

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained Deep SVDD model from pickle."""
        print(f"Loading Deep SVDD model from: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.feature_dim = model['feature_dim']
        self.hidden_dims = model['hidden_dims']
        self.error_threshold = model.get('error_threshold', 5.0)

        # Reconstruct network
        self.network = SVDDNetwork(
            input_dim=self.feature_dim,
            hidden_dims=self.hidden_dims
        )
        self.network.load_state_dict(model['network_state_dict'])
        self.network = self.network.to(self.device)
        self.network.eval()

        # Load center
        self.center = torch.FloatTensor(model['center']).to(self.device)

        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Hidden dimensions: {self.hidden_dims}")
        print(f"  Error threshold used in training: {self.error_threshold}°")
        print(f"  Device: {self.device}")

    def compute_distances(self, features):
        """
        Compute distances from hypersphere center.

        Args:
            features: (N,) array where each element is (T, D) temporal features

        Returns:
            distances: (N,) array of distances from center
        """
        if self.network is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Average over temporal dimension
        features_avg = np.array([f.mean(axis=0) for f in features])

        # Convert to torch tensor
        features_tensor = torch.FloatTensor(features_avg).to(self.device)

        # Compute distances
        with torch.no_grad():
            outputs = self.network(features_tensor)
            distances = torch.sum((outputs - self.center) ** 2, dim=1)

        return distances.cpu().numpy()

    def predict_routing(self, features, distance_threshold):
        """
        Predict routing decisions.

        Args:
            features: Dict with 'penultimate_features' key
            distance_threshold: Distance threshold (route if distance > threshold)

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            distances: (N,) distance scores
        """
        feat = features['penultimate_features']

        # Compute distances from hypersphere center
        distances = self.compute_distances(feat)

        # Route if distance > threshold (outside hypersphere = failure)
        route_to_srp = distances > distance_threshold

        return route_to_srp, distances

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25):
        """
        Find distance threshold that achieves target routing rate.

        Args:
            features: Dict with 'penultimate_features' key
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)

        Returns:
            optimal_threshold: Distance threshold
        """
        distances = self.compute_distances(features['penultimate_features'])

        # Find threshold at target percentile
        threshold = np.percentile(distances, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = distances > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"Distance threshold: {threshold:.4f}")

        return threshold
