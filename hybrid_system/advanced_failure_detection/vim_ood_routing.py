#!/usr/bin/env python3
"""
VIM (Virtual-Logit Matching) OOD Detection for Routing

Based on "ViM: Out-Of-Distribution with Virtual-logit Matching" (Wang et al., 2022)
Uses residual space of logits (null space) to detect OOD samples.

Paper insight: ID samples have low norm in residual space, OOD samples have high norm.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class VIMOODRouter:
    """VIM OOD detection using virtual logit matching."""

    def __init__(self, model_path=None, alpha=1.0):
        """
        Args:
            model_path: Path to saved VIM model
            alpha: Weight for combining principal and residual scores
        """
        self.alpha = alpha
        self.pca = None
        self.scaler = None
        self.principal_dim = None
        self.logit_mean = None
        self.logit_std = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, features):
        """
        Train VIM by computing principal subspace of logits.

        Args:
            features: Dict with 'logits_pre_sig' (N, n_forward, n_classes)
        """
        # Average over MC forward passes
        logits = features['logits_pre_sig']
        if logits.dtype == object:
            # Logits stored as array of arrays - convert and average
            logits_avg = np.array([l.mean(axis=0) for l in logits])  # (N, n_classes)
        elif len(logits.shape) == 3:
            logits_avg = logits.mean(axis=1)  # (N, n_classes)
        else:
            logits_avg = logits

        # Standardize logits
        self.scaler = StandardScaler()
        logits_scaled = self.scaler.fit_transform(logits_avg)

        # Store mean and std for later
        self.logit_mean = logits_avg.mean(axis=0)
        self.logit_std = logits_avg.std(axis=0)

        # Compute principal subspace using PCA
        # Keep 99% of variance as principal subspace
        self.pca = PCA(n_components=0.99, random_state=42)
        self.pca.fit(logits_scaled)

        self.principal_dim = self.pca.n_components_

        print(f"VIM trained:")
        print(f"  Logit dimension: {logits_avg.shape[1]}")
        print(f"  Principal dimension: {self.principal_dim}")
        print(f"  Residual dimension: {logits_avg.shape[1] - self.principal_dim}")
        print(f"  Variance explained: {self.pca.explained_variance_ratio_.sum():.4f}")

    def compute_vim_scores(self, features):
        """
        Compute VIM OOD scores.

        Higher score = more likely OOD

        Args:
            features: Dict with 'logits_pre_sig'

        Returns:
            vim_scores: Array of OOD scores (higher = more OOD)
        """
        # Average over MC forward passes
        logits = features['logits_pre_sig']
        if logits.dtype == object:
            # Logits stored as array of arrays - convert and average
            logits_avg = np.array([l.mean(axis=0) for l in logits])
        elif len(logits.shape) == 3:
            logits_avg = logits.mean(axis=1)
        else:
            logits_avg = logits

        # Standardize using training statistics
        logits_scaled = self.scaler.transform(logits_avg)

        # Project onto principal subspace
        principal_components = self.pca.transform(logits_scaled)

        # Reconstruct from principal components
        logits_reconstructed = self.pca.inverse_transform(principal_components)

        # Compute residual (what's NOT in principal subspace)
        residual = logits_scaled - logits_reconstructed

        # Compute residual norm (this is the VIM score)
        residual_norm = np.linalg.norm(residual, axis=1)

        # Also compute principal norm for reference
        principal_norm = np.linalg.norm(principal_components, axis=1)

        # VIM score: higher residual norm = more OOD
        # Optionally weight principal and residual norms
        vim_scores = residual_norm - self.alpha * principal_norm

        return vim_scores

    def predict_routing(self, features, threshold):
        """
        Predict which samples to route to SRP based on VIM scores.

        Args:
            features: Dict with CRNN features
            threshold: VIM score threshold

        Returns:
            route_to_srp: Boolean array (True = route to SRP)
            vim_scores: Array of VIM scores
        """
        vim_scores = self.compute_vim_scores(features)
        route_to_srp = vim_scores > threshold
        return route_to_srp, vim_scores

    def save(self, path):
        """Save trained VIM model."""
        model = {
            'pca': self.pca,
            'scaler': self.scaler,
            'principal_dim': self.principal_dim,
            'logit_mean': self.logit_mean,
            'logit_std': self.logit_std,
            'alpha': self.alpha
        }
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"VIM model saved to {path}")

    def load(self, path):
        """Load trained VIM model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.pca = model['pca']
        self.scaler = model['scaler']
        self.principal_dim = model['principal_dim']
        self.logit_mean = model['logit_mean']
        self.logit_std = model['logit_std']
        self.alpha = model['alpha']
        print(f"VIM model loaded from {path}")


if __name__ == "__main__":
    # Test VIM routing
    import sys
    sys.path.append("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection")

    # Load test features
    features_path = "features/test_3x12cm_consecutive_features.npz"
    print(f"Loading features from {features_path}")
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}

    # Train VIM
    print("\nTraining VIM...")
    router = VIMOODRouter(alpha=1.0)
    router.train(features)

    # Test different thresholds
    print("\nTesting thresholds:")
    vim_scores = router.compute_vim_scores(features)

    for percentile in [70, 75, 80, 85, 90]:
        threshold = np.percentile(vim_scores, percentile)
        route_to_srp, _ = router.predict_routing(features, threshold)
        routing_rate = route_to_srp.sum() / len(route_to_srp) * 100
        print(f"  p{percentile} (threshold={threshold:.4f}): {routing_rate:.1f}% routing ({route_to_srp.sum()} samples)")
