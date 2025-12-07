#!/usr/bin/env python3
"""
LLR (Likelihood Ratio) OOD Detection for Routing

Based on likelihood ratio methods from OOD detection literature.
Trains a density model on in-distribution features and uses likelihood ratios
to detect out-of-distribution samples.

Key idea: ID samples have high likelihood under the trained density model,
OOD samples have low likelihood.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class LLROODRouter:
    """LLR OOD detection using Gaussian Mixture Model for density estimation."""

    def __init__(self, model_path=None, n_components=10):
        """
        Args:
            model_path: Path to saved LLR model
            n_components: Number of Gaussian components in the mixture model
        """
        self.n_components = n_components
        self.gmm = None
        self.scaler = None
        self.background_ll = None  # Background log-likelihood for ratio

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, train_features):
        """
        Train LLR by fitting a Gaussian Mixture Model to training features.

        Args:
            train_features: Dict with 'penultimate_features'
        """
        # Average over MC forward passes if needed
        penult_features = train_features['penultimate_features']
        if penult_features.dtype == object:
            features_avg = np.array([f.mean(axis=0) if hasattr(f, 'mean') else f
                                    for f in penult_features])
        elif len(penult_features.shape) == 3:
            features_avg = np.array([f.mean(axis=0) for f in penult_features])
        else:
            features_avg = penult_features

        print(f"Training LLR with {len(features_avg)} samples...")
        print(f"Feature dimension: {features_avg.shape[1]}")

        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_avg)

        # Fit Gaussian Mixture Model
        print(f"Fitting GMM with {self.n_components} components...")
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=200,
            random_state=42,
            verbose=1
        )
        self.gmm.fit(features_scaled)

        # Compute background likelihood as reference
        # Use average log-likelihood on training data as background
        train_ll = self.gmm.score_samples(features_scaled)
        self.background_ll = train_ll.mean()

        print(f"\nLLR trained:")
        print(f"  GMM components: {self.n_components}")
        print(f"  Converged: {self.gmm.converged_}")
        print(f"  Training log-likelihood range: [{train_ll.min():.2f}, {train_ll.max():.2f}]")
        print(f"  Background log-likelihood: {self.background_ll:.2f}")

    def compute_llr_scores(self, features):
        """
        Compute LLR OOD scores.

        Higher score = more likely OOD (lower likelihood ratio)

        Args:
            features: Dict with 'penultimate_features'

        Returns:
            llr_scores: Array of OOD scores (higher = more OOD)
        """
        # Average over MC forward passes if needed
        penult_features = features['penultimate_features']
        if penult_features.dtype == object:
            features_avg = np.array([f.mean(axis=0) if hasattr(f, 'mean') else f
                                    for f in penult_features])
        elif len(penult_features.shape) == 3:
            features_avg = np.array([f.mean(axis=0) for f in penult_features])
        else:
            features_avg = penult_features

        # Standardize features
        features_scaled = self.scaler.transform(features_avg)

        # Compute log-likelihood under GMM
        log_likelihood = self.gmm.score_samples(features_scaled)

        # Compute likelihood ratio: background_ll - log_likelihood
        # Higher score = lower likelihood = more OOD
        # (We subtract from background so that low likelihood = high score)
        llr_scores = self.background_ll - log_likelihood

        return llr_scores

    def predict_routing(self, features, threshold):
        """
        Predict which samples to route to SRP based on LLR scores.

        Args:
            features: Dict with CRNN features
            threshold: LLR score threshold

        Returns:
            route_to_srp: Boolean array (True = route to SRP)
            llr_scores: Array of LLR scores
        """
        llr_scores = self.compute_llr_scores(features)
        route_to_srp = llr_scores > threshold
        return route_to_srp, llr_scores

    def save(self, path):
        """Save trained LLR model."""
        model = {
            'gmm': self.gmm,
            'scaler': self.scaler,
            'background_ll': self.background_ll,
            'n_components': self.n_components
        }
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"LLR model saved to {path}")

    def load(self, path):
        """Load trained LLR model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.gmm = model['gmm']
        self.scaler = model['scaler']
        self.background_ll = model['background_ll']
        self.n_components = model['n_components']
        print(f"LLR model loaded from {path}")


if __name__ == "__main__":
    # Test LLR routing
    import sys
    sys.path.append("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection")

    # Load training features for density estimation
    print("=" * 70)
    print("Loading TRAINING features for density model...")
    print("=" * 70)
    train_path = "features/train_combined_features.npz"
    train_data = np.load(train_path, allow_pickle=True)
    train_features = {key: train_data[key] for key in train_data.files}
    print(f"Loaded {len(train_features['gt_angles'])} training samples")

    # Train LLR with different numbers of components
    for n_comp in [5, 10, 20]:
        print(f"\n{'='*70}")
        print(f"Training LLR with n_components={n_comp}")
        print('='*70)

        router = LLROODRouter(n_components=n_comp)
        router.train(train_features)

        # Test on test features
        print(f"\n{'='*70}")
        print("Testing on test set...")
        print('='*70)
        test_path = "features/test_3x12cm_consecutive_features.npz"
        test_data = np.load(test_path, allow_pickle=True)
        test_features = {key: test_data[key] for key in test_data.files}

        # Compute LLR scores
        llr_scores = router.compute_llr_scores(test_features)
        print(f"\nLLR score stats:")
        print(f"  Range: [{llr_scores.min():.2f}, {llr_scores.max():.2f}]")
        print(f"  Mean: {llr_scores.mean():.2f}")
        print(f"  Median: {np.median(llr_scores):.2f}")
        print(f"  Std: {llr_scores.std():.2f}")

        # Test different thresholds
        print("\nTesting thresholds:")
        for percentile in [70, 75, 80, 85, 90]:
            threshold = np.percentile(llr_scores, percentile)
            route_to_srp, _ = router.predict_routing(test_features, threshold)
            routing_rate = route_to_srp.sum() / len(route_to_srp) * 100
            print(f"  p{percentile} (threshold={threshold:.2f}): {routing_rate:.1f}% routing ({route_to_srp.sum()} samples)")
