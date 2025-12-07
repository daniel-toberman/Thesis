#!/usr/bin/env python3
"""
GradNorm OOD Router for Failure Detection

Based on "Generalized Out-of-Distribution Detection" survey paper.
Uses vector norm of gradients backpropagated from KL divergence.

Key idea: Gradient norms behave differently for ID vs OOD samples.
"""

import numpy as np
import pickle
from pathlib import Path


class GradNormOODRouter:
    """
    Failure detector using Gradient Norm-based OOD detection.

    Computes approximation of gradient norms using pre-computed features.
    High gradient norm indicates OOD.
    """

    def __init__(self, model_path=None):
        """
        Initialize router.

        Args:
            model_path: Path to trained GradNorm model pickle file
        """
        self.train_mean = None
        self.train_std = None
        self.error_threshold = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained GradNorm model from pickle."""
        print(f"Loading GradNorm model from: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.train_mean = model['train_mean']
        self.train_std = model['train_std']
        self.error_threshold = model.get('error_threshold', 5.0)

        print(f"  Train mean norm: {self.train_mean:.4f}")
        print(f"  Train std norm: {self.train_std:.4f}")
        print(f"  Error threshold used in training: {self.error_threshold}°")

    def train(self, train_features, error_threshold=5.0):
        """
        Train GradNorm model on in-distribution data.

        Args:
            train_features: Dict with 'logits_pre_sig' and 'penultimate_features' keys
            error_threshold: Error threshold defining "failure"
        """
        self.error_threshold = error_threshold

        # Compute approximate gradient norms for training data
        grad_norms = self._approximate_grad_norms(
            train_features['logits_pre_sig'],
            train_features['penultimate_features']
        )

        # Store statistics
        self.train_mean = grad_norms.mean()
        self.train_std = grad_norms.std()

        print(f"GradNorm trained on {len(grad_norms)} samples")
        print(f"  Mean gradient norm: {self.train_mean:.4f}")
        print(f"  Std gradient norm: {self.train_std:.4f}")

    def _approximate_grad_norms(self, logits, features):
        """
        Approximate gradient norms from pre-computed features.

        Since we don't have direct access to the computational graph,
        we approximate gradient norms using:
        1. Softmax probabilities from logits
        2. Feature magnitudes
        3. Prediction uncertainty

        Args:
            logits: (N,) array where each element is (T, D) temporal logits
            features: (N,) array where each element is (T, F) temporal features

        Returns:
            grad_norms: (N,) array of approximate gradient norms
        """
        grad_norms = []

        for i in range(len(logits)):
            # Average over time
            logit_avg = logits[i].mean(axis=0)  # (D,)
            feature_avg = features[i].mean(axis=0)  # (F,)

            # Compute softmax probabilities
            exp_logits = np.exp(logit_avg - logit_avg.max())
            probs = exp_logits / exp_logits.sum()

            # Compute entropy (uncertainty in prediction)
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Approximate gradient norm as combination of:
            # 1. Feature magnitude (larger features = larger gradients)
            # 2. Prediction uncertainty (high entropy = larger gradients)
            # 3. Probability spread (uniform probs = larger gradients)

            feature_norm = np.linalg.norm(feature_avg)
            prob_variance = np.var(probs)

            # Combine signals
            # OOD samples tend to have:
            # - Different feature magnitudes
            # - Higher or lower entropy
            # - Different probability distributions
            approx_grad_norm = feature_norm * (1 + entropy) * (1 + prob_variance)

            grad_norms.append(approx_grad_norm)

        return np.array(grad_norms)

    def compute_gradnorm_scores(self, features):
        """
        Compute GradNorm-based OOD scores.

        Args:
            features: Dict with 'logits_pre_sig' and 'penultimate_features' keys

        Returns:
            gradnorm_scores: (N,) array of normalized gradient norms
        """
        if self.train_mean is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        # Compute approximate gradient norms
        grad_norms = self._approximate_grad_norms(
            features['logits_pre_sig'],
            features['penultimate_features']
        )

        # Normalize by training statistics
        # Z-score: (x - mean) / std
        normalized_scores = np.abs((grad_norms - self.train_mean) / (self.train_std + 1e-8))

        return normalized_scores

    def predict_routing(self, features, gradnorm_threshold):
        """
        Predict routing decisions.

        Args:
            features: Dict with 'logits_pre_sig' and 'penultimate_features' keys
            gradnorm_threshold: GradNorm score threshold (route if score > threshold)

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            gradnorm_scores: (N,) GradNorm scores
        """
        # Compute GradNorm scores
        gradnorm_scores = self.compute_gradnorm_scores(features)

        # Route if score > threshold (abnormal gradient = OOD = route)
        route_to_srp = gradnorm_scores > gradnorm_threshold

        return route_to_srp, gradnorm_scores

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25):
        """
        Find GradNorm threshold that achieves target routing rate.

        Args:
            features: Dict with 'logits_pre_sig' and 'penultimate_features' keys
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)

        Returns:
            optimal_threshold: GradNorm score threshold
        """
        gradnorm_scores = self.compute_gradnorm_scores(features)

        # Find threshold at target percentile
        threshold = np.percentile(gradnorm_scores, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = gradnorm_scores > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"GradNorm threshold: {threshold:.6f}")

        return threshold

    def save_model(self, save_path):
        """Save trained GradNorm model to pickle file."""
        model = {
            'train_mean': self.train_mean,
            'train_std': self.train_std,
            'error_threshold': self.error_threshold
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"GradNorm model saved to: {save_path}")
