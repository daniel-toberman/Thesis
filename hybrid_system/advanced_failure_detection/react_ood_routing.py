#!/usr/bin/env python3
"""
ReAct (Rectified Activations) OOD Router for Failure Detection

Based on "Generalized Out-of-Distribution Detection" survey paper.
Truncates high activations that cause overconfidence on OOD samples.

Key idea: "using mismatched BatchNorm statistics...can trigger abnormally
high unit activations" on OOD data.
"""

import numpy as np
import pickle
from pathlib import Path


class ReActOODRouter:
    """
    Failure detector using ReAct (Rectified Activations).

    Applies activation clipping to penultimate features before computing
    confidence scores. High clipped activations indicate OOD.
    """

    def __init__(self, model_path=None, clip_percentile=90):
        """
        Initialize router.

        Args:
            model_path: Path to trained ReAct model pickle file
            clip_percentile: Percentile for activation clipping threshold
        """
        self.clip_percentile = clip_percentile
        self.clip_threshold = None
        self.temperature = None
        self.error_threshold = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained ReAct model from pickle."""
        print(f"Loading ReAct model from: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.clip_threshold = model['clip_threshold']
        self.temperature = model.get('temperature', 1.0)
        self.clip_percentile = model.get('clip_percentile', 90)
        self.error_threshold = model.get('error_threshold', 5.0)

        print(f"  Clip threshold: {self.clip_threshold:.4f}")
        print(f"  Clip percentile: {self.clip_percentile}%")
        print(f"  Temperature: {self.temperature:.4f}")
        print(f"  Error threshold used in training: {self.error_threshold}°")

    def train(self, train_features, error_threshold=5.0):
        """
        Train ReAct model on in-distribution data.

        Args:
            train_features: Dict with 'penultimate_features' and 'logits_pre_sig' keys
            error_threshold: Error threshold defining "failure"
        """
        self.error_threshold = error_threshold

        # Extract and average penultimate features from ID training data
        features = train_features['penultimate_features']
        features_avg = np.array([f.mean(axis=0) for f in features])

        # Compute activation clipping threshold at specified percentile
        # This is the threshold above which activations are considered abnormally high
        all_activations = features_avg.flatten()
        self.clip_threshold = np.percentile(all_activations, self.clip_percentile)

        # Set temperature for energy computation
        self.temperature = 1.0

        print(f"ReAct trained on {len(features)} samples")
        print(f"  Clip threshold (p{self.clip_percentile}): {self.clip_threshold:.4f}")

    def apply_react(self, features):
        """
        Apply ReAct: clip activations above threshold.

        Args:
            features: (N, D) array of features

        Returns:
            clipped_features: (N, D) array with clipped activations
        """
        if self.clip_threshold is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        # Clip activations at threshold
        clipped_features = np.minimum(features, self.clip_threshold)
        return clipped_features

    def compute_react_scores(self, features, logits):
        """
        Compute ReAct-based OOD scores.

        Two approaches:
        1. Measure amount of clipping (high clipping = OOD)
        2. Compute energy on clipped features (low energy with clipping = OOD)

        Args:
            features: (N,) array where each element is (T, D) temporal features
            logits: (N,) array where each element is (T, D') temporal logits

        Returns:
            react_scores: (N,) array of ReAct scores
        """
        # Process features
        features_avg = np.array([f.mean(axis=0) for f in features])

        # Apply ReAct clipping
        clipped_features = self.apply_react(features_avg)

        # Measure clipping amount: sum of differences
        # High clipping = many high activations = likely OOD
        clipping_amount = (features_avg - clipped_features).sum(axis=1)

        # Also compute energy scores from logits for comparison
        # (ReAct can be combined with energy-based detection)
        logits_avg = np.array([l.mean(axis=0) for l in logits])

        # Energy = -T * log(sum(exp(logit_i / T)))
        max_logit = logits_avg.max(axis=1, keepdims=True)
        exp_logits = np.exp((logits_avg - max_logit) / self.temperature)
        sum_exp = exp_logits.sum(axis=1)
        energy = -self.temperature * (np.log(sum_exp) + (max_logit.squeeze() / self.temperature))

        # Combine clipping amount with energy
        # High clipping OR high energy = OOD
        # Use clipping amount as primary signal
        react_scores = clipping_amount

        return react_scores

    def predict_routing(self, features, react_threshold):
        """
        Predict routing decisions.

        Args:
            features: Dict with 'penultimate_features' and 'logits_pre_sig' keys
            react_threshold: ReAct score threshold (route if score > threshold)

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            react_scores: (N,) ReAct scores
        """
        # Compute ReAct scores
        react_scores = self.compute_react_scores(
            features['penultimate_features'],
            features['logits_pre_sig']
        )

        # Route if ReAct score > threshold (high clipping = OOD = route)
        route_to_srp = react_scores > react_threshold

        return route_to_srp, react_scores

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25):
        """
        Find ReAct threshold that achieves target routing rate.

        Args:
            features: Dict with 'penultimate_features' and 'logits_pre_sig' keys
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)

        Returns:
            optimal_threshold: ReAct score threshold
        """
        react_scores = self.compute_react_scores(
            features['penultimate_features'],
            features['logits_pre_sig']
        )

        # Find threshold at target percentile
        threshold = np.percentile(react_scores, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = react_scores > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"ReAct threshold: {threshold:.6f}")

        return threshold

    def save_model(self, save_path):
        """Save trained ReAct model to pickle file."""
        model = {
            'clip_threshold': self.clip_threshold,
            'temperature': self.temperature,
            'clip_percentile': self.clip_percentile,
            'error_threshold': self.error_threshold
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"ReAct model saved to: {save_path}")
