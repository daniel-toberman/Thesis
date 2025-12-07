#!/usr/bin/env python3
"""
Energy-Based OOD Router for Failure Detection
"""

import numpy as np
import pickle
from pathlib import Path


class EnergyOODRouter:
    """
    Failure detector using Energy-Based OOD detection.

    Routes to SRP when energy score is above threshold (high uncertainty).
    """

    def __init__(self, model_path=None):
        """
        Initialize router.

        Args:
            model_path: Path to trained model pickle file
        """
        self.temperature = None
        self.energy_threshold = None
        self.error_threshold = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained Energy OOD model from pickle."""
        print(f"Loading Energy OOD model from: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.temperature = model['temperature']
        self.error_threshold = model.get('error_threshold', 5.0)

        print(f"  Temperature: {self.temperature:.4f}")
        print(f"  Error threshold used in training: {self.error_threshold}°")

    def compute_energy_scores(self, logits):
        """
        Compute energy scores from logits.

        Args:
            logits: Array of logits (N,) of shape (T, D) each

        Returns:
            energy_scores: (N,) array of energy scores
        """
        if self.temperature is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Handle temporal dimension
        logits_avg = np.array([l.mean(axis=0) for l in logits])

        # Energy = -T * log(sum(exp(logit_i / T)))
        # For numerical stability
        max_logit = logits_avg.max(axis=1, keepdims=True)
        exp_logits = np.exp((logits_avg - max_logit) / self.temperature)
        sum_exp = exp_logits.sum(axis=1)
        energy = -self.temperature * (np.log(sum_exp) + (max_logit.squeeze() / self.temperature))

        return energy

    def predict_routing(self, features, energy_threshold):
        """
        Predict routing decisions.

        Args:
            features: Dict with 'logits_pre_sig' key
            energy_threshold: Energy threshold (route if energy > threshold)

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            energy_scores: (N,) energy scores
        """
        logits = features['logits_pre_sig']

        # Compute energy scores
        energy_scores = self.compute_energy_scores(logits)

        # Route if energy > threshold (high uncertainty)
        route_to_srp = energy_scores > energy_threshold

        return route_to_srp, energy_scores

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25):
        """
        Find energy threshold that achieves target routing rate.

        Args:
            features: Dict with 'logits_pre_sig' key
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)

        Returns:
            optimal_threshold: Energy threshold
        """
        energy_scores = self.compute_energy_scores(features['logits_pre_sig'])

        # Find threshold at target percentile
        threshold = np.percentile(energy_scores, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = energy_scores > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"Energy threshold: {threshold:.2f}")

        return threshold
