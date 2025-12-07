#!/usr/bin/env python3
"""
Simple Max Probability Threshold Routing

Routes based on maximum softmax probability from CRNN outputs.
This is a fundamental baseline for confidence-based routing.

Route to SRP when: max_prob < threshold (low confidence)
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.special import softmax


class MaxProbRouter:
    """Simple routing based on maximum softmax probability."""

    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to saved model (not needed for this simple method)
        """
        # No training needed - this is a simple post-hoc method
        pass

    def train(self, features):
        """
        No training needed for max probability routing.

        Args:
            features: Dict with 'logits_pre_sig'
        """
        print("MaxProb routing: No training needed (post-hoc method)")

    def compute_max_prob_scores(self, features):
        """
        Compute maximum softmax probabilities.

        Lower score = less confident = more likely to benefit from SRP

        Args:
            features: Dict with 'logits_pre_sig'

        Returns:
            max_probs: Array of maximum probabilities (higher = more confident)
        """
        # Average over MC forward passes if needed
        logits = features['logits_pre_sig']
        if logits.dtype == object:
            # Logits stored as array of arrays - convert and average
            logits_avg = np.array([l.mean(axis=0) for l in logits])
        elif len(logits.shape) == 3:
            logits_avg = logits.mean(axis=1)
        else:
            logits_avg = logits

        # Compute softmax probabilities
        probs = softmax(logits_avg, axis=1)

        # Get maximum probability for each sample
        max_probs = probs.max(axis=1)

        return max_probs

    def predict_routing(self, features, threshold):
        """
        Predict which samples to route to SRP based on max probability.

        Route to SRP when max_prob < threshold (low confidence)

        Args:
            features: Dict with CRNN features
            threshold: Maximum probability threshold

        Returns:
            route_to_srp: Boolean array (True = route to SRP)
            max_probs: Array of maximum probabilities
        """
        max_probs = self.compute_max_prob_scores(features)

        # Route to SRP when confidence is LOW (max_prob < threshold)
        route_to_srp = max_probs < threshold

        return route_to_srp, max_probs

    def save(self, path):
        """Save model (not needed for this simple method)."""
        print("MaxProb routing: No model to save (parameter-free method)")

    def load(self, path):
        """Load model (not needed for this simple method)."""
        print("MaxProb routing: No model to load (parameter-free method)")


if __name__ == "__main__":
    # Test MaxProb routing
    import sys
    sys.path.append("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection")

    # Load test features
    features_path = "features/test_3x12cm_consecutive_features.npz"
    print(f"Loading features from {features_path}")
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}

    # Create router
    print("\nTesting MaxProb routing...")
    router = MaxProbRouter()
    router.train(features)

    # Compute max probabilities
    max_probs = router.compute_max_prob_scores(features)
    print(f"\nMax probability range: [{max_probs.min():.4f}, {max_probs.max():.4f}]")
    print(f"Mean max probability: {max_probs.mean():.4f}")
    print(f"Median max probability: {np.median(max_probs):.4f}")

    # Test different thresholds
    print("\nTesting thresholds:")
    for percentile in [10, 20, 30, 40, 50]:
        threshold = np.percentile(max_probs, percentile)
        route_to_srp, _ = router.predict_routing(features, threshold)
        routing_rate = route_to_srp.sum() / len(route_to_srp) * 100
        print(f"  p{percentile} (threshold={threshold:.4f}): {routing_rate:.1f}% routing ({route_to_srp.sum()} samples)")
