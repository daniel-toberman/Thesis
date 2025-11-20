"""
ConfidNet-based Routing for Hybrid CRNN-SRP System

Uses trained ConfidNet model to predict CRNN failures and route to SRP.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

from confidnet_model import ConfidNet


class ConfidNetRouter:
    """
    Failure detector using trained ConfidNet model.

    Routes to SRP when predicted confidence is below threshold.
    """

    def __init__(self, model_path=None, device='mps'):
        """
        Initialize router.

        Args:
            model_path: Path to trained ConfidNet checkpoint
            device: Device to run on
        """
        self.device = device
        self.model = None
        self.confidence_threshold = None
        self.error_threshold = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained ConfidNet model from checkpoint."""
        print(f"Loading ConfidNet model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract training args
        args = checkpoint.get('args', {})
        hidden_dims = args.get('hidden_dims', [128, 64])
        dropout = args.get('dropout', 0.3)
        self.error_threshold = args.get('error_threshold', 5.0)

        # Initialize model
        self.model = ConfidNet(
            input_dim=256,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"  Model loaded successfully")
        print(f"  Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
        print(f"  Error threshold used in training: {self.error_threshold}°")

    def predict_confidences(self, features):
        """
        Predict confidence scores for test samples.

        Args:
            features: Dict with 'penultimate_features' key (N, T, 256)

        Returns:
            confidences: (N,) confidence scores [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        penultimate_features = features['penultimate_features']
        n_samples = len(penultimate_features)

        # Average over time dimension
        features_avg = np.array([feat.mean(axis=0) for feat in penultimate_features])  # (N, 256)

        # Convert to tensor
        features_tensor = torch.from_numpy(features_avg).float().to(self.device)

        # Predict confidence
        with torch.no_grad():
            confidences = self.model.predict_confidence(features_tensor)

        return confidences.cpu().numpy()

    def evaluate_routing(self, features, confidence_threshold):
        """
        Evaluate routing decisions at given confidence threshold.

        Args:
            features: Dict with feature arrays
            confidence_threshold: Threshold for routing (route to SRP if confidence < threshold)

        Returns:
            dict with routing metrics
        """
        # Predict confidences
        confidences = self.predict_confidences(features)

        # Route to SRP if confidence < threshold
        route_to_srp = confidences < confidence_threshold

        # Get ground truth (should CRNN have been routed?)
        errors = features['abs_errors']
        should_route = errors > self.error_threshold  # True failures

        # Compute metrics
        n_samples = len(errors)
        n_routed = route_to_srp.sum()
        routing_rate = n_routed / n_samples

        # Routing accuracy metrics
        precision = precision_score(should_route, route_to_srp, zero_division=0)
        recall = recall_score(should_route, route_to_srp, zero_division=0)
        f1 = f1_score(should_route, route_to_srp, zero_division=0)

        # How many correct predictions did we route? (false positives)
        correct_predictions = errors <= self.error_threshold
        false_positives = route_to_srp & correct_predictions
        fp_rate = false_positives.sum() / correct_predictions.sum() if correct_predictions.sum() > 0 else 0

        # Catastrophic failure capture (>30°)
        catastrophic = errors > 30
        catastrophic_caught = route_to_srp & catastrophic
        catastrophic_capture_rate = catastrophic_caught.sum() / catastrophic.sum() if catastrophic.sum() > 0 else 0

        return {
            'confidence_threshold': confidence_threshold,
            'confidences': confidences,
            'route_to_srp': route_to_srp,
            'routing_rate': routing_rate,
            'n_routed': n_routed,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fp_rate,
            'catastrophic_capture_rate': catastrophic_capture_rate,
        }

    def find_optimal_threshold(self, features, search_range=(0.1, 0.9), n_steps=50):
        """
        Find optimal confidence threshold by grid search.

        Optimizes F1 score for routing decisions.

        Args:
            features: Dict with feature arrays
            search_range: (min, max) threshold range
            n_steps: Number of thresholds to test

        Returns:
            dict with optimal threshold and performance metrics
        """
        print(f"\nSearching for optimal confidence threshold...")
        print(f"  Range: {search_range[0]:.2f} to {search_range[1]:.2f}")
        print(f"  Steps: {n_steps}")

        thresholds = np.linspace(search_range[0], search_range[1], n_steps)

        best_f1 = 0
        best_threshold = None
        best_results = None

        results_history = []

        for threshold in thresholds:
            results = self.evaluate_routing(features, threshold)
            results_history.append(results)

            if results['f1_score'] > best_f1:
                best_f1 = results['f1_score']
                best_threshold = threshold
                best_results = results

        print(f"\n✅ Optimal threshold found: {best_threshold:.4f}")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  Precision: {best_results['precision']:.4f}")
        print(f"  Recall: {best_results['recall']:.4f}")
        print(f"  Routing rate: {best_results['routing_rate']*100:.1f}%")
        print(f"  False positive rate: {best_results['false_positive_rate']*100:.1f}%")
        print(f"  Catastrophic capture: {best_results['catastrophic_capture_rate']*100:.1f}%")

        return {
            'best_threshold': best_threshold,
            'best_results': best_results,
            'all_results': results_history,
            'thresholds': thresholds
        }


if __name__ == "__main__":
    # Test router
    print("Testing ConfidNet Router...")

    # Load test features
    features_path = "features/test_3x12cm_consecutive_features.npz"
    print(f"\nLoading test features: {features_path}")
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"  Loaded {len(features['predicted_angles'])} samples")

    # Initialize router (without loading model for now)
    router = ConfidNetRouter()

    # Test loading model (uncomment when model is trained)
    # router.load_model("models/confidnet/best_model.ckpt")

    # Test threshold search (uncomment when model is trained)
    # optimal_results = router.find_optimal_threshold(features)

    print("\n✅ Router module test complete!")
    print("\nTo use this router:")
    print("  1. Train ConfidNet: python train_confidnet.py")
    print("  2. Load model: router.load_model('models/confidnet/best_model.ckpt')")
    print("  3. Find threshold: router.find_optimal_threshold(features)")
    print("  4. Route samples: router.evaluate_routing(features, threshold)")
