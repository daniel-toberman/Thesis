#!/usr/bin/env python3
"""
MC Dropout Ensemble Router for Failure Detection

Uses Monte Carlo Dropout to approximate Bayesian posterior.
High prediction variance indicates uncertainty -> route to SRP.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys


class MCDropoutRouter:
    """
    Failure detector using MC Dropout Ensemble.

    Runs CRNN multiple times with dropout enabled to estimate uncertainty.
    Routes to SRP when prediction variance is high.
    """

    def __init__(self, crnn_model_path=None, n_samples=10, device='cpu'):
        """
        Initialize router.

        Args:
            crnn_model_path: Path to trained CRNN model (.pt file)
            n_samples: Number of forward passes for MC sampling
            device: Device to use ('cpu', 'mps', 'cuda')
        """
        self.model = None
        self.n_samples = n_samples
        self.device = device

        if crnn_model_path is not None:
            self.load_model(crnn_model_path)

    def load_model(self, model_path):
        """Load trained CRNN model."""
        print(f"Loading CRNN model from: {model_path}")

        # Load model with weights_only=False to allow full model loading
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model = checkpoint['model']
        else:
            self.model = checkpoint

        self.model = self.model.to(self.device)

        # Enable dropout at test time (set to train mode)
        self.model.train()

        print(f"  Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  MC samples: {self.n_samples}")

    def enable_dropout(self, model):
        """Enable dropout layers during inference."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(self, audio_features):
        """
        Run MC Dropout to estimate prediction uncertainty.

        Args:
            audio_features: (N,) array where each element is audio input for CRNN

        Returns:
            predictions_mean: (N,) mean predictions
            predictions_std: (N,) standard deviation (uncertainty)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        n_samples_total = len(audio_features)
        all_predictions = []

        # Enable dropout
        self.enable_dropout(self.model)

        with torch.no_grad():
            # Run multiple forward passes
            for i in range(self.n_samples):
                batch_predictions = []

                for audio in audio_features:
                    # Convert to tensor
                    if not isinstance(audio, torch.Tensor):
                        audio = torch.FloatTensor(audio)
                    audio = audio.unsqueeze(0).to(self.device)  # Add batch dimension

                    # Forward pass
                    output = self.model(audio)

                    # Extract angle prediction
                    if isinstance(output, tuple):
                        prediction = output[0].squeeze().cpu().numpy()
                    else:
                        prediction = output.squeeze().cpu().numpy()

                    batch_predictions.append(prediction)

                all_predictions.append(batch_predictions)

        # Stack predictions: (n_samples, n_samples_total)
        all_predictions = np.array(all_predictions)

        # Compute mean and std across MC samples
        predictions_mean = all_predictions.mean(axis=0)
        predictions_std = all_predictions.std(axis=0)

        return predictions_mean, predictions_std

    def compute_variance_from_logits(self, logits_pre_sig):
        """
        Compute prediction variance from pre-sigmoid logits.

        This is a faster alternative that uses existing logits if dropout was enabled.

        Args:
            logits_pre_sig: (N,) array where each element is (T, D) temporal logits

        Returns:
            variance: (N,) array of prediction variances
        """
        # For each sample, compute variance across the output dimension
        variances = []

        for logit in logits_pre_sig:
            # logit shape: (T, D) where T is time steps, D is output dimension
            # Average over time
            logit_avg = logit.mean(axis=0)  # Shape: (D,)

            # Compute variance of the distribution
            # High variance = more uncertain
            variance = np.var(logit_avg)
            variances.append(variance)

        return np.array(variances)

    def compute_entropy_from_logits(self, logits_pre_sig):
        """
        Compute predictive entropy as uncertainty measure.

        Args:
            logits_pre_sig: (N,) array where each element is (T, D) temporal logits

        Returns:
            entropy: (N,) array of predictive entropy
        """
        entropies = []

        for logit in logits_pre_sig:
            # Average over time
            logit_avg = logit.mean(axis=0)  # Shape: (D,)

            # Apply softmax to get probabilities
            exp_logits = np.exp(logit_avg - logit_avg.max())
            probs = exp_logits / exp_logits.sum()

            # Compute entropy: -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        return np.array(entropies)

    def predict_routing(self, features, variance_threshold=None, use_entropy=False):
        """
        Predict routing decisions using pre-computed logits.

        Args:
            features: Dict with 'logits_pre_sig' key
            variance_threshold: Variance threshold (route if variance > threshold)
            use_entropy: If True, use entropy instead of variance

        Returns:
            route_to_srp: (N,) boolean array (True = route to SRP)
            uncertainty: (N,) uncertainty scores (variance or entropy)
        """
        logits = features['logits_pre_sig']

        # Compute uncertainty
        if use_entropy:
            uncertainty = self.compute_entropy_from_logits(logits)
        else:
            uncertainty = self.compute_variance_from_logits(logits)

        # Route if uncertainty > threshold (high uncertainty = route)
        if variance_threshold is not None:
            route_to_srp = uncertainty > variance_threshold
        else:
            route_to_srp = None

        return route_to_srp, uncertainty

    def find_optimal_threshold(self, features, abs_errors, target_routing_rate=0.25,
                             use_entropy=False):
        """
        Find uncertainty threshold that achieves target routing rate.

        Args:
            features: Dict with 'logits_pre_sig' key
            abs_errors: (N,) array of absolute errors
            target_routing_rate: Desired routing percentage (0.25 = 25%)
            use_entropy: If True, use entropy instead of variance

        Returns:
            optimal_threshold: Uncertainty threshold
        """
        _, uncertainty = self.predict_routing(features, variance_threshold=None,
                                              use_entropy=use_entropy)

        # Find threshold at target percentile
        threshold = np.percentile(uncertainty, (1 - target_routing_rate) * 100)

        # Verify actual routing rate
        route_to_srp = uncertainty > threshold
        actual_rate = route_to_srp.sum() / len(route_to_srp)

        metric_name = "Entropy" if use_entropy else "Variance"
        print(f"Target routing rate: {target_routing_rate*100:.1f}%")
        print(f"Actual routing rate: {actual_rate*100:.1f}%")
        print(f"{metric_name} threshold: {threshold:.6f}")

        return threshold
