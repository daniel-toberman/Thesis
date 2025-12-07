#!/usr/bin/env python3
"""
DICE (Diversified Input via Class-specific Embeddings) OOD Detection for Routing

Based on "DICE: Leveraging Sparsification for Out-of-Distribution Detection" (Sun & Li, 2022)
Ranks weights by contribution and uses only salient weights for OOD detection.

Paper claims: "provably reduces the output variance for OOD data"

Key idea: Prune low-contribution weights at inference to reduce OOD prediction variance.
"""

import numpy as np
import pickle
from pathlib import Path


class DICEOODRouter:
    """DICE OOD detection using weight sparsification."""

    def __init__(self, model_path=None, sparsity_percentile=90):
        """
        Args:
            model_path: Path to saved DICE model
            sparsity_percentile: Percentile of weights to keep (e.g., 90 = keep top 10%)
        """
        self.sparsity_percentile = sparsity_percentile
        self.weight_contributions = None  # Contribution scores for each feature dimension
        self.weight_mask = None  # Binary mask for salient features
        self.feature_dim = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, features):
        """
        Train DICE by computing feature importance/contribution scores.

        We approximate weight contribution by measuring correlation between
        features and prediction confidence.

        Args:
            features: Dict with 'penultimate_features', 'logits_pre_sig', 'abs_errors'
        """
        # Average over MC forward passes if needed
        penult_features = features['penultimate_features']
        if penult_features.dtype == object:
            # Features stored as array of arrays - convert and average
            features_avg = np.array([f.mean(axis=0) if hasattr(f, 'mean') else f for f in penult_features])
        elif len(penult_features.shape) == 3:
            features_avg = np.array([f.mean(axis=0) for f in penult_features])
        else:
            features_avg = penult_features

        logits = features['logits_pre_sig']
        if logits.dtype == object:
            # Logits stored as array of arrays - convert and average
            logits_avg = np.array([l.mean(axis=0) for l in logits])
        elif len(logits.shape) == 3:
            logits_avg = logits.mean(axis=1)
        else:
            logits_avg = logits

        abs_errors = features['abs_errors']

        self.feature_dim = features_avg.shape[1]

        # Compute feature importance using multiple signals
        # 1. Feature variance (higher variance = more informative)
        # 2. Feature magnitude (features that activate strongly)
        # 3. Correlation with prediction confidence (if possible)

        feature_variance = features_avg.var(axis=0)
        feature_magnitude = np.abs(features_avg).mean(axis=0)

        # Normalize both
        feature_variance_norm = feature_variance / (feature_variance.max() + 1e-8)
        feature_magnitude_norm = feature_magnitude / (feature_magnitude.max() + 1e-8)

        # Try to compute correlation with confidence
        # Use inverse error as confidence (lower error = higher confidence)
        confidence = 1.0 / (abs_errors + 1.0)

        feature_corr = []
        for feat_dim in range(self.feature_dim):
            feat_values = features_avg[:, feat_dim]

            # Only compute correlation if there's variation
            if feat_values.std() > 1e-8 and confidence.std() > 1e-8:
                corr = np.corrcoef(feat_values, confidence)[0, 1]
                if not np.isnan(corr):
                    feature_corr.append(abs(corr))
                else:
                    feature_corr.append(0.0)
            else:
                feature_corr.append(0.0)

        feature_corr = np.array(feature_corr)
        feature_corr_norm = feature_corr / (feature_corr.max() + 1e-8)

        # Combine signals: variance + magnitude + correlation
        self.weight_contributions = (
            0.4 * feature_variance_norm +
            0.3 * feature_magnitude_norm +
            0.3 * feature_corr_norm
        )

        # Create sparsity mask: keep only top percentile of weights
        threshold = np.percentile(self.weight_contributions, self.sparsity_percentile)
        self.weight_mask = self.weight_contributions >= threshold

        n_kept = self.weight_mask.sum()
        sparsity_achieved = (1 - n_kept / self.feature_dim) * 100

        print(f"DICE trained:")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Sparsity percentile: {self.sparsity_percentile}")
        print(f"  Features kept: {n_kept} / {self.feature_dim}")
        print(f"  Sparsity achieved: {sparsity_achieved:.1f}%")
        print(f"  Weight contribution range: [{self.weight_contributions.min():.4f}, {self.weight_contributions.max():.4f}]")

    def apply_sparsification(self, features):
        """
        Apply weight sparsification by masking out low-contribution features.

        Args:
            features: Feature array (N, feature_dim)

        Returns:
            sparsified_features: Features with low-contribution dimensions zeroed out
        """
        sparsified = features.copy()
        sparsified[:, ~self.weight_mask] = 0
        return sparsified

    def compute_dice_scores(self, features):
        """
        Compute DICE OOD scores.

        Higher score = more likely OOD

        The idea: After sparsification, OOD samples will have different variance
        in their predictions compared to ID samples.

        Args:
            features: Dict with 'penultimate_features', 'logits_pre_sig'

        Returns:
            dice_scores: Array of OOD scores (higher = more OOD)
        """
        # Average over MC forward passes if needed
        penult_features = features['penultimate_features']
        if penult_features.dtype == object:
            # Features stored as array of arrays - convert and average
            features_avg = np.array([f.mean(axis=0) if hasattr(f, 'mean') else f for f in penult_features])
        elif len(penult_features.shape) == 3:
            features_avg = np.array([f.mean(axis=0) for f in penult_features])
        else:
            features_avg = penult_features

        # Apply sparsification
        sparsified_features = self.apply_sparsification(features_avg)

        # Compute OOD score as the difference between original and sparsified feature norms
        # Intuition: ID samples should be robust to sparsification (small change)
        # OOD samples will have large change (rely on pruned features)

        original_norms = np.linalg.norm(features_avg, axis=1)
        sparsified_norms = np.linalg.norm(sparsified_features, axis=1)

        # Relative change in feature norm
        norm_change = np.abs(original_norms - sparsified_norms) / (original_norms + 1e-8)

        # Also consider variance of pruned features (higher = more OOD)
        pruned_features = features_avg[:, ~self.weight_mask]
        pruned_variance = pruned_features.var(axis=1)

        # Combine both signals
        dice_scores = norm_change + 0.1 * pruned_variance

        return dice_scores

    def predict_routing(self, features, threshold):
        """
        Predict which samples to route to SRP based on DICE scores.

        Args:
            features: Dict with CRNN features
            threshold: DICE score threshold

        Returns:
            route_to_srp: Boolean array (True = route to SRP)
            dice_scores: Array of DICE scores
        """
        dice_scores = self.compute_dice_scores(features)
        route_to_srp = dice_scores > threshold
        return route_to_srp, dice_scores

    def save(self, path):
        """Save trained DICE model."""
        model = {
            'weight_contributions': self.weight_contributions,
            'weight_mask': self.weight_mask,
            'feature_dim': self.feature_dim,
            'sparsity_percentile': self.sparsity_percentile
        }
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"DICE model saved to {path}")

    def load(self, path):
        """Load trained DICE model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.weight_contributions = model['weight_contributions']
        self.weight_mask = model['weight_mask']
        self.feature_dim = model['feature_dim']
        self.sparsity_percentile = model['sparsity_percentile']
        print(f"DICE model loaded from {path}")


if __name__ == "__main__":
    # Test DICE routing
    import sys
    sys.path.append("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection")

    # Load test features
    features_path = "features/test_3x12cm_consecutive_features.npz"
    print(f"Loading features from {features_path}")
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}

    # Train DICE with different sparsity levels
    for sparsity_pct in [80, 90, 95]:
        print(f"\n{'='*70}")
        print(f"Training DICE with sparsity_percentile={sparsity_pct}")
        print('='*70)

        router = DICEOODRouter(sparsity_percentile=sparsity_pct)
        router.train(features)

        # Test different thresholds
        print("\nTesting thresholds:")
        dice_scores = router.compute_dice_scores(features)

        for percentile in [70, 75, 80, 85, 90]:
            threshold = np.percentile(dice_scores, percentile)
            route_to_srp, _ = router.predict_routing(features, threshold)
            routing_rate = route_to_srp.sum() / len(route_to_srp) * 100
            print(f"  p{percentile} (threshold={threshold:.4f}): {routing_rate:.1f}% routing ({route_to_srp.sum()} samples)")
