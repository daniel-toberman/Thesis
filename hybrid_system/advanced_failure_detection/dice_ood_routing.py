"""
DICE OOD Detection
Faithful NumPy implementation based on:
"DICE: Leveraging Sparsification for Out-of-Distribution Detection"
Sun & Li, NeurIPS 2022
"""

import numpy as np


def logsumexp(x, axis=-1):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)))


class DICEOODRouter:
    """
    DICE (Density-aware / Directed Sparsification) OOD router.

    This class:
    1. Computes contribution matrix V = E[W ⊙ h(x)] on ID data
    2. Keeps top-(1-p) percentile of contributions
    3. Applies sparsified weights at inference
    4. Uses energy score for OOD detection
    """

    def __init__(self, clip_percentile=80):
        """
        clip_percentile: percentile of contributions to KEEP
        (e.g. 80 -> keep top 20%, sparsity p=0.8)
        """
        self.clip_percentile = clip_percentile

        self.mask = None          # M ∈ {0,1}^{m×C}
        self.W_sparse = None      # sparsified weights
        self.b = None             # bias

        print(f"Initialized DICE OOD Router with clip_percentile={clip_percentile}")

    def train(self, features):
        """
        Train DICE mask using in-distribution data.

        Required entries in `features`:
        - features["penultimate"]: shape (N, m)
        - features["W"]: shape (m, C)
        - features["b"]: shape (C,)
        """

        H = features["penultimate"]   # (N, m)
        W = features["W"]             # (m, C)
        b = features["b"]             # (C,)

        if H.ndim != 2:
            raise ValueError("penultimate features must be 2D (N, m)")
        if W.ndim != 2:
            raise ValueError("W must be 2D (m, C)")
        if b.ndim != 1:
            raise ValueError("b must be 1D (C,)")

        N, m = H.shape
        m_w, C = W.shape

        if m != m_w:
            raise ValueError("Mismatch between feature dim and weight dim")

        # ---------------------------------------------------------
        # 1. Compute contribution matrix V = E[W ⊙ h(x)]
        # ---------------------------------------------------------
        # Expand H to (N, m, 1) and W to (1, m, C)
        # Result: (N, m, C)
        contributions = H[:, :, None] * W[None, :, :]
        V = contributions.mean(axis=0)   # (m, C)

        # ---------------------------------------------------------
        # 2. Build sparsification mask
        # ---------------------------------------------------------
        flat_V = V.flatten()
        threshold = np.percentile(flat_V, self.clip_percentile)

        M = (V > threshold).astype(np.float32)

        # ---------------------------------------------------------
        # 3. Store sparsified weights
        # ---------------------------------------------------------
        self.mask = M
        self.W_sparse = W * M
        self.b = b

        kept_ratio = M.mean()
        print(
            f"DICE training complete | "
            f"Kept weights: {kept_ratio * 100:.2f}%"
        )

    def compute_dice_scores(self, features):
        """
        Compute DICE energy scores.

        Required entries in `features`:
        - features["penultimate"]: shape (N, m)

        Returns:
        - energy scores (higher = more ID-like)
        """

        if self.W_sparse is None:
            raise RuntimeError("DICE router must be trained before inference")

        H = features["penultimate"]   # (N, m)

        # ---------------------------------------------------------
        # 4. Compute sparsified logits
        # f_DICE(x) = (M ⊙ W)^T h(x) + b
        # ---------------------------------------------------------
        logits = H @ self.W_sparse + self.b  # (N, C)

        # ---------------------------------------------------------
        # 5. Energy score
        # S(x) = logsumexp(logits)
        # ---------------------------------------------------------
        scores = logsumexp(logits, axis=1)

        return scores

    def predict_routing(self, features, threshold):
        """
        Predict routing based on DICE OOD score.

        Returns:
        - route_to_srp: boolean mask (True = OOD)
        - ood_scores: energy scores
        """

        ood_scores = self.compute_dice_scores(features)
        route_to_srp = ood_scores < threshold  # lower energy => OOD
        return route_to_srp, ood_scores
