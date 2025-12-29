"""
DICE OOD Detection
Adapted for pipelines without explicit W and b
Regression-safe, variable-length-safe
"""

import numpy as np


def logsumexp(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)))


class DICEOODRouter:
    def __init__(self, clip_percentile=80):
        self.clip_percentile = clip_percentile
        self.W_sparse = None
        self.b = None
        self.is_regression = True  # your case

        print(f"Initialized DICE OOD Router (percentile={clip_percentile})")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _stack_frames(self, features):
        """
        Convert list-of-(T_i, D) into (N, D)
        """
        H_list = features["penultimate_features"]
        H = np.vstack(H_list)
        return H

    def _expand_targets(self, features):
        """
        Repeat per-sample target for each frame
        """
        Y_list = features["logits_pre_sig"]
        H_list = features["penultimate_features"]

        Y_expanded = []
        for y, h in zip(Y_list, H_list):
            T = h.shape[0]
            Y_expanded.append(np.repeat(y, T))

        Y = np.concatenate(Y_expanded)[:, None]
        return Y

    def _fit_linear_head(self, H, Y):
        """
        Fit Y ≈ H W + b using least squares
        """
        assert H.ndim == 2
        assert Y.ndim == 2

        N = H.shape[0]
        H_aug = np.hstack([H, np.ones((N, 1))])

        theta, _, _, _ = np.linalg.lstsq(H_aug, Y, rcond=None)

        W = theta[:-1]
        b = theta[-1]

        return W, b

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, features):
        """
        Train DICE mask using ID data only.

        Required keys:
        - penultimate_features: list of (T_i, D)
        - logits_pre_sig: list or array of scalars
        """

        H = self._stack_frames(features)          # (N, D)
        Y = self._expand_targets(features)        # (N, 1)

        # ---------------------------------------------------------
        # 1. Fit surrogate linear head
        # ---------------------------------------------------------
        W, b = self._fit_linear_head(H, Y)

        # ---------------------------------------------------------
        # 2. Compute contribution matrix
        # V_j = E[ |h_j * W_j| ]
        # ---------------------------------------------------------
        V = np.mean(np.abs(H * W.T), axis=0)      # (D,)

        # ---------------------------------------------------------
        # 3. Sparsification mask
        # ---------------------------------------------------------
        threshold = np.percentile(V, self.clip_percentile)
        M = (V > threshold).astype(np.float32)

        self.W_sparse = W * M[:, None]
        self.b = b

        kept = 100.0 * M.mean()
        print(f"DICE trained | kept weights: {kept:.2f}%")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def compute_dice_scores(self, features):
        """
        Compute frame-level DICE scores
        """
        if self.W_sparse is None:
            raise RuntimeError("DICE router not trained")

        H = self._stack_frames(features)           # (N, D)
        logits = H @ self.W_sparse + self.b        # (N, 1)

        # Regression energy
        scores = np.abs(logits.squeeze())

        return scores

    def predict_routing(self, features, threshold):
        """
        Lower score => more OOD
        """
        scores = self.compute_dice_scores(features)
        route_to_srp = scores < threshold
        return route_to_srp, scores
