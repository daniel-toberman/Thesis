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

    def _compute_V_streaming(self, H, W, batch_size=50000):
        """
        Compute V_j = E[|h_j * W_j|] without large temporaries
        """
        N, D = H.shape
        V = np.zeros(D, dtype=np.float64)

        for i in range(0, N, batch_size):
            h_batch = H[i:i + batch_size]  # (B, D)
            V += np.sum(np.abs(h_batch * W.T), axis=0)

        V /= N
        return V

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

    def _fit_linear_head_streaming(self, features):
        """
        Fit Y ≈ H W + b using streaming normal equations.
        Guaranteed to return (W, b).
        """

        H_list = features["penultimate_features"]
        Y_list = features["logits_pre_sig"]

        # --- sanity checks ---
        assert len(H_list) > 0
        assert len(H_list) == len(Y_list)

        D = H_list[0].shape[1]

        XtX = np.zeros((D + 1, D + 1), dtype=np.float64)
        Xty = np.zeros((D + 1, 1), dtype=np.float64)

        for h, y_vec in zip(H_list, Y_list):
            y = np.mean(y_vec)  # ← SCALAR TARGET
            T = h.shape[0]

            # augment with bias
            ones = np.ones((T, 1), dtype=h.dtype)
            h_aug = np.hstack([h, ones])  # (T, D+1)

            XtX += h_aug.T @ h_aug
            Xty += h_aug.T @ np.full((T, 1), y, dtype=np.float64)

        # ridge regularization
        lam = 1e-4
        XtX += lam * np.eye(D + 1)

        theta = np.linalg.solve(XtX, Xty)

        W = theta[:-1]  # (D, 1)
        b = theta[-1]  # (1,)

        # --- ABSOLUTE GUARANTEE ---
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

        W, b = self._fit_linear_head_streaming(features)
        H = self._stack_frames(features)

        # ---------------------------------------------------------
        # 2. Compute contribution matrix
        # V_j = E[ |h_j * W_j| ]
        # ---------------------------------------------------------
        V = self._compute_V_streaming(H, W)

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
