"""
Evidential OOD Router for Failure Detection
Automatically adapts to 72-bin or 360-bin models
based on the loaded checkpoint.
"""

import numpy as np
import torch
import torch.nn as nn

def downsample_logits_360_to_72(logits_360):
    """
    Convert 360-dim logits into 72 bins by averaging every 5 bins.
    """
    logits_360 = logits_360.reshape(-1, 72, 5)
    return logits_360.mean(axis=2)


class EvidentialHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        logits = self.mlp(x)
        evidence = torch.nn.functional.softplus(logits) * 0.1
        return evidence


class EvidentialRouter:
    def __init__(self, model_path='evidential_head_72bins.pth', device='cpu'):

        self.device = torch.device(device)

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        # 🔥 Automatically detect model size from checkpoint
        state_dict = checkpoint['model_state_dict']

        # First layer weight shape: (hidden_dim, input_dim)
        first_weight = state_dict['mlp.0.weight']
        input_dim = first_weight.shape[1]

        self.num_classes = input_dim  # 72 or 360

        # Build matching model
        self.model = EvidentialHead(
            input_dim=input_dim,
            output_dim=input_dim
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.mean = torch.FloatTensor(checkpoint['input_mean']).to(self.device)
        self.std = torch.FloatTensor(checkpoint['input_std']).to(self.device)

        print(f"[Router] Loaded evidential model with {self.num_classes} bins.")

    # --------------------------------------------------
    # Main Uncertainty Computation
    # --------------------------------------------------
    def compute_uncertainty(self, features):
        """
        Computes uncertainty score:
            U = K / S

        Higher U = more uncertain.
        """

        logits_raw = features['logits_pre_sig']

        processed = []
        for l in logits_raw:
            if l.ndim == 2:
                l = np.mean(l, axis=0)
            processed.append(l)

        X = torch.FloatTensor(np.array(processed)).to(self.device)

        # 🔥 Safety check
        if X.shape[1] != self.num_classes:
            X = downsample_logits_360_to_72(X)

        with torch.no_grad():
            evidence = self.model(X)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1)
            U = self.num_classes / S

        return U.cpu().numpy()
