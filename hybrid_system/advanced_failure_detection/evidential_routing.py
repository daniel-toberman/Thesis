"""
Evidential OOD Router for Failure Detection
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class EvidentialHead(nn.Module):
    def __init__(self, input_dim=360, hidden_dim=512, output_dim=360):
        super().__init__()
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
        evidence = torch.nn.functional.softplus(logits)
        return evidence

class EvidentialRouter:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = EvidentialHead().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.mean = torch.FloatTensor(checkpoint['input_mean']).to(self.device)
        self.std = torch.FloatTensor(checkpoint['input_std']).to(self.device)
        
    def compute_uncertainty(self, features):
        """
        Computes uncertainty score u = K / S.
        Higher u = more uncertain.
        """
        logits_raw = features['logits_pre_sig']
        
        # Prepare inputs (average over time if needed)
        processed = []
        for l in logits_raw:
            if l.ndim == 2: processed.append(np.mean(l, axis=0))
            else: processed.append(l)
        
        X = torch.FloatTensor(np.array(processed)).to(self.device)
        
        # Standardize using training stats
        X = (X - self.mean) / self.std
        
        with torch.no_grad():
            evidence = self.model(X)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1) # Dirichlet strength
            U = 360 / S # Uncertainty score
            
        return U.cpu().numpy()
