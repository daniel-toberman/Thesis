"""
ConfidNet Router for Hybrid System

This script defines the routing logic for using a pre-trained ConfidNet model
as an OOD detector in the hybrid CRNN-SRP system.
"""

import torch
import numpy as np
from pathlib import Path

# Assuming confidnet_model.py is in the same directory
from confidnet_model import ConfidNet

class ConfidNetRouter:
    """
    A router that uses a pre-trained ConfidNet model to make routing decisions.
    """
    def __init__(self, model_path=None, device='cpu'):
        """
        Initializes the ConfidNetRouter.

        Args:
            model_path (str or Path): Path to the trained ConfidNet model checkpoint.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        if model_path is None:
            # Default path where the training script saves the model
            model_path = Path(__file__).parent / 'confidnet_3mic_trained.pth'

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ConfidNet model not found at {model_path}")
        
        self.model_path = model_path
        self.model = None  # Model will be initialized lazily

    def _initialize_model(self, input_dim):
        """Initializes the model with the correct input dimension."""
        self.model = ConfidNet(input_dim=input_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"ConfidNet model loaded from {self.model_path} with input_dim={input_dim}")

    def compute_scores(self, features):
        """
        Computes confidence scores for the given features using the ConfidNet model.

        Args:
            features (dict): A dictionary containing CRNN features, including
                             'penultimate_features'.

        Returns:
            np.ndarray: An array of confidence scores.
        """
        penultimate_features = features['penultimate_features']

        # This logic must match the training script's preprocessing
        max_shape = [0, 0]
        for feat in penultimate_features:
             if feat.ndim > 1:
                max_shape[0] = max(max_shape[0], feat.shape[0])
                max_shape[1] = max(max_shape[1], feat.shape[1])
        
        padded_features_list = []
        for feat in penultimate_features:
            if feat.ndim == 1:
                 pad_width = ((0, max_shape[0] - feat.shape[0]),)
                 padded_feat = np.pad(feat, pad_width, 'constant')
                 padded_feat = np.pad(padded_feat, ((0,0), (0, max_shape[1] - padded_feat.shape[1])), 'constant') if padded_feat.ndim > 1 else np.pad(padded_feat, (0, max_shape[1] - padded_feat.shape[0]), 'constant')
            else:
                pad_width = ((0, max_shape[0] - feat.shape[0]), (0, max_shape[1] - feat.shape[1]))
                padded_feat = np.pad(feat, pad_width, 'constant')
            padded_features_list.append(padded_feat.flatten())
        
        x_tensor = torch.tensor(np.array(padded_features_list), dtype=torch.float32)

        # Lazy initialization of the model
        if self.model is None:
            input_dim = x_tensor.shape[1]
            self._initialize_model(input_dim)

        x_tensor = x_tensor.to(self.device)
        
        with torch.no_grad():
            confidence_scores = self.model.predict_confidence(x_tensor)
        
        return confidence_scores.cpu().numpy()
