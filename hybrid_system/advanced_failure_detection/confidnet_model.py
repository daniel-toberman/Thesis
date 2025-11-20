"""
ConfidNet: Learned Confidence Estimation for Failure Detection

Based on: Corbière et al., "Addressing Failure Prediction by Learning Model Confidence", NeurIPS 2019
Adapted for DOA estimation failure detection.
"""

import torch
import torch.nn as nn


class ConfidNet(nn.Module):
    """
    Confidence estimation network that predicts CRNN failure probability.

    Takes penultimate layer features from CRNN and outputs a confidence score
    indicating how likely the CRNN prediction is to be correct.

    Architecture:
        Input: (batch_size, 256) - penultimate features from CRNN
        Hidden layers with dropout and batch normalization
        Output: (batch_size, 1) - confidence score [0, 1]
    """

    def __init__(self, input_dim=256, hidden_dims=[128, 64], dropout=0.3):
        super(ConfidNet, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation here, will use sigmoid in training/inference)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch_size, 256) penultimate features

        Returns:
            logits: (batch_size, 1) raw logits (use sigmoid for confidence)
        """
        return self.network(x)

    def predict_confidence(self, x):
        """
        Predict confidence scores (sigmoid applied).

        Args:
            x: (batch_size, 256) or (256,) penultimate features

        Returns:
            confidence: (batch_size,) or scalar confidence scores [0, 1]
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(x)
            confidence = torch.sigmoid(logits).squeeze(-1)

        return confidence


class ConfidNetLoss(nn.Module):
    """
    Loss function for ConfidNet training.

    Binary cross-entropy loss for predicting correctness:
    - Target = 1 if CRNN prediction is correct (error <= threshold)
    - Target = 0 if CRNN prediction is incorrect (error > threshold)

    Supports pos_weight for handling class imbalance.
    """

    def __init__(self, pos_weight=None, device='cpu'):
        """
        Args:
            pos_weight: Weight for positive class (correct=1) to handle imbalance
            device: Device to place pos_weight tensor on
        """
        super(ConfidNetLoss, self).__init__()
        self.pos_weight = pos_weight
        if pos_weight is not None:
            # Use float32 for MPS compatibility
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Compute loss.

        Args:
            logits: (batch_size, 1) raw logits from ConfidNet
            targets: (batch_size,) binary labels (1=correct, 0=incorrect)

        Returns:
            loss: scalar loss value
        """
        targets = targets.unsqueeze(-1).float()
        return self.bce_loss(logits, targets)


def get_confidnet_model(input_dim=256, hidden_dims=[128, 64], dropout=0.3, device='mps'):
    """
    Factory function to create ConfidNet model.

    Args:
        input_dim: Input feature dimension (default: 256)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        device: Device to place model on

    Returns:
        model: ConfidNet model on specified device
    """
    model = ConfidNet(input_dim, hidden_dims, dropout)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    print("Testing ConfidNet model...")

    model = get_confidnet_model(device='cpu')
    print(f"Model architecture:\n{model}\n")

    # Test forward pass
    batch_size = 32
    features = torch.randn(batch_size, 256)

    logits = model(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {logits.shape}")

    # Test confidence prediction
    confidence = model.predict_confidence(features)
    print(f"Confidence shape: {confidence.shape}")
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")

    # Test loss
    loss_fn = ConfidNetLoss()
    targets = torch.randint(0, 2, (batch_size,))
    loss = loss_fn(logits, targets)
    print(f"\nTest loss: {loss.item():.4f}")

    print("\n✅ ConfidNet model test passed!")
