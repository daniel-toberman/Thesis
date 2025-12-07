#!/usr/bin/env python3
"""
Train Deep SVDD for One-Class Failure Detection

Deep SVDD maps successful CRNN predictions to a small hypersphere.
Failures naturally fall outside the hypersphere.

Based on: Ruff et al., "Deep One-Class Classification", ICML 2018
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SVDDNetwork(nn.Module):
    """
    Deep SVDD network that maps features to hypersphere.
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(SVDDNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Output layer (maps to hypersphere space)
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FeatureDataset(Dataset):
    """Dataset for CRNN features."""

    def __init__(self, features):
        """
        Args:
            features: (N,) array where each element is (T, D) temporal features
        """
        # Average over temporal dimension
        self.features = np.array([f.mean(axis=0) for f in features])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx])


def initialize_center(network, dataloader, device):
    """Initialize hypersphere center as mean of network outputs."""
    print("\nInitializing hypersphere center...")

    network.eval()
    outputs = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = network(batch)
            outputs.append(output.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    center = outputs.mean(axis=0)

    # Avoid center at origin
    center[(abs(center) < 0.1) & (center < 0)] = -0.1
    center[(abs(center) < 0.1) & (center >= 0)] = 0.1

    print(f"  Center shape: {center.shape}")
    print(f"  Center norm: {np.linalg.norm(center):.4f}")

    return torch.FloatTensor(center)


def train_deep_svdd(train_features_path, error_threshold, output_dir,
                   hidden_dims=[128, 64, 32], epochs=50, batch_size=32,
                   lr=1e-4, device='cpu'):
    """Train Deep SVDD model."""

    print("="*100)
    print("DEEP SVDD TRAINING")
    print("="*100)

    # Load training features
    print(f"\nLoading training features from: {train_features_path}")
    data = np.load(train_features_path, allow_pickle=True)

    features = data['penultimate_features']
    errors = data['abs_errors']

    # Use only successful predictions for one-class learning
    successes = errors <= error_threshold
    success_features = features[successes]

    print(f"  Total samples: {len(features)}")
    print(f"  Successful samples: {len(success_features)} ({len(success_features)/len(features)*100:.1f}%)")
    print(f"  Feature shape (per sample): {success_features[0].shape}")

    # Get feature dimension (averaged over time)
    feature_dim = success_features[0].shape[1]
    print(f"  Feature dimension: {feature_dim}")

    # Create dataset and dataloader
    dataset = FeatureDataset(success_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    print(f"\nInitializing Deep SVDD network...")
    print(f"  Hidden dimensions: {hidden_dims}")
    network = SVDDNetwork(input_dim=feature_dim, hidden_dims=hidden_dims)
    network = network.to(device)

    # Initialize hypersphere center
    center = initialize_center(network, dataloader, device)
    center = center.to(device)

    # Setup optimizer
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-6)

    # Training loop
    print(f"\n{'='*100}")
    print("TRAINING")
    print("="*100)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")

    network.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            outputs = network(batch)

            # Deep SVDD loss: minimize distance to center
            dist = torch.sum((outputs - center) ** 2, dim=1)
            loss = torch.mean(dist)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")

    # Compute final statistics
    print(f"\n{'='*100}")
    print("COMPUTING HYPERSPHERE STATISTICS")
    print("="*100)

    network.eval()
    distances = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = network(batch)
            dist = torch.sum((outputs - center) ** 2, dim=1)
            distances.append(dist.cpu().numpy())

    distances = np.concatenate(distances)

    print(f"  Hypersphere radius (mean distance): {distances.mean():.4f}")
    print(f"  Distance std: {distances.std():.4f}")
    print(f"  Distance median: {np.median(distances):.4f}")
    print(f"  Distance 90th percentile: {np.percentile(distances, 90):.4f}")
    print(f"  Distance 95th percentile: {np.percentile(distances, 95):.4f}")
    print(f"  Distance max: {distances.max():.4f}")

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = {
        'network_state_dict': network.state_dict(),
        'center': center.cpu().numpy(),
        'feature_dim': feature_dim,
        'hidden_dims': hidden_dims,
        'error_threshold': error_threshold,
        'training_stats': {
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std()),
            'median_distance': float(np.median(distances)),
            'p90_distance': float(np.percentile(distances, 90)),
            'p95_distance': float(np.percentile(distances, 95)),
            'max_distance': float(distances.max()),
        }
    }

    model_path = output_dir / "deep_svdd_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved to: {model_path}")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(losses) + 1),
        'loss': losses
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"✅ Training history saved to: {output_dir}/training_history.csv")

    # Save distance distribution
    distances_df = pd.DataFrame({
        'distance': distances
    })
    distances_df.to_csv(output_dir / "train_distances.csv", index=False)
    print(f"✅ Training distances saved to: {output_dir}/train_distances.csv")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Deep SVDD for one-class failure detection')
    parser.add_argument('--train_features', type=str, required=True,
                        help='Path to training features .npz file')
    parser.add_argument('--error_threshold', type=float, default=5.0,
                        help='Error threshold to define successful predictions (default: 5.0°)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for model')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                        help='Hidden layer dimensions (default: 128 64 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'mps', 'cuda'],
                        help='Device to use (default: cpu)')

    args = parser.parse_args()

    train_deep_svdd(
        train_features_path=args.train_features,
        error_threshold=args.error_threshold,
        output_dir=args.output_dir,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
