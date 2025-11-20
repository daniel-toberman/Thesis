#!/usr/bin/env python3
"""
Train ConfidNet for CRNN Failure Detection

Trains a confidence estimation network to predict when CRNN will fail.
Uses penultimate layer features from CRNN on training data.

Usage:
    python train_confidnet.py --epochs 100 --device mps
    python train_confidnet.py --epochs 50 --error_threshold 10 --test_mode
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from confidnet_model import ConfidNet, ConfidNetLoss


class ConfidNetDataset(Dataset):
    """Dataset for ConfidNet training."""

    def __init__(self, features, labels):
        """
        Args:
            features: (N, T, 256) penultimate features
            labels: (N,) binary labels (1=correct, 0=incorrect)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Average features over time dimension
        feat = self.features[idx]  # (T, 256)
        feat_avg = feat.mean(axis=0)  # (256,)

        return {
            'features': torch.from_numpy(feat_avg).float(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_binary_labels(errors, error_threshold=5.0):
    """
    Create binary labels from prediction errors.

    Args:
        errors: (N,) absolute errors in degrees
        error_threshold: threshold for "correct" classification

    Returns:
        labels: (N,) binary labels (1=correct, 0=incorrect)
    """
    return (errors <= error_threshold).astype(np.int64)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * features.size(0)
        predictions = (torch.sigmoid(logits.squeeze()) > 0.5).long()
        correct += (predictions == labels).sum().item()
        total += features.size(0)

    return total_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_confidences = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(features)
            loss = criterion(logits, labels)

            # Track metrics
            total_loss += loss.item() * features.size(0)
            confidences = torch.sigmoid(logits.squeeze())
            predictions = (confidences > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += features.size(0)

            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_confidences), np.array(all_labels)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_confidence_distribution(confidences, labels, output_dir):
    """Analyze confidence distribution."""
    correct_conf = confidences[labels == 1]
    incorrect_conf = confidences[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct_conf, bins=50, alpha=0.6, label='Correct predictions', color='green')
    ax.hist(incorrect_conf, bins=50, alpha=0.6, label='Incorrect predictions', color='red')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution on Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train ConfidNet for failure detection')
    parser.add_argument('--features_path', type=str,
                        default='features/train_6cm_features.npz',
                        help='Path to training features')
    parser.add_argument('--error_threshold', type=float, default=5.0,
                        help='Error threshold for "correct" classification (degrees)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='models/confidnet',
                        help='Output directory for model and plots')
    parser.add_argument('--test_mode', action='store_true',
                        help='Quick test mode (limit samples)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CONFIDNET TRAINING")
    print("="*80)
    print(f"Features: {args.features_path}")
    print(f"Error threshold: {args.error_threshold}°")
    print(f"Validation split: {args.val_split}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Dropout: {args.dropout}")
    print(f"Device: {args.device}")

    # Load features
    print(f"\nLoading features from: {args.features_path}")
    data = np.load(args.features_path, allow_pickle=True)
    features = data['penultimate_features']  # (N, T, 256)
    errors = data['abs_errors']  # (N,)

    if args.test_mode:
        print("⚠️  TEST MODE: Using only 1000 samples")
        features = features[:1000]
        errors = errors[:1000]

    print(f"  Total samples: {len(features)}")
    print(f"  Feature shape: {features[0].shape}")

    # Create binary labels
    labels = create_binary_labels(errors, args.error_threshold)
    n_correct = labels.sum()
    n_incorrect = len(labels) - n_correct
    print(f"\nLabel distribution:")
    print(f"  Correct (≤{args.error_threshold}°):   {n_correct:>6} ({n_correct/len(labels)*100:.1f}%)")
    print(f"  Incorrect (>{args.error_threshold}°): {n_incorrect:>6} ({n_incorrect/len(labels)*100:.1f}%)")

    # Create dataset
    dataset = ConfidNetDataset(features, labels)

    # Split into train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Training:   {n_train:>6} samples")
    print(f"  Validation: {n_val:>6} samples")

    # Create dataloaders
    # Note: pin_memory not supported on MPS
    use_pin_memory = (args.device == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=use_pin_memory
    )

    # Initialize model
    print(f"\nInitializing ConfidNet model...")
    model = ConfidNet(
        input_dim=256,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate class weights for imbalanced dataset
    # Weight inversely proportional to class frequency
    pos_weight = n_incorrect / n_correct  # Weight for positive class (correct=1)
    print(f"\nClass imbalance handling:")
    print(f"  Positive weight (correct): 1.0")
    print(f"  Negative weight (incorrect): {pos_weight:.2f}")

    # Loss and optimizer (pos_weight will be moved to device in ConfidNetLoss)
    criterion = ConfidNetLoss(pos_weight=pos_weight, device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # Validate
        val_loss, val_acc, val_confidences, val_labels = validate(
            model, val_loader, criterion, args.device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print progress
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.ckpt')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")

        print()

    print(f"{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation acc:  {best_val_acc*100:.2f}%")

    # Plot training curves
    print(f"\nSaving training curves...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir)

    # Analyze final confidence distribution
    print(f"Analyzing confidence distribution...")
    analyze_confidence_distribution(val_confidences, val_labels, output_dir)

    # Save training metrics
    np.savez(
        output_dir / 'training_metrics.npz',
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc
    )

    print(f"\n✅ All results saved to: {output_dir}")
    print(f"\nTo use this model for routing:")
    print(f"  python evaluate_confidnet_hybrid.py --model_path {output_dir / 'best_model.ckpt'}")


if __name__ == "__main__":
    main()
