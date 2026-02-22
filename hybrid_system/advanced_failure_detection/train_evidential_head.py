"""
Train an Evidential Uncertainty Head on top of pre-trained CRNN logits.
72-bin (5° resolution) version.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
VAL_DATA_PATH = Path(r'C:\daniel\Thesis\train_combined_features-002.npz')
TRAIN_DATA_PATH = Path(r'C:\daniel\Thesis\train_combined_features-002.npz')

# TRAIN_DATA_PATH = Path(r'C:\daniel\Thesis\crnn features\test_6cm_features.npz')
MODEL_SAVE_PATH = Path(r'C:\daniel\Thesis\hybrid_system\advanced_failure_detection\evidential_head_72bins.pth')

BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-5
KL_ANNEALING_EPOCHS = 30

NUM_CLASSES = 72          # 🔥 changed
BIN_SIZE = 5              # 360 / 72

class EvidentialHead(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=512, output_dim=72):
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
        evidence = torch.nn.functional.softplus(logits) * 0.1
        return evidence

def kl_divergence(alpha, num_classes, device):
    beta = torch.ones((1, num_classes)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def evidential_loss(evidence, labels, epoch, annealing_epochs, num_classes, device):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)

    # Luce term
    A = torch.sum(labels * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # KL term
    alp_hat = evidence * (1 - labels) + 1
    kl_term = kl_divergence(alp_hat, num_classes, device)

    annealing_coef = min(1.0, (epoch + 1) / annealing_epochs)

    loss = torch.mean(A + annealing_coef * kl_term / num_classes)
    return loss


def downsample_logits_360_to_72(logits_360):
    """
    Convert 360-dim logits into 72 bins by averaging every 5 bins.
    """
    logits_360 = logits_360.reshape(-1, 72, 5)
    return logits_360.mean(axis=2)


def load_and_prepare_dataset(file_path, stats=None):
    print(f"Loading data from {file_path}...")
    data = np.load(file_path, allow_pickle=True)
    logits_raw = data['logits_pre_sig']
    gt_angles = data['gt_angles']

    logits_processed = []
    for logit in logits_raw:
        if logit.ndim == 2:
            logit = np.mean(logit, axis=0)
        logits_processed.append(logit)

    X = np.array(logits_processed)

    # 🔥 Downsample 360 → 72
    X = downsample_logits_360_to_72(X)

    if stats is None:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-10
    else:
        X_mean, X_std = stats

    X = torch.FloatTensor(X)

    # --- Gaussian labels over 72 bins ---
    y_indices = (np.round(gt_angles).astype(int) % 360) // BIN_SIZE

    y_onehot = np.zeros((len(y_indices), NUM_CLASSES))

    sigma = 1.0  # narrower because bins are wider now

    for i, idx in enumerate(y_indices):
        bins = np.arange(NUM_CLASSES)
        dist = np.minimum(
            np.abs(bins - idx),
            NUM_CLASSES - np.abs(bins - idx)
        )
        y_onehot[i] = np.exp(-0.5 * (dist / sigma) ** 2)

    y_onehot = y_onehot / np.sum(y_onehot, axis=1, keepdims=True)
    y = torch.FloatTensor(y_onehot)

    return TensorDataset(X, y), (X_mean, X_std)


def main():
    print("="*80)
    print("STARTING EVIDENTIAL HEAD TRAINING (72-BIN VERSION)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, train_stats = load_and_prepare_dataset(TRAIN_DATA_PATH)
    val_ds, _ = load_and_prepare_dataset(VAL_DATA_PATH, stats=train_stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = EvidentialHead().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            evidence = model(batch_X)
            loss = evidential_loss(
                evidence, batch_y,
                epoch, KL_ANNEALING_EPOCHS,
                NUM_CLASSES, device
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                evidence = model(batch_X)
                loss = evidential_loss(
                    evidence, batch_y,
                    epoch, KL_ANNEALING_EPOCHS,
                    NUM_CLASSES, device
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_mean': train_stats[0],
                'input_std': train_stats[1]
            }, MODEL_SAVE_PATH)
            print("  --> Saved new best model")

    print("\n✅ Training Complete! Run analysis script next.")


if __name__ == '__main__':
    main()
