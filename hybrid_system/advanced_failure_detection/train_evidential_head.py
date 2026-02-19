"""
Train an Evidential Uncertainty Head on top of pre-trained CRNN logits.
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
TRAIN_DATA_PATH = Path(r'C:\daniel\Thesis\train_combined_features-002.npz')
VAL_DATA_PATH = Path(r'C:\daniel\Thesis\crnn features\test_6cm_features.npz')
MODEL_SAVE_PATH = Path(r'C:\daniel\Thesis\hybrid_system\advanced_failure_detection\evidential_head.pth')

BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4 # Slightly lower LR for stability
KL_ANNEALING_EPOCHS = 100 # Longer annealing
NUM_CLASSES = 360

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
        # Using Softplus for stability as per many EDL implementations
        evidence = torch.nn.functional.softplus(logits)
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
    
    # Luce term (Uncertainty Cross Entropy)
    A = torch.sum(labels * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    
    # KL term (Regularization)
    alp_hat = evidence * (1 - labels) + 1
    kl_term = kl_divergence(alp_hat, num_classes, device)
    
    # Annealing factor
    annealing_coef = min(1.0, epoch / annealing_epochs)
    
    # We use a smaller multiplier for KL to prevent it from dominating the Luce term
    loss = torch.mean(A + 0.1 * annealing_coef * kl_term)
    return loss

def load_and_prepare_dataset(file_path, stats=None):
    print(f"Loading data from {file_path}...")
    data = np.load(file_path, allow_pickle=True)
    logits_raw = data['logits_pre_sig']
    gt_angles = data['gt_angles']
    
    logits_processed = []
    for logit in logits_raw:
        if logit.ndim == 2: logits_processed.append(np.mean(logit, axis=0))
        else: logits_processed.append(logit)
    X = np.array(logits_processed)
    
    if stats is None:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-10
    else:
        X_mean, X_std = stats
        
    X = (X - X_mean) / X_std
    X = torch.FloatTensor(X)
    
    # Hard One-Hot Labels (as per paper)
    y_indices = np.round(gt_angles).astype(int) % 360
    y = torch.zeros(len(y_indices), 360)
    y.scatter_(1, torch.LongTensor(y_indices).unsqueeze(1), 1.0)
    
    return TensorDataset(X, y), (X_mean, X_std)

def main():
    print("="*80)
    print("STARTING EVIDENTIAL HEAD TRAINING (HARD LABELS)")
    print("="*80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds, train_stats = load_and_prepare_dataset(TRAIN_DATA_PATH)
    val_ds, _ = load_and_prepare_dataset(VAL_DATA_PATH, stats=train_stats)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = EvidentialHead().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            evidence = model(batch_X)
            loss = evidential_loss(evidence, batch_y, epoch, KL_ANNEALING_EPOCHS, NUM_CLASSES, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                evidence = model(batch_X)
                loss = evidential_loss(evidence, batch_y, epoch, KL_ANNEALING_EPOCHS, NUM_CLASSES, device)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_mean': train_stats[0],
                'input_std': train_stats[1]
            }, MODEL_SAVE_PATH)

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(Path('hybrid_system/advanced_failure_detection/results/evidential_head_training.png'))
    print(f"\n✅ Training Complete! Best Val Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
