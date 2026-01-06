"""
Train ConfidNet on a Single Microphone Configuration

This script trains a ConfidNet model on the CRNN features from a single,
specified microphone configuration. The goal is to create a model that can
predict CRNN failures, which can then be evaluated on other "unseen"
configurations to test its generalization capability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle
from pathlib import Path
import os

# Import the ConfidNet model and loss from the existing script
from confidnet_model import ConfidNet, ConfidNetLoss

# --- Configuration ---
TRAIN_CONFIG = '9_2_11_4_5_14_7_8_0'
# Construct path relative to this script's location
SCRIPT_DIR = Path(__file__).parent
FEATURES_DIR = SCRIPT_DIR.parent.parent / 'crnn features'
MODEL_OUTPUT_PATH = SCRIPT_DIR / 'confidnet_3mic_trained.pth'
ERROR_THRESHOLD = 5.0  # Degrees
VAL_SPLIT_RATIO = 0.01
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# 1. Load Data
# ------------------------------------------------------------------------------
features_path = FEATURES_DIR / f'crnn_results_mics_{TRAIN_CONFIG}.pkl'
print(f"Loading training data from: {features_path}")

if not features_path.exists():
    raise FileNotFoundError(f"Training data not found at {features_path}. Please ensure the file exists.")

with open(features_path, 'rb') as f:
    crnn_results = pickle.load(f)

# Extract penultimate features and errors
penultimate_features = [d['penultimate_features'] for d in crnn_results]
abs_errors = np.array([d['crnn_error'] for d in crnn_results])

# Pad features to have the same 2D shape
max_shape = [0, 0]
for feat in penultimate_features:
    if feat.ndim > 1:
        max_shape[0] = max(max_shape[0], feat.shape[0])
        max_shape[1] = max(max_shape[1], feat.shape[1])

padded_features = []
for feat in penultimate_features:
    if feat.ndim == 1: # Handle 1D arrays by padding them to match the 2D shape
        pad_width = ((0, max_shape[0] - feat.shape[0]),)
        padded_feat = np.pad(feat, pad_width, 'constant')
        padded_feat = np.pad(padded_feat, ((0,0), (0, max_shape[1] - padded_feat.shape[1])), 'constant') if padded_feat.ndim > 1 else np.pad(padded_feat, (0, max_shape[1] - padded_feat.shape[0]), 'constant')
    else:
        pad_width = ((0, max_shape[0] - feat.shape[0]), (0, max_shape[1] - feat.shape[1]))
        padded_feat = np.pad(feat, pad_width, 'constant')
    padded_features.append(padded_feat.flatten()) # Flatten after padding

# Stack features into a single tensor
X = torch.tensor(np.array(padded_features), dtype=torch.float32)

# Create binary labels (1 for correct, 0 for incorrect)
y = torch.tensor(abs_errors <= ERROR_THRESHOLD, dtype=torch.float32)

print(f"Loaded {len(X)} samples.")
print(f"Correct (<= {ERROR_THRESHOLD}°): {y.sum().item()} | Incorrect (> {ERROR_THRESHOLD}°): {len(y) - y.sum().item()}")

# 2. Prepare DataLoader
# ------------------------------------------------------------------------------
dataset = TensorDataset(X, y)

# Split into training and validation sets
val_size = int(VAL_SPLIT_RATIO * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# 3. Initialize Model, Loss, and Optimizer
# ------------------------------------------------------------------------------
model = ConfidNet(input_dim=X.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Calculate positive class weight to handle imbalance
pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0
loss_fn = ConfidNetLoss(pos_weight=pos_weight, device=DEVICE)

print(f"Model initialized on {DEVICE}. Using pos_weight: {pos_weight:.2f}")

# 4. Training Loop
# ------------------------------------------------------------------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    for features, labels in train_loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_val_loss += loss.item()
            
            preds = torch.sigmoid(logits) > 0.5
            correct_val += (preds.squeeze() == labels.bool()).sum().item()
            total_val += labels.size(0)
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = correct_val / total_val
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
        print(f"  -> New best model saved to {MODEL_OUTPUT_PATH} (Val Acc: {best_val_acc:.4f})")

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_acc:.4f}")