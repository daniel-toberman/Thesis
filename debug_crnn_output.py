#!/usr/bin/env python3
"""
Debug CRNN output formats to understand the angle representation.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch

# Add SSL to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SSL'))
from run_CRNN import MyModel, MyDataModule

def debug_single_batch():
    """Debug a single batch to understand data formats."""

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load model
    checkpoint_path = "08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
    model = MyModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.dev = device
    model.eval()
    model = model.to(device)

    # Get test data
    datamodule = MyDataModule()
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()

    # Get first batch
    batch = next(iter(test_dataloader))

    print("=== RAW BATCH INFO ===")
    print(f"Batch 0 (mic signals): {batch[0].shape}")
    print(f"Batch 1 (targets): {batch[1].shape}")
    print(f"Batch 2 (VAD): {batch[2].shape}")

    # Look at raw targets
    print(f"\nTargets sample [0,0,:10]: {batch[1][0,0,:10]}")
    print(f"Targets min/max: {batch[1].min():.3f} / {batch[1].max():.3f}")
    print(f"Targets shape: {batch[1].shape}")

    with torch.no_grad():
        mic_sig_batch = batch[0].to(device)
        targets_batch = batch[1].to(device)
        vad_batch = batch[2].to(device)

        # Preprocess
        data_batch = model.data_preprocess(mic_sig_batch, targets_batch)
        in_batch = data_batch[0]
        gt_batch = [data_batch[1], vad_batch]

        print("\n=== AFTER PREPROCESSING ===")
        print(f"Input batch shape: {in_batch.shape}")
        print(f"GT batch[0] shape: {gt_batch[0].shape}")
        print(f"GT batch[1] shape: {gt_batch[1].shape}")
        print(f"GT batch[0] sample [0,:5,:5]:\n{gt_batch[0][0,:5,:5]}")
        print(f"GT batch[0] min/max: {gt_batch[0].min():.3f} / {gt_batch[0].max():.3f}")

        # Get model predictions
        pred_batch = model(in_batch)

        print(f"\n=== MODEL PREDICTIONS ===")
        print(f"Pred batch shape: {pred_batch.shape}")
        print(f"Pred batch sample [0,:5,:5]:\n{pred_batch[0,:5,:5]}")
        print(f"Pred batch min/max: {pred_batch.min():.3f} / {pred_batch.max():.3f}")

        # Check what the model's metric calculation gives
        metric = model.get_metric(pred_batch=pred_batch, gt_batch=gt_batch, idx=0, tar_type='spect')
        print(f"\n=== MODEL METRICS ===")
        for k, v in metric.items():
            print(f"{k}: {v.item():.3f}")

        # Debug the angle extraction
        print(f"\n=== ANGLE EXTRACTION DEBUG ===")

        # Method 1: Using topk like the model
        vad_pred, doa_pred = pred_batch.topk(1, dim=-1)
        print(f"TopK method - doa_pred shape: {doa_pred.shape}")
        print(f"TopK method - first example angles over time: {doa_pred[0,:10,0]}")

        # Method 2: Using argmax
        doa_argmax = pred_batch.argmax(dim=-1)
        print(f"Argmax method - shape: {doa_argmax.shape}")
        print(f"Argmax method - first example angles over time: {doa_argmax[0,:10]}")

        # Check GT angles
        if len(gt_batch[0].shape) == 3:
            gt_vad, gt_doa = gt_batch[0].topk(1, dim=-1)
            print(f"GT TopK method - gt_doa shape: {gt_doa.shape}")
            print(f"GT TopK method - first example angles over time: {gt_doa[0,:10,0]}")

        gt_argmax = gt_batch[0].argmax(dim=-1)
        print(f"GT Argmax method - shape: {gt_argmax.shape}")
        print(f"GT Argmax method - first example angles over time: {gt_argmax[0,:10]}")

if __name__ == "__main__":
    debug_single_batch()