#!/usr/bin/env python3
"""
Extract detailed per-example results from CRNN model for hybrid analysis.
This will help identify where CRNN fails and SRP-PHAT might complement it.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add SSL to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SSL'))
from run_CRNN import MyModel, MyDataModule

def load_model_and_data(checkpoint_path):
    """Load trained CRNN model and test dataloader."""

    # Determine device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load model from checkpoint with proper device mapping
    model = MyModel.load_from_checkpoint(checkpoint_path, map_location=device)

    # Override the device setting to current device
    model.dev = device
    model.eval()

    # Create datamodule
    datamodule = MyDataModule()
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()

    return model, test_dataloader

def extract_predictions(model, test_dataloader, max_examples=None):
    """Extract detailed predictions from CRNN model."""

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)

    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if max_examples and batch_idx * test_dataloader.batch_size >= max_examples:
                break

            mic_sig_batch = batch[0].to(device)
            targets_batch = batch[1].to(device)  # Raw GT angles in degrees
            vad_batch = batch[2].to(device)

            # Store raw GT angles before preprocessing
            raw_gt_angles = targets_batch.cpu().numpy()  # [batch, time, 1]

            # Preprocess data
            data_batch = model.data_preprocess(mic_sig_batch, targets_batch)
            in_batch = data_batch[0]
            gt_batch = [data_batch[1], vad_batch]

            # Get predictions
            pred_batch = model(in_batch)

            # Align shapes if needed
            if pred_batch.shape[1] > gt_batch[0].shape[1]:
                pred_batch = pred_batch[:, :gt_batch[0].shape[1], :]
                # Also trim raw GT
                raw_gt_angles = raw_gt_angles[:, :gt_batch[0].shape[1], :]
            else:
                gt_batch[0] = gt_batch[0][:, :pred_batch.shape[1], :]
                gt_batch[1] = gt_batch[1][:, :pred_batch.shape[1], :]
                raw_gt_angles = raw_gt_angles[:, :pred_batch.shape[1], :]

            # Calculate metrics per example using model's method
            metric = model.get_metric(pred_batch=pred_batch, gt_batch=gt_batch, idx=batch_idx, tar_type='spect')

            # Use the model's own metric calculation for accuracy
            batch_mae = metric.get('MAE', 0).item() if 'MAE' in metric else 0
            batch_acc = metric.get('ACC', 0).item() if 'ACC' in metric else 0

            # Extract predicted angles using argmax (same as model)
            pred_angles_batch = pred_batch.argmax(dim=-1).cpu().numpy()  # [batch, time]

            batch_size = pred_batch.shape[0]
            for b in range(batch_size):
                # Use median over time for stability
                pred_angle = float(np.median(pred_angles_batch[b]))
                gt_angle = float(np.median(raw_gt_angles[b, :, 0]))  # Use raw GT angles

                # Calculate error (circular distance)
                error = abs((pred_angle - gt_angle + 180) % 360 - 180)

                results.append({
                    'batch_idx': batch_idx,
                    'example_idx': batch_idx * test_dataloader.batch_size + b,
                    'pred_angle': pred_angle,
                    'gt_angle': gt_angle,
                    'error_degrees': error,
                    'batch_mae': batch_mae,
                    'batch_acc': batch_acc,
                })

            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx}/{len(test_dataloader)} - Batch MAE: {batch_mae:.2f}°")

    return pd.DataFrame(results)

def analyze_failures(results_df, threshold_error=15):
    """Analyze CRNN failure patterns."""

    # Overall stats
    print(f"CRNN Results Summary:")
    print(f"Mean MAE: {results_df['error_degrees'].mean():.2f}°")
    print(f"Median MAE: {results_df['error_degrees'].median():.2f}°")
    print(f"Accuracy (< 20°): {(results_df['error_degrees'] < 20).mean()*100:.1f}%")

    # Find failures
    failures = results_df[results_df['error_degrees'] > threshold_error]
    print(f"\nCRNN Failures (error > {threshold_error}°): {len(failures)}/{len(results_df)} ({len(failures)/len(results_df)*100:.1f}%)")

    if len(failures) > 0:
        print(f"Failure cases - MAE: {failures['error_degrees'].mean():.1f}° (std: {failures['error_degrees'].std():.1f}°)")

        # Show worst cases
        worst_cases = failures.nlargest(10, 'error_degrees')
        print(f"\nWorst 10 cases:")
        print(worst_cases[['example_idx', 'pred_angle', 'gt_angle', 'error_degrees']])

    return failures

def compare_with_srp(crnn_results_df, srp_results_path):
    """Compare CRNN results with SRP-PHAT results if available."""

    if not Path(srp_results_path).exists():
        print(f"SRP results not found at {srp_results_path}")
        return None

    srp_df = pd.read_csv(srp_results_path)

    # Simple comparison - would need to align by filename/index in practice
    print(f"\nComparison Summary:")
    print(f"CRNN MAE: {crnn_results_df['error_degrees'].mean():.2f}°")
    print(f"SRP-PHAT MAE: {srp_df['abs_err_deg'].mean():.2f}°")

    # Find cases where SRP might be better
    # This is simplified - in practice need proper alignment
    crnn_failures = crnn_results_df[crnn_results_df['error_degrees'] > 20]
    print(f"CRNN failures (>20°): {len(crnn_failures)} cases")

    return None

def main():
    checkpoint_path = "08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"

    print("Loading CRNN model and test data...")
    model, test_dataloader = load_model_and_data(checkpoint_path)

    print("Extracting detailed predictions...")
    results_df = extract_predictions(model, test_dataloader, max_examples=500)  # Limit for faster analysis

    print("Analyzing failure patterns...")
    failures = analyze_failures(results_df)

    # Save results
    results_df.to_csv("crnn_detailed_results.csv", index=False)
    failures.to_csv("crnn_failures.csv", index=False)
    print("Results saved to crnn_detailed_results.csv and crnn_failures.csv")

    # Compare with SRP if available
    srp_path = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08_srp_results.csv"
    compare_with_srp(results_df, srp_path)

if __name__ == "__main__":
    main()