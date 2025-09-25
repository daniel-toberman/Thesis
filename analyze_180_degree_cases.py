#!/usr/bin/env python3
"""
Analyze 180° cases in both test and validation sets to understand if this is
a systematic weakness of CRNN that we can focus on for hybrid approaches.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch

# Add SSL to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SSL'))
from run_CRNN import MyModel, MyDataModule

def analyze_180_cases(model, dataloader, dataset_name, device):
    """Analyze cases around 180° to understand CRNN performance."""

    results = []
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mic_sig_batch = batch[0].to(device)
            targets_batch = batch[1].to(device)
            vad_batch = batch[2].to(device)

            # Store raw GT angles before preprocessing
            raw_gt_angles = targets_batch.cpu().numpy()

            # Preprocess and predict
            data_batch = model.data_preprocess(mic_sig_batch, targets_batch)
            in_batch = data_batch[0]
            gt_batch = [data_batch[1], vad_batch]
            pred_batch = model(in_batch)

            # Align shapes
            if pred_batch.shape[1] > gt_batch[0].shape[1]:
                pred_batch = pred_batch[:, :gt_batch[0].shape[1], :]
                raw_gt_angles = raw_gt_angles[:, :gt_batch[0].shape[1], :]
            else:
                gt_batch[0] = gt_batch[0][:, :pred_batch.shape[1], :]
                gt_batch[1] = gt_batch[1][:, :pred_batch.shape[1], :]
                raw_gt_angles = raw_gt_angles[:, :pred_batch.shape[1], :]

            # Extract predictions
            pred_angles_batch = pred_batch.argmax(dim=-1).cpu().numpy()

            batch_size = pred_batch.shape[0]
            for b in range(batch_size):
                pred_angle = float(np.median(pred_angles_batch[b]))
                gt_angle = float(np.median(raw_gt_angles[b, :, 0]))
                error = abs((pred_angle - gt_angle + 180) % 360 - 180)

                results.append({
                    'dataset': dataset_name,
                    'batch_idx': batch_idx,
                    'example_idx': batch_idx * dataloader.batch_size + b,
                    'pred_angle': pred_angle,
                    'gt_angle': gt_angle,
                    'error_degrees': error,
                })

            total_batches += 1
            if batch_idx % 100 == 0:
                print(f"  Processed {dataset_name} batch {batch_idx}")

    return pd.DataFrame(results)

def analyze_angle_regions(df, dataset_name):
    """Analyze performance in different angular regions."""

    print(f"\n=== {dataset_name.upper()} SET ANALYSIS ===")
    print(f"Total examples: {len(df)}")

    # Define angular regions
    regions = {
        'Front (0±30°)': (df['gt_angle'] <= 30) | (df['gt_angle'] >= 330),
        'Right (90±30°)': (df['gt_angle'] >= 60) & (df['gt_angle'] <= 120),
        'Back (180±30°)': (df['gt_angle'] >= 150) & (df['gt_angle'] <= 210),
        'Left (270±30°)': (df['gt_angle'] >= 240) & (df['gt_angle'] <= 300),
    }

    print(f"\nPerformance by Angular Region:")
    print(f"{'Region':<15} {'Count':<8} {'MAE':<8} {'Std':<8} {'Failures >15°':<15}")
    print("-" * 60)

    for region_name, mask in regions.items():
        region_data = df[mask]
        if len(region_data) > 0:
            mae = region_data['error_degrees'].mean()
            std = region_data['error_degrees'].std()
            failures = (region_data['error_degrees'] > 15).sum()
            print(f"{region_name:<15} {len(region_data):<8} {mae:<8.2f} {std:<8.2f} {failures:<15}")
        else:
            print(f"{region_name:<15} {0:<8} {'N/A':<8} {'N/A':<8} {0:<15}")

    # Specific analysis for 180° cases
    back_180_exact = df[df['gt_angle'] == 180.0]
    back_180_range = df[(df['gt_angle'] >= 170) & (df['gt_angle'] <= 190)]

    print(f"\n180° Specific Analysis:")
    print(f"Exactly 180°: {len(back_180_exact)} cases")
    if len(back_180_exact) > 0:
        print(f"  MAE: {back_180_exact['error_degrees'].mean():.2f}°")
        print(f"  Failures (>15°): {(back_180_exact['error_degrees'] > 15).sum()}")
        print(f"  Worst case: {back_180_exact['error_degrees'].max():.1f}°")

    print(f"180±10° range: {len(back_180_range)} cases")
    if len(back_180_range) > 0:
        print(f"  MAE: {back_180_range['error_degrees'].mean():.2f}°")
        print(f"  Failures (>15°): {(back_180_range['error_degrees'] > 15).sum()}")

    # Show worst cases in 180° region
    if len(back_180_range) > 0:
        worst_180 = back_180_range.nlargest(5, 'error_degrees')
        print(f"\nWorst 5 cases in 180±10° region:")
        print(worst_180[['example_idx', 'pred_angle', 'gt_angle', 'error_degrees']])

    return back_180_exact, back_180_range

def main():
    checkpoint_path = "08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print("Loading CRNN model...")
    model = MyModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.dev = device
    model.eval()
    model = model.to(device)

    print("Loading data...")
    datamodule = MyDataModule()
    datamodule.setup('fit')  # This sets up train/val
    datamodule.setup('test')  # This sets up test

    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    print("Analyzing validation set...")
    val_results = analyze_180_cases(model, val_dataloader, 'validation', device)

    print("Analyzing test set...")
    test_results = analyze_180_cases(model, test_dataloader, 'test', device)

    # Analyze each dataset
    val_180_exact, val_180_range = analyze_angle_regions(val_results, 'validation')
    test_180_exact, test_180_range = analyze_angle_regions(test_results, 'test')

    # Combined analysis
    combined_results = pd.concat([val_results, test_results], ignore_index=True)

    print(f"\n=== COMBINED ANALYSIS ===")
    print(f"Total examples: {len(combined_results)}")

    # Overall 180° analysis
    combined_180_exact = combined_results[combined_results['gt_angle'] == 180.0]
    combined_180_range = combined_results[(combined_results['gt_angle'] >= 170) & (combined_results['gt_angle'] <= 190)]

    print(f"\nCombined 180° Analysis:")
    print(f"Exactly 180°: {len(combined_180_exact)} cases")
    print(f"180±10° range: {len(combined_180_range)} cases")

    if len(combined_180_range) > 0:
        print(f"180° region MAE: {combined_180_range['error_degrees'].mean():.2f}°")
        print(f"180° region failures (>15°): {(combined_180_range['error_degrees'] > 15).sum()}")
        failure_rate = (combined_180_range['error_degrees'] > 15).mean() * 100
        print(f"180° region failure rate: {failure_rate:.1f}%")

        # Compare with overall performance
        overall_mae = combined_results['error_degrees'].mean()
        overall_failures = (combined_results['error_degrees'] > 15).mean() * 100
        print(f"\nComparison with overall performance:")
        print(f"Overall MAE: {overall_mae:.2f}°")
        print(f"Overall failure rate (>15°): {overall_failures:.1f}%")
        print(f"180° region is {combined_180_range['error_degrees'].mean() / overall_mae:.1f}x worse")

    # Save detailed results
    val_results.to_csv("validation_detailed_results.csv", index=False)
    test_results.to_csv("test_detailed_results.csv", index=False)
    combined_180_range.to_csv("180_degree_cases.csv", index=False)

    print(f"\nSaved detailed results to CSV files")
    print(f"Focus area identified: {len(combined_180_range)} cases in 180±10° region")

if __name__ == "__main__":
    main()