#!/usr/bin/env python3
"""
Combine 6cm and 3x12cm Training Features

Creates a combined training set with both easy (6cm) and hard (3x12cm) examples.
This provides better class balance for training failure detectors.
"""

import numpy as np
from pathlib import Path

def combine_features(features_6cm_path, features_3x12cm_path, output_path):
    """
    Combine two feature files.

    Args:
        features_6cm_path: Path to 6cm training features
        features_3x12cm_path: Path to 3x12cm training features
        output_path: Where to save combined features
    """
    print("="*80)
    print("COMBINING TRAINING FEATURES")
    print("="*80)

    # Load 6cm features
    print(f"\nLoading 6cm features: {features_6cm_path}")
    data_6cm = np.load(features_6cm_path, allow_pickle=True)
    features_6cm = {key: data_6cm[key] for key in data_6cm.files}
    n_6cm = len(features_6cm['predicted_angles'])
    print(f"  Samples: {n_6cm}")
    print(f"  MAE: {features_6cm['abs_errors'].mean():.2f}°")
    print(f"  Success (≤5°): {(features_6cm['abs_errors'] <= 5).sum()} ({(features_6cm['abs_errors'] <= 5).mean()*100:.1f}%)")

    # Load 3x12cm features
    print(f"\nLoading 3x12cm features: {features_3x12cm_path}")
    data_3x12 = np.load(features_3x12cm_path, allow_pickle=True)
    features_3x12 = {key: data_3x12[key] for key in data_3x12.files}
    n_3x12 = len(features_3x12['predicted_angles'])
    print(f"  Samples: {n_3x12}")
    print(f"  MAE: {features_3x12['abs_errors'].mean():.2f}°")
    print(f"  Success (≤5°): {(features_3x12['abs_errors'] <= 5).sum()} ({(features_3x12['abs_errors'] <= 5).mean()*100:.1f}%)")

    # Combine all keys
    print(f"\nCombining features...")
    combined = {}

    for key in features_6cm.keys():
        if key in features_3x12:
            # For variable-length arrays (object dtype), concatenate lists
            if features_6cm[key].dtype == object:
                combined[key] = np.concatenate([features_6cm[key], features_3x12[key]])
            # For regular arrays, concatenate directly
            else:
                combined[key] = np.concatenate([features_6cm[key], features_3x12[key]])
        else:
            print(f"  Warning: key '{key}' not found in 3x12cm features")

    # Combined statistics
    n_total = len(combined['predicted_angles'])
    mae_total = combined['abs_errors'].mean()
    success_total = (combined['abs_errors'] <= 5).mean()

    print(f"\nCombined dataset:")
    print(f"  Total samples: {n_total}")
    print(f"  6cm contribution: {n_6cm} ({n_6cm/n_total*100:.1f}%)")
    print(f"  3x12cm contribution: {n_3x12} ({n_3x12/n_total*100:.1f}%)")
    print(f"  Overall MAE: {mae_total:.2f}°")
    print(f"  Overall success (≤5°): {(combined['abs_errors'] <= 5).sum()} ({success_total*100:.1f}%)")

    # Class distribution at different thresholds
    print(f"\nClass distribution for training:")
    for threshold in [3, 5, 10]:
        n_correct = (combined['abs_errors'] <= threshold).sum()
        n_incorrect = n_total - n_correct
        print(f"  Threshold {threshold}°: {n_correct} correct ({n_correct/n_total*100:.1f}%), {n_incorrect} incorrect ({n_incorrect/n_total*100:.1f}%)")

    # Save combined features
    print(f"\nSaving combined features to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)

    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n✅ Features combined successfully!")

    return combined


if __name__ == "__main__":
    # File paths
    features_6cm = Path("features/train_6cm_features.npz")
    features_3x12cm = Path("features/train_3x12cm_consecutive_features.npz")
    output = Path("features/train_combined_features.npz")

    # Check input files exist
    if not features_6cm.exists():
        print(f"ERROR: {features_6cm} not found")
        exit(1)

    if not features_3x12cm.exists():
        print(f"ERROR: {features_3x12cm} not found")
        print("Run: python extract_features.py --split train --array_config 3x12cm_consecutive --device mps")
        exit(1)

    # Combine
    combined = combine_features(features_6cm, features_3x12cm, output)

    print("\n" + "="*80)
    print("READY FOR TRAINING")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Retrain Temperature + Mahalanobis:")
    print(f"     python run_pipeline.py --skip_extraction --features_path {output}")
    print(f"  2. Train ConfidNet:")
    print(f"     python train_confidnet.py --features_path {output} --epochs 100")
