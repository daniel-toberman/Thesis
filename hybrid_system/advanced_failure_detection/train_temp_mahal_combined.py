#!/usr/bin/env python3
"""
Train Temperature + Mahalanobis on Combined Training Data

Fits temperature scaling and Mahalanobis OOD detector on combined 6cm + 3x12cm training features.

Usage:
    python train_temp_mahal_combined.py --n_pca 64
"""

import argparse
import numpy as np
from pathlib import Path
import pickle

from temperature_scaling import optimize_temperature
from mahalanobis_ood import MahalanobisOODDetector


def main():
    parser = argparse.ArgumentParser(description='Train Temperature + Mahalanobis on combined data')
    parser.add_argument('--train_features', type=str,
                        default='features/train_combined_features.npz',
                        help='Path to combined training features')
    parser.add_argument('--n_pca_components', type=int, default=64,
                        help='Number of PCA components for Mahalanobis')
    parser.add_argument('--error_threshold', type=float, default=5.0,
                        help='Error threshold for "correct" classification (degrees)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for trained models')

    args = parser.parse_args()

    print("="*80)
    print("TRAINING TEMPERATURE + MAHALANOBIS ON COMBINED DATA")
    print("="*80)
    print(f"Training features: {args.train_features}")
    print(f"PCA components: {args.n_pca_components}")
    print(f"Error threshold: {args.error_threshold}°")

    # Load training features
    print(f"\nLoading training features...")
    data = np.load(args.train_features, allow_pickle=True)

    # Get logits (averaged over time)
    print("  Averaging logits over time dimension...")
    logits_raw = data['logits_pre_sig']
    avg_logits = []
    for logit in logits_raw:
        # logit shape: (T, 360)
        avg_logits.append(logit.mean(axis=0))  # (360,)
    train_logits = np.array(avg_logits)  # (N, 360)

    train_features = data['penultimate_features']
    train_gt_angles = data['gt_angles']
    train_errors = data['abs_errors']
    train_correct = (train_errors <= args.error_threshold)

    print(f"  Total samples: {len(train_errors)}")
    print(f"  Correct (≤{args.error_threshold}°):   {train_correct.sum():>6} ({train_correct.mean()*100:.1f}%)")
    print(f"  Incorrect (>{args.error_threshold}°): {(~train_correct).sum():>6} ({(~train_correct).mean()*100:.1f}%)")
    print(f"  Mean error: {train_errors.mean():.2f}°")

    # 1. Optimize temperature scaling
    print("\n" + "="*80)
    print("FITTING TEMPERATURE SCALING")
    print("="*80)

    optimal_temp, results = optimize_temperature(
        train_logits,
        train_gt_angles,
        initial_temp=1.0,
        method='nll'
    )

    print(f"\n✅ Optimal temperature: {optimal_temp:.3f}")
    print(f"   NLL before: {results.get('nll_before', 'N/A')}")
    print(f"   NLL after: {results.get('nll_after', 'N/A')}")

    # Save temperature
    temp_path = Path(args.output_dir) / 'temperature_combined.npy'
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(temp_path, optimal_temp)
    print(f"   Saved to: {temp_path}")

    # 2. Fit Mahalanobis detector
    print("\n" + "="*80)
    print("FITTING MAHALANOBIS OOD DETECTOR")
    print("="*80)

    # Average features over time dimension
    print("\nAveraging features over time dimension...")
    train_features_avg = []
    for feat in train_features:
        # feat shape: (T, 256)
        feat_avg = feat.mean(axis=0)  # (256,)
        train_features_avg.append(feat_avg)
    train_features_avg = np.array(train_features_avg)

    print(f"  Feature shape after averaging: {train_features_avg.shape}")

    # Fit detector on correct samples only
    correct_features = train_features_avg[train_correct]
    correct_angles = train_gt_angles[train_correct]
    print(f"\nFitting on {len(correct_features)} correct samples...")

    mahal_detector = MahalanobisOODDetector(n_components=args.n_pca_components)
    mahal_detector.fit(correct_features, correct_angles, angle_resolution=10)

    print(f"\n✅ Mahalanobis detector fitted")
    print(f"   PCA components: {args.n_pca_components}")
    print(f"   Explained variance: {mahal_detector.pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Save detector
    mahal_path = Path(args.output_dir) / 'mahalanobis_combined.pkl'
    with open(mahal_path, 'wb') as f:
        pickle.dump(mahal_detector, f)
    print(f"\n✅ Saved to: {mahal_path}")

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTrained models:")
    print(f"  1. Temperature scaling: {temp_path}")
    print(f"  2. Mahalanobis detector: {mahal_path}")

    print(f"\nNext steps:")
    print(f"  1. Evaluate hybrid system:")
    print(f"     python evaluate_temp_mahal_hybrid.py \\")
    print(f"       --temperature_path {temp_path} \\")
    print(f"       --mahalanobis_path {mahal_path}")
    print(f"  2. Compare with ConfidNet:")
    print(f"     python evaluate_confidnet_hybrid.py \\")
    print(f"       --model_path models/confidnet_combined/best_model.ckpt")


if __name__ == "__main__":
    main()
