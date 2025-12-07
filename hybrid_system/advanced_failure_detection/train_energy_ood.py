#!/usr/bin/env python3
"""
Train Energy-Based OOD Detection for Failure Routing

Energy score: E(x) = -T * log(sum(exp(logit_i / T)))
Lower energy = more confident (in-distribution)
Higher energy = less confident (out-of-distribution, should route)

Based on: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy.optimize import minimize_scalar
import pickle


def compute_energy_scores(logits, temperature=1.0):
    """
    Compute energy scores from logits.

    Args:
        logits: Array of logits - can be (N,) of (T, D) arrays or (N, D) array
        temperature: Temperature parameter T

    Returns:
        energy_scores: (N,) array of energy scores
    """
    # Handle case where logits is array of arrays
    if isinstance(logits, np.ndarray) and logits.ndim == 1:
        # Check if first element is an array (not a scalar)
        if hasattr(logits[0], '__len__') and len(logits[0].shape) > 0:
            # logits is (N,) array where each element is (T, D) or (D,)
            # Average over time dimension if needed
            if logits[0].ndim == 2:
                logits = np.array([l.mean(axis=0) for l in logits])
            else:
                logits = np.stack(logits)

    # Handle temporal dimension if present in full array
    if logits.ndim == 3:
        logits = logits.mean(axis=1)

    # Now logits should be (N, D)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits array, got shape {logits.shape}")

    # Energy = -T * log(sum(exp(logit_i / T)))
    # For numerical stability: E = -T * (logsumexp(logits / T))
    max_logit = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp((logits - max_logit) / temperature)
    sum_exp = exp_logits.sum(axis=1)
    energy = -temperature * (np.log(sum_exp) + (max_logit.squeeze() / temperature))

    return energy


def calibrate_temperature(train_logits, train_errors, error_threshold=5.0):
    """
    Find optimal temperature T that best separates successes from failures.

    Minimizes the overlap between energy distributions.
    """

    print(f"\nCalibrating temperature parameter...")
    print(f"  Using error threshold: {error_threshold}°")

    successes = train_errors <= error_threshold
    failures = train_errors > error_threshold

    n_success = successes.sum()
    n_failure = failures.sum()

    print(f"  Successes: {n_success} ({n_success/len(train_errors)*100:.1f}%)")
    print(f"  Failures: {n_failure} ({n_failure/len(train_errors)*100:.1f}%)")

    def objective(T):
        """
        Objective: Maximize separation between success and failure energy distributions.
        We want: low energy for successes, high energy for failures.
        """
        energies = compute_energy_scores(train_logits, temperature=T)

        success_energies = energies[successes]
        failure_energies = energies[failures]

        # Want to maximize: (mean_failure_energy - mean_success_energy) / (std_success + std_failure)
        # Equivalent to minimizing the negative
        separation = (failure_energies.mean() - success_energies.mean()) / (success_energies.std() + failure_energies.std() + 1e-8)

        return -separation  # Minimize negative separation

    # Search for optimal temperature in range [0.1, 10.0]
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')

    optimal_T = result.x
    optimal_separation = -result.fun

    print(f"\n  Optimal temperature: {optimal_T:.4f}")
    print(f"  Separation score: {optimal_separation:.4f}")

    # Compute statistics at optimal T
    energies = compute_energy_scores(train_logits, temperature=optimal_T)
    success_energies = energies[successes]
    failure_energies = energies[failures]

    print(f"\n  Success energies: mean={success_energies.mean():.2f}, std={success_energies.std():.2f}")
    print(f"  Failure energies: mean={failure_energies.mean():.2f}, std={failure_energies.std():.2f}")

    return optimal_T


def train_energy_ood(train_features_path, error_threshold, output_dir):
    """Train Energy-Based OOD detector."""

    print("="*100)
    print("ENERGY-BASED OOD TRAINING")
    print("="*100)

    # Load training features
    print(f"\nLoading training features from: {train_features_path}")
    data = np.load(train_features_path, allow_pickle=True)

    logits = data['logits_pre_sig']
    errors = data['abs_errors']

    print(f"  Total samples: {len(logits)}")
    print(f"  Logits shape (per sample): {logits[0].shape}")

    # Calibrate temperature
    optimal_T = calibrate_temperature(logits, errors, error_threshold=error_threshold)

    # Compute energy distribution for threshold selection
    energies = compute_energy_scores(logits, temperature=optimal_T)

    print(f"\n{'='*100}")
    print("ENERGY DISTRIBUTION")
    print("="*100)
    print(f"  Min: {energies.min():.2f}")
    print(f"  Max: {energies.max():.2f}")
    print(f"  Mean: {energies.mean():.2f}")
    print(f"  Std: {energies.std():.2f}")
    print(f"  Median: {np.median(energies):.2f}")
    print(f"  Q1: {np.percentile(energies, 25):.2f}")
    print(f"  Q3: {np.percentile(energies, 75):.2f}")

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = {
        'temperature': optimal_T,
        'error_threshold': error_threshold,
        'energy_stats': {
            'min': float(energies.min()),
            'max': float(energies.max()),
            'mean': float(energies.mean()),
            'std': float(energies.std()),
            'median': float(np.median(energies)),
            'q1': float(np.percentile(energies, 25)),
            'q3': float(np.percentile(energies, 75)),
        }
    }

    model_path = output_dir / "energy_ood_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved to: {model_path}")

    # Save energy scores for analysis
    energy_df = pd.DataFrame({
        'energy': energies,
        'abs_error': errors,
        'is_failure': errors > error_threshold
    })
    energy_df.to_csv(output_dir / "train_energy_scores.csv", index=False)
    print(f"✅ Training energy scores saved to: {output_dir}/train_energy_scores.csv")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Energy-Based OOD detector')
    parser.add_argument('--train_features', type=str, required=True,
                        help='Path to training features .npz file')
    parser.add_argument('--error_threshold', type=float, default=5.0,
                        help='Error threshold to define failures (default: 5.0°)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for model')

    args = parser.parse_args()

    train_energy_ood(
        train_features_path=args.train_features,
        error_threshold=args.error_threshold,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
