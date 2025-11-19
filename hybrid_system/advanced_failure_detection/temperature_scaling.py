"""
Temperature Scaling for Calibration

Implements temperature scaling calibration method (Guo et al., 2017).
Optimizes a single temperature parameter T to calibrate the model's confidence scores.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On Calibration of Modern Neural Networks. ICML 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict


class TemperatureScaling(nn.Module):
    """Temperature scaling module with single learnable parameter."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (N, C) raw logits before sigmoid

        Returns:
            calibrated_probs: (N, C) calibrated probabilities
        """
        return torch.sigmoid(logits / self.temperature)


def compute_ece(predictions: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).

    For angular predictions, we use the max probability as confidence and check
    if the predicted angle matches the ground truth (within 1 degree tolerance).

    Args:
        predictions: (N, 360) probability distributions
        labels: (N,) ground truth angles in degrees
        n_bins: Number of bins for ECE computation

    Returns:
        ece: Expected Calibration Error
    """
    # Get predicted angles and confidences
    predicted_angles = predictions.argmax(axis=1)
    confidences = predictions.max(axis=1)

    # Check if predictions are correct (within 1 degree)
    correct = np.abs((predicted_angles - labels + 180) % 360 - 180) <= 1.0

    # Bin predictions by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find predictions in this bin
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])

        if in_bin.sum() > 0:
            # Accuracy in bin
            bin_accuracy = correct[in_bin].mean()
            # Average confidence in bin
            bin_confidence = confidences[in_bin].mean()
            # Weight by number of samples
            bin_weight = in_bin.sum() / len(predictions)

            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

    return ece


def optimize_temperature(
    logits_pre_sig: np.ndarray,
    gt_angles: np.ndarray,
    initial_temp: float = 1.0,
    method: str = 'nll'
) -> Tuple[float, Dict]:
    """
    Optimize temperature parameter on validation set.

    Args:
        logits_pre_sig: (N, 360) raw logits before sigmoid
        gt_angles: (N,) ground truth angles
        initial_temp: Initial temperature
        method: 'nll' for negative log-likelihood or 'ece' for calibration error

    Returns:
        optimal_temp: Optimized temperature value
        results: Dictionary with optimization details
    """
    print("\nOptimizing temperature parameter...")

    # Convert to torch tensors
    logits_tensor = torch.from_numpy(logits_pre_sig).float()

    def objective(temp: float) -> float:
        """Objective function to minimize."""
        if temp <= 0:
            return 1e10  # Invalid temperature

        # Apply temperature scaling
        scaled_logits = logits_tensor / temp
        probs = torch.sigmoid(scaled_logits)

        if method == 'nll':
            # Use negative log-likelihood
            # For angular predictions, we use cross-entropy with one-hot targets
            targets = torch.zeros_like(probs)
            for i, angle in enumerate(gt_angles):
                targets[i, int(angle) % 360] = 1.0

            # Binary cross-entropy loss (since we use sigmoid, not softmax)
            loss = -(targets * torch.log(probs + 1e-10) +
                     (1 - targets) * torch.log(1 - probs + 1e-10)).sum(dim=1).mean()

            return loss.item()

        elif method == 'ece':
            # Use ECE as objective
            probs_np = probs.numpy()
            return compute_ece(probs_np, gt_angles)

    # Optimize temperature (search in range [0.1, 10.0])
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')

    optimal_temp = result.x

    # Compute metrics before and after calibration
    probs_before = torch.sigmoid(logits_tensor).numpy()
    probs_after = torch.sigmoid(logits_tensor / optimal_temp).numpy()

    ece_before = compute_ece(probs_before, gt_angles)
    ece_after = compute_ece(probs_after, gt_angles)

    results = {
        'optimal_temperature': optimal_temp,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'ece_improvement': ece_before - ece_after,
        'optimization_success': result.success,
        'optimization_message': result.message if hasattr(result, 'message') else 'Success'
    }

    print(f"  Optimal temperature: {optimal_temp:.4f}")
    print(f"  ECE before: {ece_before:.4f}")
    print(f"  ECE after: {ece_after:.4f}")
    print(f"  Improvement: {ece_before - ece_after:.4f}")

    return optimal_temp, results


def apply_temperature_scaling(
    logits_pre_sig: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Args:
        logits_pre_sig: (N, T, 360) or (N, 360) raw logits
        temperature: Temperature parameter

    Returns:
        calibrated_probs: Same shape as input, calibrated probabilities
    """
    logits_tensor = torch.from_numpy(logits_pre_sig).float()
    calibrated_probs = torch.sigmoid(logits_tensor / temperature)
    return calibrated_probs.numpy()


def evaluate_routing_with_calibration(
    test_features: Dict,
    temperature: float,
    threshold: float = 0.04
) -> Dict:
    """
    Evaluate routing accuracy with temperature-calibrated confidences.

    Args:
        test_features: Dictionary with test set features
        temperature: Calibrated temperature parameter
        threshold: Confidence threshold for routing

    Returns:
        results: Dictionary with routing metrics
    """
    print(f"\nEvaluating routing with temperature={temperature:.4f}, threshold={threshold:.4f}")

    # Apply temperature scaling
    if test_features['logits_pre_sig'][0].ndim == 2:  # (T, 360)
        # Average over time first
        avg_logits = np.array([logits.mean(axis=0) for logits in test_features['logits_pre_sig']])
    else:
        avg_logits = test_features['logits_pre_sig']

    calibrated_probs = apply_temperature_scaling(avg_logits, temperature)

    # Get max probability as confidence
    confidences = calibrated_probs.max(axis=1)

    # Get predicted angles
    predicted_angles = calibrated_probs.argmax(axis=1)
    gt_angles = test_features['gt_angles']

    # Compute errors
    errors = np.abs((predicted_angles - gt_angles + 180) % 360 - 180)

    # Routing decision: route to SRP if confidence < threshold
    route_to_srp = confidences < threshold

    # Success = correctly identify failures (error > 5 degrees)
    actual_failures = errors > 5.0
    detected_failures = route_to_srp

    # Confusion matrix
    tp = np.sum(actual_failures & detected_failures)  # Correctly detected failures
    tn = np.sum(~actual_failures & ~detected_failures)  # Correctly kept successes
    fp = np.sum(~actual_failures & detected_failures)  # False alarms
    fn = np.sum(actual_failures & ~detected_failures)  # Missed failures

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(errors)

    results = {
        'threshold': threshold,
        'temperature': temperature,
        'n_samples': len(errors),
        'n_routed_to_srp': route_to_srp.sum(),
        'routing_rate': route_to_srp.mean(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
    }

    print(f"  Routing rate: {results['routing_rate']*100:.1f}% ({results['n_routed_to_srp']}/{results['n_samples']})")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")

    return results


def grid_search_threshold(
    test_features: Dict,
    temperature: float,
    threshold_range: np.ndarray = None
) -> Tuple[float, Dict]:
    """
    Grid search for optimal routing threshold.

    Args:
        test_features: Dictionary with test set features
        temperature: Calibrated temperature parameter
        threshold_range: Array of thresholds to try

    Returns:
        best_threshold: Optimal threshold value
        all_results: List of results for all thresholds
    """
    if threshold_range is None:
        threshold_range = np.linspace(0.01, 0.20, 20)

    print("\n" + "="*80)
    print("Grid Search for Optimal Threshold")
    print("="*80)

    all_results = []
    best_f1 = 0.0
    best_threshold = threshold_range[0]

    for threshold in threshold_range:
        results = evaluate_routing_with_calibration(test_features, temperature, threshold)
        all_results.append(results)

        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_threshold = threshold

    print("\n" + "="*80)
    print(f"Best threshold: {best_threshold:.4f} (F1={best_f1:.3f})")
    print("="*80)

    return best_threshold, all_results


if __name__ == "__main__":
    """Example usage for testing."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, required=True,
                        help='Path to training features .npz file')
    parser.add_argument('--test_features', type=str, required=True,
                        help='Path to test features .npz file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results')

    args = parser.parse_args()

    # Load features
    print("Loading features...")
    train_data = np.load(args.train_features, allow_pickle=True)
    test_data = np.load(args.test_features, allow_pickle=True)

    # Get average logits if needed
    if train_data['logits_pre_sig'][0].ndim == 2:
        train_avg_logits = np.array([logits.mean(axis=0) for logits in train_data['logits_pre_sig']])
    else:
        train_avg_logits = train_data['logits_pre_sig']

    # Optimize temperature on training/validation set
    optimal_temp, opt_results = optimize_temperature(
        train_avg_logits,
        train_data['gt_angles']
    )

    # Convert test data to dict format
    test_features = {key: test_data[key] for key in test_data.files}

    # Grid search for best threshold
    best_threshold, all_results = grid_search_threshold(
        test_features,
        optimal_temp
    )

    # Save results
    output_path = Path(args.output_dir) / 'temperature_scaling_results.npz'
    np.savez(
        output_path,
        optimal_temperature=optimal_temp,
        optimization_results=opt_results,
        best_threshold=best_threshold,
        all_threshold_results=all_results
    )

    print(f"\nResults saved to: {output_path}")
