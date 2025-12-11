#!/usr/bin/env python3
"""
Evaluate All OOD Methods with Optimal F1 Thresholds

Uses optimal thresholds from distribution analysis to compute hybrid metrics.
SRP results are cached, so this runs quickly.
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# Import all router classes
from energy_ood_routing import EnergyOODRouter
from mc_dropout_routing import MCDropoutRouter
from knn_ood_routing import KNNOODRouter
from mahalanobis_ood_routing import MahalanobisOODRouter
from gradnorm_ood_routing import GradNormOODRouter
from vim_ood_routing import VIMOODRouter
from she_ood_routing import SHEOODRouter
from dice_ood_routing import DICEOODRouter
from llr_ood_routing import LLROODRouter
from max_prob_routing import MaxProbRouter


def load_cached_srp_results():
    """Load cached SRP predictions."""
    srp_path = Path("features/test_3x12cm_srp_results.pkl")
    with open(srp_path, 'rb') as f:
        srp_results = pickle.load(f)
    print(f"Loaded cached SRP results: {len(srp_results)} samples")
    return srp_results


def load_test_features():
    """Load test features and metadata."""
    features_path = Path("features/test_3x12cm_consecutive_features.npz")
    data = np.load(features_path, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"Loaded test features: {len(features['abs_errors'])} samples")
    return features


def load_optimal_thresholds():
    """Load optimal F1 thresholds from distribution analysis."""
    # Load original methods
    df1 = pd.read_csv('results/ood_distributions/distribution_analysis_summary.csv')

    # Load temperature-scaled methods
    df2 = pd.read_csv('results/ood_distributions/temperature_scaling_summary.csv')

    # Combine
    df = pd.concat([df1, df2], ignore_index=True)

    thresholds = {}
    for _, row in df.iterrows():
        method = row['method']
        optimal_thresh = row['optimal_threshold']
        optimal_f1 = row['optimal_f1']
        roc_auc = row['roc_auc']
        thresholds[method] = {
            'threshold': optimal_thresh,
            'f1': optimal_f1,
            'roc_auc': roc_auc
        }

    print(f"Loaded optimal thresholds for {len(thresholds)} methods")
    return thresholds


def compute_method_scores_with_temp(method_name, features, val_features, temperature=None):
    """
    Compute OOD scores for a method, with optional temperature scaling.

    Args:
        method_name: Base method name (e.g., 'energy', 'max_prob')
        features: Test features
        val_features: Validation features (for training if needed)
        temperature: Temperature scaling factor (optional)
    """
    from scipy.special import softmax
    from scipy.stats import entropy

    # Apply temperature scaling to logits if temperature is provided
    if temperature is not None and temperature != 1.0:
        features = features.copy()
        features['logits_pre_sig'] = features['logits_pre_sig'] / temperature

        val_features = val_features.copy()
        val_features['logits_pre_sig'] = val_features['logits_pre_sig'] / temperature

    # Compute scores based on method
    if method_name == 'energy':
        router = EnergyOODRouter()
        router.temperature = 1.0 if temperature is None else temperature
        scores = router.compute_energy_scores(features['logits_pre_sig'])

    elif method_name == 'max_prob':
        router = MaxProbRouter()
        scores = router.compute_max_prob_scores(features)

    elif method_name == 'mc_dropout_entropy':
        router = MCDropoutRouter()
        scores = router.compute_entropy_from_logits(features['logits_pre_sig'])

    elif method_name == 'mc_dropout_variance':
        router = MCDropoutRouter()
        scores = router.compute_variance_from_logits(features['logits_pre_sig'])

    elif method_name == 'vim':
        router = VIMOODRouter()
        router.train(val_features)
        scores = router.compute_vim_scores(features)

    elif method_name == 'she':
        router = SHEOODRouter()
        router.train(val_features)
        scores = router.compute_she_scores(features)

    elif method_name == 'gradnorm':
        router = GradNormOODRouter()
        router.train(val_features)
        scores = router.compute_gradnorm_scores(features)

    elif method_name.startswith('knn_k'):
        k = int(method_name.split('_k')[1])
        router = KNNOODRouter(k=k)
        router.train(val_features)
        scores = router.compute_knn_distances(features)

    elif method_name == 'mahalanobis':
        router = MahalanobisOODRouter()
        router.train(val_features)
        scores = router.compute_mahalanobis_distances(features)

    elif method_name.startswith('dice_'):
        sparsity = int(method_name.split('_')[1])
        router = DICEOODRouter(sparsity_percentile=sparsity)
        router.train(val_features)
        scores = router.compute_dice_scores(features)

    elif method_name.startswith('llr_gmm'):
        n_components = int(method_name.split('gmm')[1])
        router = LLROODRouter(n_components=n_components)
        # Load combined training features for LLR
        train_combined_path = 'features/train_combined_features.npz'
        train_combined_data = np.load(train_combined_path, allow_pickle=True)
        train_combined = {key: train_combined_data[key] for key in train_combined_data.files}
        router.train(train_combined)
        scores = router.compute_llr_scores(features)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    return scores


def evaluate_hybrid_with_threshold(method_display_name, base_method, temperature, threshold,
                                   test_features, val_features, srp_results):
    """
    Evaluate hybrid performance with given threshold.

    Returns:
        dict with routing stats and hybrid metrics
    """
    # Compute scores
    scores = compute_method_scores_with_temp(base_method, test_features, val_features, temperature)

    # Determine routing direction (higher score = route or lower score = route)
    # Auto-detect based on score distribution vs errors
    fail_mask = test_features['abs_errors'] > 5.0
    ascending = scores[fail_mask].mean() > scores[~fail_mask].mean()

    # Apply threshold to get routing decisions
    if ascending:
        route_decisions = scores > threshold
    else:
        route_decisions = scores < threshold

    # Routing statistics
    n_routed = route_decisions.sum()
    routing_rate = n_routed / len(route_decisions) * 100

    # Ground truth: should route if error > 5°
    should_route = test_features['abs_errors'] > 5.0

    # Precision, Recall, F1
    precision = precision_score(should_route, route_decisions, zero_division=0)
    recall = recall_score(should_route, route_decisions, zero_division=0)
    f1 = f1_score(should_route, route_decisions, zero_division=0)

    # False positive rate
    n_success = (~should_route).sum()
    fp = (route_decisions & ~should_route).sum()
    fp_rate = fp / n_success if n_success > 0 else 0

    # Compute hybrid predictions
    crnn_preds = test_features['predicted_angles']
    srp_preds_array = srp_results['srp_pred'].values

    # Hybrid: use SRP where routed, CRNN otherwise
    hybrid_preds = np.where(route_decisions, srp_preds_array, crnn_preds)

    # Compute hybrid metrics
    ground_truth = test_features['gt_angles']
    hybrid_errors = np.abs(hybrid_preds - ground_truth)

    # Handle wraparound at 360/0 degrees
    hybrid_errors = np.minimum(hybrid_errors, 360 - hybrid_errors)

    hybrid_mae = hybrid_errors.mean()
    hybrid_median = np.median(hybrid_errors)
    hybrid_success = (hybrid_errors <= 5.0).sum() / len(hybrid_errors) * 100

    # Baseline CRNN metrics
    crnn_mae = test_features['abs_errors'].mean()
    crnn_success = (test_features['abs_errors'] <= 5.0).sum() / len(test_features['abs_errors']) * 100

    # Improvements
    delta_mae = hybrid_mae - crnn_mae
    delta_success = hybrid_success - crnn_success

    return {
        'method': method_display_name,
        'base_method': base_method,
        'temperature': temperature,
        'threshold': threshold,
        'routing_rate': routing_rate,
        'n_routed': int(n_routed),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp_rate,
        'hybrid_mae': hybrid_mae,
        'hybrid_median': hybrid_median,
        'hybrid_success': hybrid_success,
        'delta_mae': delta_mae,
        'delta_success': delta_success,
        'crnn_mae': crnn_mae,
        'crnn_success': crnn_success
    }


def main():
    """Main execution."""
    print("="*100)
    print("EVALUATING ALL METHODS WITH OPTIMAL F1 THRESHOLDS")
    print("="*100)

    # Load data
    print("\nLoading data...")
    test_features = load_test_features()
    srp_results = load_cached_srp_results()

    # Load validation features (for training methods that need it)
    val_features_path = Path("features/train_6cm_features.npz")
    val_data = np.load(val_features_path, allow_pickle=True)
    val_features = {key: val_data[key] for key in val_data.files}
    print(f"Loaded validation features: {len(val_features['abs_errors'])} samples")

    # Load optimal thresholds
    optimal_thresholds = load_optimal_thresholds()

    # Define all methods to evaluate
    method_configs = [
        # Original methods
        ('energy', 'energy', None),
        ('vim', 'vim', None),
        ('she', 'she', None),
        ('gradnorm', 'gradnorm', None),
        ('max_prob', 'max_prob', None),
        ('knn_k5', 'knn_k5', None),
        ('knn_k10', 'knn_k10', None),
        ('knn_k20', 'knn_k20', None),
        ('mc_dropout_entropy', 'mc_dropout_entropy', None),
        ('mc_dropout_variance', 'mc_dropout_variance', None),
        ('mahalanobis', 'mahalanobis', None),
        ('dice_80', 'dice_80', None),
        ('dice_90', 'dice_90', None),
        ('llr_gmm5', 'llr_gmm5', None),

        # Temperature-scaled methods
        ('energy_T1.00', 'energy', 1.0),
        ('max_prob_T4.00', 'max_prob', 4.0),
        ('mc_dropout_entropy_T2.00', 'mc_dropout_entropy', 2.0),
        ('mc_dropout_variance_T3.00', 'mc_dropout_variance', 3.0),
        ('vim_T0.50', 'vim', 0.5),
    ]

    results = []

    for i, (method_key, base_method, temperature) in enumerate(method_configs, 1):
        print(f"\n{'='*100}")
        print(f"[{i}/{len(method_configs)}] Evaluating: {method_key}")
        print(f"{'='*100}")

        if method_key not in optimal_thresholds:
            print(f"  ⚠️  No optimal threshold found for {method_key}, skipping")
            continue

        threshold_info = optimal_thresholds[method_key]
        threshold = threshold_info['threshold']
        expected_f1 = threshold_info['f1']

        print(f"  Optimal threshold: {threshold:.6f}")
        print(f"  Expected F1: {expected_f1:.4f}")

        try:
            result = evaluate_hybrid_with_threshold(
                method_key, base_method, temperature, threshold,
                test_features, val_features, srp_results
            )

            print(f"\n  Results:")
            print(f"    Routing: {result['routing_rate']:.1f}% ({result['n_routed']} samples)")
            print(f"    Precision: {result['precision']:.3f}")
            print(f"    Recall: {result['recall']:.3f}")
            print(f"    F1: {result['f1']:.3f}")
            print(f"    Hybrid MAE: {result['hybrid_mae']:.2f}° (Δ {result['delta_mae']:.2f}°)")
            print(f"    Success: {result['hybrid_success']:.1f}% (Δ {result['delta_success']:.1f}%)")

            results.append(result)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}")

    df = pd.DataFrame(results)

    # Sort by hybrid MAE
    df = df.sort_values('hybrid_mae')

    # Save CSV
    output_dir = Path('results/optimal_thresholds')
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'all_methods_optimal_thresholds.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}")

    # Create routing statistics table
    routing_df = df[['method', 'precision', 'recall', 'f1', 'routing_rate', 'n_routed']].copy()
    routing_df = routing_df.sort_values('f1', ascending=False)
    routing_csv = output_dir / 'routing_statistics.csv'
    routing_df.to_csv(routing_csv, index=False)
    print(f"✅ Saved: {routing_csv}")

    # Create hybrid results table
    hybrid_df = df[['method', 'routing_rate', 'f1', 'hybrid_mae', 'hybrid_median', 'hybrid_success', 'delta_mae']].copy()
    hybrid_df = hybrid_df.sort_values('hybrid_mae')
    hybrid_csv = output_dir / 'hybrid_results.csv'
    hybrid_df.to_csv(hybrid_csv, index=False)
    print(f"✅ Saved: {hybrid_csv}")

    print(f"\n{'='*100}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*100}")
    print(f"Total methods evaluated: {len(results)}/{len(method_configs)}")
    print(f"Results directory: {output_dir}")


if __name__ == '__main__':
    main()
