"""
Analyze OOD Method Scores vs. CRNN Prediction Error

This script investigates the correlation between the OOD scores produced by
various methods (GradNorm, KNN, Mahalanobis, DICE, ConfidNet) and the actual
absolute error of the CRNN model's predictions.
"""
import os
import torch
import numpy as np
import pickle
from pathlib import Path # Import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Import all necessary routers
from gradnorm_ood_routing import GradNormOODRouter
from knn_ood_routing import KNNOODRouter
from mahalanobis_ood_routing import MahalanobisOODRouter
from dice_ood_routing import DICEOODRouter
from confidnet_routing import ConfidNetRouter
from evidential_routing import EvidentialRouter

# --- Configuration ---
CONFIGS_3MIC = [
    '9_10_11_4_5_6_7_8_0', '1_10_11_12_5_6_7_8_0', '1_2_11_12_13_6_7_8_0',
    '9_2_3_12_5_6_15_8_0', '1_10_3_4_13_6_7_16_0', '9_2_11_4_5_14_7_8_0',
    '1_10_3_12_5_6_15_8_0', '9_2_3_4_13_6_15_8_0', '1_10_11_4_5_6_15_8_0',
    '9_10_3_12_5_6_7_16_0'
]
# METHODS_TO_ANALYZE = {
#     'GradNorm': GradNormOODRouter,
#     'KNN (k=10)': lambda: KNNOODRouter(k=10),
#     'Mahalanobis': MahalanobisOODRouter,
#     'DICE (80%)': lambda: DICEOODRouter(clip_percentile=80),
#     'ConfidNet': ConfidNetRouter,
#     'Evidential': lambda: EvidentialRouter(MODEL_SAVE_PATH)
# }

METHODS_TO_ANALYZE = {
    'Evidential': lambda: EvidentialRouter(MODEL_SAVE_PATH)
}
SCRIPT_DIR = Path(__file__).parent
FEATURES_DIR = SCRIPT_DIR.parent.parent / 'crnn features'
VAL_FEATURES_PATH = Path(r'C:\daniel\Thesis\hybrid_system\advanced_failure_detection\srp_features_end_result\train_combined_features.npz')
MODEL_SAVE_PATH = SCRIPT_DIR / 'evidential_head.pth'
RESULTS_DIR = SCRIPT_DIR / 'results' / 'ood_analysis'

def load_features_from_config(config_name):
    """Loads features for a single microphone configuration."""
    features_path = FEATURES_DIR / f'crnn_results_mics_{config_name}.pkl'
    if not features_path.exists():
        print(f"Warning: Feature file not found for config {config_name}. Skipping.")
        return None
    with open(features_path, 'rb') as f:
        crnn_results = pickle.load(f)
    
    features = {
        'abs_errors': np.array([d['crnn_error'] for d in crnn_results]),
        'penultimate_features': np.array([d['penultimate_features'] for d in crnn_results], dtype=object),
        'logits_pre_sig': np.array([d['logits_pre_sig'] for d in crnn_results], dtype=object)
    }
    return features

def get_scores(router, features):
    """A helper to call the correct score computation method for each router."""
    if isinstance(router, GradNormOODRouter):
        return router.compute_gradnorm_scores(features)
    elif isinstance(router, KNNOODRouter):
        return router.compute_knn_distances(features)
    elif isinstance(router, MahalanobisOODRouter):
        return router.compute_mahalanobis_distances(features)
    elif isinstance(router, DICEOODRouter):
        dice_scores_per_frame = router.compute_dice_scores(features)
        return np.array([s.mean() for s in dice_scores_per_frame])
    elif isinstance(router, ConfidNetRouter):
        return router.compute_scores(features)
    elif isinstance(router, EvidentialRouter):
        return router.compute_uncertainty(features)
    else:
        raise TypeError(f"Unsupported router type: {type(router)}")

def main():
    """Main execution function."""
    print("="*80)
    print("STARTING OOD METHOD VS. ERROR ANALYSIS")
    print("="*80)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Validation Data (for training OOD methods)
    print(f"Loading validation (in-distribution) data from: {VAL_FEATURES_PATH}")
    if not VAL_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Validation data not found at {VAL_FEATURES_PATH}")
    val_data = np.load(VAL_FEATURES_PATH, allow_pickle=True)
    val_features = {key: val_data[key] for key in val_data.files}
    
    # 2. Load and Combine Test Data from all 10 configurations
    print("Loading features for all 10 '3-mics-changed' configurations...")
    all_penultimate_features = []
    all_logits = []
    all_abs_errors = []

    for config in CONFIGS_3MIC:
        data = load_features_from_config(config)
        if data:
            all_penultimate_features.extend(data['penultimate_features'])
            all_logits.extend(data['logits_pre_sig'])
            all_abs_errors.extend(data['abs_errors'])
    
    if not all_abs_errors:
        raise ValueError("No feature data could be loaded. Aborting.")

    test_features = {
        'penultimate_features': np.array(all_penultimate_features, dtype=object),
        'logits_pre_sig': np.array(all_logits, dtype=object),
        'abs_errors': np.array(all_abs_errors)
    }
    crnn_errors = test_features['abs_errors']
    print(f"Total samples loaded: {len(crnn_errors)}")

    # 3. Iterate through methods, train, score, and plot
    for method_name, router_class in METHODS_TO_ANALYZE.items():
        print(f"\n--- Analyzing {method_name} ---")

        # Initialize and train router
        router = router_class()
        if hasattr(router, 'train'):
            print("Training router...")
            router.train(val_features)
        
        # Compute scores
        print("Computing scores...")
        ood_scores = get_scores(router, test_features)

        # Analysis and Plotting
        correlation, p_value = pearsonr(crnn_errors, ood_scores)
        print(f"Pearson Correlation for {method_name}: {correlation:.4f} (p-value: {p_value:.4g})")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        correct_mask = crnn_errors <= 5.0
        
        sns.scatterplot(x=crnn_errors[correct_mask], y=ood_scores[correct_mask], ax=ax, alpha=0.4, label='Correct (Error <= 5°)', color='green', s=15)
        sns.scatterplot(x=crnn_errors[~correct_mask], y=ood_scores[~correct_mask], ax=ax, alpha=0.4, label='Incorrect (Error > 5°)', color='red', s=15)
        
        ax.set_title(f'{method_name} Score vs. CRNN Absolute Error', fontsize=16)
        ax.set_xlabel('CRNN Absolute Error (°)', fontsize=12)
        ax.set_ylabel(f'{method_name} Score (Higher = More OOD)', fontsize=12)
        ax.axvline(x=5.0, color='blue', linestyle='--', label='Error Threshold (5°)')
        ax.legend()
        
        ax.text(0.95, 0.95, f'Pearson Correlation: {correlation:.3f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        fig.tight_layout()
        plot_path = RESULTS_DIR / f'{method_name.replace(" ", "_").replace("(", "").replace(")", "")}_vs_error.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Analysis plot saved to: {plot_path}")
        plt.close(fig)

    print("\n✅ All analyses complete!")

if __name__ == '__main__':
    main()
