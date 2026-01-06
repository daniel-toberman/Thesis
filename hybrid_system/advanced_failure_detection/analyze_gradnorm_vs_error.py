"""
Analyze GradNorm Scores vs. CRNN Prediction Error

This script investigates the correlation between the OOD scores produced by
the GradNorm method and the actual absolute error of the CRNN model's predictions.

The primary goal is to understand why GradNorm's performance degrades at high
routing rates. The hypothesis is that GradNorm assigns high scores to some
"benign outliers" which the CRNN actually predicts correctly.

The script performs the following steps:
1.  Loads the features for all 10 "3-mics-changed" configurations.
2.  Trains a GradNorm router on the in-distribution (6cm) data.
3.  Computes GradNorm scores for all samples from the 10 configurations.
4.  Generates a scatter plot of CRNN Error vs. GradNorm Score.
5.  Calculates and prints the Pearson correlation coefficient.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from gradnorm_ood_routing import GradNormOODRouter
from confidnet_model import ConfidNet, ConfidNetLoss # Needed for feature loading helpers if any

# --- Configuration ---
CONFIGS_3MIC = [
    '9_10_11_4_5_6_7_8_0', '1_10_11_12_5_6_7_8_0', '1_2_11_12_13_6_7_8_0',
    '9_2_3_12_5_6_15_8_0', '1_10_3_4_13_6_7_16_0', '9_2_11_4_5_14_7_8_0',
    '1_10_3_12_5_6_15_8_0', '9_2_3_4_13_6_15_8_0', '1_10_11_4_5_6_15_8_0',
    '9_10_3_12_5_6_7_16_0'
]
SCRIPT_DIR = Path(__file__).parent
FEATURES_DIR = SCRIPT_DIR.parent.parent / 'crnn features'
VAL_FEATURES_PATH = FEATURES_DIR / "test_6cm_features.npz"
OUTPUT_PLOT_PATH = SCRIPT_DIR / 'results' / 'gradnorm_vs_error_analysis.png'

def load_features_from_config(config_name):
    """Loads features for a single microphone configuration."""
    features_path = FEATURES_DIR / f'crnn_results_mics_{config_name}.pkl'
    if not features_path.exists():
        print(f"Warning: Feature file not found for config {config_name}. Skipping.")
        return None
    with open(features_path, 'rb') as f:
        crnn_results = pickle.load(f)
    
    # Ensure all features are numpy arrays before returning
    features = {
        'abs_errors': np.array([d['crnn_error'] for d in crnn_results]),
        'penultimate_features': np.array([d['penultimate_features'] for d in crnn_results], dtype=object),
        'logits_pre_sig': np.array([d['logits_pre_sig'] for d in crnn_results], dtype=object)
    }
    return features

def main():
    """Main execution function."""
    print("="*80)
    print("STARTING GRADNORM VS. ERROR ANALYSIS")
    print("="*80)

    # 1. Load Validation Data (for training the OOD method)
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
    print(f"Total samples loaded: {len(test_features['abs_errors'])}")

    # 3. Train GradNorm Router and Compute Scores
    print("\nTraining GradNorm router...")
    gradnorm_router = GradNormOODRouter()
    gradnorm_router.train(val_features)
    
    print("Computing GradNorm scores for test data...")
    gradnorm_scores = gradnorm_router.compute_gradnorm_scores(test_features)

    # 4. Analyze and Plot
    print("Generating scatter plot...")
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    crnn_errors = test_features['abs_errors']
    
    # Calculate Pearson correlation
    correlation, p_value = pearsonr(crnn_errors, gradnorm_scores)
    print(f"\nPearson Correlation between CRNN Error and GradNorm Score: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separate correct and incorrect predictions for coloring
    correct_mask = crnn_errors <= 5.0
    
    sns.scatterplot(x=crnn_errors[correct_mask], y=gradnorm_scores[correct_mask], ax=ax, alpha=0.5, label='Correct (Error <= 5°)', color='green')
    sns.scatterplot(x=crnn_errors[~correct_mask], y=gradnorm_scores[~correct_mask], ax=ax, alpha=0.5, label='Incorrect (Error > 5°)', color='red')
    
    ax.set_title('GradNorm Score vs. CRNN Absolute Error (All 10 "3-Mics-Changed" Configs)', fontsize=16)
    ax.set_xlabel('CRNN Absolute Error (°)', fontsize=12)
    ax.set_ylabel('GradNorm Score (Higher = More OOD)', fontsize=12)
    ax.axvline(x=5.0, color='blue', linestyle='--', label='Error Threshold (5°)')
    ax.legend()
    
    # Add annotation for correlation
    ax.text(0.95, 0.95, f'Pearson Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    fig.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    print(f"\n✅ Analysis plot saved to: {OUTPUT_PLOT_PATH}")

if __name__ == '__main__':
    main()
