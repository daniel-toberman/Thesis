#!/usr/bin/env python3
"""
Weighted Combination Analysis for Hybrid OOD Routing

This script evaluates a weighted combination of a CRNN-based confidence metric
and an SRP-based confidence metric to find the optimal balance for routing
decisions in a hybrid system.

The combined score is calculated as:
  Score = (alpha * normalized_CRNN_score) + ((1 - alpha) * normalized_SRP_score)

The script iterates through different microphone configurations, `alpha` values,
and decision thresholds to find the combination that yields the lowest
hybrid Mean Absolute Error (MAE).
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy.stats import entropy
from sklearn.preprocessing import minmax_scale

# Import router and helper functions from the parent directory
from gradnorm_ood_routing import GradNormOODRouter

# --- Configuration ---
CRNN_METRIC_NAME = 'gradnorm'
SRP_METRIC_NAME = 'srp_entropy'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'weighted_combination'
VAL_FEATURES_PATH = Path(r'C:\daniel\Thesis\hybrid_system\advanced_failure_detection\srp_features_end_result\train_combined_features.npz')
SRP_BASE_DIR = Path('srp_features_end_result')
RAW_SRP_BASE_DIR = Path('srp_features_raw')
CRNN_BASE_DIR = '../../crnn features' # Changed to string
# --- End Configuration ---

def load_crnn_features(crnn_path):
    """Load CRNN features and metadata."""
    if not os.path.exists(crnn_path):
        print(f"  ❌ CRNN features file not found: {crnn_path}")
        return None
    if str(crnn_path).endswith('.pkl'):
        with open(crnn_path, 'rb') as f: crnn_results_list = pickle.load(f)
        features = {
            'abs_errors': np.array([d['crnn_error'] for d in crnn_results_list]),
            'predicted_angles': np.array([d['crnn_pred'] for d in crnn_results_list]),
            'gt_angles': np.array([d['gt_angle'] for d in crnn_results_list]),
            'logits_pre_sig': np.array([d['logits_pre_sig'] for d in crnn_results_list], dtype=object),
            'penultimate_features': np.array([d['penultimate_features'] for d in crnn_results_list], dtype=object),
        }
    else:
        raise ValueError(f"Unsupported file type: {crnn_path}")
    print(f"  Loaded CRNN features from {Path(crnn_path).name}: {len(features['abs_errors'])} samples")
    return features

def load_cached_srp_results(srp_path):
    """Load cached SRP predictions from a pickle file."""
    with open(srp_path, 'rb') as f: srp_results_list = pickle.load(f)
    srp_results_df = pd.DataFrame(srp_results_list)
    print(f"  Loaded cached SRP results from {Path(srp_path).name}: {len(srp_results_df)} samples")
    return srp_results_df

def load_srp_raw_features(raw_srp_path):
    """Load raw SRP features (including srp_map) from a pickle file."""
    if not os.path.exists(raw_srp_path):
        print(f"  ⚠️ Raw SRP features file not found: {raw_srp_path}")
        return None
    with open(raw_srp_path, 'rb') as f: raw_features_list = pickle.load(f)
    print(f"  Loaded raw SRP features from {Path(raw_srp_path).name}: {len(raw_features_list)} samples")
    return raw_features_list

def calculate_srp_entropy_scores(raw_srp_features):
    """Calculates entropy from a list of raw SRP feature dicts."""
    scores = []
    for sample in raw_srp_features:
        srp_map = sample['srp_map']
        if srp_map is None or len(srp_map) == 0:
            scores.append(np.nan)
            continue
        total_power = np.sum(srp_map)
        if total_power == 0: normalized_map = np.ones_like(srp_map) / len(srp_map)
        else: normalized_map = srp_map / total_power
        scores.append(entropy(normalized_map + 1e-10))
    return np.array(scores)

def get_gradnorm_scores(features, val_features):
    """Initializes GradNorm router and computes scores."""
    router = GradNormOODRouter()
    router.train(val_features)
    return router.compute_gradnorm_scores(features)

def evaluate_for_threshold(scores, threshold, ascending, test_features, srp_results):
    """Calculates performance metrics for a single threshold."""
    route_decisions = scores > threshold if ascending else scores < threshold
    routing_rate = np.mean(route_decisions) * 100
    
    hybrid_preds = np.where(route_decisions, srp_results['srp_pred'].values, test_features['predicted_angles'])
    gt_angles = test_features['gt_angles']
    hybrid_errors = np.minimum(np.abs(hybrid_preds - gt_angles), 360 - np.abs(hybrid_preds - gt_angles))
    
    hybrid_mae = np.mean(hybrid_errors[~np.isnan(hybrid_preds)])
    hybrid_success = np.mean(hybrid_errors[~np.isnan(hybrid_preds)] <= 5.0) * 100
    
    return routing_rate, hybrid_mae, hybrid_success

def count_changed_mics(mic_config_str):
    """Count the number of mics with a label >= 9."""
    mics = mic_config_str.split('_')
    return sum(1 for m_str in mics if m_str.isdigit() and int(m_str) >= 9)

def main():
    """Main execution function."""
    print("="*80)
    print("STARTING WEIGHTED COMBINATION ANALYSIS")
    print(f"CRNN Metric: {CRNN_METRIC_NAME} | SRP Metric: {SRP_METRIC_NAME}")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading validation features from: {VAL_FEATURES_PATH}")
    val_features = load_crnn_features(str(VAL_FEATURES_PATH).replace('.npz', '.pkl')) # Adapt for pkl format if needed
    if val_features is None:
        val_features = np.load(VAL_FEATURES_PATH, allow_pickle=True) # Fallback for npz
        print("Loaded validation features as NPZ.")

    srp_mic_config_files = glob.glob(str(SRP_BASE_DIR / 'srp_results_mics_*.pkl'))
    
    # --- Filter for specific number of changed mics ---
    num_mics_to_process = 3
    print(f"\nFiltering for configurations with exactly {num_mics_to_process} changed mics...")
    filtered_srp_files = []
    for srp_path in srp_mic_config_files:
        mic_config_name = Path(srp_path).stem.replace('srp_results_mics_', '')
        if count_changed_mics(mic_config_name) == num_mics_to_process:
            filtered_srp_files.append(srp_path)
    print(f"Found {len(filtered_srp_files)} configurations to process out of {len(srp_mic_config_files)} total.")
    # --- End of filter ---

    all_results = []
    
    for srp_path in filtered_srp_files:
        mic_config_name = Path(srp_path).stem.replace('srp_results_mics_', '')
        print(f"\nProcessing mic config: {mic_config_name}")

        crnn_path = f"{CRNN_BASE_DIR}/crnn_results_mics_{mic_config_name}.pkl" # Changed to f-string with string base path
        raw_srp_path = RAW_SRP_BASE_DIR / Path(srp_path).name
        
        test_features = load_crnn_features(crnn_path)
        srp_results = load_cached_srp_results(srp_path)
        raw_srp_features = load_srp_raw_features(raw_srp_path)
        
        if test_features is None or srp_results is None or raw_srp_features is None:
            print("  Skipping config due to missing data.")
            continue

        # --- Calculate Scores ---
        srp_scores = calculate_srp_entropy_scores(raw_srp_features)
        crnn_scores = get_gradnorm_scores(test_features, val_features)

        # --- Normalize Scores ---
        # We need to ensure that for BOTH metrics, a higher score means "more confident" / "less OOD".
        # We check the correlation with error: if positive, higher score = more error, so we invert it.
        
        # SRP Entropy: Higher entropy means more uncertainty/error. Correlation should be positive.
        if np.corrcoef(srp_scores[~np.isnan(srp_scores)], test_features['abs_errors'][~np.isnan(srp_scores)])[0, 1] > 0:
            # Positive correlation with error, so invert score. Higher score will now mean more confident.
            norm_srp_scores = 1 - minmax_scale(srp_scores)
        else:
            norm_srp_scores = minmax_scale(srp_scores)

        # GradNorm: Higher gradnorm means more OOD/error. Correlation should be positive.
        if np.corrcoef(crnn_scores, test_features['abs_errors'])[0, 1] > 0:
            # Positive correlation with error, so invert score.
            norm_crnn_scores = 1 - minmax_scale(crnn_scores)
        else:
            norm_crnn_scores = minmax_scale(crnn_scores)

        num_changed_mics = count_changed_mics(mic_config_name)

        # --- Iterate through alphas and thresholds ---
        for alpha in np.linspace(0, 1, 11): # 11 steps for 0.0, 0.1, ..., 1.0
            
            combined_scores = (alpha * norm_crnn_scores) + ((1 - alpha) * norm_srp_scores)

            # Now that both normalized scores are "higher is better", we can consistently
            # say that for the combined score, "higher is better".
            # Therefore, we always route when the score is LOW (ascending=False).
            ascending = False
            
            # Use percentiles for thresholds for robustness
            thresholds = np.percentile(combined_scores, np.linspace(1, 99, 50))
            
            for threshold in thresholds:
                routing_rate, hybrid_mae, hybrid_success = evaluate_for_threshold(
                    combined_scores, threshold, ascending, test_features, srp_results
                )
                all_results.append({
                    'alpha': alpha,
                    'mic_config': mic_config_name,
                    'num_changed_mics': num_changed_mics,
                    'threshold': threshold,
                    'routing_rate': routing_rate,
                    'hybrid_mae': hybrid_mae,
                    'hybrid_success': hybrid_success,
                })

    results_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / 'weighted_combination_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Full results saved to: {csv_path}")

    # --- Plotting ---
    print("\nGenerating heatmap plot...")
    try:
        # Bin routing rates for smoother plotting
        results_df['routing_bin'] = pd.cut(results_df['routing_rate'], bins=np.arange(0, 101, 5), labels=np.arange(2.5, 100, 5))
        
        # Group by alpha and routing bin, then find the mean MAE
        heatmap_data = results_df.groupby(['alpha', 'routing_bin'])['hybrid_mae'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='alpha', columns='routing_bin', values='hybrid_mae')

        plt.figure(figsize=(20, 10))
        sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="viridis_r", 
                    cbar_kws={'label': 'Mean Absolute Error (°)'})
        plt.title('Hybrid MAE vs. Alpha and Routing Rate', fontsize=16)
        plt.xlabel('Routing Rate (%)')
        plt.ylabel('Alpha (Weight for CRNN GradNorm)')
        plt.gca().invert_yaxis() # Alpha 1.0 (CRNN-only) at top
        
        plot_path = OUTPUT_DIR / 'weighted_combination_heatmap.png'
        plt.savefig(plot_path, dpi=300)
        print(f"✅ Heatmap saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"  Could not generate heatmap: {e}")

    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == '__main__':
    main()
