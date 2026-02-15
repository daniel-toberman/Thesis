"""
Cascade (Joint-Criteria) Hybrid OOD Routing Analysis

This script evaluates a cascade routing strategy where a sample is routed to the
SRP system only if it meets two conditions simultaneously:
1.  Its CRNN confidence is LOW (e.g., GradNorm score > threshold).
2.  Its SRP confidence is HIGH (e.g., SRP Entropy score < threshold).

The script performs a 2D grid search over the thresholds for both metrics
and visualizes the resulting hybrid performance (MAE, success rate) as a heatmap.
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

# Import necessary routers and helper functions
from gradnorm_ood_routing import GradNormOODRouter

# --- Configuration ---
OUTPUT_DIR = Path(__file__).parent / 'results' / 'cascade_routing'
VAL_FEATURES_PATH = Path(r'C:/daniel/Thesis/hybrid_system/advanced_failure_detection/srp_features_end_result/train_combined_features.npz')
SRP_BASE_DIR = Path('srp_features_end_result')
RAW_SRP_BASE_DIR = Path('srp_features_raw')
CRNN_BASE_DIR = '../../crnn features'

# Define the 2D search space (percentiles)
GRADNORM_THRESHOLDS_PERCENTILES = np.arange(0, 101, 5) # 0 to 100 in steps of 5
SRP_ENTROPY_THRESHOLDS_PERCENTILES = np.arange(0, 101, 5) # 0 to 100 in steps of 5

# --- Helper Functions (copied/adapted from previous scripts) ---
def load_crnn_features(crnn_path):
    # (implementation is identical to evaluate_two_step_routing.py)
    if not os.path.exists(crnn_path): return None
    with open(crnn_path, 'rb') as f: data = pickle.load(f)
    return {
        'abs_errors': np.array([d['crnn_error'] for d in data]),
        'predicted_angles': np.array([d['crnn_pred'] for d in data]),
        'gt_angles': np.array([d['gt_angle'] for d in data]),
        'logits_pre_sig': np.array([d['logits_pre_sig'] for d in data], dtype=object),
        'penultimate_features': np.array([d['penultimate_features'] for d in data], dtype=object),
    }

def load_cached_srp_results(srp_path):
    with open(srp_path, 'rb') as f: return pd.DataFrame(pickle.load(f))

def load_srp_raw_features(raw_srp_path):
    if not os.path.exists(raw_srp_path): return None
    with open(raw_srp_path, 'rb') as f: return pickle.load(f)

def calculate_srp_entropy_scores(raw_srp_features):
    scores = []
    for sample in raw_srp_features:
        srp_map = sample.get('srp_map')
        if srp_map is None or len(srp_map) == 0: scores.append(np.nan); continue
        total_power = np.sum(srp_map)
        normalized_map = (np.ones_like(srp_map) / len(srp_map)) if total_power == 0 else (srp_map / total_power)
        scores.append(entropy(normalized_map + 1e-10))
    return np.array(scores)

def get_gradnorm_scores(features, val_features):
    router = GradNormOODRouter(); router.train(val_features)
    return router.compute_gradnorm_scores(features)

def evaluate_performance(route_decisions, test_features, srp_results):
    srp_preds = srp_results['srp_pred'].values
    crnn_preds = test_features['predicted_angles']
    gt_angles = test_features['gt_angles']
    hybrid_preds = np.where(route_decisions, srp_preds, crnn_preds)
    hybrid_errors = np.minimum(np.abs(hybrid_preds - gt_angles), 360 - np.abs(hybrid_preds - gt_angles))
    valid_errors = hybrid_errors[~np.isnan(hybrid_errors)]
    hybrid_mae = np.mean(valid_errors) if len(valid_errors) > 0 else np.nan
    hybrid_success = np.mean(valid_errors <= 5.0) * 100 if len(valid_errors) > 0 else np.nan
    routing_rate = np.mean(route_decisions) * 100
    return routing_rate, hybrid_mae, hybrid_success

def count_changed_mics(mic_config_str):
    return sum(1 for m in mic_config_str.split('_') if m.isdigit() and int(m) >= 9)

# --- Main Execution ---
def main():
    print("="*80)
    print("STARTING CASCADE (JOINT-CRITERIA) ROUTING ANALYSIS")
    print(f"Step 1: CRNN GradNorm | Step 2: SRP Entropy")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading validation features from: {VAL_FEATURES_PATH}")
    if VAL_FEATURES_PATH.suffix == '.pkl': val_features = load_crnn_features(str(VAL_FEATURES_PATH))
    else: val_data = np.load(VAL_FEATURES_PATH, allow_pickle=True); val_features = {k: val_data[k] for k in val_data.files}
    if val_features is None: print("Could not load validation features. Exiting."); return

    srp_mic_config_files = glob.glob(str(SRP_BASE_DIR / 'srp_results_mics_*.pkl'))
    num_mics_to_process = 3
    print(f"Filtering for configurations with exactly {num_mics_to_process} changed mics...")
    filtered_srp_files = [p for p in srp_mic_config_files if count_changed_mics(Path(p).stem.replace('srp_results_mics_', '')) == num_mics_to_process]
    print(f"Found {len(filtered_srp_files)} configurations to process.")

    all_results = []
    
    for srp_path in filtered_srp_files:
        mic_config_name = Path(srp_path).stem.replace('srp_results_mics_', '')
        print(f"Processing mic config: {mic_config_name}")

        crnn_path = f"{CRNN_BASE_DIR}/crnn_results_mics_{mic_config_name}.pkl"
        raw_srp_path = RAW_SRP_BASE_DIR / Path(srp_path).name
        
        test_features = load_crnn_features(crnn_path)
        srp_results = load_cached_srp_results(srp_path)
        raw_srp_features = load_srp_raw_features(raw_srp_path)
        
        if test_features is None or srp_results is None or raw_srp_features is None: continue

        gradnorm_scores = get_gradnorm_scores(test_features, val_features)
        srp_entropy_scores = calculate_srp_entropy_scores(raw_srp_features)
        
        srp_reliable_mask = ~np.isnan(srp_results['srp_pred'].values)
        
        # --- 2D Grid Search ---
        for gn_perc in GRADNORM_THRESHOLDS_PERCENTILES:
            # Higher percentile for GradNorm means routing MORE uncertain samples
            gradnorm_threshold = np.percentile(gradnorm_scores, gn_perc)
            
            for srp_perc in SRP_ENTROPY_THRESHOLDS_PERCENTILES:
                # Lower percentile for SRP Entropy means routing MORE confident samples
                srp_entropy_threshold = np.percentile(srp_entropy_scores[~np.isnan(srp_entropy_scores)], srp_perc)
                
                # Cascade Logic: Route if GradNorm is high AND SRP Entropy is low AND SRP is reliable
                route_decisions = (
                    (gradnorm_scores > gradnorm_threshold) &
                    (srp_entropy_scores < srp_entropy_threshold) &
                    srp_reliable_mask
                )
                
                routing_rate, hybrid_mae, hybrid_success = evaluate_performance(
                    route_decisions, test_features, srp_results
                )
                
                all_results.append({
                    'mic_config': mic_config_name,
                    'num_changed_mics': count_changed_mics(mic_config_name),
                    'gradnorm_percentile': gn_perc,
                    'srp_entropy_percentile': srp_perc,
                    'routing_rate': routing_rate,
                    'hybrid_mae': hybrid_mae,
                    'hybrid_success': hybrid_success,
                })

    results_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / 'cascade_routing_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Full results saved to: {csv_path}")

    # --- Plotting Heatmaps ---
    print("Generating heatmaps...")
    
    # Aggregate results for plotting (mean over mic configs)
    heatmap_mae = results_df.pivot_table(
        index='gradnorm_percentile', columns='srp_entropy_percentile', values='hybrid_mae', aggfunc='mean'
    )
    heatmap_success = results_df.pivot_table(
        index='gradnorm_percentile', columns='srp_entropy_percentile', values='hybrid_success', aggfunc='mean'
    )
    heatmap_routing = results_df.pivot_table(
        index='gradnorm_percentile', columns='srp_entropy_percentile', values='routing_rate', aggfunc='mean'
    )
    
    # Plot MAE Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_mae, annot=True, fmt=".2f", cmap="viridis_r", cbar_kws={'label': 'Mean Absolute Error (°)'})
    plt.title('Cascade Routing: Hybrid MAE', fontsize=18)
    plt.xlabel('SRP Entropy Percentile (< threshold routes)', fontsize=12)
    plt.ylabel('GradNorm Percentile (> threshold routes)', fontsize=12)
    plt.savefig(OUTPUT_DIR / 'heatmap_cascade_mae.png', dpi=300)
    plt.close()

    # Plot Success Rate Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_success, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Success Rate (%)'})
    plt.title('Cascade Routing: Hybrid Success Rate', fontsize=18)
    plt.xlabel('SRP Entropy Percentile (< threshold routes)', fontsize=12)
    plt.ylabel('GradNorm Percentile (> threshold routes)', fontsize=12)
    plt.savefig(OUTPUT_DIR / 'heatmap_cascade_success.png', dpi=300)
    plt.close()
    
    # Plot Routing Rate Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_routing, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Routing Rate (%)'})
    plt.title('Cascade Routing: Total Routing Rate', fontsize=18)
    plt.xlabel('SRP Entropy Percentile (< threshold routes)', fontsize=12)
    plt.ylabel('GradNorm Percentile (> threshold routes)', fontsize=12)
    plt.savefig(OUTPUT_DIR / 'heatmap_cascade_routing_rate.png', dpi=300)
    plt.close()

    print(f"✅ ANALYSIS COMPLETE!")

if __name__ == '__main__':
    main()
