"""
Two-Step Hybrid OOD Routing Analysis

This script implements and evaluates a two-step routing strategy:
1.  **Step 1 (CRNN GradNorm)**: Route a predefined percentage of the most uncertain
    samples (highest GradNorm score) to the SRP system.
2.  **Step 2 (SRP Entropy)**: From the *remaining* samples (those not routed
    by GradNorm), route additional samples to SRP based on their SRP Entropy
    score, exploring a range of thresholds for this second step.

The script evaluates the overall hybrid system performance (MAE, success rate,
total routing rate) for various combinations of these two steps and visualizes
the results in heatmaps.
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
REVERSE_ORDER = True # If True, routes by SRP Entropy first, then GradNorm.
OUTPUT_DIR = Path(__file__).parent / 'results' / 'two_step_routing'
VAL_FEATURES_PATH = Path(r'C:/daniel/Thesis/hybrid_system/advanced_failure_detection/srp_features_end_result/train_combined_features.npz')
SRP_BASE_DIR = Path('srp_features_end_result')
RAW_SRP_BASE_DIR = Path('srp_features_raw')
CRNN_BASE_DIR = '../../crnn features'

# GradNorm routing rates for the FIRST step (as percentages of total samples)
GRADNORM_ROUTING_RATES = np.linspace(0.0, 0.90, 10) # 10% to 50% in 5% steps

# SRP Entropy thresholds for the SECOND step (as percentiles of the *remaining* unrouted data)
SRP_ENTROPY_THRESHOLDS_PERCENTILES = np.linspace(1, 99, 20) # 20 steps from 1st to 99th percentile

# --- Helper Functions (copied/adapted from evaluate_threshold_sensitivity.py) ---
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
    # print(f"  Loaded CRNN features from {Path(crnn_path).name}: {len(features['abs_errors'])} samples")
    return features

def load_cached_srp_results(srp_path):
    """Load cached SRP predictions from a pickle file."""
    with open(srp_path, 'rb') as f: srp_results_list = pickle.load(f)
    srp_results_df = pd.DataFrame(srp_results_list)
    # print(f"  Loaded cached SRP results from {Path(srp_path).name}: {len(srp_results_df)} samples")
    return srp_results_df

def load_srp_raw_features(raw_srp_path):
    """Load raw SRP features (including srp_map) from a pickle file."""
    if not os.path.exists(raw_srp_path):
        print(f"  ⚠️ Raw SRP features file not found: {raw_srp_path}")
        return None
    with open(raw_srp_path, 'rb') as f: raw_features_list = pickle.load(f)
    # print(f"  Loaded raw SRP features from {Path(raw_srp_path).name}: {len(raw_features_list)} samples")
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

def evaluate_performance(route_decisions, test_features, srp_results):
    """Calculates MAE, success rate, and routing rate for given routing decisions."""
    # Ensure srp_pred is aligned and has same length as test_features
    # Assuming srp_results and test_features are already aligned by sample index
    srp_preds = srp_results['srp_pred'].values
    crnn_preds = test_features['predicted_angles']
    gt_angles = test_features['gt_angles']

    # Create hybrid predictions
    # If route_decisions is True, use SRP. If False, use CRNN.
    # If srp_preds is NaN for a routed sample, the hybrid_pred for that sample will be NaN.
    hybrid_preds = np.where(route_decisions, srp_preds, crnn_preds)
    
    # Calculate errors.
    hybrid_errors = np.minimum(np.abs(hybrid_preds - gt_angles), 360 - np.abs(hybrid_preds - gt_angles))
    
    # Only consider non-NaN errors for MAE and success rate calculation
    valid_errors_mask = ~np.isnan(hybrid_errors)
    valid_errors = hybrid_errors[valid_errors_mask]
    
    if len(valid_errors) == 0:
        hybrid_mae = np.nan
        hybrid_success = np.nan
    else:
        hybrid_mae = np.mean(valid_errors)
        hybrid_success = np.mean(valid_errors <= 5.0) * 100
    
    routing_rate = np.mean(route_decisions) * 100
    
    return routing_rate, hybrid_mae, hybrid_success

def count_changed_mics(mic_config_str):
    """Count the number of mics with a label >= 9."""
    mics = mic_config_str.split('_')
    return sum(1 for m_str in mics if m_str.isdigit() and int(m_str) >= 9)

# --- Main Execution ---
def main():
    print("="*80)
    print("STARTING TWO-STEP HYBRID OOD ROUTING ANALYSIS")
    print(f"Step 1: CRNN GradNorm | Step 2: SRP Entropy")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading validation features from: {VAL_FEATURES_PATH}")
    # Load validation features using the appropriate loading mechanism (pkl or npz)
    if VAL_FEATURES_PATH.suffix == '.pkl':
        val_features = load_crnn_features(str(VAL_FEATURES_PATH))
    else: # Assume .npz
        val_data = np.load(VAL_FEATURES_PATH, allow_pickle=True)
        val_features = {key: val_data[key] for key in val_data.files}
    if val_features is None:
        print("Could not load validation features. Exiting.")
        return

    srp_mic_config_files = glob.glob(str(SRP_BASE_DIR / 'srp_results_mics_*.pkl'))
    
    # Optional: Filter for a specific number of changed mics if desired (e.g., 3)
    num_mics_to_process = 3
    print(f"""
Filtering for configurations with exactly {num_mics_to_process} changed mics...""")
    filtered_srp_files = []
    for srp_path in srp_mic_config_files:
        mic_config_name = Path(srp_path).stem.replace('srp_results_mics_', '')
        if count_changed_mics(mic_config_name) == num_mics_to_process:
            filtered_srp_files.append(srp_path)
    srp_mic_config_files = filtered_srp_files
    print(f"Found {len(srp_mic_config_files)} configurations to process.")

    all_results = []
    
    for srp_path in srp_mic_config_files:
        mic_config_name = Path(srp_path).stem.replace('srp_results_mics_', '')
        print(f"""
Processing mic config: {mic_config_name}""")

        crnn_path = f"{CRNN_BASE_DIR}/crnn_results_mics_{mic_config_name}.pkl"
        raw_srp_path = RAW_SRP_BASE_DIR / Path(srp_path).name
        
        test_features = load_crnn_features(crnn_path)
        srp_results = load_cached_srp_results(srp_path)
        raw_srp_features = load_srp_raw_features(raw_srp_path)
        
        if test_features is None or srp_results is None or raw_srp_features is None:
            print("  Skipping config due to missing data.")
            continue

        # --- Calculate base scores ---
        gradnorm_scores = get_gradnorm_scores(test_features, val_features)
        srp_entropy_scores = calculate_srp_entropy_scores(raw_srp_features)
        
        num_samples = len(gradnorm_scores)
        if num_samples == 0:
            print("  No samples to process for this config.")
            continue

        # Mask of samples where SRP is reliable (not NaN)
        srp_reliable_mask = ~np.isnan(srp_results['srp_pred'].values)
        
    # --- Main logic block ---
    if REVERSE_ORDER:
        print("Executing in REVERSED order: Step 1: SRP Entropy, Step 2: GradNorm")
        # --- Step 1: SRP Entropy Routing ---
        for srp_routing_rate_target in GRADNORM_ROUTING_RATES: # Reusing the same rate percentages
            # SRP Entropy: lower score is better (more confident), so we route the lowest scores.
            # We want the bottom X% of scores, so we take the X-th percentile as the threshold.
            srp_entropy_threshold = np.percentile(srp_entropy_scores[srp_reliable_mask], srp_routing_rate_target * 100)
            
            # Initial SRP routing decision (if score < threshold)
            srp_route_initial = (srp_entropy_scores < srp_entropy_threshold)
            srp_route_decisions = srp_route_initial & srp_reliable_mask

            # Samples NOT routed by SRP, that are still available for GradNorm routing
            remaining_mask = ~srp_route_decisions & srp_reliable_mask

            # --- Step 2: GradNorm Routing for Remaining Samples ---
            gradnorm_scores_remaining = gradnorm_scores[remaining_mask]
            
            if len(gradnorm_scores_remaining) == 0:
                # Evaluate with only the SRP step if no samples remain
                total_routing_decisions = srp_route_decisions
                total_routing_rate, hybrid_mae, hybrid_success = evaluate_performance(
                    total_routing_decisions, test_features, srp_results
                )
                all_results.append({
                    'mic_config': mic_config_name, 'num_changed_mics': count_changed_mics(mic_config_name),
                    'srp_entropy_routing_rate_target': srp_routing_rate_target,
                    'gradnorm_threshold_percentile': 0,
                    'total_routing_rate': total_routing_rate, 'hybrid_mae': hybrid_mae, 'hybrid_success': hybrid_success,
                })
                continue

            for gradnorm_threshold_percentile in SRP_ENTROPY_THRESHOLDS_PERCENTILES: # Reusing percentile steps
                # GradNorm: higher score is more uncertain. We route the top X% of *remaining* samples.
                gradnorm_threshold = np.percentile(gradnorm_scores_remaining, gradnorm_threshold_percentile)
                gradnorm_route_decisions_step2 = gradnorm_scores[remaining_mask] > gradnorm_threshold
                
                # --- Combine ALL routing decisions ---
                total_routing_decisions = np.copy(srp_route_decisions)
                total_routing_decisions[remaining_mask] = gradnorm_route_decisions_step2
                
                total_routing_rate, hybrid_mae, hybrid_success = evaluate_performance(
                    total_routing_decisions, test_features, srp_results
                )
                
                all_results.append({
                    'mic_config': mic_config_name, 'num_changed_mics': count_changed_mics(mic_config_name),
                    'srp_entropy_routing_rate_target': srp_routing_rate_target,
                    'gradnorm_threshold_percentile': gradnorm_threshold_percentile,
                    'total_routing_rate': total_routing_rate, 'hybrid_mae': hybrid_mae, 'hybrid_success': hybrid_success,
                })
    else: # Original Order: GradNorm -> SRP Entropy
        print("Executing in default order: Step 1: GradNorm, Step 2: SRP Entropy")
        # --- Step 1: GradNorm Routing ---
        for gradnorm_routing_rate_target in GRADNORM_ROUTING_RATES:
            # GradNorm: higher score is more uncertain. We route the top X% of samples.
            gradnorm_threshold = np.percentile(gradnorm_scores, (1 - gradnorm_routing_rate_target) * 100)
            
            gradnorm_route_initial = (gradnorm_scores > gradnorm_threshold)
            gradnorm_route_decisions = gradnorm_route_initial & srp_reliable_mask

            # Samples NOT routed by GradNorm
            remaining_mask = ~gradnorm_route_decisions & srp_reliable_mask

            # --- Step 2: SRP Entropy Routing for Remaining Samples ---
            srp_entropy_scores_remaining = srp_entropy_scores[remaining_mask]
            
            if len(srp_entropy_scores_remaining) == 0:
                total_routing_decisions = gradnorm_route_decisions
                total_routing_rate, hybrid_mae, hybrid_success = evaluate_performance(
                    total_routing_decisions, test_features, srp_results
                )
                all_results.append({
                    'mic_config': mic_config_name, 'num_changed_mics': count_changed_mics(mic_config_name),
                    'gradnorm_routing_rate_target': gradnorm_routing_rate_target,
                    'srp_entropy_threshold_percentile': 0,
                    'total_routing_rate': total_routing_rate, 'hybrid_mae': hybrid_mae, 'hybrid_success': hybrid_success,
                })
                continue

            for srp_entropy_threshold_percentile in SRP_ENTROPY_THRESHOLDS_PERCENTILES:
                # SRP Entropy: lower score is better. We route the bottom X% of *remaining* samples.
                srp_entropy_threshold = np.percentile(srp_entropy_scores_remaining, srp_entropy_threshold_percentile)
                srp_route_decisions_step2 = srp_entropy_scores[remaining_mask] < srp_entropy_threshold
                
                # --- Combine ALL routing decisions ---
                total_routing_decisions = np.copy(gradnorm_route_decisions)
                total_routing_decisions[remaining_mask] = srp_route_decisions_step2
                
                total_routing_rate, hybrid_mae, hybrid_success = evaluate_performance(
                    total_routing_decisions, test_features, srp_results
                )
                
                all_results.append({
                    'mic_config': mic_config_name, 'num_changed_mics': count_changed_mics(mic_config_name),
                    'gradnorm_routing_rate_target': gradnorm_routing_rate_target,
                    'srp_entropy_threshold_percentile': srp_entropy_threshold_percentile,
                    'total_routing_rate': total_routing_rate, 'hybrid_mae': hybrid_mae, 'hybrid_success': hybrid_success,
                })

    results_df = pd.DataFrame(all_results)
    order_str = "srp_first" if REVERSE_ORDER else "gradnorm_first"
    csv_path = OUTPUT_DIR / f'two_step_routing_results_{order_str}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"""
✅ Full results saved to: {csv_path}""")

    # --- Plotting Line Graphs ---
    print("\nGenerating line plots...")
    
    # Determine the correct columns for plotting based on the order
    if REVERSE_ORDER:
        primary_rate_col = 'srp_entropy_routing_rate_target'
        secondary_percentile_col = 'gradnorm_threshold_percentile'
        primary_metric_name = "SRP Entropy"
        secondary_metric_name = "GradNorm"
    else:
        primary_rate_col = 'gradnorm_routing_rate_target'
        secondary_percentile_col = 'srp_entropy_threshold_percentile'
        primary_metric_name = "GradNorm"
        secondary_metric_name = "SRP Entropy"

    unique_primary_targets = np.sort(results_df[primary_rate_col].unique())

    for target_rate in unique_primary_targets:
        df_subset = results_df[results_df[primary_rate_col] == target_rate]
        
        plot_data = df_subset.groupby(secondary_percentile_col).agg(
            hybrid_mae=('hybrid_mae', 'mean'),
            hybrid_success=('hybrid_success', 'mean'),
            total_routing_rate=('total_routing_rate', 'mean')
        ).reset_index()

        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        title = f'Two-Step Routing (Initial {primary_metric_name} Routing: {target_rate*100:.0f}%)'
        fig.suptitle(title, fontsize=18)

        # Plot MAE
        sns.lineplot(data=plot_data, x=secondary_percentile_col, y='hybrid_mae', ax=axes[0], marker='o')
        axes[0].set_title(f'Hybrid MAE vs. {secondary_metric_name} Threshold')
        axes[0].set_ylabel('Mean Absolute Error (°)')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Plot Success Rate
        sns.lineplot(data=plot_data, x=secondary_percentile_col, y='hybrid_success', ax=axes[1], marker='o', color='g')
        axes[1].set_title(f'Hybrid Success Rate vs. {secondary_metric_name} Threshold')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # Plot Total Routing Rate
        sns.lineplot(data=plot_data, x=secondary_percentile_col, y='total_routing_rate', ax=axes[2], marker='o', color='r')
        axes[2].set_title(f'Total Routing Rate vs. {secondary_metric_name} Threshold')
        axes[2].set_xlabel(f'{secondary_metric_name} Threshold Percentile (of remaining data)')
        axes[2].set_ylabel('Total Routing Rate (%)')
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_path = OUTPUT_DIR / f'lineplot_performance_{order_str}_initial_{target_rate*100:.0f}pc.png'
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)


    print(f"""
✅ ANALYSIS COMPLETE!""")

if __name__ == '__main__':
    main()
