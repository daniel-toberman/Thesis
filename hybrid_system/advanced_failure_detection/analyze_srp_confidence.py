"""
Analyze SRP Confidence Metrics

This script calculates various confidence metrics from raw SRP maps and
analyzes their correlation with the actual SRP prediction error.

The goal is to identify metrics that can effectively indicate the reliability
of an SRP prediction.
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, entropy
from scipy.signal import find_peaks
import pandas as pd

# --- Configuration ---
SRP_FEATURES_RAW_DIR = Path('hybrid_system/advanced_failure_detection/srp_features_raw')
OUTPUT_DIR = Path('hybrid_system/advanced_failure_detection/results/srp_confidence_analysis')

def count_changed_mics(filename):
    """Parses the number of changed mics (ID >= 9) from the filename."""
    try:
        config_str = filename.stem.split('srp_results_mics_')[1]
        mics = [int(m) for m in config_str.split('_') if m.isdigit()]
        return sum(1 for m in mics if m >= 9)
    except (IndexError, ValueError):
        return 0 # Default if parsing fails

def calculate_srp_metrics(srp_map, srp_pred_angle, grid_positions):
    """
    Calculates various confidence metrics from a single SRP map.

    Args:
        srp_map (np.ndarray): The raw SRP power map (e.g., 360 values).
        srp_pred_angle (float): The predicted angle from SRP.
        grid_positions (np.ndarray): The angles corresponding to the srp_map values.

    Returns:
        dict: A dictionary of calculated confidence metrics.
    """
    metrics = {}
    
    # Normalize srp_map to get a probability distribution
    total_power = np.sum(srp_map)
    if total_power == 0:
        normalized_map = np.ones_like(srp_map) / len(srp_map)
    else:
        normalized_map = srp_map / total_power

    max_prob = np.max(normalized_map)
    peak_idx = np.argmax(normalized_map)
    metrics['max_prob'] = max_prob

    # Peak Width (at 50% of max height)
    peak_threshold = max_prob * 0.5
    above_threshold = normalized_map >= peak_threshold
    
    if np.all(above_threshold):
        metrics['peak_width'] = np.nan # Return NaN for flat maps
    else:
        rolled_above = np.roll(above_threshold, -peak_idx)
        width = np.where(rolled_above == False)[0][0]
        metrics['peak_width'] = width * (360.0 / len(srp_map))

    metrics['entropy'] = entropy(normalized_map + 1e-10)
    metrics['peak_to_average_ratio'] = max_prob / (np.mean(normalized_map) + 1e-10)

    peaks, _ = find_peaks(normalized_map, height=0)
    if len(peaks) > 1:
        sorted_peak_heights = np.sort(normalized_map[peaks])[::-1]
        metrics['second_peak_ratio'] = sorted_peak_heights[1] / (sorted_peak_heights[0] + 1e-10)
    else:
        metrics['second_peak_ratio'] = 0.0

    angles_rad = np.deg2rad(grid_positions)
    cos_sum = np.sum(normalized_map * np.cos(angles_rad))
    sin_sum = np.sum(normalized_map * np.sin(angles_rad))
    centroid_rad = np.arctan2(sin_sum, cos_sum)
    centroid_deg = np.degrees(centroid_rad) % 360
    diff_angle = np.abs(srp_pred_angle - centroid_deg)
    metrics['centroid_deviation'] = np.min([diff_angle, 360 - diff_angle])

    return metrics

def main():
    """Main execution function."""
    print("="*80)
    print("STARTING SRP CONFIDENCE ANALYSIS")
    print("="*80)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_list = []

    srp_files = list(SRP_FEATURES_RAW_DIR.glob('srp_results_mics_*.pkl'))
    if not srp_files:
        raise FileNotFoundError(f"No SRP raw feature files found in {SRP_FEATURES_RAW_DIR}")

    print(f"Found {len(srp_files)} SRP raw feature files.")

    for srp_file in srp_files:
        print(f"Processing {srp_file.name}...")
        num_mics_changed = count_changed_mics(srp_file)
        with open(srp_file, 'rb') as f:
            data_list = pickle.load(f)
        
        for sample_data in data_list:
            srp_map = sample_data['srp_map']
            if 'grid' in sample_data and sample_data['grid'] is not None:
                grid_positions = sample_data['grid'][:, 0]
            else:
                grid_positions = np.arange(360)

            calculated_metrics = calculate_srp_metrics(srp_map, sample_data['srp_pred'], grid_positions)
            
            result_row = {
                'srp_error': sample_data['srp_error'],
                'num_mics_changed': num_mics_changed
            }
            result_row.update(calculated_metrics)
            results_list.append(result_row)
    
    results_df = pd.DataFrame(results_list)

    print("\nCalculating correlations and generating plots...")
    for metric_name in [m for m in results_df.columns if m not in ['srp_error', 'num_mics_changed']]:
        
        # Drop rows with NaN/Inf for the current metric for clean plotting and correlation
        metric_df = results_df[['srp_error', 'num_mics_changed', metric_name]].dropna()

        if len(metric_df) < 2:
            print(f"  Skipping plot for {metric_name}: Not enough valid data points.")
            continue
            
        correlation, p_value = pearsonr(metric_df['srp_error'], metric_df[metric_name])
        print(f"Correlation (SRP Error vs. {metric_name}): {correlation:.4f} (p-value: {p_value:.4g})")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = sns.scatterplot(
            data=metric_df,
            x='srp_error', 
            y=metric_name,
            hue='num_mics_changed',
            palette='viridis',
            ax=ax, 
            alpha=0.6, 
            s=20
        )
        
        ax.set_title(f'SRP Error vs. {metric_name}', fontsize=16)
        ax.set_xlabel('SRP Absolute Error (°)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        handles, labels = scatter.get_legend_handles_labels()
        ax.legend(handles, labels, title='Changed Mics')
        
        ax.text(0.95, 0.95, f'Pearson Correlation: {correlation:.3f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plot_path = OUTPUT_DIR / f'srp_error_vs_{metric_name}_colorcoded.png'
        plt.savefig(plot_path, dpi=300)
        print(f"  Plot saved to: {plot_path}")
        plt.close(fig)

    print("\n✅ SRP Confidence Analysis Complete!")

if __name__ == '__main__':
    main()
