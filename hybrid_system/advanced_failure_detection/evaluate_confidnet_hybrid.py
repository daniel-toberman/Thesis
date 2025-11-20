#!/usr/bin/env python3
"""
Evaluate Hybrid CRNN-SRP System with ConfidNet Routing

This script:
1. Loads trained ConfidNet model
2. Finds optimal confidence threshold on validation set (or uses provided threshold)
3. Routes low-confidence cases to SRP
4. Runs SRP on routed cases
5. Combines CRNN (kept) + SRP (routed) for hybrid performance
6. Compares with CRNN-only and other routing baselines

Usage:
    python evaluate_confidnet_hybrid.py --model_path models/confidnet/best_model.ckpt
    python evaluate_confidnet_hybrid.py --model_path models/confidnet/best_model.ckpt --confidence_threshold 0.5
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import soundfile as sf
from scipy import signal
from tqdm import tqdm

# SRP imports
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/xsrpMain")
from xsrp.conventional_srp import ConventionalSrp
from SSL.utils_ import audiowu_high_array_geometry

# Import ConfidNet router
from confidnet_routing import ConfidNetRouter

# Paths
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
FEATURES_PATH = Path("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/test_3x12cm_consecutive_features.npz")

# Configuration
MIC_ORDER_SRP = [0, 9, 10, 11, 4, 5, 6, 7, 8]  # SRP order (mic 0 first - reference)

# SRP parameters
SRP_CONFIG = {
    'n_dft_bins': 16384,
    'freq_min': 300,
    'freq_max': 4000,
    'grid_cells': 360,
    'mode': 'gcc_phat_freq',
    'n_avg_samples': 1,
}


def load_audio_multichannel(example_path, use_mic_id, fs=16000):
    """Load audio from multiple microphone channels."""
    base_path = example_path.replace('_CH0.wav', '')

    signals = []
    for mic_id in use_mic_id:
        ch_path = f"{base_path}_CH{mic_id}.wav"
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f"Missing channel file: {ch_path}")

        sig, file_fs = sf.read(ch_path, dtype="float64")

        # Resample if needed
        if file_fs != fs:
            sig = signal.resample(sig, int(len(sig) * fs / file_fs))

        signals.append(sig)

    return np.array(signals)  # Shape: (n_mics, n_samples)


def run_srp_on_case(audio_path, mic_order_srp, gt_angle):
    """Run SRP on a single audio case."""
    # Load audio with SRP mic ordering
    try:
        audio = load_audio_multichannel(audio_path, mic_order_srp)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        return None, None

    # Get microphone positions
    all_mic_positions = audiowu_high_array_geometry()
    mic_positions = all_mic_positions[mic_order_srp, :2]

    # Initialize SRP
    srp = ConventionalSrp(
        fs=16000,
        grid_type="doa_1D",
        n_grid_cells=SRP_CONFIG['grid_cells'],
        mic_positions=mic_positions,
        room_dims=None,
        mode=SRP_CONFIG['mode'],
        interpolation=True,
        n_average_samples=SRP_CONFIG['n_avg_samples'],
        n_dft_bins=SRP_CONFIG['n_dft_bins']
    )

    # Run SRP
    est_vec, srp_map, grid = srp.forward(audio)

    # Calculate azimuth
    srp_pred = float(np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0)

    # Calculate error
    error = abs((srp_pred - gt_angle + 180) % 360 - 180)

    return srp_pred, error


def main():
    parser = argparse.ArgumentParser(description='Evaluate ConfidNet-based hybrid system')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained ConfidNet checkpoint')
    parser.add_argument('--confidence_threshold', type=float, default=None,
                        help='Confidence threshold (if None, will search for optimal)')
    parser.add_argument('--error_threshold', type=float, default=15.0,
                        help='Error threshold for defining routing ground truth (default: 15°)')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')

    args = parser.parse_args()

    print("="*100)
    print("HYBRID CRNN-SRP WITH CONFIDNET ROUTING")
    print("="*100)

    # Load features
    print(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    n_samples = len(features['predicted_angles'])
    print(f"  Total samples: {n_samples}")

    # Load test CSV
    print(f"\nLoading test dataset info from: {CSV_PATH}")
    df_test = pd.read_csv(CSV_PATH)
    print(f"  Total test samples: {len(df_test)}")

    # Load ConfidNet router
    print("\n" + "="*100)
    print("LOADING CONFIDNET ROUTER")
    print("="*100)

    router = ConfidNetRouter(model_path=args.model_path, device=args.device)

    # Override error threshold for routing evaluation
    router.error_threshold = args.error_threshold
    print(f"\nUsing error threshold for routing: {args.error_threshold}° (overriding training value)")

    # Find optimal threshold or use provided one
    if args.confidence_threshold is None:
        print("\nSearching for optimal confidence threshold...")
        optimal_results = router.find_optimal_threshold(features)
        confidence_threshold = optimal_results['best_threshold']
        routing_results = optimal_results['best_results']
    else:
        confidence_threshold = args.confidence_threshold
        print(f"\nUsing provided confidence threshold: {confidence_threshold:.4f}")
        routing_results = router.evaluate_routing(features, confidence_threshold)

    route_to_srp = routing_results['route_to_srp']
    n_srp = route_to_srp.sum()
    n_crnn = (~route_to_srp).sum()

    print(f"\n{'='*100}")
    print(f"ROUTING DECISIONS:")
    print(f"{'='*100}")
    print(f"  Confidence threshold: {confidence_threshold:.4f}")
    print(f"  Cases for SRP:  {n_srp:>4} ({n_srp/n_samples*100:>5.1f}%)")
    print(f"  Cases for CRNN: {n_crnn:>4} ({n_crnn/n_samples*100:>5.1f}%)")
    print(f"\n  Routing Metrics:")
    print(f"    Precision: {routing_results['precision']:.4f}")
    print(f"    Recall:    {routing_results['recall']:.4f}")
    print(f"    F1 Score:  {routing_results['f1_score']:.4f}")
    print(f"    False Positive Rate: {routing_results['false_positive_rate']*100:.1f}%")
    print(f"    Catastrophic Capture: {routing_results['catastrophic_capture_rate']*100:.1f}%")

    # Analyze SRP cases
    srp_indices = np.where(route_to_srp)[0]
    srp_errors = features['abs_errors'][srp_indices]

    print(f"\nSRP Cases Breakdown:")
    print(f"  Catastrophic (>30°):  {(srp_errors > 30).sum():>4} ({(srp_errors > 30).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  Bad (10-30°):         {((srp_errors >= 10) & (srp_errors <= 30)).sum():>4} ({((srp_errors >= 10) & (srp_errors <= 30)).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  Moderate (5-10°):     {((srp_errors >= 5) & (srp_errors < 10)).sum():>4} ({((srp_errors >= 5) & (srp_errors < 10)).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  Good (≤5°):           {(srp_errors <= 5).sum():>4} ({(srp_errors <= 5).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  CRNN MAE on these:    {srp_errors.mean():.2f}°")

    print(f"\n{'='*100}")
    print(f"RUNNING SRP ON {n_srp} CASES")
    print(f"{'='*100}")
    print(f"Configuration:")
    print(f"  Mic ordering: {MIC_ORDER_SRP}")
    print(f"  n_dft_bins:   {SRP_CONFIG['n_dft_bins']}")
    print(f"  freq_range:   {SRP_CONFIG['freq_min']}-{SRP_CONFIG['freq_max']} Hz")
    print(f"  grid_cells:   {SRP_CONFIG['grid_cells']}")

    # Run SRP on selected cases
    srp_results = []

    for idx in tqdm(srp_indices, desc="Running SRP"):
        global_idx = int(features['global_indices'][idx])

        # Get audio path from test CSV
        if global_idx >= len(df_test):
            print(f"  Warning: global_idx {global_idx} out of range")
            continue

        test_row = df_test.iloc[global_idx]
        audio_filename = test_row['filename']
        audio_path = os.path.join(DATA_ROOT, audio_filename.replace('.flac', '_CH0.wav'))
        gt_angle = test_row['angle(°)']

        srp_pred, srp_error = run_srp_on_case(audio_path, MIC_ORDER_SRP, gt_angle)

        if srp_pred is not None:
            srp_results.append({
                'sample_idx': idx,
                'global_idx': global_idx,
                'gt_angle': gt_angle,
                'crnn_pred': features['predicted_angles'][idx],
                'crnn_error': features['abs_errors'][idx],
                'srp_pred': srp_pred,
                'srp_error': srp_error,
                'confidence': routing_results['confidences'][idx],
            })

    print(f"\n✅ SRP processing complete: {len(srp_results)}/{n_srp} cases successful")

    # Create results DataFrame
    df_srp = pd.DataFrame(srp_results)

    # Save SRP results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    srp_output = output_dir / "confidnet_hybrid_srp_results.csv"
    df_srp.to_csv(srp_output, index=False)
    print(f"\n💾 Saved SRP results to: {srp_output}")

    # Calculate hybrid performance
    print(f"\n{'='*100}")
    print(f"HYBRID SYSTEM PERFORMANCE")
    print(f"{'='*100}")

    # CRNN-only cases
    crnn_indices = np.where(~route_to_srp)[0]
    crnn_errors = features['abs_errors'][crnn_indices]
    crnn_mae = crnn_errors.mean()
    crnn_median = np.median(crnn_errors)

    # SRP cases
    srp_mae = df_srp['srp_error'].mean()
    srp_median = df_srp['srp_error'].median()

    # Hybrid
    hybrid_errors = []
    hybrid_errors.extend(crnn_errors)
    hybrid_errors.extend(df_srp['srp_error'].values)
    hybrid_mae = np.mean(hybrid_errors)
    hybrid_median = np.median(hybrid_errors)

    # Overall CRNN-only (for comparison)
    crnn_all_mae = features['abs_errors'].mean()
    crnn_all_median = np.median(features['abs_errors'])

    print(f"\nCRNN-only (all {n_samples} cases):")
    print(f"  MAE:    {crnn_all_mae:.2f}°")
    print(f"  Median: {crnn_all_median:.2f}°")
    print(f"  Success (≤5°): {(features['abs_errors'] <= 5).sum()} / {n_samples} ({(features['abs_errors'] <= 5).sum()/n_samples*100:.1f}%)")

    print(f"\nCRNN-only ({len(crnn_errors)} cases kept on CRNN):")
    print(f"  MAE:    {crnn_mae:.2f}°")
    print(f"  Median: {crnn_median:.2f}°")
    print(f"  Success (≤5°): {(crnn_errors <= 5).sum()} / {len(crnn_errors)} ({(crnn_errors <= 5).sum()/len(crnn_errors)*100:.1f}%)")

    print(f"\nSRP-only ({len(df_srp)} cases routed to SRP):")
    print(f"  MAE:    {srp_mae:.2f}°")
    print(f"  Median: {srp_median:.2f}°")
    print(f"  Success (≤5°): {(df_srp['srp_error'] <= 5).sum()} / {len(df_srp)} ({(df_srp['srp_error'] <= 5).sum()/len(df_srp)*100:.1f}%)")
    print(f"  vs CRNN on same cases: {srp_errors.mean():.2f}°")
    print(f"  Improvement: {srp_errors.mean() - srp_mae:.2f}°")

    print(f"\n🎯 HYBRID SYSTEM ({len(crnn_errors)} CRNN + {len(df_srp)} SRP):")
    print(f"  MAE:    {hybrid_mae:.2f}°")
    print(f"  Median: {hybrid_median:.2f}°")

    # Success rates
    hybrid_success_count = (crnn_errors <= 5).sum() + (df_srp['srp_error'] <= 5).sum()
    hybrid_success_rate = hybrid_success_count / n_samples * 100
    crnn_all_success_rate = (features['abs_errors'] <= 5).sum() / n_samples * 100

    print(f"  Success (≤5°): {hybrid_success_count} / {n_samples} ({hybrid_success_rate:.1f}%)")
    print(f"\n  vs CRNN-only MAE: {crnn_all_mae - hybrid_mae:+.2f}° ({(crnn_all_mae - hybrid_mae)/crnn_all_mae*100:+.1f}%)")
    print(f"  vs CRNN-only median: {crnn_all_median - hybrid_median:+.2f}° ({(crnn_all_median - hybrid_median)/crnn_all_median*100:+.1f}%)")
    print(f"  vs CRNN-only success: {hybrid_success_rate - crnn_all_success_rate:+.1f}%")

    # Routing accuracy
    srp_better = (df_srp['srp_error'] < df_srp['crnn_error']).sum()
    routing_accuracy = srp_better / len(df_srp) * 100

    print(f"\nRouting Accuracy:")
    print(f"  SRP better than CRNN: {srp_better} / {len(df_srp)} ({routing_accuracy:.1f}%)")

    # Catastrophic rescue
    catastrophic_indices = srp_indices[srp_errors > 30]
    if len(catastrophic_indices) > 0:
        catastrophic_srp_results = df_srp[df_srp['crnn_error'] > 30]
        if len(catastrophic_srp_results) > 0:
            print(f"\nCatastrophic Failure Rescue:")
            print(f"  Cases: {len(catastrophic_srp_results)}")
            print(f"  CRNN MAE on these: {catastrophic_srp_results['crnn_error'].mean():.2f}°")
            print(f"  SRP MAE on these: {catastrophic_srp_results['srp_error'].mean():.2f}°")
            print(f"  Improvement: {catastrophic_srp_results['crnn_error'].mean() - catastrophic_srp_results['srp_error'].mean():.2f}°")

    print(f"\n{'='*100}")

    # Save summary
    summary = {
        'method': 'ConfidNet',
        'n_samples': n_samples,
        'n_routed_srp': len(df_srp),
        'routing_rate': n_srp / n_samples,
        'confidence_threshold': confidence_threshold,
        'precision': routing_results['precision'],
        'recall': routing_results['recall'],
        'f1_score': routing_results['f1_score'],
        'false_positive_rate': routing_results['false_positive_rate'],
        'crnn_only_mae': crnn_all_mae,
        'crnn_only_median': crnn_all_median,
        'crnn_only_success': crnn_all_success_rate,
        'hybrid_mae': hybrid_mae,
        'hybrid_median': hybrid_median,
        'hybrid_success': hybrid_success_rate,
        'mae_improvement': crnn_all_mae - hybrid_mae,
        'median_improvement': crnn_all_median - hybrid_median,
        'success_improvement': hybrid_success_rate - crnn_all_success_rate,
        'routing_accuracy': routing_accuracy,
    }

    summary_df = pd.DataFrame([summary])
    summary_output = output_dir / "confidnet_hybrid_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"💾 Saved summary to: {summary_output}")

    return df_srp, hybrid_mae, hybrid_median


if __name__ == "__main__":
    main()
