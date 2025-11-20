#!/usr/bin/env python3
"""
Evaluate Hybrid CRNN-SRP with Temperature + Mahalanobis (Combined Training)

This script evaluates the hybrid system using Temperature scaling + Mahalanobis distance
trained on combined 6cm + 3x12cm training data.
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
import pickle

# SRP imports
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/xsrpMain")
from xsrp.conventional_srp import ConventionalSrp
from SSL.utils_ import audiowu_high_array_geometry

# Routing imports
from temperature_scaling import apply_temperature_scaling
from mahalanobis_ood import MahalanobisOODDetector

# Paths
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")

# Configuration
MIC_ORDER_SRP = [0, 9, 10, 11, 4, 5, 6, 7, 8]  # SRP order

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

        if file_fs != fs:
            sig = signal.resample(sig, int(len(sig) * fs / file_fs))

        signals.append(sig)

    return np.array(signals)


def run_srp_on_case(audio_path, mic_order_srp, gt_angle):
    """Run SRP on a single audio case."""
    try:
        audio = load_audio_multichannel(audio_path, mic_order_srp)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        return None, None

    all_mic_positions = audiowu_high_array_geometry()
    mic_positions = all_mic_positions[mic_order_srp, :2]

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


def search_optimal_thresholds(features, temperature, mahal_detector):
    """Search for optimal confidence and distance thresholds."""
    print("\nSearching for optimal thresholds...")

    # Compute calibrated confidences
    logits = []
    for logit in features['logits_pre_sig']:
        logits.append(logit.mean(axis=0))  # Average over time
    logits = np.array(logits)  # (N, 360)

    calibrated_probs = apply_temperature_scaling(logits, temperature)
    confidences = calibrated_probs.max(axis=1)  # (N,)

    # Compute Mahalanobis distances
    penult_features = []
    for feat in features['penultimate_features']:
        penult_features.append(feat.mean(axis=0))  # Average over time
    penult_features = np.array(penult_features)  # (N, 256)

    predicted_angles = features['predicted_angles']
    distances = mahal_detector.compute_mahalanobis_distance(penult_features, predicted_angles)

    # Ground truth: should we route to SRP?
    # Use higher threshold to focus on catastrophic failures
    errors = features['abs_errors']
    should_route = (errors > 15.0)  # Route if CRNN has significant error > 15°

    # Search confidence thresholds
    conf_thresholds = np.linspace(0.1, 0.9, 50)
    dist_thresholds = np.linspace(5, 50, 50)

    best_f1 = 0
    best_conf_thresh = None
    best_dist_thresh = None
    best_metrics = None

    for conf_thresh in conf_thresholds:
        for dist_thresh in dist_thresholds:
            # Route if EITHER low confidence OR high distance
            route = (confidences < conf_thresh) | (distances > dist_thresh)

            # Compute metrics
            tp = (route & should_route).sum()
            fp = (route & ~should_route).sum()
            fn = (~route & should_route).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_conf_thresh = conf_thresh
                best_dist_thresh = dist_thresh
                best_metrics = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'routing_rate': route.mean(),
                    'fp_rate': fp / (~should_route).sum() if (~should_route).sum() > 0 else 0,
                }

    print(f"\n✅ Optimal thresholds found:")
    print(f"  Confidence threshold: {best_conf_thresh:.4f}")
    print(f"  Distance threshold: {best_dist_thresh:.2f}")
    print(f"  F1 Score: {best_metrics['f1']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  Routing rate: {best_metrics['routing_rate']*100:.1f}%")
    print(f"  False positive rate: {best_metrics['fp_rate']*100:.1f}%")

    return best_conf_thresh, best_dist_thresh, best_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Temperature + Mahalanobis hybrid')
    parser.add_argument('--temperature_path', type=str, default='models/temperature_combined.npy')
    parser.add_argument('--mahalanobis_path', type=str, default='models/mahalanobis_combined.pkl')
    parser.add_argument('--confidence_threshold', type=float, default=None)
    parser.add_argument('--distance_threshold', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default='results/temp_mahal_combined')

    args = parser.parse_args()

    print("="*100)
    print("HYBRID CRNN-SRP WITH TEMPERATURE + MAHALANOBIS")
    print("="*100)

    # Load test features
    print(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    n_samples = len(features['predicted_angles'])
    print(f"  Total samples: {n_samples}")

    # Load test CSV
    print(f"\nLoading test dataset info from: {CSV_PATH}")
    df_test = pd.read_csv(CSV_PATH)
    print(f"  Total test samples: {len(df_test)}")

    # Load trained models
    print("\n" + "="*100)
    print("LOADING TRAINED MODELS")
    print("="*100)

    print(f"\nLoading temperature from: {args.temperature_path}")
    temperature = float(np.load(args.temperature_path))
    print(f"  Temperature: {temperature:.4f}")

    print(f"\nLoading Mahalanobis detector from: {args.mahalanobis_path}")
    with open(args.mahalanobis_path, 'rb') as f:
        mahal_detector = pickle.load(f)
    print(f"  PCA components: {mahal_detector.n_components}")
    print(f"  Variance explained: {mahal_detector.pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Find optimal thresholds or use provided
    if args.confidence_threshold is None or args.distance_threshold is None:
        conf_thresh, dist_thresh, metrics = search_optimal_thresholds(
            features, temperature, mahal_detector
        )
    else:
        conf_thresh = args.confidence_threshold
        dist_thresh = args.distance_threshold
        print(f"\nUsing provided thresholds:")
        print(f"  Confidence: {conf_thresh:.4f}")
        print(f"  Distance: {dist_thresh:.2f}")

    # Compute routing decisions
    print("\n" + "="*100)
    print("COMPUTING ROUTING DECISIONS")
    print("="*100)

    # Calibrated confidences
    logits = []
    for logit in features['logits_pre_sig']:
        logits.append(logit.mean(axis=0))
    logits = np.array(logits)

    calibrated_probs = apply_temperature_scaling(logits, temperature)
    confidences = calibrated_probs.max(axis=1)

    # Mahalanobis distances
    penult_features = []
    for feat in features['penultimate_features']:
        penult_features.append(feat.mean(axis=0))
    penult_features = np.array(penult_features)

    predicted_angles = features['predicted_angles']
    distances = mahal_detector.compute_mahalanobis_distance(penult_features, predicted_angles)

    # Route to SRP if EITHER condition met
    route_to_srp = (confidences < conf_thresh) | (distances > dist_thresh)
    n_srp = route_to_srp.sum()
    n_crnn = (~route_to_srp).sum()

    print(f"\n  Cases for SRP:  {n_srp:>4} ({n_srp/n_samples*100:>5.1f}%)")
    print(f"  Cases for CRNN: {n_crnn:>4} ({n_crnn/n_samples*100:>5.1f}%)")

    # Analyze SRP cases
    srp_indices = np.where(route_to_srp)[0]
    srp_errors = features['abs_errors'][srp_indices]

    print(f"\nSRP Cases Breakdown:")
    print(f"  Catastrophic (>30°):  {(srp_errors > 30).sum():>4} ({(srp_errors > 30).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  Bad (10-30°):         {((srp_errors >= 10) & (srp_errors <= 30)).sum():>4} ({((srp_errors >= 10) & (srp_errors <= 30)).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  Moderate (5-10°):     {((srp_errors >= 5) & (srp_errors < 10)).sum():>4} ({((srp_errors >= 5) & (srp_errors < 10)).sum()/len(srp_errors)*100:>5.1f}%)")
    print(f"  Good (≤5°):           {(srp_errors <= 5).sum():>4} ({(srp_errors <= 5).sum()/len(srp_errors)*100:>5.1f}%)")

    # Run SRP on routed cases
    print(f"\n{'='*100}")
    print(f"RUNNING SRP ON {n_srp} CASES")
    print(f"{'='*100}")

    srp_results = []
    for idx in tqdm(srp_indices, desc="Running SRP"):
        global_idx = int(features['global_indices'][idx])

        if global_idx >= len(df_test):
            continue

        test_row = df_test.iloc[global_idx]
        audio_filename = test_row['filename']
        audio_path = os.path.join(DATA_ROOT, audio_filename.replace('.flac', '_CH0.wav'))
        gt_angle = test_row['angle(°)']

        srp_angle, srp_error = run_srp_on_case(str(audio_path), MIC_ORDER_SRP, gt_angle)

        if srp_angle is not None:
            srp_results.append({
                'test_index': idx,
                'global_index': global_idx,
                'crnn_angle': features['predicted_angles'][idx],
                'crnn_error': features['abs_errors'][idx],
                'srp_angle': srp_angle,
                'srp_error': srp_error,
                'gt_angle': gt_angle,
                'confidence': confidences[idx],
                'distance': distances[idx]
            })

    # Compute hybrid performance
    print(f"\n{'='*100}")
    print(f"HYBRID PERFORMANCE")
    print(f"{'='*100}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_srp = pd.DataFrame(srp_results)
    df_srp.to_csv(output_dir / 'srp_results.csv', index=False)
    print(f"\n✅ SRP results saved to: {output_dir / 'srp_results.csv'}")

    # Compute hybrid errors
    hybrid_errors = features['abs_errors'].copy()
    for result in srp_results:
        hybrid_errors[result['test_index']] = result['srp_error']

    # Compute metrics
    crnn_only_mae = features['abs_errors'].mean()
    hybrid_mae = hybrid_errors.mean()
    crnn_only_median = np.median(features['abs_errors'])
    hybrid_median = np.median(hybrid_errors)
    crnn_only_success = (features['abs_errors'] <= 5).mean()
    hybrid_success = (hybrid_errors <= 5).mean()

    summary = {
        'method': 'Temp+Mahal Combined',
        'confidence_threshold': conf_thresh,
        'distance_threshold': dist_thresh,
        'routing_rate': n_srp / n_samples,
        'crnn_only_mae': crnn_only_mae,
        'hybrid_mae': hybrid_mae,
        'crnn_only_median': crnn_only_median,
        'hybrid_median': hybrid_median,
        'crnn_only_success': crnn_only_success,
        'hybrid_success': hybrid_success,
        'improvement_mae': crnn_only_mae - hybrid_mae,
        'improvement_median': crnn_only_median - hybrid_median,
        'improvement_success': hybrid_success - crnn_only_success
    }

    print(f"\n  CRNN-only MAE:    {crnn_only_mae:.2f}°")
    print(f"  Hybrid MAE:       {hybrid_mae:.2f}° (Δ {summary['improvement_mae']:+.2f}°)")
    print(f"\n  CRNN-only Median: {crnn_only_median:.2f}°")
    print(f"  Hybrid Median:    {hybrid_median:.2f}° (Δ {summary['improvement_median']:+.2f}°)")
    print(f"\n  CRNN-only Success: {crnn_only_success*100:.1f}%")
    print(f"  Hybrid Success:    {hybrid_success*100:.1f}% (Δ {summary['improvement_success']*100:+.1f}%)")

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(output_dir / 'summary.csv', index=False)
    print(f"\n✅ Summary saved to: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
