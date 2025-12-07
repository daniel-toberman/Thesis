#!/usr/bin/env python3
"""
Oracle baseline evaluation: Routes cases with worst CRNN errors to SRP.
Represents theoretical upper bound on routing performance using ground truth.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import soundfile as sf
from scipy import signal

# SRP imports
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/xsrpMain")
from xsrp.conventional_srp import ConventionalSrp
from SSL.utils_ import audiowu_high_array_geometry

# Paths
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")
TEST_CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")

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


def evaluate_oracle_baseline(routing_pct, output_dir):
    """Evaluate oracle baseline that routes worst X% of CRNN predictions to SRP."""

    print("="*100)
    print(f"ORACLE BASELINE EVALUATION: ROUTE WORST {routing_pct}% OF CRNN PREDICTIONS")
    print("="*100)

    # Load features
    print(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}
    print(f"  Total samples: {len(features['predicted_angles'])}")

    # Load test dataset info
    print(f"\nLoading test dataset info from: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"  Total test samples: {len(test_df)}")

    # Get CRNN errors
    errors = features['abs_errors']
    predicted_angles = features['predicted_angles']
    gt_angles = features['gt_angles']

    # Define microphone layout
    mic_order_srp = np.array([
        [0, 0, 0], [0.03, 0, 0], [0.06, 0, 0],
        [0, 0.03, 0], [0.03, 0.03, 0], [0.06, 0.03, 0],
        [0, 0.06, 0], [0.03, 0.06, 0], [0.06, 0.06, 0]
    ])

    # Oracle routing: route worst X% by CRNN error
    n_total = len(errors)
    n_to_route = int(n_total * routing_pct / 100)

    # Get indices of worst predictions
    worst_indices = np.argsort(errors)[-n_to_route:]
    route_to_srp = np.zeros(n_total, dtype=bool)
    route_to_srp[worst_indices] = True

    print(f"\n{'='*100}")
    print("ORACLE ROUTING DECISIONS")
    print("="*100)
    print(f"\n  Target routing: {routing_pct}%")
    print(f"  Cases for SRP:   {route_to_srp.sum():4d} ({route_to_srp.sum()/n_total*100:.1f}%)")
    print(f"  Cases for CRNN: {(~route_to_srp).sum():4d} ({(~route_to_srp).sum()/n_total*100:.1f}%)")

    # Routing quality metrics
    should_route_5deg = (errors > 5)
    should_route_30deg = (errors > 30)

    tp = (route_to_srp & should_route_5deg).sum()
    fp = (route_to_srp & ~should_route_5deg).sum()
    fn = (~route_to_srp & should_route_5deg).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nRouting Quality (5° threshold):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Catastrophic (>30°) capture: {(route_to_srp & should_route_30deg).sum()}/{should_route_30deg.sum()} ({(route_to_srp & should_route_30deg).sum()/should_route_30deg.sum()*100:.1f}%)")

    # Analyze routed cases
    routed_errors = errors[route_to_srp]
    print(f"\nCRNN performance on routed cases:")
    print(f"  MAE: {routed_errors.mean():.2f}°")
    print(f"  Median: {np.median(routed_errors):.2f}°")
    print(f"  Min: {routed_errors.min():.2f}°, Max: {routed_errors.max():.2f}°")

    # Breakdown by error severity
    catastrophic = (routed_errors > 30).sum()
    bad = ((routed_errors > 10) & (routed_errors <= 30)).sum()
    moderate = ((routed_errors > 5) & (routed_errors <= 10)).sum()
    good = (routed_errors <= 5).sum()

    print(f"\nRouted cases breakdown:")
    print(f"  Catastrophic (>30°):   {catastrophic:4d} ({catastrophic/len(routed_errors)*100:5.1f}%)")
    print(f"  Bad (10-30°):          {bad:4d} ({bad/len(routed_errors)*100:5.1f}%)")
    print(f"  Moderate (5-10°):      {moderate:4d} ({moderate/len(routed_errors)*100:5.1f}%)")
    print(f"  Good (≤5°):            {good:4d} ({good/len(routed_errors)*100:5.1f}%)")

    # Run SRP on routed cases
    print(f"\n{'='*100}")
    print(f"RUNNING SRP ON {route_to_srp.sum()} CASES")
    print("="*100)

    srp_results = []
    routed_indices = np.where(route_to_srp)[0]

    for idx in tqdm(routed_indices, desc="Running SRP"):
        test_row = test_df.iloc[idx]

        # Construct audio path (filename includes test/ma_speech prefix)
        audio_filename = test_row['filename']
        audio_path = os.path.join(DATA_ROOT, audio_filename.replace('.flac', '_CH0.wav'))

        # Run SRP
        srp_pred, srp_error = run_srp_on_case(audio_path, MIC_ORDER_SRP, gt_angles[idx])

        srp_results.append({
            'sample_idx': idx,
            'crnn_pred': predicted_angles[idx],
            'srp_pred': srp_pred,
            'gt_angle': gt_angles[idx],
            'crnn_error': errors[idx],
            'srp_error': srp_error
        })

    srp_df = pd.DataFrame(srp_results)

    # Compute hybrid performance
    print(f"\n{'='*100}")
    print("HYBRID PERFORMANCE")
    print("="*100)

    # Start with CRNN predictions
    hybrid_errors = errors.copy()

    # Replace routed cases with SRP
    for result in srp_results:
        idx = result['sample_idx']
        hybrid_errors[idx] = result['srp_error']

    # CRNN-only metrics
    crnn_mae = errors.mean()
    crnn_median = np.median(errors)
    crnn_success = (errors <= 5).sum() / len(errors) * 100

    # Hybrid metrics
    hybrid_mae = hybrid_errors.mean()
    hybrid_median = np.median(hybrid_errors)
    hybrid_success = (hybrid_errors <= 5).sum() / len(hybrid_errors) * 100

    print(f"\n  CRNN-only MAE:    {crnn_mae:.2f}°")
    print(f"  Hybrid MAE:       {hybrid_mae:.2f}° (Δ {hybrid_mae - crnn_mae:+.2f}°)")

    print(f"\n  CRNN-only Median: {crnn_median:.2f}°")
    print(f"  Hybrid Median:    {hybrid_median:.2f}° (Δ {hybrid_median - crnn_median:+.2f}°)")

    print(f"\n  CRNN-only Success: {crnn_success:.1f}%")
    print(f"  Hybrid Success:    {hybrid_success:.1f}% (Δ {hybrid_success - crnn_success:+.1f}%)")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    srp_df.to_csv(output_dir / "srp_results.csv", index=False)
    print(f"\n✅ SRP results saved to: {output_dir}/srp_results.csv")

    # Save summary
    summary_df = pd.DataFrame([{
        'method': 'Oracle',
        'routing_pct': routing_pct,
        'n_routed': route_to_srp.sum(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'crnn_only_mae': crnn_mae,
        'crnn_only_median': crnn_median,
        'crnn_only_success': crnn_success,
        'hybrid_mae': hybrid_mae,
        'hybrid_median': hybrid_median,
        'hybrid_success': hybrid_success,
        'mae_improvement': crnn_mae - hybrid_mae,
        'median_improvement': crnn_median - hybrid_median,
        'success_improvement': hybrid_success - crnn_success,
        'catastrophic_capture_rate': (route_to_srp & should_route_30deg).sum() / should_route_30deg.sum(),
        'false_positive_rate': fp / (~should_route_5deg).sum() if (~should_route_5deg).sum() > 0 else 0
    }])

    summary_df.to_csv(output_dir / "oracle_hybrid_summary.csv", index=False)
    print(f"✅ Summary saved to: {output_dir}/oracle_hybrid_summary.csv")


def main():
    parser = argparse.ArgumentParser(description='Evaluate oracle baseline routing')
    parser.add_argument('--routing_pct', type=float, required=True,
                        help='Percentage of worst cases to route to SRP (e.g., 25, 30)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    args = parser.parse_args()

    evaluate_oracle_baseline(args.routing_pct, args.output_dir)


if __name__ == "__main__":
    main()
