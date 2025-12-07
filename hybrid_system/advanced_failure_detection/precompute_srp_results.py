#!/usr/bin/env python3
"""
Pre-compute SRP results for entire test set.

Run this once to save SRP predictions for all test samples.
Then hybrid evaluations can reuse these results instantly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

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

# Paths
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
FEATURES_PATH = Path("features/test_3x12cm_consecutive_features.npz")
OUTPUT_PATH = Path("features/test_3x12cm_srp_results.pkl")

# Configuration
MIC_ORDER_SRP = [0, 9, 10, 11, 4, 5, 6, 7, 8]

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

    return np.array(signals)


def run_srp_on_case(audio_path, mic_order_srp, gt_angle):
    """Run SRP on a single audio case."""
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

    # Calculate angular error
    angle_diff = abs(srp_pred - gt_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return srp_pred, angle_diff


def main():
    print("="*100)
    print("PRE-COMPUTING SRP RESULTS FOR ENTIRE TEST SET")
    print("="*100)

    # Load test features
    print(f"\nLoading test features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = {key: data[key] for key in data.files}

    n_samples = len(features['gt_angles'])
    gt_angles = features['gt_angles']
    crnn_preds = features['predicted_angles']
    crnn_errors = features['abs_errors']

    print(f"Test samples: {n_samples}")
    print(f"CRNN-only MAE: {crnn_errors.mean():.2f}°")

    # Load CSV for audio paths
    df = pd.read_csv(CSV_PATH)

    # Pre-compute SRP for all samples
    print(f"\nRunning SRP on ALL {n_samples} test samples...")
    print("This will take ~2-3 hours but only needs to be done ONCE.")
    print("Then all future hybrid evaluations will be instant!\n")

    srp_results = []

    for idx in tqdm(range(n_samples), desc="Running SRP"):
        global_idx = int(features['global_indices'][idx])

        if global_idx >= len(df):
            print(f"  Warning: global_idx {global_idx} out of range")
            srp_results.append({
                'sample_idx': idx,
                'global_idx': global_idx,
                'gt_angle': gt_angles[idx],
                'crnn_pred': crnn_preds[idx],
                'crnn_error': crnn_errors[idx],
                'srp_pred': None,
                'srp_error': None
            })
            continue

        row = df.iloc[global_idx]
        audio_filename = row['filename']
        audio_path = os.path.join(str(DATA_ROOT), audio_filename.replace('.flac', '_CH0.wav'))

        gt_angle = gt_angles[idx]
        crnn_pred = crnn_preds[idx]
        crnn_error = crnn_errors[idx]

        # Run SRP
        srp_pred, srp_error = run_srp_on_case(audio_path, MIC_ORDER_SRP, gt_angle)

        srp_results.append({
            'sample_idx': idx,
            'global_idx': global_idx,
            'gt_angle': gt_angle,
            'crnn_pred': crnn_pred,
            'crnn_error': crnn_error,
            'srp_pred': srp_pred,
            'srp_error': srp_error
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(srp_results)

    # Filter valid SRP results
    valid_srp = results_df[results_df['srp_pred'].notna()]

    print("\n" + "="*100)
    print("SRP RESULTS SUMMARY")
    print("="*100)
    print(f"Total samples: {len(results_df)}")
    print(f"Valid SRP results: {len(valid_srp)}")
    print(f"Failed SRP: {len(results_df) - len(valid_srp)}")

    if len(valid_srp) > 0:
        print(f"\nSRP-only performance:")
        print(f"  MAE: {valid_srp['srp_error'].mean():.2f}°")
        print(f"  Median: {valid_srp['srp_error'].median():.2f}°")
        print(f"  Success rate (≤5°): {(valid_srp['srp_error'] <= 5).sum() / len(valid_srp) * 100:.1f}%")

        print(f"\nCRNN-only performance (same samples):")
        print(f"  MAE: {valid_srp['crnn_error'].mean():.2f}°")
        print(f"  Median: {valid_srp['crnn_error'].median():.2f}°")
        print(f"  Success rate (≤5°): {(valid_srp['crnn_error'] <= 5).sum() / len(valid_srp) * 100:.1f}%")

    # Save results
    print(f"\nSaving results to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(results_df, f)

    print(f"✅ Pre-computed SRP results saved!")
    print(f"\nNow you can run hybrid evaluations instantly using:")
    print(f"  python evaluate_ood_hybrid_fast.py --method <method> --threshold <threshold>")


if __name__ == "__main__":
    main()
