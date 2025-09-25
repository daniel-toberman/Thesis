#!/usr/bin/env python3
"""
Run SRP-PHAT on full test/validation sets to find ALL cases where
SRP performs excellently (<2° error) - not just on CRNN failures.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'SSL'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'xsrpMain/xsrp'))

from conventional_srp import ConventionalSrp
from utils_ import audiowu_high_array_geometry
import soundfile as sf

# SRP Configuration (optimized)
USE_MIC_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MIC_POSITIONS = audiowu_high_array_geometry()[USE_MIC_ID, :2]
SRP_GRID_CELLS = 720
SRP_MODE = "gcc_phat_freq"
N_AVG_SAMPLES = 200
N_DFT_BINS = 1024

def load_multichannel_audio(base_audio_path, wav_use_len=4):
    """Load multichannel audio for SRP processing."""

    try:
        channels = []
        for mic_id in USE_MIC_ID:
            ch_path = f"{base_audio_path}_CH{mic_id}.wav"

            if not os.path.exists(ch_path):
                return None, None

            signal, fs = sf.read(ch_path, dtype="float64")

            # Use first 4 seconds like CRNN test
            target_len_samples = int(wav_use_len * fs)
            if len(signal) >= target_len_samples:
                signal_segment = signal[:target_len_samples]
            else:
                signal_segment = signal

            channels.append(signal_segment)

        multichannel_signal = np.stack(channels, axis=0)
        return multichannel_signal, fs

    except Exception as e:
        return None, None

def run_srp_on_signal(multichannel_signal, fs):
    """Run SRP-PHAT on multichannel signal."""

    try:
        srp = ConventionalSrp(
            fs=fs,
            grid_type="doa_1D",
            n_grid_cells=SRP_GRID_CELLS,
            mic_positions=MIC_POSITIONS,
            room_dims=None,
            mode=SRP_MODE,
            interpolation=True,
            n_average_samples=N_AVG_SAMPLES,
            n_dft_bins=N_DFT_BINS
        )

        est_vec, srp_map, grid = srp.forward(multichannel_signal)
        az = float(np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0)
        return az

    except Exception as e:
        return None

def process_dataset(dataset_name, max_examples=500):
    """Process full dataset to find excellent SRP cases."""

    if dataset_name == 'test':
        csv_path = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
    else:
        csv_path = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/val/val_static_source_location_08.csv"

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} examples from {dataset_name} set")
    except FileNotFoundError:
        print(f"CSV not found: {csv_path}")
        return []

    # Limit for testing
    df = df.head(max_examples)
    print(f"Processing first {len(df)} examples...")

    results = []
    excellent_cases = []  # <2° error
    good_cases = []       # <5° error

    base_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processing {dataset_name} example {idx}/{len(df)}")

        try:
            # Get ground truth angle
            gt_angle = row.get('angle(°)', row.get('angle', 0))

            # Get filename and construct path
            filename = str(row['filename']).replace('\\', '/')
            if filename.endswith('.flac'):
                base_filename = filename.replace('.flac', '')
            else:
                base_filename = filename.replace('.wav', '')

            base_audio_path = os.path.join(base_dir, base_filename)

            # Load audio
            multichannel_signal, fs = load_multichannel_audio(base_audio_path)
            if multichannel_signal is None:
                continue

            # Run SRP
            srp_pred = run_srp_on_signal(multichannel_signal, fs)
            if srp_pred is None:
                continue

            # Calculate error
            srp_error = abs((srp_pred - gt_angle + 180) % 360 - 180)

            result = {
                'dataset': dataset_name,
                'example_idx': idx,
                'gt_angle': gt_angle,
                'srp_pred': srp_pred,
                'srp_error': srp_error,
                'audio_file': os.path.basename(base_filename)
            }

            results.append(result)

            # Categorize by performance
            if srp_error < 2.0:
                excellent_cases.append(result)
            elif srp_error < 5.0:
                good_cases.append(result)

        except Exception as e:
            continue

    print(f"\n{dataset_name} Results:")
    print(f"  Total processed: {len(results)}")
    print(f"  Excellent (<2°): {len(excellent_cases)} ({len(excellent_cases)/len(results)*100:.1f}%)")
    print(f"  Good (<5°): {len(good_cases)} ({len(good_cases)/len(results)*100:.1f}%)")
    print(f"  Mean SRP error: {np.mean([r['srp_error'] for r in results]):.1f}°")

    return results, excellent_cases, good_cases

def main():
    """Find all cases where SRP performs excellently."""

    print("=== FINDING ALL EXCELLENT SRP CASES ===")
    print("Searching for cases where SRP achieves <2° error")
    print()

    # Process both datasets (limited for quick estimate)
    test_results, test_excellent, test_good = process_dataset('test', max_examples=50)
    val_results, val_excellent, val_good = process_dataset('validation', max_examples=50)

    # Combined analysis
    all_excellent = test_excellent + val_excellent
    all_good = test_good + val_good
    all_results = test_results + val_results

    print(f"\n=== COMBINED ANALYSIS ===")
    print(f"Total examples processed: {len(all_results)}")
    print(f"Excellent SRP cases (<2°): {len(all_excellent)} ({len(all_excellent)/len(all_results)*100:.1f}%)")
    print(f"Good SRP cases (<5°): {len(all_good)} ({len(all_good)/len(all_results)*100:.1f}%)")

    if len(all_excellent) > 0:
        print(f"\n=== TOP EXCELLENT SRP CASES (<2° error) ===")
        excellent_df = pd.DataFrame(all_excellent)
        excellent_df = excellent_df.sort_values('srp_error')

        print("Rank | Dataset | GT°   | SRP° | Error | Audio File")
        print("-" * 60)

        for idx, (_, row) in enumerate(excellent_df.head(20).iterrows()):
            print(f"{idx+1:4d} | {row['dataset']:7s} | {row['gt_angle']:5.1f} | {row['srp_pred']:4.1f} | {row['srp_error']:5.1f} | {row['audio_file']}")

    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("full_srp_results_sample.csv", index=False)

        if len(all_excellent) > 0:
            excellent_df = pd.DataFrame(all_excellent)
            excellent_df.to_csv("srp_excellent_cases.csv", index=False)
            print(f"\nExcellent cases saved to: srp_excellent_cases.csv")

        print(f"Full results saved to: full_srp_results_sample.csv")

    return len(all_excellent), len(all_results)

if __name__ == "__main__":
    main()