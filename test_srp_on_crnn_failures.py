#!/usr/bin/env python3
"""
Test SRP-PHAT specifically on CRNN's worst failure cases to evaluate
if classical methods can help where neural networks struggle most.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add SSL to path for dataset access
sys.path.append(os.path.join(os.path.dirname(__file__), 'SSL'))
from run_CRNN import MyDataModule

# Add xsrp to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'xsrpMain/xsrp'))
from conventional_srp import ConventionalSrp
from utils_ import audiowu_high_array_geometry

# Import SRP utilities
import soundfile as sf

# SRP Configuration (optimized from previous analysis)
USE_MIC_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MIC_POSITIONS = audiowu_high_array_geometry()[USE_MIC_ID, :2]
SRP_GRID_CELLS = 720  # High resolution
SRP_MODE = "gcc_phat_freq"
N_AVG_SAMPLES = 200  # High averaging
N_DFT_BINS = 1024

def load_worst_cases():
    """Load the worst CRNN failure cases from our previous analysis."""

    try:
        val_df = pd.read_csv("validation_detailed_results.csv")
        test_df = pd.read_csv("test_detailed_results.csv")
        combined_df = pd.concat([val_df, test_df], ignore_index=True)
    except FileNotFoundError:
        print("Error: Run analyze_180_degree_cases.py first to generate data files")
        return None

    # Get worst cases (>20° error)
    worst_cases = combined_df[combined_df['error_degrees'] > 20].copy()
    worst_cases = worst_cases.sort_values('error_degrees', ascending=False)

    print(f"Found {len(worst_cases)} cases with >20° CRNN error")
    print(f"Error range: {worst_cases['error_degrees'].min():.1f}° - {worst_cases['error_degrees'].max():.1f}°")

    return worst_cases

def get_audio_file_for_example(dataset_name, example_idx):
    """
    Map dataset example indices to actual audio files.
    """

    if dataset_name == 'test':
        csv_path = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
    else:  # validation
        csv_path = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/val/val_static_source_location_08.csv"

    try:
        df = pd.read_csv(csv_path)

        if example_idx < len(df):
            row = df.iloc[example_idx]
            filename = str(row['filename']).replace('\\', '/')

            # Convert from .flac to .wav and remove extension for base name
            if filename.endswith('.flac'):
                base_filename = filename.replace('.flac', '')
            elif filename.endswith('.wav'):
                base_filename = filename.replace('.wav', '')
            else:
                base_filename = filename

            # Construct path to base audio file (without channel suffix)
            base_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
            base_audio_path = os.path.join(base_dir, base_filename)

            # Get angle - try different column names
            angle = row.get('angle(°)', row.get('angle', row.get('target_angle', 0)))

            return base_audio_path, row['real_st'], row['real_ed'], angle
        else:
            return None, None, None, None

    except Exception as e:
        print(f"Error mapping example {example_idx}: {e}")
        return None, None, None, None

def load_multichannel_audio(base_audio_path, wav_use_len=4, target_fs=16000):
    """Load multichannel audio for SRP processing, using first 4 seconds like CRNN test."""

    try:
        # Load all channels
        channels = []
        for mic_id in USE_MIC_ID:
            # Construct channel-specific file path
            ch_path = f"{base_audio_path}_CH{mic_id}.wav"

            if not os.path.exists(ch_path):
                print(f"Channel file not found: {ch_path}")
                return None, None

            signal, fs = sf.read(ch_path, dtype="float64")

            # Use first 4 seconds like CRNN test (mimicking RecordData validation logic)
            target_len_samples = int(wav_use_len * fs)

            if len(signal) >= target_len_samples:
                signal_segment = signal[:target_len_samples]
            else:
                # If shorter than 4s, use entire signal
                signal_segment = signal

            channels.append(signal_segment)

        # Stack channels (C, T) format
        multichannel_signal = np.stack(channels, axis=0)
        print(f"  Loaded {len(channels)} channels, segment length: {multichannel_signal.shape[1]} samples ({multichannel_signal.shape[1]/fs:.2f}s)")
        return multichannel_signal, fs

    except Exception as e:
        print(f"Error loading audio {base_audio_path}: {e}")
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
        print(f"Error running SRP: {e}")
        return None

def test_srp_on_failures():
    """Test SRP-PHAT on CRNN failure cases."""

    print("Loading worst CRNN failure cases...")
    worst_cases = load_worst_cases()
    if worst_cases is None:
        return

    results = []
    successful_tests = 0

    # Test top failures (limit to avoid long runtime)
    test_cases = worst_cases.head(20)  # Top 20 worst cases

    print(f"\nTesting SRP-PHAT on {len(test_cases)} worst CRNN cases:")
    print("-" * 80)

    for idx, (_, case) in enumerate(test_cases.iterrows()):
        print(f"\n[{idx+1}/{len(test_cases)}] Testing case {case['example_idx']}")
        print(f"  CRNN: GT={case['gt_angle']:.1f}°, Pred={case['pred_angle']:.1f}°, Error={case['error_degrees']:.1f}°")

        # Map to audio file (this is approximate)
        audio_path, start_sample, end_sample, csv_gt_angle = get_audio_file_for_example(
            case['dataset'], case['example_idx']
        )

        if audio_path is None:
            print(f"  Could not map to audio file")
            continue

        # Load audio (use first 4 seconds like CRNN test)
        multichannel_signal, fs = load_multichannel_audio(audio_path)
        if multichannel_signal is None:
            continue

        # Run SRP
        srp_pred = run_srp_on_signal(multichannel_signal, fs)
        if srp_pred is None:
            continue

        # Calculate SRP error
        gt_angle = case['gt_angle']
        srp_error = abs((srp_pred - gt_angle + 180) % 360 - 180)
        crnn_error = case['error_degrees']

        # Compare methods
        improvement = crnn_error - srp_error
        better_method = "SRP" if improvement > 0 else "CRNN"

        print(f"  SRP:  Pred={srp_pred:.1f}°, Error={srp_error:.1f}°")
        print(f"  Comparison: {better_method} better by {abs(improvement):.1f}°")

        results.append({
            'case_idx': idx,
            'example_idx': case['example_idx'],
            'dataset': case['dataset'],
            'gt_angle': gt_angle,
            'crnn_pred': case['pred_angle'],
            'crnn_error': crnn_error,
            'srp_pred': srp_pred,
            'srp_error': srp_error,
            'improvement': improvement,
            'better_method': better_method,
            'audio_file': os.path.basename(audio_path) if audio_path else None
        })

        successful_tests += 1

    if len(results) == 0:
        print("No successful tests - check file paths and data mapping")
        return

    # Analyze results
    print(f"\n" + "="*80)
    print(f"ANALYSIS OF {len(results)} SUCCESSFUL TESTS")
    print("="*80)

    results_df = pd.DataFrame(results)

    # Overall comparison
    srp_better_count = (results_df['improvement'] > 0).sum()
    crnn_better_count = (results_df['improvement'] <= 0).sum()

    print(f"\nMethod Comparison:")
    print(f"  SRP-PHAT better: {srp_better_count}/{len(results)} ({srp_better_count/len(results)*100:.1f}%)")
    print(f"  CRNN better: {crnn_better_count}/{len(results)} ({crnn_better_count/len(results)*100:.1f}%)")

    # Error statistics
    print(f"\nError Statistics:")
    print(f"  CRNN MAE on failures: {results_df['crnn_error'].mean():.1f}° ± {results_df['crnn_error'].std():.1f}°")
    print(f"  SRP MAE on failures:  {results_df['srp_error'].mean():.1f}° ± {results_df['srp_error'].std():.1f}°")
    print(f"  Average improvement:   {results_df['improvement'].mean():.1f}° ± {results_df['improvement'].std():.1f}°")

    # Best improvements
    print(f"\nBest SRP Improvements:")
    best_improvements = results_df[results_df['improvement'] > 0].nlargest(5, 'improvement')
    if len(best_improvements) > 0:
        for _, row in best_improvements.iterrows():
            print(f"  GT={row['gt_angle']:.1f}°: CRNN={row['crnn_error']:.1f}°, SRP={row['srp_error']:.1f}° (improved by {row['improvement']:.1f}°)")

    # Worst cases where SRP failed
    print(f"\nWorst SRP Performance:")
    worst_srp = results_df[results_df['improvement'] < 0].nsmallest(3, 'improvement')
    if len(worst_srp) > 0:
        for _, row in worst_srp.iterrows():
            print(f"  GT={row['gt_angle']:.1f}°: CRNN={row['crnn_error']:.1f}°, SRP={row['srp_error']:.1f}° (worse by {abs(row['improvement']):.1f}°)")

    # Save results
    results_df.to_csv("srp_vs_crnn_on_failures.csv", index=False)
    print(f"\nResults saved to: srp_vs_crnn_on_failures.csv")

    return results_df

def main():
    test_srp_on_failures()

if __name__ == "__main__":
    main()