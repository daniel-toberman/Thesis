#!/usr/bin/env python3
"""
Test Hybrid CRNN-SRP System on 3x12cm Consecutive Configuration

This script:
1. Loads CRNN predictions for 3x12cm consecutive (mics [9,10,11,4,5,6,7,8,0])
2. Identifies cases with low confidence (max_prob < 0.04) 
3. Runs SRP on those cases with correct mic ordering ([0,9,10,11,4,5,6,7,8])
4. Compares hybrid vs CRNN-only performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
from pathlib import Path
import soundfile as sf
from scipy import signal

# SRP imports
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/xsrpMain")
from xsrp.conventional_srp import ConventionalSrp
from SSL.utils_ import audiowu_high_array_geometry

# Paths
CRNN_RESULTS = Path("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/analysis/geometry_robustness/partial_replacement/crnn_6cm_3x12cm_consecutive_results.csv")
BASE_DIR = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")

# Configuration
CONFIDENCE_THRESHOLD = 0.04  # max_prob < 0.04 → use SRP
MIC_ORDER_CRNN = [9, 10, 11, 4, 5, 6, 7, 8, 0]  # CRNN order (mic 0 last)
MIC_ORDER_SRP = [0, 9, 10, 11, 4, 5, 6, 7, 8]   # SRP order (mic 0 first - reference)

# SRP parameters (optimized from Phase 1)
SRP_CONFIG = {
    'n_dft_bins': 16384,  # 2**14 as user selected
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

    # Get microphone positions (for 9-channel array with specific geometry)
    # Get all 32 mic positions, then select the ones we need (x,y only for 2D)
    all_mic_positions = audiowu_high_array_geometry()  # Returns (32, 3) array
    mic_positions = all_mic_positions[mic_order_srp, :2]  # Select x,y positions for our mics

    # Initialize SRP (matching run_SRP.py pattern)
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

    # Calculate azimuth from estimated vector
    srp_pred = float(np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0)

    # Calculate error (circular distance)
    error = abs((srp_pred - gt_angle + 180) % 360 - 180)

    return srp_pred, error

def main():
    print("="*100)
    print("HYBRID CRNN-SRP SYSTEM TEST: 3x12cm Consecutive Configuration")
    print("="*100)
    
    # Load CRNN predictions
    print(f"\nLoading CRNN results from: {CRNN_RESULTS}")
    df_crnn = pd.read_csv(CRNN_RESULTS)
    print(f"  Total samples: {len(df_crnn)}")
    
    # Load test CSV to get audio paths
    print(f"\nLoading test dataset info from: {CSV_PATH}")
    df_test = pd.read_csv(CSV_PATH)
    print(f"  Total test samples: {len(df_test)}")
    
    # Identify cases for SRP
    df_crnn['use_srp'] = df_crnn['max_prob'] < CONFIDENCE_THRESHOLD
    n_srp = df_crnn['use_srp'].sum()
    n_crnn = (~df_crnn['use_srp']).sum()
    
    print(f"\n{'='*100}")
    print(f"ROUTING DECISION (Threshold: max_prob < {CONFIDENCE_THRESHOLD}):")
    print(f"{'='*100}")
    print(f"  Cases for SRP:  {n_srp:>4} ({n_srp/len(df_crnn)*100:>5.1f}%)")
    print(f"  Cases for CRNN: {n_crnn:>4} ({n_crnn/len(df_crnn)*100:>5.1f}%)")
    
    # Analyze SRP cases
    srp_cases = df_crnn[df_crnn['use_srp']]
    print(f"\nSRP Cases Breakdown:")
    print(f"  Catastrophic (>30°):  {(srp_cases['abs_error'] > 30).sum():>4} ({(srp_cases['abs_error'] > 30).sum()/len(srp_cases)*100:>5.1f}%)")
    print(f"  Bad (10-30°):         {((srp_cases['abs_error'] >= 10) & (srp_cases['abs_error'] <= 30)).sum():>4} ({((srp_cases['abs_error'] >= 10) & (srp_cases['abs_error'] <= 30)).sum()/len(srp_cases)*100:>5.1f}%)")
    print(f"  Moderate (5-10°):     {((srp_cases['abs_error'] >= 5) & (srp_cases['abs_error'] < 10)).sum():>4} ({((srp_cases['abs_error'] >= 5) & (srp_cases['abs_error'] < 10)).sum()/len(srp_cases)*100:>5.1f}%)")
    print(f"  Good (≤5°):           {(srp_cases['abs_error'] < 5).sum():>4} ({(srp_cases['abs_error'] < 5).sum()/len(srp_cases)*100:>5.1f}%)")
    print(f"  CRNN MAE on these:    {srp_cases['abs_error'].mean():.2f}°")
    
    print(f"\n{'='*100}")
    print(f"RUNNING SRP ON {n_srp} CASES")
    print(f"{'='*100}")
    print(f"Configuration:")
    print(f"  Mic ordering: {MIC_ORDER_SRP} (mic 0 first - reference)")
    print(f"  n_dft_bins:   {SRP_CONFIG['n_dft_bins']}")
    print(f"  freq_range:   {SRP_CONFIG['freq_min']}-{SRP_CONFIG['freq_max']} Hz")
    print(f"  grid_cells:   {SRP_CONFIG['grid_cells']}")
    
    # Run SRP on selected cases
    srp_results = []
    
    for idx, row in srp_cases.iterrows():
        global_idx = int(row['global_idx'])
        
        # Get audio path from test CSV
        if global_idx >= len(df_test):
            print(f"  Warning: global_idx {global_idx} out of range")
            continue
        
        test_row = df_test.iloc[global_idx]

        # Get audio path - filename contains full relative path from BASE_DIR
        # Format: test/ma_noisy_speech/Room/static/P0006/TEST_S_XXX.flac
        # Convert to: BASE_DIR/test/ma_noisy_speech/Room/static/P0006/TEST_S_XXX_CH0.wav
        audio_filename = test_row['filename']
        audio_path = os.path.join(BASE_DIR, audio_filename.replace('.flac', '_CH0.wav'))

        gt_angle = test_row['angle(°)']
        
        if (len(srp_results) + 1) % 50 == 0:
            print(f"  Processing case {len(srp_results)+1}/{n_srp}...")
        
        srp_pred, srp_error = run_srp_on_case(audio_path, MIC_ORDER_SRP, gt_angle)
        
        if srp_pred is not None:
            srp_results.append({
                'global_idx': global_idx,
                'gt_angle': gt_angle,
                'crnn_pred': row['pred_angle'],
                'crnn_error': row['abs_error'],
                'srp_pred': srp_pred,
                'srp_error': srp_error,
                'max_prob': row['max_prob'],
            })
    
    print(f"\n✅ SRP processing complete: {len(srp_results)}/{n_srp} cases successful")
    
    # Create results DataFrame
    df_srp = pd.DataFrame(srp_results)
    
    # Save SRP results
    output_dir = Path("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/analysis/geometry_robustness/partial_replacement")
    srp_output = output_dir / "hybrid_3x12cm_srp_results.csv"
    df_srp.to_csv(srp_output, index=False)
    print(f"\n💾 Saved SRP results to: {srp_output}")
    
    # Calculate hybrid performance
    print(f"\n{'='*100}")
    print(f"HYBRID SYSTEM PERFORMANCE")
    print(f"{'='*100}")
    
    # CRNN-only cases
    crnn_only = df_crnn[~df_crnn['use_srp']]
    crnn_mae = crnn_only['abs_error'].mean()
    crnn_median = crnn_only['abs_error'].median()
    
    # SRP cases
    srp_mae = df_srp['srp_error'].mean()
    srp_median = df_srp['srp_error'].median()
    
    # Hybrid (CRNN for most, SRP for low confidence)
    hybrid_errors = []
    hybrid_errors.extend(crnn_only['abs_error'].values)  # CRNN cases
    hybrid_errors.extend(df_srp['srp_error'].values)      # SRP cases
    hybrid_mae = np.mean(hybrid_errors)
    hybrid_median = np.median(hybrid_errors)
    
    # Overall CRNN-only (for comparison)
    crnn_all_mae = df_crnn['abs_error'].mean()
    crnn_all_median = df_crnn['abs_error'].median()
    
    print(f"\nCRNN-only (all {len(df_crnn)} cases):")
    print(f"  MAE:    {crnn_all_mae:.2f}°")
    print(f"  Median: {crnn_all_median:.2f}°")
    
    print(f"\nCRNN-only ({len(crnn_only)} cases kept on CRNN):")
    print(f"  MAE:    {crnn_mae:.2f}°")
    print(f"  Median: {crnn_median:.2f}°")
    
    print(f"\nSRP-only ({len(df_srp)} cases routed to SRP):")
    print(f"  MAE:    {srp_mae:.2f}°")
    print(f"  Median: {srp_median:.2f}°")
    print(f"  vs CRNN on same cases: {srp_cases['abs_error'].mean():.2f}°")
    print(f"  Improvement: {srp_cases['abs_error'].mean() - srp_mae:.2f}°")
    
    print(f"\n🎯 HYBRID SYSTEM ({len(crnn_only)} CRNN + {len(df_srp)} SRP):")
    print(f"  MAE:    {hybrid_mae:.2f}°")
    print(f"  Median: {hybrid_median:.2f}°")
    print(f"  vs CRNN-only: {crnn_all_mae - hybrid_mae:+.2f}° ({(crnn_all_mae - hybrid_mae)/crnn_all_mae*100:+.1f}%)")
    
    # Success rates
    crnn_success = (crnn_only['abs_error'] <= 5).sum() / len(crnn_only) * 100
    srp_success = (df_srp['srp_error'] <= 5).sum() / len(df_srp) * 100
    hybrid_success = (len(crnn_only[crnn_only['abs_error'] <= 5]) + len(df_srp[df_srp['srp_error'] <= 5])) / len(df_crnn) * 100
    crnn_all_success = (df_crnn['abs_error'] <= 5).sum() / len(df_crnn) * 100
    
    print(f"\nSuccess Rate (≤5°):")
    print(f"  CRNN-only (all):      {crnn_all_success:.1f}%")
    print(f"  CRNN (kept cases):    {crnn_success:.1f}%")
    print(f"  SRP (routed cases):   {srp_success:.1f}%")
    print(f"  Hybrid:               {hybrid_success:.1f}%")
    print(f"  Improvement:          {hybrid_success - crnn_all_success:+.1f}%")
    
    print(f"\n{'='*100}")
    
    return df_srp, hybrid_mae, hybrid_median

if __name__ == "__main__":
    main()
