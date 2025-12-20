#!/bin/bash
#
# Batch cache SRP results for 30 different microphone configurations
#
# This script runs precompute_srp.py-style computations for multiple configs
# Results are saved with filenames reflecting the actual mic IDs used

cd /Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection

echo "=========================================="
echo "Batch SRP Caching for 30 Configurations"
echo "=========================================="
echo ""

# Define all 30 configurations
# Format: "mic_id1,mic_id2,...,mic_id9"

# 2 mics from 12cm (10 configs)
CONFIGS_2MIC=(
    "9,10,3,4,5,6,7,8,0"
    "1,10,11,4,5,6,7,8,0"
    "1,2,11,12,5,6,7,8,0"
    "9,2,3,4,13,6,7,8,0"
    "1,10,3,4,5,14,7,8,0"
    "9,2,11,4,5,6,7,8,0"
    "1,10,3,12,5,6,7,8,0"
    "1,2,11,4,13,6,7,8,0"
    "1,2,3,12,5,14,7,8,0"
    "1,2,3,4,13,6,15,8,0"
)

# 3 mics from 12cm (10 configs)
CONFIGS_3MIC=(
    "9,10,11,4,5,6,7,8,0"
    "1,10,11,12,5,6,7,8,0"
    "1,2,11,12,13,6,7,8,0"
    "9,2,3,12,5,6,15,8,0"
    "1,10,3,4,13,6,7,16,0"
    "9,2,11,4,5,14,7,8,0"
    "1,10,3,12,5,6,15,8,0"
    "9,2,3,4,13,6,15,8,0"
    "1,10,11,4,5,6,15,8,0"
    "9,10,3,12,5,6,7,16,0"
)

# 4 mics from 12cm (10 configs)
CONFIGS_4MIC=(
    "9,10,11,12,5,6,7,8,0"
    "1,10,11,12,13,6,7,8,0"
    "1,2,11,12,13,14,7,8,0"
    "9,2,11,4,13,6,15,8,0"
    "1,10,3,12,5,14,7,16,0"
    "9,10,11,4,5,14,15,8,0"
    "9,10,3,12,13,6,7,8,0"
    "1,10,11,4,13,14,7,8,0"
    "9,2,11,12,5,6,15,8,0"
    "1,10,3,4,13,14,15,8,0"
)

# Combine all configs
ALL_CONFIGS=("${CONFIGS_2MIC[@]}" "${CONFIGS_3MIC[@]}" "${CONFIGS_4MIC[@]}")

# Process each configuration
TOTAL=${#ALL_CONFIGS[@]}
COUNT=0

for MIC_CONFIG in "${ALL_CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))

    # Create filename with mic IDs
    MIC_STR=$(echo "$MIC_CONFIG" | tr ',' '_')
    OUTPUT_FILE="features/srp_results_mics_${MIC_STR}.pkl"

    # Skip if already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$COUNT/$TOTAL] Skipping (exists): $MIC_CONFIG"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$COUNT/$TOTAL] Processing: $MIC_CONFIG"
    echo "Output: $OUTPUT_FILE"
    echo "=========================================="

    # Run precompute_srp.py with custom mic config
    python3 - <<EOF
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
OUTPUT_PATH = Path("$OUTPUT_FILE")

# Configuration
MIC_ORDER_SRP = [${MIC_CONFIG}]

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
    base_path = example_path.replace('_CH0.wav', '')
    signals = []
    for mic_id in use_mic_id:
        ch_path = f"{base_path}_CH{mic_id}.wav"
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f"Missing: {ch_path}")
        sig, file_fs = sf.read(ch_path, dtype="float64")
        if file_fs != fs:
            sig = signal.resample(sig, int(len(sig) * fs / file_fs))
        signals.append(sig)
    return np.array(signals)

def run_srp_on_case(audio_path, mic_order_srp, gt_angle):
    try:
        audio = load_audio_multichannel(audio_path, mic_order_srp)
    except FileNotFoundError as e:
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

    est_vec, srp_map, grid = srp.forward(audio)
    srp_pred = float(np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0)

    angle_diff = abs(srp_pred - gt_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return srp_pred, angle_diff

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Processing {len(df)} samples...")

# Pre-compute SRP for all samples
srp_results = []

for idx in tqdm(range(len(df)), desc="Running SRP"):
    row = df.iloc[idx]
    audio_filename = row['filename']
    audio_path = os.path.join(str(DATA_ROOT), audio_filename.replace('.flac', '_CH0.wav'))
    gt_angle = row['angle(°)']

    srp_pred, srp_error = run_srp_on_case(audio_path, MIC_ORDER_SRP, gt_angle)

    srp_results.append({
        'sample_idx': idx,
        'global_idx': idx,
        'gt_angle': gt_angle,
        'srp_pred': srp_pred,
        'srp_error': srp_error
    })

# Save results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(srp_results, f)

# Print statistics
valid_srp = [r for r in srp_results if r['srp_pred'] is not None]
if valid_srp:
    errors = [r['srp_error'] for r in valid_srp]
    mae = np.mean(errors)
    success_rate = sum(1 for e in errors if e <= 5) / len(errors) * 100
    print(f"\nMAE: {mae:.2f}°")
    print(f"Success rate (≤5°): {success_rate:.1f}%")
    print(f"Valid: {len(valid_srp)}/{len(srp_results)}")
    print(f"Saved to: {OUTPUT_PATH}")
EOF

    if [ $? -eq 0 ]; then
        echo "✓ Success: $MIC_CONFIG"
    else
        echo "✗ Failed: $MIC_CONFIG"
    fi
done

echo ""
echo "=========================================="
echo "Batch SRP caching complete!"
echo "Results in: features/srp_results_mics_*.pkl"
echo "=========================================="
