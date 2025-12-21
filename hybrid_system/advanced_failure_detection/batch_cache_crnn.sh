#!/bin/bash
#
# Batch cache CRNN predictions for 30 different microphone configurations
#
# This script runs CRNN inference for multiple configs and saves all outputs:
# - Predicted angles
# - Logits (pre-sigmoid) for OOD methods
# - Penultimate features for distance-based OOD
# - Probabilities (post-sigmoid) for ConfidNet, max prob, etc.
# - Ground truth and errors

cd /Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection

echo "=========================================="
echo "Batch CRNN Caching for 30 Configurations"
echo "=========================================="
echo ""

# Define all 30 configurations
# Format: "mic_id1,mic_id2,...,mic_id9"
# IMPORTANT: CRNN expects mic 0 LAST (training default)

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
    OUTPUT_FILE="features/crnn_results_mics_${MIC_STR}.pkl"

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

    # Run CRNN inference with custom mic config
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
import torch

# CRNN imports (CRNN.py is in SSL directory)
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/SSL")
from CRNN import CRNN

# SSL imports
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/SSL")
from utils_ import audiowu_high_array_geometry

# Paths
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
CHECKPOINT_PATH = "/Users/danieltoberman/Documents/git/Thesis/08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
OUTPUT_PATH = Path("$OUTPUT_FILE")

# Configuration
MIC_ORDER_CRNN = [${MIC_CONFIG}]

# CRNN preprocessing parameters
CRNN_CONFIG = {
    'win_len': 512,
    'nfft': 512,
    'win_shift_ratio': 0.625,
    'fre_range_used': range(1, 257),  # 1 to 256
    'eps': 1e-6,
}

def load_crnn_model(checkpoint_path, device='cpu'):
    """Load trained CRNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CRNN()

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', '').replace('arch.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_audio_multichannel(example_path, use_mic_id, fs=16000):
    base_path = example_path.replace('_CH0.wav', '')
    signals = []
    for mic_id in use_mic_id:
        ch_path = f"{base_path}_CH{mic_id}.wav"
        if not os.path.exists(ch_path):
            raise FileNotFoundError(f"Missing: {ch_path}")
        sig, file_fs = sf.read(ch_path, dtype="float32")
        if file_fs != fs:
            sig = signal.resample(sig, int(len(sig) * fs / file_fs))
        signals.append(sig)
    return np.array(signals, dtype=np.float32)

def run_crnn_on_case(model, audio_path, mic_order, device='cpu'):
    """Run CRNN inference and return all outputs."""
    try:
        audio_signals = load_audio_multichannel(audio_path, mic_order)
    except FileNotFoundError as e:
        return None

    # Preprocessing
    win_len = CRNN_CONFIG['win_len']
    nfft = CRNN_CONFIG['nfft']
    win_shift = int(win_len * CRNN_CONFIG['win_shift_ratio'])
    eps = CRNN_CONFIG['eps']

    # Convert to tensor
    mic_sig = torch.from_numpy(audio_signals.T).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # STFT
        window = torch.hann_window(window_length=win_len, device=device)
        nb, nsample, nch = mic_sig.shape
        nf = int(nfft / 2) + 1
        nt = int(np.floor(nsample / win_shift + 1))
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64, device=device)

        for ch_idx in range(nch):
            stft[:, :, :, ch_idx] = torch.stft(
                mic_sig[:, :, ch_idx],
                n_fft=nfft,
                hop_length=win_shift,
                win_length=win_len,
                window=window,
                center=True,
                normalized=False,
                return_complex=True
            )

        # Permute and normalize
        stft = stft.permute(0, 3, 1, 2)
        mag = torch.abs(stft)
        mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
        mean_value = mean_value[:, None, None, None].expand(mag.shape)

        stft_real = torch.real(stft) / (mean_value + eps)
        stft_imag = torch.imag(stft) / (mean_value + eps)
        real_image_batch = torch.cat((stft_real, stft_imag), dim=1)

        # Select frequency range
        x = real_image_batch[:, :, CRNN_CONFIG['fre_range_used'], :]

        # Extract features and predictions
        logits_pre_sig, penultimate_features = model.forward_with_intermediates(x)
        predictions = torch.sigmoid(logits_pre_sig)

        # Time-averaged prediction
        avg_prediction = predictions.mean(dim=1).squeeze(0)
        predicted_angle = torch.argmax(avg_prediction).item()

        # Get max probability (confidence)
        max_prob = torch.max(avg_prediction).item()

        return {
            'logits_pre_sig': logits_pre_sig.squeeze(0).cpu().numpy(),  # (T, 360)
            'penultimate_features': penultimate_features.squeeze(0).cpu().numpy(),  # (T, 256)
            'predictions': predictions.squeeze(0).cpu().numpy(),  # (T, 360)
            'avg_prediction': avg_prediction.cpu().numpy(),  # (360,)
            'predicted_angle': predicted_angle,
            'max_prob': max_prob,
        }

# Load model
print("Loading CRNN model...")
# Try to use MPS (Apple Silicon GPU) if available, otherwise use CPU
if torch.backends.mps.is_available():
    device = 'mps'
    print("Using MPS (Apple GPU) for acceleration")
elif torch.cuda.is_available():
    device = 'cuda'
    print("Using CUDA GPU for acceleration")
else:
    device = 'cpu'
    print("Using CPU (no GPU available)")

model = load_crnn_model(CHECKPOINT_PATH, device=device)
print(f"Model loaded on {device}")

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Processing {len(df)} samples...")

# Run CRNN on all samples
crnn_results = []

for idx in tqdm(range(len(df)), desc="Running CRNN"):
    row = df.iloc[idx]
    audio_filename = row['filename']
    audio_path = os.path.join(str(DATA_ROOT), audio_filename.replace('.flac', '_CH0.wav'))
    gt_angle = row['angle(°)']

    outputs = run_crnn_on_case(model, audio_path, MIC_ORDER_CRNN, device=device)

    if outputs is None:
        crnn_error = None
        crnn_pred = None
        max_prob = None
        logits_pre_sig = None
        penultimate_features = None
        predictions = None
        avg_prediction = None
    else:
        crnn_pred = outputs['predicted_angle']
        max_prob = outputs['max_prob']
        logits_pre_sig = outputs['logits_pre_sig']
        penultimate_features = outputs['penultimate_features']
        predictions = outputs['predictions']
        avg_prediction = outputs['avg_prediction']

        # Calculate error
        angle_diff = abs(crnn_pred - gt_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        crnn_error = angle_diff

    crnn_results.append({
        'sample_idx': idx,
        'global_idx': idx,
        'gt_angle': gt_angle,
        'crnn_pred': crnn_pred,
        'crnn_error': crnn_error,
        'max_prob': max_prob,
        'logits_pre_sig': logits_pre_sig,
        'penultimate_features': penultimate_features,
        'predictions': predictions,
        'avg_prediction': avg_prediction,
    })

# Save results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(crnn_results, f)

# Print statistics
valid_crnn = [r for r in crnn_results if r['crnn_pred'] is not None]
if valid_crnn:
    errors = [r['crnn_error'] for r in valid_crnn]
    mae = np.mean(errors)
    success_rate = sum(1 for e in errors if e <= 5) / len(errors) * 100
    avg_max_prob = np.mean([r['max_prob'] for r in valid_crnn])
    print(f"\\nMAE: {mae:.2f}°")
    print(f"Success rate (≤5°): {success_rate:.1f}%")
    print(f"Avg max prob: {avg_max_prob:.4f}")
    print(f"Valid: {len(valid_crnn)}/{len(crnn_results)}")
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
echo "Batch CRNN caching complete!"
echo "Results in: features/crnn_results_mics_*.pkl"
echo "=========================================="
