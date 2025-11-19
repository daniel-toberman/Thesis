"""
Feature Extraction Script for Advanced Failure Detection

Extracts penultimate layer features and predictions from trained CRNN model
for multiple microphone array configurations.

Usage:
    python extract_features.py --split train --array_config 6cm
    python extract_features.py --split test --array_config 3x12cm_consecutive
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm

# Add SSL directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'SSL'))
from utils_ import audiowu_high_array_geometry

# Add CRNN directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '08_CRNN'))
from CRNN import CRNN

# Configuration
CHECKPOINT_PATH = "/Users/danieltoberman/Documents/git/Thesis/08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
DATA_ROOT = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
CSV_ROOT = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08"

# Microphone configurations
# IMPORTANT: CRNN expects mic 0 to be LAST (training default: [1,2,3,4,5,6,7,8,0])
# SRP expects mic 0 to be FIRST (reference mic)
MIC_CONFIGS = {
    '6cm': [1, 2, 3, 4, 5, 6, 7, 8, 0],  # Training geometry - CRNN order (mic 0 last)
    '3x12cm_consecutive': [9, 10, 11, 4, 5, 6, 7, 8, 0],  # 3 mics replaced - CRNN order (mic 0 last)
    '12cm': list(range(9, 17)) + [0],  # Full 12cm array - CRNN order (mic 0 last)
    '18cm': list(range(17, 25)) + [0],  # Full 18cm array - CRNN order (mic 0 last)
    '1x12cm_pos1': [9, 2, 3, 4, 5, 6, 7, 8, 0],  # Single mic replaced - CRNN order (mic 0 last)
    '2x12cm_opposite': [9, 2, 3, 12, 5, 6, 7, 8, 0],  # Two opposite mics - CRNN order (mic 0 last)
}


def load_crnn_model(checkpoint_path, device='mps'):
    """Load trained CRNN model from checkpoint."""
    print(f"Loading CRNN model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = CRNN()

    # Load state dict (handle Lightning checkpoint format)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' or 'arch.' prefix if present
        state_dict = {k.replace('model.', '').replace('arch.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on device: {device}")
    return model


def load_multichannel_audio(audio_path, start, end, mic_ids, sr=16000):
    """
    Load multichannel audio for specific microphone configuration.

    NOTE: The start/end parameters are not used because the audio files
    are already pre-segmented. The CSV indices refer to the master recording.

    Args:
        audio_path: Path to audio file (will be modified to add _CH{i}.wav)
        start: Start sample (unused - for API compatibility)
        end: End sample (unused - for API compatibility)
        mic_ids: List of microphone IDs to load
        sr: Sample rate

    Returns:
        signals: np.array of shape (n_mics, n_samples)
    """
    # Convert .flac to .wav format if needed
    if audio_path.endswith('.flac'):
        audio_path = audio_path.replace('.flac', '')

    base_path = audio_path.replace('.wav', '')

    signals = []
    for mic_id in mic_ids:
        mic_path = f"{base_path}_CH{mic_id}.wav"

        if not os.path.exists(mic_path):
            raise FileNotFoundError(f"Audio file not found: {mic_path}")

        # Load entire audio file (already pre-segmented)
        signal, fs = sf.read(mic_path, dtype='float32')

        # Resample if needed
        if fs != sr:
            from scipy import signal as scipy_signal
            signal = scipy_signal.resample(signal, int(len(signal) * sr / fs))

        signals.append(signal)

    return np.array(signals, dtype=np.float32)


def extract_features_for_sample(model, audio_signals, device='mps', sr=16000):
    """
    Extract features and predictions from CRNN model.

    Args:
        model: CRNN model
        audio_signals: np.array of shape (n_mics, n_samples)
        device: Device to run on
        sr: Sample rate

    Returns:
        dict with:
            - logits_pre_sig: (T, 360) logits before sigmoid
            - penultimate_features: (T, 256) features from penultimate layer
            - predictions: (T, 360) final predictions after sigmoid
            - predicted_angle: scalar, predicted angle in degrees
    """
    # CRNN preprocessing parameters (from run_CRNN.py)
    win_len = 512
    nfft = 512
    win_shift_ratio = 0.625
    fre_range_used = range(1, int(nfft / 2) + 1, 1)  # 1 to 256
    eps = 1e-6

    # Convert to tensor: (n_mics, n_samples) -> (1, n_samples, n_mics)
    mic_sig = torch.from_numpy(audio_signals.T).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # 1. STFT
        win_shift = int(win_len * win_shift_ratio)
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

        # 2. Permute to (batch, channels, freq, time)
        stft = stft.permute(0, 3, 1, 2)
        nb, nc, nf, nt = stft.shape

        # 3. Normalize by mean magnitude
        mag = torch.abs(stft)
        mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
        mean_value = mean_value[:, None, None, None].expand(mag.shape)

        # 4. Split into real and imaginary parts, normalize, and concatenate
        stft_real = torch.real(stft) / (mean_value + eps)
        stft_imag = torch.imag(stft) / (mean_value + eps)
        real_image_batch = torch.cat((stft_real, stft_imag), dim=1)  # (1, 18, nf, nt)

        # 5. Select frequency range
        x = real_image_batch[:, :, fre_range_used, :]  # (1, 18, 256, nt)

        # 6. Extract features using forward_with_intermediates
        logits_pre_sig, penultimate_features = model.forward_with_intermediates(x)

        # Get final predictions
        predictions = torch.sigmoid(logits_pre_sig)

        # Average over time dimension
        avg_prediction = predictions.mean(dim=1).squeeze(0)  # (360,)

        # Get predicted angle
        predicted_angle = torch.argmax(avg_prediction).item()

    return {
        'logits_pre_sig': logits_pre_sig.squeeze(0).cpu().numpy(),  # (T, 360)
        'penultimate_features': penultimate_features.squeeze(0).cpu().numpy(),  # (T, 256)
        'predictions': predictions.squeeze(0).cpu().numpy(),  # (T, 360)
        'avg_prediction': avg_prediction.cpu().numpy(),  # (360,)
        'predicted_angle': predicted_angle
    }


def process_dataset(model, csv_path, array_config, split='train', device='mps', max_samples=None):
    """
    Process entire dataset and extract features.

    Args:
        model: CRNN model
        csv_path: Path to CSV file
        array_config: Microphone configuration name
        split: 'train' or 'test'
        device: Device to run on
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        dict with extracted features for all samples
    """
    print(f"\nProcessing {split} set with {array_config} configuration...")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from CSV")

    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples for testing")

    # Get microphone IDs
    mic_ids = MIC_CONFIGS[array_config]
    print(f"Using microphones: {mic_ids}")

    # Storage for features
    all_features = {
        'penultimate_features': [],  # (N, T, 256)
        'logits_pre_sig': [],  # (N, T, 360)
        'predictions': [],  # (N, T, 360)
        'avg_predictions': [],  # (N, 360)
        'predicted_angles': [],  # (N,)
        'gt_angles': [],  # (N,)
        'global_indices': [],  # (N,)
        'filenames': [],  # (N,)
    }

    errors = []

    # Process each sample
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting features"):
        try:
            # Get file info
            filename = row['filename']
            start_sample = int(row['real_st'])
            end_sample = int(row['real_ed'])
            gt_angle = float(row['angle(°)'])

            # Construct full path
            audio_path = os.path.join(DATA_ROOT, filename)

            # Load audio
            audio_signals = load_multichannel_audio(
                audio_path, start_sample, end_sample, mic_ids
            )

            # Extract features
            features = extract_features_for_sample(model, audio_signals, device)

            # Store results
            all_features['penultimate_features'].append(features['penultimate_features'])
            all_features['logits_pre_sig'].append(features['logits_pre_sig'])
            all_features['predictions'].append(features['predictions'])
            all_features['avg_predictions'].append(features['avg_prediction'])
            all_features['predicted_angles'].append(features['predicted_angle'])
            all_features['gt_angles'].append(gt_angle)
            all_features['global_indices'].append(idx)
            all_features['filenames'].append(filename)

        except Exception as e:
            import traceback
            errors.append((idx, filename, str(e), traceback.format_exc()))
            if idx < 3:  # Print detailed error for first 3 failures
                print(f"\nError processing {idx}: {filename}")
                print(f"Error: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
            continue

    print(f"\nSuccessfully processed {len(all_features['predicted_angles'])} samples")
    if errors:
        print(f"Failed to process {len(errors)} samples")

    # Convert lists to numpy arrays for storage efficiency
    all_features['predicted_angles'] = np.array(all_features['predicted_angles'])
    all_features['gt_angles'] = np.array(all_features['gt_angles'])
    all_features['global_indices'] = np.array(all_features['global_indices'])
    all_features['avg_predictions'] = np.array(all_features['avg_predictions'])  # (N, 360)

    # Store variable-length features as object arrays
    all_features['penultimate_features'] = np.array(all_features['penultimate_features'], dtype=object)
    all_features['logits_pre_sig'] = np.array(all_features['logits_pre_sig'], dtype=object)
    all_features['predictions'] = np.array(all_features['predictions'], dtype=object)

    # Compute errors
    errors_deg = np.abs((all_features['predicted_angles'] - all_features['gt_angles'] + 180) % 360 - 180)
    all_features['abs_errors'] = errors_deg

    print(f"\nPerformance Summary:")
    print(f"  MAE: {errors_deg.mean():.2f}°")
    print(f"  Median: {np.median(errors_deg):.2f}°")
    print(f"  Success (≤5°): {(errors_deg <= 5).sum()} / {len(errors_deg)} ({(errors_deg <= 5).mean() * 100:.1f}%)")

    return all_features


def save_features(features, output_path):
    """Save extracted features to disk."""
    print(f"\nSaving features to: {output_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy archive
    np.savez_compressed(
        output_path,
        **features
    )

    print(f"Features saved successfully")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract features from trained CRNN")
    parser.add_argument('--split', type=str, choices=['train', 'test'], required=True,
                        help='Dataset split to process')
    parser.add_argument('--array_config', type=str,
                        choices=list(MIC_CONFIGS.keys()), required=True,
                        help='Microphone array configuration')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features',
                        help='Output directory for features')

    args = parser.parse_args()

    # Setup paths
    csv_path = f"{CSV_ROOT}/{args.split}/{args.split}_static_source_location_08.csv"
    output_path = Path(args.output_dir) / f"{args.split}_{args.array_config}_features.npz"

    print("="*80)
    print("CRNN Feature Extraction")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Array config: {args.array_config}")
    print(f"CSV path: {csv_path}")
    print(f"Output path: {output_path}")
    print(f"Device: {args.device}")

    # Check if output already exists
    if output_path.exists():
        response = input(f"\nOutput file already exists: {output_path}\nOverwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborting.")
            return

    # Load model
    model = load_crnn_model(CHECKPOINT_PATH, device=args.device)

    # Process dataset
    features = process_dataset(
        model, csv_path, args.array_config,
        split=args.split, device=args.device,
        max_samples=args.max_samples
    )

    # Save features
    save_features(features, output_path)

    print("\n" + "="*80)
    print("Feature extraction complete!")
    print("="*80)


if __name__ == "__main__":
    main()
