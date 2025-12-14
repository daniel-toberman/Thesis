"""
Feature Extraction Script for IPDnet - Advanced Failure Detection

Extracts penultimate layer features and predictions from trained IPDnet model.
Adapts the CRNN feature extraction pipeline for IPDnet's IPD-based architecture.

Usage:
    python extract_ipdnet_features.py --split train --array_config 6cm
    python extract_ipdnet_features.py --split test --array_config 3x12cm_consecutive
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
from SingleTinyIPDnet import SingleTinyIPDnet
from utils_ import audiowu_high_array_geometry
import Module as at_module

# Configuration
CHECKPOINT_PATH = "/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/version_*/checkpoints/best*.ckpt"  # Update after training
DATA_ROOT = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
CSV_ROOT = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08"

# Microphone configurations - IPDNET expects mic 0 FIRST (reference mic)
MIC_CONFIGS = {
    '6cm': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Training geometry - mic 0 first (reference)
    '3x12cm_consecutive': [0, 9, 10, 11, 4, 5, 6, 7, 8],  # 3 mics replaced - mic 0 first
    '12cm': [0] + list(range(9, 17)),  # Full 12cm array - mic 0 first
    '18cm': [0] + list(range(17, 25)),  # Full 18cm array - mic 0 first
}


def load_ipdnet_model(checkpoint_path, device='mps'):
    """Load trained IPDnet model from checkpoint."""
    print(f"Loading IPDnet model from: {checkpoint_path}")

    # Handle wildcard paths
    if '*' in checkpoint_path:
        import glob
        matching_files = glob.glob(checkpoint_path)
        if not matching_files:
            raise FileNotFoundError(f"No checkpoint found matching: {checkpoint_path}")
        checkpoint_path = matching_files[0]
        print(f"Found checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model with correct input size (16 for 9 mics: 8 pairs * 2 for real+imag)
    model = SingleTinyIPDnet(input_size=16, hidden_size=128)

    # Load state dict (handle Lightning checkpoint format)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'arch.' prefix if present
        state_dict = {k.replace('arch.', ''): v for k, v in state_dict.items()}
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

    Args:
        audio_path: Path to audio file
        start: Start sample (unused - files are pre-segmented)
        end: End sample (unused)
        mic_ids: List of microphone IDs to load
        sr: Sample rate

    Returns:
        signals: np.array of shape (n_mics, n_samples)
    """
    if audio_path.endswith('.flac'):
        audio_path = audio_path.replace('.flac', '')

    base_path = audio_path.replace('.wav', '')

    signals = []
    for mic_id in mic_ids:
        mic_path = f"{base_path}_CH{mic_id}.wav"

        if not os.path.exists(mic_path):
            raise FileNotFoundError(f"Audio file not found: {mic_path}")

        signal, file_sr = sf.read(mic_path, dtype='float32')

        if file_sr != sr:
            raise ValueError(f"Expected sample rate {sr}, got {file_sr} for {mic_path}")

        signals.append(signal)

    return np.stack(signals, axis=0)  # (n_mics, n_samples)


def compute_ipd_features(audio_signals, sr=16000, n_fft=512, hop_length=160):
    """
    Compute IPD (Inter-channel Phase Difference) features from multi-channel audio.

    Args:
        audio_signals: (n_mics, n_samples) numpy array
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT

    Returns:
        ipd_features: (16, n_freq, n_frames) for 9 mics (8 pairs * 2 for real+imag)
    """
    # Convert to torch
    signals = torch.from_numpy(audio_signals).float()

    # Compute STFT for all channels
    stft = torch.stft(
        signals,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        center=True,
        normalized=False,
        return_complex=True
    )  # (n_mics, n_freq, n_frames)

    # Reference microphone is first (mic 0)
    ref_stft = stft[0:1, :, :]  # (1, n_freq, n_frames)
    other_stft = stft[1:, :, :]  # (n_mics-1, n_freq, n_frames)

    # Compute IPD: phase difference between reference and other mics
    ipd_complex = ref_stft * torch.conj(other_stft)  # (n_mics-1, n_freq, n_frames)

    # Normalize by magnitude
    mag = torch.abs(stft)
    mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
    mean_value = mean_value[:, None, None].expand(mag.shape)

    # Extract real and imaginary parts, normalized
    ipd_real = torch.real(ipd_complex) / (mean_value[1:, :, :] + 1e-8)
    ipd_imag = torch.imag(ipd_complex) / (mean_value[1:, :, :] + 1e-8)

    # Concatenate: (2*(n_mics-1), n_freq, n_frames)
    ipd_features = torch.cat((ipd_real, ipd_imag), dim=0)

    return ipd_features


def extract_features_for_sample(model, audio_signals, device='mps'):
    """
    Extract features for a single audio sample using IPDnet.

    Args:
        model: IPDnet model
        audio_signals: (n_mics, n_samples) numpy array
        device: Device to run on

    Returns:
        dict with extracted features
    """
    model.eval()

    with torch.no_grad():
        # 1. Compute IPD features
        ipd_features = compute_ipd_features(audio_signals)  # (16, n_freq, n_frames)

        # 2. Select frequency range (256 bins used in training)
        fre_range_used = range(1, 257)
        ipd_input = ipd_features[:, fre_range_used, :].unsqueeze(0)  # (1, 16, 256, n_frames)

        # 3. Move to device
        ipd_input = ipd_input.to(device)

        # 4. Extract features using forward_with_intermediates
        ipd_predictions, penultimate_features = model.forward_with_intermediates(ipd_input)

        # ipd_predictions: (1, time//5, 1, 512, features)
        # penultimate_features: (1, time//5, feature_dim) - aggregated over frequency

        # 5. Convert IPD to DOA angle (placeholder - needs PredDOA module)
        # For now, use a simple placeholder prediction
        # TODO: Implement proper IPD->DOA conversion using at_module.PredDOA
        predicted_angle = 0.0  # Placeholder

    return {
        'logits_pre_sig': ipd_predictions.squeeze(0).cpu().numpy(),  # (T//5, 1, 512, features)
        'penultimate_features': penultimate_features.squeeze(0).cpu().numpy(),  # (T//5, feature_dim)
        'predictions': ipd_predictions.squeeze(0).cpu().numpy(),  # Same as logits for IPDnet
        'avg_prediction': penultimate_features.mean(dim=1).squeeze(0).cpu().numpy(),  # (feature_dim,)
        'predicted_angle': predicted_angle
    }


def process_dataset(model, csv_path, array_config, split='train', device='mps', max_samples=None):
    """
    Process entire dataset and extract features.

    Returns:
        dict with extracted features for all samples
    """
    print(f"\nProcessing {split} set with {array_config} configuration...")

    # Load CSV
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    print(f"Found {len(df)} samples in CSV")

    # Get microphone IDs
    mic_ids = MIC_CONFIGS[array_config]
    print(f"Using microphone configuration: {mic_ids}")

    # Initialize storage
    all_features = {
        'penultimate_features': [],
        'logits_pre_sig': [],
        'predictions': [],
        'avg_predictions': [],
        'predicted_angles': [],
        'gt_angles': [],
        'global_indices': [],
        'filenames': []
    }

    errors = []

    # Process each sample
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting features"):
        try:
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
            if idx < 3:
                print(f"\nError processing {idx}: {filename}")
                print(f"Error: {e}")
            continue

    print(f"\nSuccessfully processed {len(all_features['predicted_angles'])} samples")
    if errors:
        print(f"Failed to process {len(errors)} samples")

    # Convert to numpy arrays
    all_features['predicted_angles'] = np.array(all_features['predicted_angles'])
    all_features['gt_angles'] = np.array(all_features['gt_angles'])
    all_features['global_indices'] = np.array(all_features['global_indices'])
    all_features['avg_predictions'] = np.array(all_features['avg_predictions'])

    # Variable-length features as object arrays
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
    print("\nNote: IPD→DOA conversion is placeholder. Actual performance will be computed after implementing PredDOA conversion.")

    return all_features


def main():
    parser = argparse.ArgumentParser(description='Extract IPDnet features')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test', 'val'])
    parser.add_argument('--array_config', type=str, required=True, choices=list(MIC_CONFIGS.keys()))
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    print("=" * 80)
    print(f"IPDnet Feature Extraction - {args.split.upper()} SET")
    print("=" * 80)

    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model = load_ipdnet_model(args.checkpoint, device=device)

    # Get CSV path
    csv_path = os.path.join(CSV_ROOT, args.split, f"{args.split}_static_source_location_08.csv")

    # Extract features
    features = process_dataset(
        model, csv_path, args.array_config, args.split, device, args.max_samples
    )

    # Save features
    output_dir = Path(__file__).parent / 'features' / 'ipdnet'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.split}_{args.array_config}_ipdnet_features.npz"
    np.savez_compressed(output_file, **features)

    print(f"\n✓ Features saved to: {output_file}")


if __name__ == "__main__":
    main()
