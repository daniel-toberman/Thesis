"""
Validation Script for Cached CRNN and SRP Results

Tests cached predictions against fresh inference to ensure accuracy.
Compares:
- Predicted angles
- Errors
- Features (CRNN only)
- Logits and predictions (CRNN only)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'SSL'))
from utils_ import audiowu_high_array_geometry

import numpy as np
import pandas as pd
import pickle
import torch
import soundfile as sf
from scipy import signal
from pathlib import Path
import argparse

# CRNN imports
from CRNN import CRNN

# SRP imports
sys.path.append("/Users/danieltoberman/Documents/git/Thesis/xsrpMain")
from xsrp.conventional_srp import ConventionalSrp

# Paths
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
CRNN_CHECKPOINT = "/Users/danieltoberman/Documents/git/Thesis/08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"

# CRNN config
CRNN_CONFIG = {
    'win_len': 512,
    'nfft': 512,
    'win_shift_ratio': 0.625,
    'fre_range_used': range(1, 257),
    'eps': 1e-6,
}

# SRP config
SRP_CONFIG = {
    'n_dft_bins': 16384,
    'freq_min': 300,
    'freq_max': 4000,
    'grid_cells': 360,
    'mode': 'gcc_phat_freq',
    'n_avg_samples': 1,
}


def load_crnn_model(checkpoint_path, device='mps'):
    """Load CRNN model."""
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
    """Load multichannel audio."""
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


def run_crnn_inference(model, audio_path, mic_order, device='mps'):
    """Run fresh CRNN inference."""
    try:
        audio_signals = load_audio_multichannel(audio_path, mic_order)
    except FileNotFoundError:
        return None

    # Preprocessing
    win_len = CRNN_CONFIG['win_len']
    nfft = CRNN_CONFIG['nfft']
    win_shift = int(win_len * CRNN_CONFIG['win_shift_ratio'])
    eps = CRNN_CONFIG['eps']

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

        # Normalize
        stft = stft.permute(0, 3, 1, 2)
        mag = torch.abs(stft)
        mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
        mean_value = mean_value[:, None, None, None].expand(mag.shape)

        stft_real = torch.real(stft) / (mean_value + eps)
        stft_imag = torch.imag(stft) / (mean_value + eps)
        real_image_batch = torch.cat((stft_real, stft_imag), dim=1)
        x = real_image_batch[:, :, CRNN_CONFIG['fre_range_used'], :]

        # Forward pass
        logits_pre_sig, penultimate_features = model.forward_with_intermediates(x)
        predictions = torch.sigmoid(logits_pre_sig)
        avg_prediction = predictions.mean(dim=1).squeeze(0)
        predicted_angle = torch.argmax(avg_prediction).item()
        max_prob = torch.max(avg_prediction).item()

        return {
            'logits_pre_sig': logits_pre_sig.squeeze(0).cpu().numpy(),
            'penultimate_features': penultimate_features.squeeze(0).cpu().numpy(),
            'predictions': predictions.squeeze(0).cpu().numpy(),
            'avg_prediction': avg_prediction.cpu().numpy(),
            'predicted_angle': predicted_angle,
            'max_prob': max_prob,
        }


def run_srp_inference(audio_path, mic_order, gt_angle):
    """Run fresh SRP inference."""
    try:
        audio = load_audio_multichannel(audio_path, mic_order)
    except FileNotFoundError:
        return None, None

    all_mic_positions = audiowu_high_array_geometry()
    mic_positions = all_mic_positions[mic_order, :2]

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


def validate_crnn_cache(cache_file, num_samples=10, device='mps'):
    """Validate CRNN cached results against fresh inference."""
    print(f"\n{'='*60}")
    print(f"Validating CRNN Cache: {cache_file.name}")
    print(f"{'='*60}")

    # Extract mic config from filename
    mic_str = cache_file.stem.replace('crnn_results_mics_', '')
    mic_config = [int(x) for x in mic_str.split('_')]
    print(f"Mic config: {mic_config}")

    # Load cached results
    with open(cache_file, 'rb') as f:
        cached_results = pickle.load(f)

    print(f"Total cached samples: {len(cached_results)}")

    # Load CSV for ground truth
    df = pd.read_csv(CSV_PATH)

    # Load model
    print("Loading CRNN model...")
    model = load_crnn_model(CRNN_CHECKPOINT, device=device)
    print(f"Model loaded on {device}")

    # Randomly sample indices to validate
    valid_indices = [i for i, r in enumerate(cached_results) if r['crnn_pred'] is not None]
    sample_indices = np.random.choice(valid_indices, min(num_samples, len(valid_indices)), replace=False)

    print(f"\nValidating {len(sample_indices)} random samples...\n")

    discrepancies = []
    max_angle_diff = 0
    max_feature_diff = 0
    max_logit_diff = 0

    for idx in sample_indices:
        cached = cached_results[idx]
        row = df.iloc[idx]

        audio_filename = row['filename']
        audio_path = os.path.join(str(DATA_ROOT), audio_filename.replace('.flac', '_CH0.wav'))

        # Run fresh inference
        fresh = run_crnn_inference(model, audio_path, mic_config, device=device)

        if fresh is None:
            print(f"Sample {idx}: Skipped (missing audio)")
            continue

        # Compare predictions
        angle_match = cached['crnn_pred'] == fresh['predicted_angle']
        angle_diff = abs(cached['crnn_pred'] - fresh['predicted_angle'])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Compare features (using mean absolute difference)
        feature_diff = np.mean(np.abs(cached['penultimate_features'] - fresh['penultimate_features']))
        logit_diff = np.mean(np.abs(cached['logits_pre_sig'] - fresh['logits_pre_sig']))
        pred_diff = np.mean(np.abs(cached['predictions'] - fresh['predictions']))

        max_angle_diff = max(max_angle_diff, angle_diff)
        max_feature_diff = max(max_feature_diff, feature_diff)
        max_logit_diff = max(max_logit_diff, logit_diff)

        status = "✓" if angle_match and feature_diff < 1e-4 else "✗"
        print(f"{status} Sample {idx:4d}: "
              f"Angle diff: {angle_diff:5.1f}°  "
              f"Feature diff: {feature_diff:.2e}  "
              f"Logit diff: {logit_diff:.2e}")

        if not angle_match or feature_diff > 1e-3:
            discrepancies.append({
                'sample_idx': idx,
                'angle_diff': angle_diff,
                'feature_diff': feature_diff,
                'logit_diff': logit_diff,
            })

    # Summary
    print(f"\n{'='*60}")
    print("CRNN Validation Summary:")
    print(f"  Samples validated: {len(sample_indices)}")
    print(f"  Discrepancies found: {len(discrepancies)}")
    print(f"  Max angle difference: {max_angle_diff:.1f}°")
    print(f"  Max feature difference: {max_feature_diff:.2e}")
    print(f"  Max logit difference: {max_logit_diff:.2e}")

    if len(discrepancies) == 0:
        print("\n✓ CRNN cache validation PASSED - all samples match!")
    else:
        print(f"\n✗ CRNN cache validation FAILED - {len(discrepancies)} discrepancies")

    print(f"{'='*60}\n")

    return len(discrepancies) == 0


def validate_srp_cache(cache_file, num_samples=10):
    """Validate SRP cached results against fresh inference."""
    print(f"\n{'='*60}")
    print(f"Validating SRP Cache: {cache_file.name}")
    print(f"{'='*60}")

    # Extract mic config from filename
    mic_str = cache_file.stem.replace('srp_results_mics_', '')
    mic_config = [int(x) for x in mic_str.split('_')]
    print(f"Mic config: {mic_config}")

    # Load cached results
    with open(cache_file, 'rb') as f:
        cached_results = pickle.load(f)

    print(f"Total cached samples: {len(cached_results)}")

    # Load CSV for ground truth
    df = pd.read_csv(CSV_PATH)

    # Randomly sample indices to validate
    valid_indices = [i for i, r in enumerate(cached_results) if r['srp_pred'] is not None]
    sample_indices = np.random.choice(valid_indices, min(num_samples, len(valid_indices)), replace=False)

    print(f"\nValidating {len(sample_indices)} random samples...\n")

    discrepancies = []
    max_angle_diff = 0

    for idx in sample_indices:
        cached = cached_results[idx]
        row = df.iloc[idx]

        audio_filename = row['filename']
        audio_path = os.path.join(str(DATA_ROOT), audio_filename.replace('.flac', '_CH0.wav'))
        gt_angle = row['angle(°)']

        # Run fresh inference
        fresh_pred, fresh_error = run_srp_inference(audio_path, mic_config, gt_angle)

        if fresh_pred is None:
            print(f"Sample {idx}: Skipped (missing audio)")
            continue

        # Compare predictions
        angle_diff = abs(cached['srp_pred'] - fresh_pred)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        max_angle_diff = max(max_angle_diff, angle_diff)

        status = "✓" if angle_diff < 0.1 else "✗"
        print(f"{status} Sample {idx:4d}: "
              f"Cached: {cached['srp_pred']:6.2f}°  "
              f"Fresh: {fresh_pred:6.2f}°  "
              f"Diff: {angle_diff:5.2f}°")

        if angle_diff > 0.1:
            discrepancies.append({
                'sample_idx': idx,
                'cached_pred': cached['srp_pred'],
                'fresh_pred': fresh_pred,
                'angle_diff': angle_diff,
            })

    # Summary
    print(f"\n{'='*60}")
    print("SRP Validation Summary:")
    print(f"  Samples validated: {len(sample_indices)}")
    print(f"  Discrepancies found: {len(discrepancies)}")
    print(f"  Max angle difference: {max_angle_diff:.2f}°")

    if len(discrepancies) == 0:
        print("\n✓ SRP cache validation PASSED - all samples match!")
    else:
        print(f"\n✗ SRP cache validation FAILED - {len(discrepancies)} discrepancies")

    print(f"{'='*60}\n")

    return len(discrepancies) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate cached CRNN and SRP results")
    parser.add_argument('--type', choices=['crnn', 'srp', 'both'], default='both',
                        help='Which cache to validate')
    parser.add_argument('--cache_file', type=str, default=None,
                        help='Specific cache file to validate (optional)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of random samples to validate per cache')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cpu', 'mps', 'cuda'],
                        help='Device for CRNN inference')

    args = parser.parse_args()

    features_dir = Path("features")

    all_passed = True

    # Validate specific file or all files
    if args.cache_file:
        cache_file = Path(args.cache_file)
        if not cache_file.exists():
            print(f"Error: Cache file not found: {cache_file}")
            return

        if 'crnn' in cache_file.name:
            passed = validate_crnn_cache(cache_file, args.num_samples, args.device)
        elif 'srp' in cache_file.name:
            passed = validate_srp_cache(cache_file, args.num_samples)
        else:
            print(f"Error: Unknown cache type in filename: {cache_file.name}")
            return

        all_passed = passed

    else:
        # Validate all CRNN caches
        if args.type in ['crnn', 'both']:
            crnn_caches = sorted(features_dir.glob('crnn_results_mics_*.pkl'))
            if len(crnn_caches) == 0:
                print("No CRNN cache files found")
            else:
                print(f"\nFound {len(crnn_caches)} CRNN cache files")
                for cache_file in crnn_caches[:3]:  # Validate first 3 by default
                    passed = validate_crnn_cache(cache_file, args.num_samples, args.device)
                    all_passed = all_passed and passed

        # Validate all SRP caches
        if args.type in ['srp', 'both']:
            srp_caches = sorted(features_dir.glob('srp_results_mics_*.pkl'))
            if len(srp_caches) == 0:
                print("No SRP cache files found")
            else:
                print(f"\nFound {len(srp_caches)} SRP cache files")
                for cache_file in srp_caches[:3]:  # Validate first 3 by default
                    passed = validate_srp_cache(cache_file, args.num_samples)
                    all_passed = all_passed and passed

    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED - check output above")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
