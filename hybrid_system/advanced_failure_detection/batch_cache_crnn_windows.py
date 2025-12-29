#!/usr/bin/env python3
"""
Batch cache CRNN predictions for 30 different microphone configurations on Windows.
This script is a Python-native replacement for the original .sh script.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from scipy import signal
from pathlib import Path
from tqdm import tqdm

# Adjust system path to find the CRNN module
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "SSL"))
try:
    from CRNN import CRNN
except ImportError:
    print("Error: Could not import CRNN module. Please ensure the SSL directory is in the python path.")
    sys.exit(1)

# --- Configuration Section ---
DATA_ROOT = Path(r"C:\daniel\extracted")
CSV_PATH = Path(r"C:\daniel\test\test_static_source_location_08.csv")
CHECKPOINT_PATH = project_root / "08_CRNN" / "checkpoints" / "best_valid_loss0.0220.ckpt"
OUTPUT_DIR = project_root / "crnn features"

CRNN_CONFIG = dict(win_len=512, nfft=512, win_shift_ratio=0.625, fre_range_used=range(1, 257), eps=1e-6)

CONFIGS_2MIC = [
    "9,10,3,4,5,6,7,8,0", "1,10,11,4,5,6,7,8,0", "1,2,11,12,5,6,7,8,0",
    "9,2,3,4,13,6,7,8,0", "1,10,3,4,5,14,7,8,0", "9,2,11,4,5,6,7,8,0",
    "1,10,3,12,5,6,7,8,0", "1,2,11,4,13,6,7,8,0", "1,2,3,12,5,14,7,8,0",
    "1,2,3,4,13,6,15,8,0"
]

CONFIGS_3MIC = [
    "9,10,11,4,5,6,7,8,0", "1,10,11,12,5,6,7,8,0", "1,2,11,12,13,6,7,8,0",
    "9,2,3,12,5,6,15,8,0", "1,10,3,4,13,6,7,16,0", "9,2,11,4,5,14,7,8,0",
    "1,10,3,12,5,6,15,8,0", "9,2,3,4,13,6,15,8,0", "1,10,11,4,5,6,15,8,0",
    "9,10,3,12,5,6,7,16,0"
]

CONFIGS_4MIC = [
    "9,10,11,12,5,6,7,8,0", "1,10,11,12,13,6,7,8,0", "1,2,11,12,13,14,7,8,0",
    "9,2,11,4,13,6,15,8,0", "1,10,3,12,5,14,7,16,0", "9,10,11,4,5,14,15,8,0",
    "9,10,3,12,13,6,7,8,0", "1,10,11,4,13,14,7,8,0", "9,2,11,12,5,6,15,8,0",
    "1,10,3,4,13,14,15,8,0"
]

ALL_CONFIGS = CONFIGS_2MIC + CONFIGS_3MIC + CONFIGS_4MIC
# --- End of Configuration ---

def load_model():
    """Loads the CRNN model from the checkpoint."""
    print(f"Loading model from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = CRNN()
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict({k.replace("model.", "").replace("arch.", ""): v for k, v in state.items()})
    model.eval()
    print("Model loaded successfully.")
    return model

def load_audio(path, mic_ids, fs=16000):
    """Loads and resamples audio channels for a given file path."""
    base = str(path).replace("_CH0.wav", "")
    sigs = []
    for m in mic_ids:
        s, f = sf.read(f"{base}_CH{m}.wav", dtype="float32")
        if f != fs:
            s = signal.resample(s, int(len(s) * fs / f))
        sigs.append(s)
    return np.stack(sigs)

def get_stft(audio, **kwargs):
    """Performs STFT on the audio and extracts features."""
    # audio shape: (num_mics, num_samples)
    num_mics = audio.shape[0]
    stfts = []
    for i in range(num_mics):
        stfts.append(signal.stft(audio[i, :],
                                 window=signal.get_window("hann", kwargs["win_len"]),
                                 nperseg=kwargs["win_len"],
                                 noverlap=int(kwargs["win_len"] * kwargs["win_shift_ratio"]),
                                 nfft=kwargs["nfft"])[2])
    stfts = np.stack(stfts, axis=0)[:, kwargs["fre_range_used"], :]
    
    # stfts shape: (num_mics, freq_bins, time_frames)
    # The model expects 9 mic channels, with the last one being the reference.
    ref_stft = stfts[-1, :, :]
    other_stfts = stfts[:-1, :, :]
    
    # Calculate IPD and Magnitude features
    ipd = np.angle(other_stfts * np.conj(ref_stft))
    mag = np.abs(other_stfts)
    ref_mag = np.abs(ref_stft)
    
    # Normalize magnitudes
    mag /= (ref_mag + kwargs["eps"])
    ref_mag /= (np.mean(ref_mag) + kwargs["eps"])

    # Concatenate features: 8 IPD, 8 Mag, 1 Ref Mag, 1 Ref IPD (all zeros)
    features = np.concatenate([ipd, mag, ref_mag[None, ...], np.zeros_like(ref_mag)[None, ...]], axis=0)
    return features.astype(np.float32)

def main():
    """Main execution function."""
    print("="*50)
    print("Batch CRNN Caching for 30 Configurations (Windows)")
    print("="*50)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = load_model()
    df = pd.read_csv(CSV_PATH)
    
    total = len(ALL_CONFIGS)
    for i, mic_config_str in enumerate(ALL_CONFIGS, 1):
        mic_str_filename = mic_config_str.replace(',', '_')
        output_file = OUTPUT_DIR / f"crnn_results_mics_{mic_str_filename}.pkl"
        
        print(f"\n[{i}/{total}] Processing config: {mic_config_str}")

        if output_file.exists():
            print(f"  Skipping (exists): {output_file.name}")
            continue

        mic_order = [int(m) for m in mic_config_str.split(',')]
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Config {i}"):
            relative_filename = row["filename"].replace(".flac", "_CH0.wav")
            if relative_filename.startswith("test/"):
                relative_filename = relative_filename[len("test/"):]
            path = DATA_ROOT / relative_filename
            
            try:
                audio = load_audio(path, mic_order)
                features = get_stft(audio, **CRNN_CONFIG)
            except Exception as e:
                print(f"Warning: Could not load or process audio for {path}. Error: {e}")
                continue

            x = torch.from_numpy(features).unsqueeze(0) # Shape: [1, 18, F, T]
            
            with torch.no_grad():
                logits, feats = model.forward_with_intermediates(x)

            logits = logits[0]
            feats = feats[0]

            probs = torch.sigmoid(logits)
            avg = probs.mean(dim=0)
            pred = int(torch.argmax(avg))
            err = abs(pred - row["angle(°)"])
            if err > 180: err = 360 - err

            results.append(dict(
                sample_idx=idx,
                global_idx=row.get('global_idx', -1),
                gt_angle=row["angle(°)"],
                crnn_pred=pred,
                crnn_error=err,
                max_prob=torch.max(avg).item(),
                logits_pre_sig=logits.cpu().numpy(),
                penultimate_features=feats.cpu().numpy(),
                predictions=probs.cpu().numpy(),
                avg_prediction=avg.cpu().numpy()
            ))

        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        
        print(f"  Saved {len(results)} samples to {output_file.name}")

    print("\n==========================================")
    print("Batch CRNN caching complete!")
    print("==========================================")

if __name__ == '__main__':
    main()