import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from scipy import signal

# SRP imports
from xsrp.conventional_srp import ConventionalSrp
from SSL.utils_ import audiowu_high_array_geometry


# ========================
# Paths (EDIT THESE)
# ========================

DATA_ROOT = Path(r"C:\daniel\extracted")
CSV_PATH = Path(r"C:\daniel\test\test_static_source_location_08.csv")
OUTPUT_DIR = Path(r"../hybrid_system/advanced_failure_detection/srp_features_raw")

FS = 16000

# 2 mics from 12cm (10 configs)
CONFIGS_2MIC = [
    [9,10,3,4,5,6,7,8,0],
    [1,10,11,4,5,6,7,8,0],
    [1,2,11,12,5,6,7,8,0],
    [9,2,3,4,13,6,7,8,0],
    [1,10,3,4,5,14,7,8,0],
    [9,2,11,4,5,6,7,8,0],
    [1,10,3,12,5,6,7,8,0],
    [1,2,11,4,13,6,7,8,0],
    [1,2,3,12,5,14,7,8,0],
    [1,2,3,4,13,6,15,8,0]
]

# 3 mics from 12cm (10 configs)

# 4 mics from 12cm (10 configs)
CONFIGS_4MIC = [
    [9,10,11,12,5,6,7,8,0],
    [1,10,11,12,13,6,7,8,0],
    [1,2,11,12,13,14,7,8,0],
    [9,2,11,4,13,6,15,8,0],
    [1,10,3,12,5,14,7,16,0],
    [9,10,11,4,5,14,15,8,0],
    [9,10,3,12,13,6,7,8,0],
    [1,10,11,4,13,14,7,8,0],
    [9,2,11,12,5,6,15,8,0],
    [1,10,3,4,13,14,15,8,0]
]


# Combine all configs
ALL_CONFIGS = CONFIGS_2MIC + CONFIGS_3MIC + CONFIGS_4MIC

# Process each configuration
TOTAL = len(ALL_CONFIGS)
COUNT = 0


# ========================
# SRP configuration
# ========================

SRP_CONFIG = {
    "n_dft_bins": 16384,
    "freq_min": 300,
    "freq_max": 4000,
    "grid_cells": 360,
    "mode": "gcc_phat_freq",
    "n_avg_samples": 1,
}


# ========================
# Utilities
# ========================

def load_audio_multichannel(example_path, mic_ids):
    base = example_path.replace("_CH0.wav", "")
    signals = []

    for mic_id in mic_ids:
        ch_path = f"{base}_CH{mic_id}.wav"
        if not os.path.exists(ch_path):
            raise FileNotFoundError(ch_path)

        sig, fs = sf.read(ch_path, dtype="float64")
        if fs != FS:
            sig = signal.resample(sig, int(len(sig) * FS / fs))
        signals.append(sig)

    return np.asarray(signals)


# ========================
# Main caching function
# ========================

def cache_srp_raw(mic_ids):
    mic_str = "_".join(str(m) for m in mic_ids)
    output_path = OUTPUT_DIR / f"srp_results_mics_{mic_str}.pkl"

    print(f"\nCaching RAW SRP for mic config: {mic_ids}")
    print(f"Output file: {output_path}")

    df = pd.read_csv(CSV_PATH)

    all_mic_positions = audiowu_high_array_geometry()
    mic_positions = all_mic_positions[mic_ids, :2]

    srp = ConventionalSrp(
        fs=FS,
        grid_type="doa_1D",
        n_grid_cells=SRP_CONFIG["grid_cells"],
        mic_positions=mic_positions,
        mode=SRP_CONFIG["mode"],
        interpolation=True,
        n_average_samples=SRP_CONFIG["n_avg_samples"],
        n_dft_bins=SRP_CONFIG["n_dft_bins"],
    )

    results = []

    for idx in tqdm(range(len(df)), desc="Running SRP"):
        row = df.iloc[idx]

        audio_path = DATA_ROOT / row["filename"].replace(".flac", "_CH0.wav")
        gt_angle = float(row["angle(°)"])

        audio = load_audio_multichannel(str(audio_path), mic_ids)
        est_vec, srp_map, grid = srp.forward(audio)

        srp_pred = float(
            np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0
        )

        err = abs(srp_pred - gt_angle)
        err = min(err, 360 - err)

        total = np.sum(srp_map)
        if total != 0:
            srp_map_normalized = srp_map / total
        else:
            srp_map_normalized = np.zeros_like(srp_map)

        results.append({
            "sample_idx": idx,
            "gt_angle": gt_angle,
            "srp_pred": srp_pred,
            "srp_error": err,          # evaluation only
            "srp_map": srp_map.astype(np.float32),
            "grid": grid.positions.astype(np.float32) if grid is not None else None,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(results)} samples to {output_path}")


# ========================
# Example usage
# ========================

if __name__ == "__main__":
    for MIC_CONFIG in ALL_CONFIGS:
        COUNT += 1
        print(f"Processing {COUNT}/{TOTAL}: {MIC_CONFIG}")
        cache_srp_raw(MIC_CONFIG)
