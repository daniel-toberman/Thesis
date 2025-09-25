# run_SRP.py
import argparse
import subprocess
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import re

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'SSL'))
from utils_ import audiowu_high_array_geometry

# Novel noise settings - will be set by command line arguments
USE_NOVEL_NOISE = False
NOVEL_NOISE_SCENE = "BadmintonCourt1"
NOVEL_NOISE_SNR = 5.0
NOVEL_NOISE_ROOT = "/Users/danieltoberman/Documents/RealMAN_9_channels/extracted/train/ma_noise"

# === CONFIG (defaults; can override with CLI) ===
# === CONFIG (defaults; can override with CLI) ===
BASE_DIR = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
CSV_PATH = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"

USE_MIC_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MIC_POSITIONS = audiowu_high_array_geometry()[USE_MIC_ID, :2]
SRP_GRID_CELLS = 360
SRP_MODE = "gcc_phat_time"   # Try time domain - might be better for small arrays
N_AVG_SAMPLES = 100         # Even more averaging for stability
N_DFT_BINS = 1024           # Smaller DFT for faster processing

# Debug: Print array info
print(f"Microphone positions (m):\n{MIC_POSITIONS}")
print(f"Array diameter: {np.max(np.linalg.norm(MIC_POSITIONS, axis=1)) * 2:.3f}m")

# Local imports
from xsrpMain.xsrp.conventional_srp import ConventionalSrp

# Visualization + signal features (avoid name clash)
from xsrpMain.xsrp.signal_features.gcc_phat import gcc_phat
from xsrpMain.xsrp.signal_features import cross_correlation as cc_raw
from xsrpMain.visualization.cross_correlation import plot_cross_correlation as plot_cc  # viz util



USE_MIC_ID = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def load_novel_noise(scene_name, target_length, fs, example_idx=0):
    """Load novel noise from high T60 scene for testing generalization.
    Uses example_idx to ensure reproducible noise selection for comparison."""
    scene_noise_dir = os.path.join(NOVEL_NOISE_ROOT, scene_name)

    # Get list of available noise files (sorted for reproducibility)
    noise_files = []
    if os.path.exists(scene_noise_dir):
        for root, dirs, files in os.walk(scene_noise_dir):
            # Look for .high_CH0.wav pattern in novel noise files
            noise_files.extend([os.path.join(root, f) for f in files if f.endswith('_CH0.wav')])
        noise_files = sorted(noise_files)  # Ensure deterministic order

    if not noise_files:
        print(f"Warning: No noise files found in {scene_noise_dir}")
        return np.zeros((len(USE_MIC_ID), target_length))

    # Select noise file based on example index for reproducibility
    noise_idx = example_idx % len(noise_files)
    selected_noise = noise_files[noise_idx]
    # Handle both .high_CH0.wav and _CH0.wav patterns
    if '.high_CH0.wav' in selected_noise:
        base_path = selected_noise.replace('.high_CH0.wav', '.high')
    else:
        base_path = selected_noise.replace('_CH0.wav', '')

    # Use example-specific seed for segment selection
    rng = np.random.default_rng(42 + example_idx)

    # Load all channels of the selected noise
    noise_channels = []
    for i in USE_MIC_ID:
        noise_ch_path = f"{base_path}_CH{i}.wav"
        if os.path.exists(noise_ch_path):
            noise_signal, noise_fs = sf.read(noise_ch_path, dtype="float64")

            # Resample if necessary
            if noise_fs != fs:
                from scipy import signal as scipy_signal
                noise_signal = scipy_signal.resample(noise_signal, int(len(noise_signal) * fs / noise_fs))

            # Extract reproducible segment based on example index
            if len(noise_signal) >= target_length:
                start_idx = rng.integers(0, len(noise_signal) - target_length + 1)
                noise_segment = noise_signal[start_idx:start_idx + target_length]
            else:
                # Repeat if noise is shorter
                repeats = int(np.ceil(target_length / len(noise_signal)))
                noise_repeated = np.tile(noise_signal, repeats)
                noise_segment = noise_repeated[:target_length]

            noise_channels.append(noise_segment)
        else:
            # If channel doesn't exist, use zeros
            noise_channels.append(np.zeros(target_length))

    return np.stack(noise_channels, axis=0)

def add_novel_noise_to_signal(signal, fs, example_idx, snr_db=5.0):
    """Add novel noise to clean signal at specified SNR."""
    target_length = signal.shape[1]
    noise = load_novel_noise(NOVEL_NOISE_SCENE, target_length, fs, example_idx)

    # Calculate SNR coefficient
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if signal_power == 0 or noise_power == 0:
        return signal

    noise_coeff = np.sqrt(signal_power / noise_power * np.power(10, -snr_db / 10))
    noisy_signal = signal + noise_coeff * noise

    return noisy_signal

def load_segment_multichannel(wav_ch1_path: Path, st: int, ed: int, example_idx=0, use_mic_id=USE_MIC_ID):
    """
    Loads a multichannel segment from files named ..._CH{N}.wav.
    Accepts either a base path ending with .wav or a CH1 path like *_CH1.wav.
    Returns: fs (int), X with shape (channels, samples) as float64.
    """
    p = str(wav_ch1_path)
    channels = []
    for i in use_mic_id:
        temp_path = p.replace('.wav', f'_CH{i}.wav')
        if i == 0:
            info = sf.info(temp_path)
            fs_ref = info.samplerate
        single_ch_signal, fs = sf.read(temp_path, dtype="float64")
        channels.append(single_ch_signal)

    # SRP expects (C, T)
    X = np.stack(channels, axis=0)

    # Add novel noise if enabled
    global USE_NOVEL_NOISE, NOVEL_NOISE_SNR
    if USE_NOVEL_NOISE:
        X = add_novel_noise_to_signal(X, fs_ref, example_idx, snr_db=NOVEL_NOISE_SNR)

    return fs_ref, X


def run_srp_return_details(fs: int, mic_signals: np.ndarray):
    srp = ConventionalSrp(
        fs=fs,
        grid_type="doa_1D",
        n_grid_cells=720,  # Higher resolution - 0.5 degree steps
        mic_positions=MIC_POSITIONS,
        room_dims=None,
        mode="gcc_phat_freq",  # Back to frequency domain
        interpolation=True,
        n_average_samples=200,  # Even more averaging
        n_dft_bins=N_DFT_BINS
    )
    est_vec, srp_map, grid = srp.forward(mic_signals)
    az = float(np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0)
    return az, srp, np.asarray(srp_map).ravel(), grid  # 1D map, UniformSphericalGrid

def angular_error_deg(pred: float, truth: float) -> float:
    return float(abs((pred - truth + 180) % 360 - 180))

def plot_srp_azimuth(srp_map: np.ndarray, grid, az_est: float, az_gt: float | None, out: Path | None):
    az_rads = getattr(grid, "azimuth_range", np.linspace(0, 2*np.pi, len(srp_map), endpoint=False))
    az_degs = (np.degrees(az_rads) % 360.0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(az_degs, srp_map, lw=1.4)
    ax.axvline(az_est, ls="--", label=f"Estimate {az_est:.1f}°")
    if az_gt is not None:
        ax.axvline(az_gt, ls="--", label=f"GT {az_gt:.1f}°")
    ax.set(xlabel="Azimuth (deg)", ylabel="SRP value", title="SRP over azimuth", xlim=(0, 360))
    ax.legend()
    plt.tight_layout()
    if out:
        fig.savefig(out, dpi=150); plt.close(fig)
    else:
        plt.show()

def plot_pairwise_cc_or_gcc(fs: int, X: np.ndarray, out: Path | None, use_gcc=True):
    if plot_cc is None:
        print("[WARN] Visualization utils not found; skipping CC/GCC plot.")
        return
    if use_gcc and gcc_phat is not None:
        cc, lags = gcc_phat(X, ifft=True, abs=False, return_lags=True, n_dft_bins=N_DFT_BINS)
    elif cc_raw is not None:
        cc, lags = cc_raw(X)
    else:
        print("[WARN] Missing feature functions; skipping.")
        return
    axs = plot_cc(cc, sr=fs, plot_peaks=False, lags=lags, n_central_bins=min(256, cc.shape[-1]))
    if out:
        axs[0].figure.savefig(out, dpi=150); plt.close(axs[0].figure)

def process_row(row, idx, plots: bool, outdir: Path | None, save_cc: bool, save_gcc: bool):
    rel = str(row["filename"]).replace("\\", "/")
    st = int(row["real_st"]); ed = int(row["real_ed"])
    gt = None
    for c in row.index:
        if "angle" in c.lower():
            try: gt = float(row[c])
            except: gt = None
            break

    # Convert .flac to .wav if needed
    if rel.endswith('.flac'):
        rel = rel.replace('.flac', '.wav')
    wav_path = os.path.join(BASE_DIR, rel)
    fs, X = load_segment_multichannel(wav_path, st, ed, example_idx=idx)  # Pass example index for reproducible noise
    if X.shape[0] != MIC_POSITIONS.shape[0]:
        return {"idx": idx, "filename": rel, "status": f"error: mic count mismatch ({X.shape[0]} vs {MIC_POSITIONS.shape[0]})"}

    az, srp_obj, srp_map, grid = run_srp_return_details(fs, X)
    err = angular_error_deg(az, gt) if gt is not None else None

    noise_info = f" + novel noise from {NOVEL_NOISE_SCENE}" if USE_NOVEL_NOISE else ""
    print(f"[{idx}] {rel} -> {az:.1f}°" + (f" (gt {gt:.1f}°, err {err:.1f}°)" if gt is not None else "") + noise_info)

    if plots:
        outdir and outdir.mkdir(parents=True, exist_ok=True)
        plot_srp_azimuth(srp_map, grid, az, gt, outdir / f"srp_azimuth_idx{idx}.png" if outdir else None)
        if save_gcc:
            plot_pairwise_cc_or_gcc(fs, X, outdir / f"gcc_pairwise_idx{idx}.png" if outdir else None, use_gcc=True)
        if save_cc:
            plot_pairwise_cc_or_gcc(fs, X, outdir / f"cc_pairwise_idx{idx}.png" if outdir else None, use_gcc=False)

    return {
        "idx": idx, "filename": rel, "fs": fs,
        "n_channels": X.shape[0], "n_samples": X.shape[1],
        "gt_angle_deg": gt, "srp_angle_deg": az, "abs_err_deg": err, "status": "ok",
    }

def main():
    global CSV_PATH, BASE_DIR, USE_NOVEL_NOISE, NOVEL_NOISE_SCENE, NOVEL_NOISE_SNR
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--base-dir", default=BASE_DIR)
    p.add_argument("--first-only", action="store_true")
    p.add_argument("--idx", type=int, default=None, help="Row index to run (overrides --first-only)")
    p.add_argument("--plots", action="store_true", help="Enable verification plots")
    p.add_argument("--outdir", default=None, help="Where to save plots (if omitted, show on screen)")
    p.add_argument("--save_cc", action="store_true", help="Also plot plain cross-correlation per pair")
    p.add_argument("--save_gcc", action="store_true", help="Also plot GCC-PHAT per pair")
    p.add_argument("--n", type=int, default=None, help="Number of rows to run")
    p.add_argument("--random", action="store_true", help="Pick rows at random")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --random")

    # Novel noise arguments
    p.add_argument("--use_novel_noise", action="store_true", default=False,
                  help="Use novel noise from high T60 environments for testing generalization")
    p.add_argument("--novel_noise_scene", type=str, default="BadmintonCourt1",
                  choices=["BadmintonCourt1", "Cafeteria2", "ShoppingMall", "SunkenPlaza2"],
                  help="High T60 scene to use for novel noise")
    p.add_argument("--novel_noise_snr", type=float, default=5.0,
                  help="SNR in dB for novel noise addition (default: 5.0)")

    args = p.parse_args()

    # Set global novel noise settings
    USE_NOVEL_NOISE = args.use_novel_noise
    NOVEL_NOISE_SCENE = args.novel_noise_scene
    NOVEL_NOISE_SNR = args.novel_noise_snr

    if USE_NOVEL_NOISE:
        print(f"Using novel noise from scene: {NOVEL_NOISE_SCENE}")
        print(f"Novel noise SNR: {NOVEL_NOISE_SNR} dB")
        print(f"Noise source: {os.path.join(NOVEL_NOISE_ROOT, NOVEL_NOISE_SCENE)}")


    CSV_PATH = args.csv; BASE_DIR = args.base_dir
    outdir = Path(args.outdir) if args.outdir else None

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower() for c in df.columns]
    for need in ("filename", "real_st", "real_ed"):
        if need not in df.columns:
            raise ValueError(f"CSV must contain '{need}' (case-insensitive).")

    if args.idx is not None:
        rows = [(args.idx, df.iloc[args.idx])]
    elif args.first_only:
        rows = [(0, df.iloc[0])]
    else:
        if args.n is not None and args.random:
            rng = np.random.default_rng(args.seed)
            idxs = rng.choice(df.index.values, size=min(args.n, len(df)), replace=False)
            rows = [(int(i), df.loc[int(i)]) for i in idxs]
        elif args.n is not None:
            rows = list(df.iloc[:args.n].iterrows())
        else:
            rows = list(df.iterrows())

    results = []
    for idx, row in rows:
        try:
            res = process_row(row, idx, plots=args.plots, outdir=outdir,
                              save_cc=args.save_cc, save_gcc=args.save_gcc)
        except Exception as e:
            res = {"idx": idx, "filename": row['filename'], "status": f"error: {e}"}
            print(f"[ERROR] {row['filename']}: {e}")
        results.append(res)

    if len(rows) > 1:
        out_csv = Path(CSV_PATH).with_name(Path(CSV_PATH).stem + "_srp_results.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\nSaved results to: {out_csv}")
    else:
        print("\nSingle example result:")
        print(pd.Series(results[0]))

if __name__ == "__main__":
    main()
