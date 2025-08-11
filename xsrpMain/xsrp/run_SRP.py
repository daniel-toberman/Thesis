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

from SSL.utils_ import audiowu_high_array_geometry

# === CONFIG (defaults; can override with CLI) ===
BASE_DIR = r"E:\RealMAN\test\ma_noisy_speech"
CSV_PATH = r"E:\RealMAN\test\test_static_source_location.csv"
SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"
TMP_EXTRACT = r"E:\RealMAN\_tmp_extract"

MIC_POSITIONS = audiowu_high_array_geometry()
SRP_GRID_CELLS = 360
SRP_MODE = "gcc_phat_time"   # or "gcc_phat_freq"
N_AVG_SAMPLES = 3
N_DFT_BINS = 2048

# Local imports
try:
    from conventional_srp import ConventionalSrp
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from conventional_srp import ConventionalSrp

# Visualization + signal features (avoid name clash)
try:
    from gcc_phat import gcc_phat
    from cross_correlation import cross_correlation as cc_raw
    from cross_correlation import plot_cross_correlation as plot_cc  # viz util
except ImportError:
    gcc_phat = None
    plot_cc = None
    cc_raw = None


def ensure_flac_available(rel_flac: str) -> Path | None:
    abs_path = Path(BASE_DIR) / rel_flac
    if abs_path.exists():
        return abs_path
    if not SEVEN_ZIP or not Path(SEVEN_ZIP).exists():
        print(f"[WARN] {abs_path} not found and 7z.exe not configured. Skipping.")
        return None
    archive = os.path.join(BASE_DIR,rel_flac.split('/')[0] + '.rar')
    if not archive:
        print(f"[WARN] No .rar found for {rel_flac}.")
        return None
    dest = Path(TMP_EXTRACT)
    dest.mkdir(parents=True, exist_ok=True)
    inner_path = '/'.join(rel_flac.split('/')[0:])
    inner_path = inner_path.split('.')[0] + '*' + '.flac'
    cmd = [SEVEN_ZIP, "x", "-y", f"-o{dest}", str(archive), inner_path.replace("/", "\\")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] 7z failed for {rel_flac}:\n{res.stderr}")
        return None
    if dest.exists():
        return dest
    cmd2 = [SEVEN_ZIP, "x", "-y", f"-o{dest.parent}", str(archive)]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    if res2.returncode != 0:
        print(f"[ERROR] 7z full extract failed:\n{res2.stderr}")
        return None
    if dest.exists():
        return dest
    found = list(dest.parent.rglob(Path(rel_flac).name))
    return found[0] if found else None

def load_segment_multichannel(flac_path: Path, st: int, ed: int):
    x, fs = sf.read(str(flac_path), always_2d=True)
    ed = min(ed, len(x))
    return fs, x[st:ed, :].T.astype(np.float64)

def run_srp_return_details(fs: int, mic_signals: np.ndarray):
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
    rel = '/'.join(rel.split('/')[2:])
    st = int(row["real_st"]); ed = int(row["real_ed"])
    gt = None
    for c in row.index:
        if "angle" in c.lower():
            try: gt = float(row[c])
            except: gt = None
            break

    flac_path = ensure_flac_available(rel)
    if flac_path is None:
        print(f"[WARN] Missing audio for {rel}")
        return {"idx": idx, "filename": rel, "status": "missing"}

    fs, X = load_segment_multichannel(flac_path, st, ed)
    if X.shape[0] != MIC_POSITIONS.shape[0]:
        return {"idx": idx, "filename": rel, "status": f"error: mic count mismatch ({X.shape[0]} vs {MIC_POSITIONS.shape[0]})"}

    az, srp_obj, srp_map, grid = run_srp_return_details(fs, X)
    err = angular_error_deg(az, gt) if gt is not None else None

    print(f"[{idx}] {rel} -> {az:.1f}°" + (f" (gt {gt:.1f}°, err {err:.1f}°)" if gt is not None else ""))

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
    global CSV_PATH, BASE_DIR, SEVEN_ZIP
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--base-dir", default=BASE_DIR)
    p.add_argument("--first-only", action="store_true")
    p.add_argument("--idx", type=int, default=None, help="Row index to run (overrides --first-only)")
    p.add_argument("--plots", action="store_true", help="Enable verification plots")
    p.add_argument("--outdir", default=None, help="Where to save plots (if omitted, show on screen)")
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--save_cc", action="store_true", help="Also plot plain cross-correlation per pair")
    p.add_argument("--save_gcc", action="store_true", help="Also plot GCC-PHAT per pair")
    args = p.parse_args()


    CSV_PATH = args.csv; BASE_DIR = args.base_dir
    if args.no_extract: SEVEN_ZIP = None
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
