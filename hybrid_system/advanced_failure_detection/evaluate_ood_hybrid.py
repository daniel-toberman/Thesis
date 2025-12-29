#!/usr/bin/env python3
"""
Evaluate Hybrid CRNN-SRP System with OOD Method Routing

Runs actual SRP on OOD-routed samples and computes real hybrid metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import soundfile as sf
from scipy import signal
from tqdm import tqdm
import pickle

# SRP imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'xsrpMain'))
from xsrp.conventional_srp import ConventionalSrp
from SSL.utils_ import audiowu_high_array_geometry

# Import OOD routers
from energy_ood_routing import EnergyOODRouter
from mc_dropout_routing import MCDropoutRouter
from knn_ood_routing import KNNOODRouter
from react_ood_routing import ReActOODRouter
from gradnorm_ood_routing import GradNormOODRouter
from mahalanobis_ood_routing import MahalanobisOODRouter
from confidnet_routing import ConfidNetRouter
from dice_ood_routing import DICEOODRouter
from she_ood_routing import SHEOODRouter
from vim_ood_routing import VIMOODRouter
from max_prob_routing import MaxProbRouter
from llr_ood_routing import LLROODRouter


# Paths
DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")

# Configuration
MIC_ORDER_SRP = [0, 9, 10, 11, 4, 5, 6, 7, 8]  # SRP order

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

    return np.array(signals)


def run_srp_on_case(audio_path, mic_order_srp, gt_angle):
    """Run SRP on a single audio case."""
    try:
        audio = load_audio_multichannel(audio_path, mic_order_srp)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        return None, None

    # Get microphone positions
    all_mic_positions = audiowu_high_array_geometry()
    mic_positions = all_mic_positions[mic_order_srp, :2]

    # Initialize SRP
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

    # Calculate azimuth
    srp_pred = float(np.degrees(np.arctan2(est_vec[1], est_vec[0])) % 360.0)

    # Calculate error
    error = abs((srp_pred - gt_angle + 180) % 360 - 180)

    return srp_pred, error


def evaluate_ood_hybrid(method, threshold, output_dir, features_path, srp_cache_path):
    """
    Evaluate OOD method with actual SRP runs.

    Args:
        method: OOD method name
        threshold: OOD score threshold for routing
        output_dir: Where to save results
        features_path: Path to the CRNN features file (.npz)
        srp_cache_path: Path to the cached SRP results file (.pkl)
    """
    print("="*100)
    print(f"EVALUATING OOD HYBRID: {method.upper()}")
    print(f"  Features: {features_path}")
    print(f"  SRP Cache: {srp_cache_path}")
    print("="*100)

    # Load test features
    print(f"\nLoading test features from: {features_path}")
    with open(features_path, 'rb') as f:
        data_list = pickle.load(f)

    # Convert list of dictionaries to a dictionary of numpy arrays
    df_crnn = pd.DataFrame(data_list)
    features = {
        'predictions': df_crnn['crnn_pred'].to_numpy(),
        'gt_angles': df_crnn['gt_angle'].to_numpy(),
        'abs_errors': df_crnn['crnn_error'].to_numpy(),
        'penultimate_features': np.stack(df_crnn['penultimate_features'].values),
        'logits_pre_sig': np.stack(df_crnn['logits_pre_sig'].values),
        'avg_predictions': np.stack(df_crnn['avg_prediction'].values),
        'global_indices': df_crnn['global_idx'].to_numpy()
    }

    crnn_preds = features['predictions']
    gt_angles = features['gt_angles']
    abs_errors = features['abs_errors']

    n_samples = len(crnn_preds)
    print(f"Test samples: {n_samples}")
    print(f"CRNN-only MAE: {abs_errors.mean():.2f}°")

    # Load CSV for audio paths
    df = pd.read_csv(CSV_PATH)

    # Initialize router
    router_args = {'features': features, 'threshold': threshold}
    if method == 'energy':
        print("\nInitializing Energy OOD router...")
        router = EnergyOODRouter(model_path='models/energy_ood_20.0deg/energy_ood_model.pkl')
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'mc_dropout':
        print("\nInitializing MC Dropout router...")
        router = MCDropoutRouter()
        route_to_srp, scores = router.predict_routing(**router_args, use_entropy=True)
    elif method == 'knn':
        print("\nInitializing KNN OOD router...")
        router = KNNOODRouter(k=10)
        router.train(features)
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'react':
        print("\nInitializing ReAct router...")
        router = ReActOODRouter(clip_percentile=85)
        router.train(features)
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'gradnorm':
        print("\nInitializing GradNorm router...")
        router = GradNormOODRouter()
        router.train(features)
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'mahalanobis':
        print("\nInitializing Mahalanobis router...")
        router = MahalanobisOODRouter()
        router.train(features)
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'confidnet':
        print("\nInitializing ConfidNet router...")
        router = ConfidNetRouter(model_path='models/confidnet_20deg/best_model.ckpt')
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'dice':
        print("\nInitializing DICE router...")
        router = DiceOODRouter(weight_path='models/dice_weights.pkl')
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'she':
        print("\nInitializing SHE router...")
        router = SheOODRouter()
        router.train(features)
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'vim':
        print("\nInitializing VIM router...")
        router = VimOODRouter()
        router.train(features)
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'max_prob':
        print("\nInitializing Max Probability router...")
        router = MaxProbRouter()
        route_to_srp, scores = router.predict_routing(**router_args)
    elif method == 'llr':
        print("\nInitializing LLR router...")
        router = LLROODRouter(gmm_path='models/gmm_5_components.pkl')
        route_to_srp, scores = router.predict_routing(**router_args)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_routed = route_to_srp.sum()
    routing_rate = n_routed / n_samples * 100

    print(f"\nRouting decisions:")
    print(f"  Threshold: {threshold}")
    print(f"  Samples routed to SRP: {n_routed} / {n_samples} ({routing_rate:.1f}%)")

    # Compute routing quality metrics
    failures = abs_errors > 5
    precision = failures[route_to_srp].sum() / route_to_srp.sum() if route_to_srp.sum() > 0 else 0
    recall = failures[route_to_srp].sum() / failures.sum() if failures.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nRouting quality:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")

    # Check if cached SRP results exist
    routed_indices = np.where(route_to_srp)[0]
    srp_cache_path = Path(srp_cache_path)

    if srp_cache_path.exists():
        print(f"\n✅ Found cached SRP results: {srp_cache_path}")
        print(f"Loading pre-computed SRP predictions for {n_routed} routed cases...")

        # Load cached results
        with open(srp_cache_path, 'rb') as f:
            cached_df = pickle.load(f)

        # Filter to routed indices
        srp_results = []
        for idx in routed_indices:
            cached_row = cached_df[cached_df['sample_idx'] == idx]
            if len(cached_row) > 0:
                cached_row = cached_row.iloc[0]
                if pd.notna(cached_row['srp_pred']):
                    srp_results.append({
                        'sample_idx': idx,
                        'global_idx': int(cached_row['global_idx']),
                        'gt_angle': cached_row['gt_angle'],
                        'crnn_pred': cached_row['crnn_pred'],
                        'crnn_error': cached_row['crnn_error'],
                        'srp_pred': cached_row['srp_pred'],
                        'srp_error': cached_row['srp_error'],
                        'ood_score': scores[idx]
                    })

        print(f"✅ Loaded {len(srp_results)} cached SRP results instantly!")
        srp_df = pd.DataFrame(srp_results)

    else:
        # Fall back to real-time SRP computation
        print(f"\n⚠️  No cached SRP results found at: {srp_cache_path}")
        print(f"Running SRP on {n_routed} routed cases in real-time...")
        print("This will take ~1-2 hours...")
        print("Tip: Run 'python precompute_srp_results.py' once to cache all SRP results!")

        srp_results = []

        for idx in tqdm(routed_indices, desc="Running SRP"):
            global_idx = int(features['global_indices'][idx])

            if global_idx >= len(df):
                print(f"  Warning: global_idx {global_idx} out of range")
                continue

            row = df.iloc[global_idx]
            audio_filename = row['filename']
            audio_path = os.path.join(str(DATA_ROOT), audio_filename.replace('.flac', '_CH0.wav'))

            gt_angle = gt_angles[idx]
            crnn_pred = crnn_preds[idx]
            crnn_error = abs_errors[idx]

            # Run SRP
            srp_pred, srp_error = run_srp_on_case(audio_path, MIC_ORDER_SRP, gt_angle)

            if srp_pred is not None:
                srp_results.append({
                    'sample_idx': idx,
                    'global_idx': global_idx,
                    'gt_angle': gt_angle,
                    'crnn_pred': crnn_pred,
                    'crnn_error': crnn_error,
                    'srp_pred': srp_pred,
                    'srp_error': srp_error,
                    'ood_score': scores[idx]
                })

        # Create results DataFrame
        srp_df = pd.DataFrame(srp_results)

    print(f"\nSRP completed: {len(srp_df)} / {n_routed} successful")
    if not srp_df.empty:
        print(f"SRP MAE on routed: {srp_df['srp_error'].mean():.2f}°")

    # Compute hybrid performance
    print("\n" + "="*100)
    print("COMPUTING HYBRID PERFORMANCE")
    print("="*100)

    # Initialize with CRNN predictions
    hybrid_errors = abs_errors.copy()

    # Replace routed samples with SRP predictions
    if not srp_df.empty:
        for _, row in srp_df.iterrows():
            idx = int(row['sample_idx'])
            hybrid_errors[idx] = row['srp_error']

    # Compute metrics
    hybrid_mae = hybrid_errors.mean()
    hybrid_median = np.median(hybrid_errors)
    hybrid_success = (hybrid_errors <= 5).sum() / n_samples * 100

    crnn_mae = abs_errors.mean()
    crnn_median = np.median(abs_errors)
    crnn_success = (abs_errors <= 5).sum() / n_samples * 100

    print(f"\nCRNN-only:")
    print(f"  MAE: {crnn_mae:.2f}°")
    print(f"  Median: {crnn_median:.2f}°")
    print(f"  Success (≤5°): {crnn_success:.1f}%")

    print(f"\nHybrid ({method}):")
    print(f"  MAE: {hybrid_mae:.2f}°")
    print(f"  Median: {hybrid_median:.2f}°")
    print(f"  Success (≤5°): {hybrid_success:.1f}%")

    print(f"\nImprovement:")
    print(f"  Δ MAE: {hybrid_mae - crnn_mae:+.2f}°")
    print(f"  Δ Median: {hybrid_median - crnn_median:+.2f}°")
    print(f"  Δ Success: {hybrid_success - crnn_success:+.1f}%")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save SRP results
    srp_path = output_dir / "srp_results.csv"
    srp_df.to_csv(srp_path, index=False)
    print(f"\n✅ SRP results saved to: {srp_path}")

    # Save summary
    summary = {
        'method': method,
        'threshold': threshold,
        'n_samples': n_samples,
        'n_routed': n_routed,
        'routing_rate': routing_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'crnn_mae': crnn_mae,
        'crnn_median': crnn_median,
        'crnn_success': crnn_success,
        'hybrid_mae': hybrid_mae,
        'hybrid_median': hybrid_median,
        'hybrid_success': hybrid_success,
        'mae_improvement': hybrid_mae - crnn_mae,
        'median_improvement': hybrid_median - crnn_median,
        'success_improvement': hybrid_success - crnn_success
    }

    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "hybrid_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate OOD hybrid with actual SRP')
    parser.add_argument('--method', type=str, required=True,
                        choices=['energy', 'mc_dropout', 'knn', 'react', 'gradnorm', 'mahalanobis', 'confidnet', 'dice', 'she', 'vim', 'max_prob', 'llr'],
                        help='OOD method to evaluate')
    parser.add_argument('--threshold', type=float, required=True,
                        help='OOD score threshold for routing')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--features_path', type=str,
                        default="/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/test_3x12cm_consecutive_features.npz",
                        help='Path to the CRNN features file (.npz)')
    parser.add_argument('--srp_cache_path', type=str,
                        default="features/test_3x12cm_srp_results.pkl",
                        help='Path to the cached SRP results file (.pkl)')

    args = parser.parse_args()

    evaluate_ood_hybrid(
        method=args.method,
        threshold=args.threshold,
        output_dir=args.output_dir,
        features_path=args.features_path,
        srp_cache_path=args.srp_cache_path
    )


if __name__ == "__main__":
    main()
