"""
Main Pipeline for Advanced Failure Detection

Runs the complete workflow:
1. Extract features from CRNN for multiple array configurations
2. Fit temperature scaling on training data
3. Fit Mahalanobis detector on training data
4. Evaluate combined routing on test data
5. Generate visualizations and comparison reports

Usage:
    # Run full pipeline
    python run_pipeline.py

    # Run with custom configurations
    python run_pipeline.py --max_samples 1000 --n_pca_components 32
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import subprocess
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from temperature_scaling import optimize_temperature, evaluate_routing_with_calibration
from mahalanobis_ood import MahalanobisOODDetector, visualize_feature_space
from combined_routing import CombinedFailureDetector, visualize_combined_results


def run_feature_extraction(
    split: str,
    array_config: str,
    output_dir: str,
    max_samples: int = None,
    force: bool = False
) -> Path:
    """
    Run feature extraction script.

    Args:
        split: 'train' or 'test'
        array_config: Mic array configuration
        output_dir: Output directory
        max_samples: Max samples to process (None = all)
        force: Force re-extraction even if file exists

    Returns:
        output_path: Path to extracted features
    """
    output_path = Path(output_dir) / f"{split}_{array_config}_features.npz"

    # Check if already exists
    if output_path.exists() and not force:
        print(f"Features already exist: {output_path}")
        return output_path

    print("\n" + "="*80)
    print(f"Extracting features: {split} set, {array_config} array")
    print("="*80)

    # Build command
    cmd = [
        'python',
        'extract_features.py',
        '--split', split,
        '--array_config', array_config,
        '--output_dir', output_dir,
        '--device', 'mps'
    ]

    if max_samples is not None:
        cmd.extend(['--max_samples', str(max_samples)])

    # Run extraction
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        raise RuntimeError(f"Feature extraction failed for {split}/{array_config}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run advanced failure detection pipeline")
    parser.add_argument('--output_dir', type=str,
                        default='/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/results',
                        help='Output directory for all results')
    parser.add_argument('--features_dir', type=str,
                        default='/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features',
                        help='Directory to store extracted features')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per dataset (for testing)')
    parser.add_argument('--n_pca_components', type=int, default=64,
                        help='Number of PCA components for Mahalanobis')
    parser.add_argument('--force_extract', action='store_true',
                        help='Force re-extraction of features')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip feature extraction (use existing features)')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("ADVANCED FAILURE DETECTION PIPELINE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # STEP 1: Extract Features
    # =========================================================================
    if not args.skip_extraction:
        print("\n" + "="*80)
        print("STEP 1: Feature Extraction")
        print("="*80)

        # Extract features for training set (6cm - in-distribution)
        train_6cm_path = run_feature_extraction(
            'train', '6cm', features_dir,
            max_samples=args.max_samples,
            force=args.force_extract
        )

        # Extract features for test set (6cm - in-distribution)
        test_6cm_path = run_feature_extraction(
            'test', '6cm', features_dir,
            max_samples=args.max_samples,
            force=args.force_extract
        )

        # Extract features for test set (3x12cm_consecutive - out-of-distribution)
        test_3x12cm_path = run_feature_extraction(
            'test', '3x12cm_consecutive', features_dir,
            max_samples=args.max_samples,
            force=args.force_extract
        )

        print("\n✓ Feature extraction complete!")

    else:
        print("\nSkipping feature extraction, using existing features...")
        train_6cm_path = features_dir / 'train_6cm_features.npz'
        test_6cm_path = features_dir / 'test_6cm_features.npz'
        test_3x12cm_path = features_dir / 'test_3x12cm_consecutive_features.npz'

        # Check if files exist
        for path in [train_6cm_path, test_6cm_path, test_3x12cm_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required features not found: {path}")

    # Load features
    print("\nLoading features...")
    train_data = np.load(train_6cm_path, allow_pickle=True)
    test_6cm_data = np.load(test_6cm_path, allow_pickle=True)
    test_3x12cm_data = np.load(test_3x12cm_path, allow_pickle=True)

    train_features = {key: train_data[key] for key in train_data.files}
    test_6cm_features = {key: test_6cm_data[key] for key in test_6cm_data.files}
    test_3x12cm_features = {key: test_3x12cm_data[key] for key in test_3x12cm_data.files}

    print(f"  Train (6cm): {len(train_features['predicted_angles'])} samples")
    print(f"  Test (6cm): {len(test_6cm_features['predicted_angles'])} samples")
    print(f"  Test (3x12cm): {len(test_3x12cm_features['predicted_angles'])} samples")

    # =========================================================================
    # STEP 2: Fit Combined Detector
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Fitting Combined Failure Detector")
    print("="*80)

    detector = CombinedFailureDetector()
    detector.fit(
        train_features,
        val_features=None,  # Use training data for temperature optimization
        n_pca_components=args.n_pca_components
    )

    print("\n✓ Combined detector fitted!")

    # =========================================================================
    # STEP 3: Evaluate on Test Set (3x12cm - geometric mismatch)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Evaluating on Test Set (Geometric Mismatch)")
    print("="*80)

    # Grid search for optimal thresholds
    best_conf, best_dist, all_results = detector.grid_search_combined(
        test_3x12cm_features,
        strategy='or'
    )

    # Get best results
    best_results = [r for r in all_results if
                    r['confidence_threshold'] == best_conf and
                    r['distance_threshold'] == best_dist][0]

    print("\n✓ Evaluation complete!")

    # =========================================================================
    # STEP 4: Visualizations
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Generating Visualizations")
    print("="*80)

    # Combined routing visualizations
    visualize_combined_results(best_results, output_dir)

    # Feature space t-SNE
    print("\nGenerating feature space visualization...")
    visualize_feature_space(
        train_features,
        test_6cm_features,
        test_3x12cm_features,
        output_path=output_dir / 'feature_space_tsne.png',
        n_samples=2000
    )

    print("\n✓ Visualizations complete!")

    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Saving Results")
    print("="*80)

    # Save main results
    results_path = output_dir / 'pipeline_results.npz'
    np.savez(
        results_path,
        # Configuration
        temperature=detector.temperature,
        best_confidence_threshold=best_conf,
        best_distance_threshold=best_dist,
        n_pca_components=args.n_pca_components,
        # Best results
        best_precision=best_results['precision'],
        best_recall=best_results['recall'],
        best_f1=best_results['f1_score'],
        best_accuracy=best_results['accuracy'],
        best_routing_rate=best_results['routing_rate'],
        # All grid search results
        all_results=all_results,
        # Test set info
        n_test_samples=len(test_3x12cm_features['predicted_angles']),
    )

    print(f"  Main results: {results_path}")

    # Save summary report
    summary_path = output_dir / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ADVANCED FAILURE DETECTION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Temperature (calibration): {detector.temperature:.4f}\n")
        f.write(f"PCA components: {args.n_pca_components}\n")
        f.write(f"Routing strategy: OR (low confidence OR high distance)\n\n")

        f.write("OPTIMAL THRESHOLDS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Confidence threshold: {best_conf:.4f}\n")
        f.write(f"Mahalanobis distance threshold: {best_dist:.2f}\n\n")

        f.write("PERFORMANCE METRICS (Test Set: 3x12cm Geometric Mismatch)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {best_results['precision']:.3f}\n")
        f.write(f"Recall: {best_results['recall']:.3f}\n")
        f.write(f"F1 Score: {best_results['f1_score']:.3f}\n")
        f.write(f"Accuracy: {best_results['accuracy']:.3f}\n")
        f.write(f"Routing Rate: {best_results['routing_rate']*100:.1f}%\n\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")
        f.write(f"True Positives (failures detected): {best_results['true_positives']}\n")
        f.write(f"True Negatives (successes kept): {best_results['true_negatives']}\n")
        f.write(f"False Positives (false alarms): {best_results['false_positives']}\n")
        f.write(f"False Negatives (missed failures): {best_results['false_negatives']}\n\n")

        f.write("DATASET INFO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training samples (6cm): {len(train_features['predicted_angles'])}\n")
        f.write(f"Test samples (6cm): {len(test_6cm_features['predicted_angles'])}\n")
        f.write(f"Test samples (3x12cm): {len(test_3x12cm_features['predicted_angles'])}\n\n")

        f.write("FILES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write(f"- {results_path.name}\n")
        f.write(f"- combined_scatter.png\n")
        f.write(f"- combined_distributions.png\n")
        f.write(f"- feature_space_tsne.png\n")
        f.write(f"- summary_report.txt\n\n")

    print(f"  Summary report: {summary_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey Findings:")
    print(f"  • Temperature: {detector.temperature:.4f}")
    print(f"  • Optimal confidence threshold: {best_conf:.4f}")
    print(f"  • Optimal distance threshold: {best_dist:.2f}")
    print(f"  • F1 Score: {best_results['f1_score']:.3f}")
    print(f"  • Precision: {best_results['precision']:.3f}")
    print(f"  • Recall: {best_results['recall']:.3f}")
    print(f"  • Routing rate: {best_results['routing_rate']*100:.1f}%")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
