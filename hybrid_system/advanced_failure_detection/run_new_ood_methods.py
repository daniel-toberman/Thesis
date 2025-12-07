#!/usr/bin/env python3
"""
Run all new OOD methods from the survey paper.

Methods:
- KNN Distance-based
- ReAct (Rectified Activations)
- GradNorm

Runs threshold optimization for each method.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*100}")
    print(f"{description}")
    print("="*100)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    # Configuration
    features_path = "features/test_3x12cm_consecutive_features.npz"

    # Check if features exist
    if not Path(features_path).exists():
        print(f"❌ Features not found: {features_path}")
        print("Please extract features first.")
        sys.exit(1)

    results = {}

    # 1. KNN Distance-based OOD
    print("\n" + "="*100)
    print("STEP 1/3: KNN Distance-based OOD")
    print("="*100)

    for k in [5, 10, 20]:
        output_dir = f"results/ood_methods/knn_k{k}"
        cmd = [
            "python3", "evaluate_ood_methods.py",
            "--method", "knn",
            "--k", str(k),
            "--features_path", features_path,
            "--output_dir", output_dir
        ]
        success = run_command(cmd, f"KNN OOD (k={k})")
        results[f"KNN k={k}"] = success

    # 2. ReAct (Rectified Activations)
    print("\n" + "="*100)
    print("STEP 2/3: ReAct (Rectified Activations)")
    print("="*100)

    for percentile in [85, 90, 95]:
        output_dir = f"results/ood_methods/react_p{percentile}"
        cmd = [
            "python3", "evaluate_ood_methods.py",
            "--method", "react",
            "--clip_percentile", str(percentile),
            "--features_path", features_path,
            "--output_dir", output_dir
        ]
        success = run_command(cmd, f"ReAct (p{percentile})")
        results[f"ReAct p{percentile}"] = success

    # 3. GradNorm
    print("\n" + "="*100)
    print("STEP 3/4: GradNorm")
    print("="*100)

    output_dir = "results/ood_methods/gradnorm"
    cmd = [
        "python3", "evaluate_ood_methods.py",
        "--method", "gradnorm",
        "--features_path", features_path,
        "--output_dir", output_dir
    ]
    success = run_command(cmd, "GradNorm")
    results["GradNorm"] = success

    # 4. Mahalanobis Distance
    print("\n" + "="*100)
    print("STEP 4/4: Mahalanobis Distance")
    print("="*100)

    output_dir = "results/ood_methods/mahalanobis"
    cmd = [
        "python3", "evaluate_ood_methods.py",
        "--method", "mahalanobis",
        "--features_path", features_path,
        "--output_dir", output_dir
    ]
    success = run_command(cmd, "Mahalanobis Distance")
    results["Mahalanobis"] = success

    # Summary
    print("\n" + "="*100)
    print("EXECUTION SUMMARY")
    print("="*100)

    for method, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {method:25s} {status}")

    # Overall status
    all_success = all(results.values())
    if all_success:
        print("\n✅ All methods completed successfully!")
        print("\nNext steps:")
        print("1. Review threshold optimization results in results/ood_methods/*/threshold_optimization.csv")
        print("2. Choose best thresholds for 30% routing")
        print("3. Run hybrid evaluations with: run_new_ood_hybrid.py")
    else:
        print("\n❌ Some methods failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
