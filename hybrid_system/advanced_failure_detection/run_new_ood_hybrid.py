#!/usr/bin/env python3
"""
Run hybrid CRNN-SRP evaluations for all new OOD methods.

Uses optimal thresholds from threshold optimization for ~30% routing.
Runs actual SRP predictions on routed cases.
"""

import subprocess
import sys
from pathlib import Path


# Optimal thresholds for ~30% routing (from threshold optimization)
THRESHOLDS = {
    'knn': 3.0428,          # k=10, F1=0.526
    'react': 82.0905,       # p85, F1=0.387
    'gradnorm': 1.0373,     # F1=0.429
    'mahalanobis': 12.9890, # F1=0.411
}


def run_command_async(cmd, description, log_file):
    """Run a command asynchronously in the background."""
    print(f"\n{'='*100}")
    print(f"{description}")
    print("="*100)
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}\n")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    print(f"✅ {description} - STARTED (PID: {process.pid})")
    print(f"   Monitor with: tail -f {log_file}")

    return process


def main():
    print("\n" + "="*100)
    print("HYBRID CRNN-SRP EVALUATION - NEW OOD METHODS")
    print("="*100)
    print("\nThis will run SRP on routed cases for all new OOD methods.")
    print("Each evaluation takes ~1-2 hours with 30% routing (~600 samples).")
    print("\nRunning all methods in parallel (one per method)...\n")

    processes = {}

    # 1. KNN k=10
    output_dir = "results/ood_methods/knn_k10_hybrid"
    log_file = "/tmp/knn_hybrid.log"
    cmd = [
        "python3", "evaluate_ood_hybrid.py",
        "--method", "knn",
        "--threshold", str(THRESHOLDS['knn']),
        "--output_dir", output_dir
    ]
    processes['KNN k=10'] = run_command_async(cmd, "KNN k=10 Hybrid Evaluation", log_file)

    # 2. ReAct p85
    output_dir = "results/ood_methods/react_p85_hybrid"
    log_file = "/tmp/react_hybrid.log"
    cmd = [
        "python3", "evaluate_ood_hybrid.py",
        "--method", "react",
        "--threshold", str(THRESHOLDS['react']),
        "--output_dir", output_dir
    ]
    processes['ReAct p85'] = run_command_async(cmd, "ReAct p85 Hybrid Evaluation", log_file)

    # 3. GradNorm
    output_dir = "results/ood_methods/gradnorm_hybrid"
    log_file = "/tmp/gradnorm_hybrid.log"
    cmd = [
        "python3", "evaluate_ood_hybrid.py",
        "--method", "gradnorm",
        "--threshold", str(THRESHOLDS['gradnorm']),
        "--output_dir", output_dir
    ]
    processes['GradNorm'] = run_command_async(cmd, "GradNorm Hybrid Evaluation", log_file)

    # 4. Mahalanobis
    output_dir = "results/ood_methods/mahalanobis_hybrid"
    log_file = "/tmp/mahalanobis_hybrid.log"
    cmd = [
        "python3", "evaluate_ood_hybrid.py",
        "--method", "mahalanobis",
        "--threshold", str(THRESHOLDS['mahalanobis']),
        "--output_dir", output_dir
    ]
    processes['Mahalanobis'] = run_command_async(cmd, "Mahalanobis Hybrid Evaluation", log_file)

    # Summary
    print("\n" + "="*100)
    print("ALL HYBRID EVALUATIONS STARTED")
    print("="*100)
    print("\nRunning processes:")
    for method, process in processes.items():
        print(f"  {method:25s} PID: {process.pid}")

    print("\nMonitor progress with:")
    print("  tail -f /tmp/knn_hybrid.log")
    print("  tail -f /tmp/react_hybrid.log")
    print("  tail -f /tmp/gradnorm_hybrid.log")
    print("  tail -f /tmp/mahalanobis_hybrid.log")

    print("\nEach evaluation will take ~1-2 hours.")
    print("Results will be saved to results/ood_methods/*/hybrid_summary.csv")

    print("\n⏳ Waiting for all evaluations to complete...")
    print("   (This will take ~1-2 hours total)\n")

    # Wait for all processes to complete
    for method, process in processes.items():
        print(f"Waiting for {method}...")
        process.wait()
        if process.returncode == 0:
            print(f"  ✅ {method} - COMPLETED")
        else:
            print(f"  ❌ {method} - FAILED (exit code: {process.returncode})")

    # Final summary
    print("\n" + "="*100)
    print("HYBRID EVALUATIONS COMPLETED")
    print("="*100)

    all_success = all(p.returncode == 0 for p in processes.values())
    if all_success:
        print("\n✅ All hybrid evaluations completed successfully!")
        print("\nNext steps:")
        print("1. Review results in results/ood_methods/*/hybrid_summary.csv")
        print("2. Update research_summary.md with actual results")
    else:
        print("\n❌ Some evaluations failed. Check the log files for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
