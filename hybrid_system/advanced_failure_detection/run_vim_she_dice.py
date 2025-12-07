#!/usr/bin/env python3
"""
Run threshold optimization for VIM, SHE, and DICE methods.

This script runs all three methods sequentially and saves results.
"""

import subprocess
import sys
from pathlib import Path

# Methods to evaluate with their hyperparameters
METHODS = [
    {'method': 'vim', 'params': {'alpha': 1.0}, 'name': 'VIM (alpha=1.0)'},
    {'method': 'she', 'params': {}, 'name': 'SHE'},
    {'method': 'dice', 'params': {'sparsity': 90}, 'name': 'DICE (sparsity=90%)'},
    {'method': 'dice', 'params': {'sparsity': 80}, 'name': 'DICE (sparsity=80%)'},
]


def run_method(method_info):
    """Run threshold optimization for a single method."""
    method = method_info['method']
    params = method_info['params']
    name = method_info['name']

    print("\n" + "="*100)
    print(f"{name} - Threshold Optimization")
    print("="*100)

    # Build command
    cmd = [
        'python3', 'evaluate_ood_methods.py',
        '--method', method
    ]

    # Add method-specific parameters
    if method == 'dice' and 'sparsity' in params:
        cmd.extend(['--sparsity', str(params['sparsity'])])
        output_dir = f"results/ood_methods/{method}_s{params['sparsity']}"
    elif method == 'vim' and 'alpha' in params:
        cmd.extend(['--alpha', str(params['alpha'])])
        output_dir = f"results/ood_methods/{method}"
    else:
        output_dir = f"results/ood_methods/{method}"

    cmd.extend(['--output_dir', output_dir])

    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}")

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ {name} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} - FAILED")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False


def main():
    print("="*100)
    print("THRESHOLD OPTIMIZATION - VIM, SHE, DICE")
    print("="*100)
    print(f"\nRunning {len(METHODS)} method configurations...")

    results = []
    for method_info in METHODS:
        success = run_method(method_info)
        results.append((method_info['name'], success))

    # Summary
    print("\n" + "="*100)
    print("THRESHOLD OPTIMIZATION COMPLETED")
    print("="*100)

    print("\nResults:")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name:40s} {status}")

    successful = sum(1 for _, success in results if success)
    print(f"\n{successful}/{len(METHODS)} methods completed successfully")

    if successful < len(METHODS):
        print("\n⚠️  Some methods failed. Check output above for details.")
        sys.exit(1)
    else:
        print("\n✅ All methods completed successfully!")
        print("\nNext steps:")
        print("1. Review threshold optimization results in results/ood_methods/*/threshold_optimization.csv")
        print("2. Run hybrid evaluations with: python3 run_vim_she_dice_hybrid.py")


if __name__ == "__main__":
    main()
