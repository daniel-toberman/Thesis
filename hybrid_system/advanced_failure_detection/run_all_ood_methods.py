#!/usr/bin/env python3
"""
Master Runner for All OOD-Based Failure Detection Methods

Trains and evaluates:
1. Energy-Based OOD
2. Deep SVDD
3. MC Dropout Ensemble (Variance)
4. MC Dropout Ensemble (Entropy)
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import subprocess
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def run_command(cmd, description):
    """Run a shell command with error handling."""
    print("\n" + "="*100)
    print(f"RUNNING: {description}")
    print("="*100)
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        print(f"Return code: {result.returncode}")
        return False

    print(f"\n✅ COMPLETED: {description}")
    return True


def train_all_methods(train_features, error_threshold, device, output_base):
    """Train Energy OOD and Deep SVDD."""

    print("\n" + "="*100)
    print("TRAINING ALL OOD METHODS")
    print("="*100)

    results = {}

    # 1. Train Energy OOD
    print("\n\n" + "#"*100)
    print("# 1. ENERGY-BASED OOD")
    print("#"*100)

    energy_dir = Path(output_base) / f"energy_ood_{error_threshold}deg"
    cmd = [
        'python', 'train_energy_ood.py',
        '--train_features', train_features,
        '--error_threshold', str(error_threshold),
        '--output_dir', str(energy_dir)
    ]
    results['energy'] = run_command(cmd, "Training Energy OOD")

    # 2. Train Deep SVDD
    print("\n\n" + "#"*100)
    print("# 2. DEEP SVDD")
    print("#"*100)

    svdd_dir = Path(output_base) / f"deep_svdd_{error_threshold}deg"
    cmd = [
        'python', 'train_deep_svdd.py',
        '--train_features', train_features,
        '--error_threshold', str(error_threshold),
        '--output_dir', str(svdd_dir),
        '--epochs', '50',
        '--device', device
    ]
    results['deep_svdd'] = run_command(cmd, "Training Deep SVDD")

    # 3. MC Dropout (no training needed)
    print("\n\n" + "#"*100)
    print("# 3. MC DROPOUT")
    print("#"*100)
    print("MC Dropout requires no training (uses existing CRNN logits)")
    results['mc_dropout'] = True

    return results


def evaluate_all_methods(test_features, device, model_base, output_base):
    """Evaluate all OOD methods."""

    print("\n" + "="*100)
    print("EVALUATING ALL OOD METHODS")
    print("="*100)

    results = {}

    # 1. Evaluate Energy OOD
    print("\n\n" + "#"*100)
    print("# 1. ENERGY-BASED OOD")
    print("#"*100)

    energy_model = Path(model_base) / "energy_ood_model.pkl"
    energy_out = Path(output_base) / "energy_ood"

    if energy_model.exists():
        cmd = [
            'python', 'evaluate_ood_methods.py',
            '--method', 'energy',
            '--model_path', str(energy_model),
            '--features_path', test_features,
            '--output_dir', str(energy_out)
        ]
        results['energy'] = run_command(cmd, "Evaluating Energy OOD")
    else:
        print(f"❌ Model not found: {energy_model}")
        results['energy'] = False

    # 2. Evaluate Deep SVDD
    print("\n\n" + "#"*100)
    print("# 2. DEEP SVDD")
    print("#"*100)

    svdd_model = Path(model_base) / "deep_svdd_model.pkl"
    svdd_out = Path(output_base) / "deep_svdd"

    if svdd_model.exists():
        cmd = [
            'python', 'evaluate_ood_methods.py',
            '--method', 'deep_svdd',
            '--model_path', str(svdd_model),
            '--features_path', test_features,
            '--output_dir', str(svdd_out),
            '--device', device
        ]
        results['deep_svdd'] = run_command(cmd, "Evaluating Deep SVDD")
    else:
        print(f"❌ Model not found: {svdd_model}")
        results['deep_svdd'] = False

    # 3. Evaluate MC Dropout (Variance)
    print("\n\n" + "#"*100)
    print("# 3. MC DROPOUT (Variance)")
    print("#"*100)

    mc_var_out = Path(output_base) / "mc_dropout_variance"
    cmd = [
        'python', 'evaluate_ood_methods.py',
        '--method', 'mc_dropout',
        '--features_path', test_features,
        '--output_dir', str(mc_var_out)
    ]
    results['mc_dropout_variance'] = run_command(cmd, "Evaluating MC Dropout (Variance)")

    # 4. Evaluate MC Dropout (Entropy)
    print("\n\n" + "#"*100)
    print("# 4. MC DROPOUT (Entropy)")
    print("#"*100)

    mc_ent_out = Path(output_base) / "mc_dropout_entropy"
    cmd = [
        'python', 'evaluate_ood_methods.py',
        '--method', 'mc_dropout',
        '--features_path', test_features,
        '--output_dir', str(mc_ent_out),
        '--use_entropy'
    ]
    results['mc_dropout_entropy'] = run_command(cmd, "Evaluating MC Dropout (Entropy)")

    return results


def compare_results(results_base):
    """Compare results from all methods."""

    print("\n" + "="*100)
    print("COMPARING ALL OOD METHODS")
    print("="*100)

    methods = [
        ('energy_ood', 'Energy OOD'),
        ('deep_svdd', 'Deep SVDD'),
        ('mc_dropout_variance', 'MC Dropout (Variance)'),
        ('mc_dropout_entropy', 'MC Dropout (Entropy)')
    ]

    comparison_data = []

    for method_dir, method_name in methods:
        results_path = Path(results_base) / method_dir / "threshold_optimization.csv"

        if not results_path.exists():
            print(f"\n❌ Results not found: {results_path}")
            continue

        df = pd.read_csv(results_path)

        # Find best configurations
        best_f1_idx = df['f1_score'].idxmax()
        best_f1 = df.loc[best_f1_idx]

        # Best in reasonable routing range (20-35%)
        reasonable = df[(df['routing_rate'] >= 20) & (df['routing_rate'] <= 35)]
        if len(reasonable) > 0:
            best_reasonable_idx = reasonable['f1_score'].idxmax()
            best_reasonable = df.loc[best_reasonable_idx]
        else:
            best_reasonable = best_f1

        comparison_data.append({
            'method': method_name,
            'best_f1': best_f1['f1_score'],
            'best_f1_threshold': best_f1['threshold'],
            'best_f1_routing': best_f1['routing_rate'],
            'best_f1_precision': best_f1['precision'],
            'best_f1_recall': best_f1['recall'],
            'reasonable_f1': best_reasonable['f1_score'],
            'reasonable_threshold': best_reasonable['threshold'],
            'reasonable_routing': best_reasonable['routing_rate'],
            'reasonable_precision': best_reasonable['precision'],
            'reasonable_recall': best_reasonable['recall'],
            'reasonable_fp_rate': best_reasonable['fp_rate'],
            'reasonable_cat_capture': best_reasonable['catastrophic_capture']
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Display results
    print("\n" + "="*100)
    print("BEST F1 SCORES (Any Routing Rate)")
    print("="*100)
    print(comparison_df[['method', 'best_f1', 'best_f1_routing', 'best_f1_precision',
                        'best_f1_recall']].to_string(index=False))

    print("\n" + "="*100)
    print("BEST RESULTS IN 20-35% ROUTING RANGE")
    print("="*100)
    print(comparison_df[['method', 'reasonable_f1', 'reasonable_routing',
                        'reasonable_precision', 'reasonable_recall',
                        'reasonable_fp_rate', 'reasonable_cat_capture']].to_string(index=False))

    # Save comparison
    comparison_path = Path(results_base) / "method_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Comparison saved to: {comparison_path}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate all OOD methods')
    parser.add_argument('--train_features', type=str,
                        default='features/train_combined_features.npz',
                        help='Path to training features')
    parser.add_argument('--test_features', type=str,
                        default='features/test_3x12cm_consecutive_features.npz',
                        help='Path to test features')
    parser.add_argument('--error_threshold', type=float, default=20.0,
                        help='Error threshold for training (default: 20.0°)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'mps', 'cuda'],
                        help='Device to use (default: cpu)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Base directory for trained models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Base directory for results')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, only evaluate existing models')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation, only train models')

    args = parser.parse_args()

    print("="*100)
    print("OOD-BASED FAILURE DETECTION: MASTER RUNNER")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Training features: {args.train_features}")
    print(f"  Test features: {args.test_features}")
    print(f"  Error threshold: {args.error_threshold}°")
    print(f"  Device: {args.device}")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Skip training: {args.skip_training}")
    print(f"  Skip evaluation: {args.skip_evaluation}")

    # Training
    if not args.skip_training:
        train_results = train_all_methods(
            train_features=args.train_features,
            error_threshold=args.error_threshold,
            device=args.device,
            output_base=args.model_dir
        )

        print("\n" + "="*100)
        print("TRAINING SUMMARY")
        print("="*100)
        for method, success in train_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {method}")

    # Evaluation
    if not args.skip_evaluation:
        # Determine model base directory
        model_base = Path(args.model_dir) / f"energy_ood_{args.error_threshold}deg"
        if not model_base.exists():
            model_base = Path(args.model_dir) / f"deep_svdd_{args.error_threshold}deg"
        if not model_base.exists():
            model_base = Path(args.model_dir)

        # Use appropriate model directories for each method
        eval_results = {}

        # Energy OOD
        energy_base = Path(args.model_dir) / f"energy_ood_{args.error_threshold}deg"
        if energy_base.exists():
            eval_results.update(evaluate_all_methods(
                test_features=args.test_features,
                device=args.device,
                model_base=energy_base,
                output_base=Path(args.results_dir) / "ood_methods"
            ))

        # Deep SVDD
        svdd_base = Path(args.model_dir) / f"deep_svdd_{args.error_threshold}deg"
        if svdd_base.exists():
            svdd_results = evaluate_all_methods(
                test_features=args.test_features,
                device=args.device,
                model_base=svdd_base,
                output_base=Path(args.results_dir) / "ood_methods"
            )
            eval_results.update(svdd_results)

        # MC Dropout (no model needed)
        mc_out = Path(args.results_dir) / "ood_methods"
        mc_results = {
            'mc_dropout_variance': run_command([
                'python', 'evaluate_ood_methods.py',
                '--method', 'mc_dropout',
                '--features_path', args.test_features,
                '--output_dir', str(mc_out / "mc_dropout_variance")
            ], "Evaluating MC Dropout (Variance)"),
            'mc_dropout_entropy': run_command([
                'python', 'evaluate_ood_methods.py',
                '--method', 'mc_dropout',
                '--features_path', args.test_features,
                '--output_dir', str(mc_out / "mc_dropout_entropy"),
                '--use_entropy'
            ], "Evaluating MC Dropout (Entropy)")
        }
        eval_results.update(mc_results)

        print("\n" + "="*100)
        print("EVALUATION SUMMARY")
        print("="*100)
        for method, success in eval_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {method}")

        # Compare results
        compare_results(Path(args.results_dir) / "ood_methods")

    print("\n" + "="*100)
    print("ALL DONE!")
    print("="*100)


if __name__ == "__main__":
    main()
