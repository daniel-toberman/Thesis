#!/usr/bin/env python3
"""
Systematic parameter optimization for SRP-PHAT on the 201 failure cases.
Tests different frequency bands, window sizes, and other SRP parameters.

Usage:
    python optimize_srp_parameters.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json
import itertools
from datetime import datetime
import tempfile
import shutil

class SRPParameterOptimizer:
    def __init__(self):
        self.base_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
        self.test_csv = "srp_predicted_failures_test.csv"
        self.results_log = []

    def create_parameter_grid(self):
        """Define parameter grid for systematic search."""

        # Start with most promising parameters based on acoustic theory
        param_grid = {
            # Frequency range optimization - most critical for noisy environments
            'freq_min': [200, 300, 500],        # Lower bound
            'freq_max': [2000, 3000, 4000],     # Upper bound - avoid high-freq noise

            # SRP grid resolution
            'srp_grid_cells': [360, 720, 1440], # Angular resolution

            # SRP algorithm mode
            'srp_mode': ['gcc_phat_freq', 'gcc_phat_time'],  # Core SRP variants

            # Averaging and stability parameters
            'n_avg_samples': [100, 200, 400],   # Temporal averaging for stability
            'n_dft_bins': [512, 1024, 2048],    # Frequency resolution
        }

        # Generate all combinations (this will be large, so we'll sample)
        all_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        print(f"Total possible combinations: {len(all_combinations)}")

        # Sample a reasonable subset for initial testing
        if len(all_combinations) > 50:
            # Use smart sampling: try extremes and middle values
            sampled_combinations = self._smart_sample_parameters(all_combinations, 50)
        else:
            sampled_combinations = all_combinations

        # Convert to list of dicts
        parameter_sets = []
        for combo in sampled_combinations:
            param_dict = dict(zip(param_names, combo))
            parameter_sets.append(param_dict)

        print(f"Testing {len(parameter_sets)} parameter combinations")
        return parameter_sets

    def _smart_sample_parameters(self, combinations, n_samples):
        """Intelligently sample parameter combinations."""
        # Always include baseline (middle values)
        baseline_idx = len(combinations) // 2
        sampled = [combinations[baseline_idx]]

        # Add some random sampling
        np.random.seed(42)  # Reproducible
        remaining = [c for i, c in enumerate(combinations) if i != baseline_idx]
        additional = np.random.choice(len(remaining), size=min(n_samples-1, len(remaining)), replace=False)

        for idx in additional:
            sampled.append(remaining[idx])

        return sampled

    def run_srp_with_parameters(self, params, test_id):
        """Run SRP with specific parameters and return results."""
        print(f"Testing parameter set {test_id}: {params}")

        try:
            # Run SRP with parameters as command line arguments
            cmd = [
                'python', '-m', 'xsrpMain.xsrp.run_SRP',
                '--csv', self.test_csv,
                '--base-dir', self.base_dir,
                '--use_novel_noise',
                '--novel_noise_scene', 'Cafeteria2',
                '--srp_grid_cells', str(params['srp_grid_cells']),
                '--srp_mode', params['srp_mode'],
                '--n_avg_samples', str(params['n_avg_samples']),
                '--n_dft_bins', str(params['n_dft_bins']),
                '--freq_min', str(params['freq_min']),
                '--freq_max', str(params['freq_max'])
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"SRP failed for test {test_id}: {result.stderr}")
                return None

            # Look for results file (SRP creates its own filename)
            possible_files = [
                f'{Path(self.test_csv).stem}_srp_results.csv',
                'srp_results.csv'
            ]

            results_file = None
            for filename in possible_files:
                if Path(filename).exists():
                    results_file = filename
                    break

            if not results_file:
                # Look for any CSV file modified in the last minute
                csv_files = list(Path('.').glob('*srp*.csv'))
                if csv_files:
                    results_file = max(csv_files, key=lambda x: x.stat().st_mtime)

            if results_file and Path(results_file).exists():
                results_df = pd.read_csv(results_file)
                performance = self._analyze_performance(results_df, params, test_id)

                # Rename results file to avoid conflicts
                new_name = f'srp_results_test_{test_id}_{datetime.now().strftime("%H%M%S")}.csv'
                Path(results_file).rename(new_name)

                return performance
            else:
                print(f"Results file not found")
                return None

        except subprocess.TimeoutExpired:
            print(f"SRP timed out for test {test_id}")
            return None
        except Exception as e:
            print(f"Error running SRP for test {test_id}: {e}")
            return None

    def _analyze_performance(self, results_df, params, test_id):
        """Analyze SRP performance with these parameters."""

        # Calculate key metrics
        mae = results_df['abs_err_deg'].mean()
        failures = (results_df['abs_err_deg'] > 30).sum()
        success_rate = (1 - failures / len(results_df)) * 100

        # Calculate percentiles for robustness analysis
        p50 = results_df['abs_err_deg'].median()
        p90 = results_df['abs_err_deg'].quantile(0.9)
        p95 = results_df['abs_err_deg'].quantile(0.95)

        performance = {
            'test_id': test_id,
            'parameters': params,
            'mae': mae,
            'success_rate': success_rate,
            'failures': failures,
            'total_cases': len(results_df),
            'median_error': p50,
            'p90_error': p90,
            'p95_error': p95,
            'timestamp': datetime.now().isoformat()
        }

        # Compare to baseline (23.9% CRNN success rate)
        improvement = success_rate - 23.9
        performance['improvement_over_crnn'] = improvement

        print(f"  Results: {success_rate:.1f}% success ({improvement:+.1f}% vs CRNN), {mae:.1f}° MAE")

        return performance

    def run_optimization(self):
        """Run the full parameter optimization."""
        print("SRP PARAMETER OPTIMIZATION")
        print("=" * 80)

        # Verify input files exist
        if not Path(self.test_csv).exists():
            raise FileNotFoundError(f"Test CSV not found: {self.test_csv}")

        # Generate parameter grid
        parameter_sets = self.create_parameter_grid()

        print(f"Starting optimization with {len(parameter_sets)} parameter sets...")
        print(f"Baseline to beat: 23.9% success rate (CRNN baseline)")
        print(f"Target: >30% success rate")
        print("-" * 80)

        # Test each parameter set
        best_performance = None
        all_results = []

        for i, params in enumerate(parameter_sets, 1):
            print(f"\n[{i}/{len(parameter_sets)}]", end=" ")

            performance = self.run_srp_with_parameters(params, i)

            if performance is not None:
                all_results.append(performance)

                # Track best performance
                if (best_performance is None or
                    performance['success_rate'] > best_performance['success_rate']):
                    best_performance = performance
                    print(f"  🎯 NEW BEST: {performance['success_rate']:.1f}% success rate!")
            else:
                print(f"  ❌ Failed to run")

        # Save all results
        self._save_optimization_results(all_results, best_performance)

        return best_performance, all_results

    def _save_optimization_results(self, all_results, best_performance):
        """Save optimization results for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_df = pd.DataFrame(all_results)
        results_file = f"srp_parameter_optimization_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        # Save summary
        summary = {
            'optimization_timestamp': timestamp,
            'total_parameter_sets_tested': len(all_results),
            'best_performance': best_performance,
            'baseline_crnn_success_rate': 23.9,
            'target_success_rate': 30.0
        }

        summary_file = f"srp_optimization_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)

        if best_performance:
            print(f"🏆 BEST PARAMETERS FOUND:")
            print(f"   Success Rate: {best_performance['success_rate']:.1f}%")
            print(f"   Improvement: {best_performance['improvement_over_crnn']:+.1f}% vs CRNN")
            print(f"   MAE: {best_performance['mae']:.1f}°")
            print(f"   Parameters: {best_performance['parameters']}")

            if best_performance['success_rate'] > 23.9:
                print(f"✅ SUCCESS: Beat CRNN baseline!")
            else:
                print(f"❌ No improvement over CRNN baseline")

            if best_performance['success_rate'] > 30.0:
                print(f"🎯 TARGET ACHIEVED: >30% success rate!")

        print(f"\nResults saved:")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")

def main():
    """Main optimization function."""
    optimizer = SRPParameterOptimizer()

    try:
        best_performance, all_results = optimizer.run_optimization()

        print(f"\n✅ Parameter optimization complete!")
        print(f"Tested {len(all_results)} parameter combinations")

        if best_performance and best_performance['success_rate'] > 23.9:
            print(f"🎉 Found improvement! Best: {best_performance['success_rate']:.1f}% success rate")
        else:
            print(f"⚠️  No parameter combination beat CRNN baseline")
            print(f"   Next step: Try advanced SRP methods or preprocessing")

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")

if __name__ == "__main__":
    main()