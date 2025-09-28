#!/usr/bin/env python3
"""
Simple SRP parameter testing using available arguments.
Tests different SNR values and analyzes results.

Usage:
    python simple_srp_parameter_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json
from datetime import datetime

class SimpleSRPTester:
    def __init__(self):
        self.base_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
        self.test_csv = "srp_predicted_failures_test.csv"
        self.results_log = []

    def test_snr_variations(self):
        """Test different SNR values for novel noise."""
        print("TESTING SNR VARIATIONS")
        print("=" * 50)

        # Test different SNR values
        snr_values = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]

        results = []

        for snr in snr_values:
            print(f"\n🎯 Testing SNR: {snr} dB")

            result = self.run_srp_with_snr(snr)
            if result:
                result['snr'] = snr
                results.append(result)
                print(f"  Result: {result['success_rate']:.1f}% success, {result['mae']:.1f}° MAE")
            else:
                print(f"  ❌ Failed")

        return results

    def test_noise_scenes(self):
        """Test different noise scenes."""
        print("\nTESTING DIFFERENT NOISE SCENES")
        print("=" * 50)

        # Available noise scenes (you can add more based on your data)
        noise_scenes = ['Cafeteria1', 'Cafeteria2']

        results = []

        for scene in noise_scenes:
            print(f"\n🎯 Testing Scene: {scene}")

            result = self.run_srp_with_noise_scene(scene)
            if result:
                result['noise_scene'] = scene
                results.append(result)
                print(f"  Result: {result['success_rate']:.1f}% success, {result['mae']:.1f}° MAE")
            else:
                print(f"  ❌ Failed")

        return results

    def test_without_novel_noise(self):
        """Test SRP without novel noise (baseline)."""
        print("\nTESTING WITHOUT NOVEL NOISE (BASELINE)")
        print("=" * 50)

        result = self.run_srp_baseline()
        if result:
            result['condition'] = 'no_novel_noise'
            print(f"Result: {result['success_rate']:.1f}% success, {result['mae']:.1f}° MAE")
            return result
        else:
            print("❌ Failed")
            return None

    def run_srp_with_snr(self, snr_value):
        """Run SRP with specific SNR."""
        cmd = [
            'python', '-m', 'xsrpMain.xsrp.run_SRP',
            '--csv', self.test_csv,
            '--base-dir', self.base_dir,
            '--use_novel_noise',
            '--novel_noise_scene', 'Cafeteria2',
            '--novel_noise_snr', str(snr_value)
        ]

        return self._run_srp_command(cmd, f"snr_{snr_value}")

    def run_srp_with_noise_scene(self, scene):
        """Run SRP with specific noise scene."""
        cmd = [
            'python', '-m', 'xsrpMain.xsrp.run_SRP',
            '--csv', self.test_csv,
            '--base-dir', self.base_dir,
            '--use_novel_noise',
            '--novel_noise_scene', scene
        ]

        return self._run_srp_command(cmd, f"scene_{scene}")

    def run_srp_baseline(self):
        """Run SRP without novel noise."""
        cmd = [
            'python', '-m', 'xsrpMain.xsrp.run_SRP',
            '--csv', self.test_csv,
            '--base-dir', self.base_dir
        ]

        return self._run_srp_command(cmd, "baseline")

    def _run_srp_command(self, cmd, test_name):
        """Execute SRP command and analyze results."""
        try:
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  ❌ SRP failed: {result.stderr}")
                return None

            # Look for results file (SRP creates its own filename)
            # Try common patterns
            possible_files = [
                'srp_predicted_failures_test_srp_results.csv',
                f'srp_results_{test_name}.csv',
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
                # Load and analyze results
                results_df = pd.read_csv(results_file)
                performance = self._analyze_srp_performance(results_df, test_name)

                # Rename results file to avoid conflicts
                new_name = f'srp_results_{test_name}_{datetime.now().strftime("%H%M%S")}.csv'
                Path(results_file).rename(new_name)

                return performance
            else:
                print(f"  ❌ Results file not found")
                return None

        except subprocess.TimeoutExpired:
            print(f"  ❌ SRP timed out")
            return None
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

    def _analyze_srp_performance(self, results_df, test_name):
        """Analyze SRP performance."""
        if 'abs_err_deg' not in results_df.columns:
            print(f"  ❌ No abs_err_deg column found")
            return None

        # Calculate metrics
        mae = results_df['abs_err_deg'].mean()
        failures = (results_df['abs_err_deg'] > 30).sum()
        success_rate = (1 - failures / len(results_df)) * 100

        # Additional statistics
        median_error = results_df['abs_err_deg'].median()
        p90_error = results_df['abs_err_deg'].quantile(0.9)

        performance = {
            'test_name': test_name,
            'mae': mae,
            'success_rate': success_rate,
            'failures': failures,
            'total_cases': len(results_df),
            'median_error': median_error,
            'p90_error': p90_error,
            'improvement_over_crnn': success_rate - 23.9,  # CRNN baseline
            'timestamp': datetime.now().isoformat()
        }

        return performance

    def save_results(self, all_results):
        """Save all test results."""
        if not all_results:
            print("No results to save")
            return

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_srp_parameter_test_{timestamp}.csv"
        results_df.to_csv(filename, index=False)

        # Find best result
        best_result = max(all_results, key=lambda x: x['success_rate'])

        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_tests': len(all_results),
            'best_result': best_result,
            'baseline_to_beat': 23.9,
            'best_improvement': best_result['improvement_over_crnn']
        }

        summary_file = f"simple_srp_test_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        print(f"Tests completed: {len(all_results)}")
        print(f"Best performance:")
        print(f"  - Test: {best_result['test_name']}")
        print(f"  - Success rate: {best_result['success_rate']:.1f}%")
        print(f"  - Improvement vs CRNN: {best_result['improvement_over_crnn']:+.1f}%")
        print(f"  - MAE: {best_result['mae']:.1f}°")

        if best_result['success_rate'] > 23.9:
            print("✅ Found improvement over CRNN baseline!")
        else:
            print("❌ No improvement over CRNN baseline")

        print(f"\nFiles saved:")
        print(f"  - {filename}")
        print(f"  - {summary_file}")

def main():
    """Main testing function."""
    print("SIMPLE SRP PARAMETER TESTING")
    print("=" * 80)

    tester = SimpleSRPTester()

    # Verify input file exists
    if not Path(tester.test_csv).exists():
        print(f"❌ Test CSV not found: {tester.test_csv}")
        print("Please run the hybrid test first to generate this file")
        return

    all_results = []

    try:
        # Test 1: Baseline (no novel noise)
        print("=" * 80)
        baseline_result = tester.test_without_novel_noise()
        if baseline_result:
            all_results.append(baseline_result)

        # Test 2: SNR variations
        print("=" * 80)
        snr_results = tester.test_snr_variations()
        all_results.extend(snr_results)

        # Test 3: Different noise scenes (if you have them)
        # print("=" * 80)
        # scene_results = tester.test_noise_scenes()
        # all_results.extend(scene_results)

        # Save and analyze results
        if all_results:
            tester.save_results(all_results)
        else:
            print("❌ No successful test results")

    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        if all_results:
            tester.save_results(all_results)

if __name__ == "__main__":
    main()