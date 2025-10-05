#!/usr/bin/env python3
"""
Find SRP Strengths Analysis - Option 3 Approach

This script reverses the analysis approach:
1. Run SRP on the ENTIRE test dataset with all 3 array configurations (6cm, 12cm, 18cm)
2. Identify cases where SRP performs excellently (<1° error)
3. Analyze CRNN performance on those same excellent SRP cases
4. Look for genuine scenarios where classical methods have advantages

Goal: Find where SRP succeeds and check if CRNN struggles there,
rather than trying to rescue CRNN failures with SRP.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class SRPStrengthAnalyzer:
    def __init__(self, base_dir="/Users/danieltoberman/Documents/git/Thesis"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "hybrid_system" / "analysis" / "srp_strengths_analysis"
        self.results_dir.mkdir(exist_ok=True)

        # Array configurations
        self.array_configs = {
            "6cm": {"mics": [0, 1, 2, 3, 4, 5, 6, 7, 8], "description": "6cm diameter baseline"},
            "12cm": {"mics": [0] + list(range(9, 17)), "description": "12cm diameter medium circle"},
            "18cm": {"mics": [0] + list(range(17, 25)), "description": "18cm diameter large circle"}
        }

        # Test dataset path
        self.test_csv = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
        self.base_data_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"

        # Excellence threshold
        self.excellence_threshold = 1.0  # <1 degree error

    def run_srp_all_arrays(self, use_novel_noise=True):
        """Run SRP on entire test dataset with all array configurations"""
        print("🎯 Running SRP on entire test dataset with all array configurations...")

        results = {}

        for array_name, config in self.array_configs.items():
            print(f"\n📡 Testing {array_name} array ({config['description']})...")

            # Prepare output file
            output_file = self.results_dir / f"srp_full_dataset_{array_name}_results.csv"

            # Build SRP command
            cmd = [
                "python", "-m", "xsrpMain.xsrp.run_SRP",
                "--csv", self.test_csv,
                "--base-dir", self.base_data_dir,
                "--array_diameter", array_name,
                "--output_file", str(output_file)
            ]

            if use_novel_noise:
                cmd.extend([
                    "--use_novel_noise",
                    "--novel_noise_scene", "Cafeteria2",
                    "--novel_noise_snr", "5.0"
                ])

            print(f"Running command: {' '.join(cmd)}")

            try:
                # Run SRP
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

                if result.returncode == 0:
                    print(f"✅ {array_name} array completed successfully")

                    # Load and analyze results
                    if output_file.exists():
                        df = pd.read_csv(output_file)
                        results[array_name] = {
                            "data": df,
                            "file": output_file,
                            "config": config
                        }
                        print(f"📊 {array_name}: {len(df)} results processed")
                    else:
                        print(f"⚠️ Output file not found for {array_name}")
                else:
                    print(f"❌ {array_name} array failed:")
                    print(f"STDERR: {result.stderr}")

            except Exception as e:
                print(f"❌ Error running {array_name} array: {e}")

        return results

    def analyze_excellent_cases(self, srp_results):
        """Find cases where SRP performs excellently (<1° error)"""
        print(f"\n🔍 Analyzing SRP excellent cases (MAE < {self.excellence_threshold}°)...")

        excellent_cases = {}

        for array_name, result_data in srp_results.items():
            df = result_data["data"]

            # Calculate errors (assuming columns: predicted_angle, true_angle)
            if "predicted_angle" in df.columns and "true_angle" in df.columns:
                # Calculate absolute angular error
                df["angular_error"] = np.abs(df["predicted_angle"] - df["true_angle"])

                # Handle wrap-around for angles (e.g., 359° vs 1° should be 2° error, not 358°)
                wrap_error = 360 - df["angular_error"]
                df["angular_error"] = np.minimum(df["angular_error"], wrap_error)

                # Find excellent cases
                excellent_mask = df["angular_error"] < self.excellence_threshold
                excellent_df = df[excellent_mask].copy()

                print(f"📈 {array_name}: {len(excellent_df)}/{len(df)} excellent cases ({len(excellent_df)/len(df)*100:.1f}%)")

                # Save excellent cases
                excellent_file = self.results_dir / f"srp_excellent_cases_{array_name}.csv"
                excellent_df.to_csv(excellent_file, index=False)

                excellent_cases[array_name] = {
                    "data": excellent_df,
                    "file": excellent_file,
                    "count": len(excellent_df),
                    "percentage": len(excellent_df)/len(df)*100,
                    "mean_error": excellent_df["angular_error"].mean(),
                    "std_error": excellent_df["angular_error"].std()
                }

        return excellent_cases

    def get_crnn_performance_on_cases(self, case_indices, test_dataset_path=None):
        """Get CRNN performance on specific test cases"""
        print(f"\n🧠 Analyzing CRNN performance on {len(case_indices)} selected cases...")

        # For now, we'll need to run CRNN or load existing results
        # This would require running CRNN on the same test set with novel noise

        # Placeholder - in practice, you'd either:
        # 1. Load existing CRNN results from previous runs
        # 2. Run CRNN on these specific cases

        crnn_results_file = self.base_dir / "hybrid_system" / "results" / "crnn_full_dataset_results.csv"

        if crnn_results_file.exists():
            print(f"📁 Loading existing CRNN results from {crnn_results_file}")
            crnn_df = pd.read_csv(crnn_results_file)

            # Filter to selected cases (assuming some common identifier)
            # This depends on how the results are indexed
            selected_crnn = crnn_df.iloc[case_indices] if len(case_indices) <= len(crnn_df) else crnn_df

            return selected_crnn
        else:
            print("⚠️ No existing CRNN results found. You may need to run CRNN first.")
            return None

    def compare_srp_vs_crnn(self, srp_excellent_cases):
        """Compare SRP vs CRNN performance on SRP's excellent cases"""
        print(f"\n⚖️ Comparing SRP vs CRNN on SRP excellent cases...")

        comparison_results = {}

        for array_name, srp_data in srp_excellent_cases.items():
            print(f"\n🔍 Analyzing {array_name} excellent cases...")

            excellent_df = srp_data["data"]

            if len(excellent_df) == 0:
                print(f"⚠️ No excellent cases found for {array_name}")
                continue

            # Get case indices (assuming row numbers correspond to test cases)
            case_indices = excellent_df.index.tolist()

            # Get CRNN performance on these cases
            crnn_on_cases = self.get_crnn_performance_on_cases(case_indices)

            if crnn_on_cases is not None:
                # Calculate CRNN errors on these cases
                # This assumes CRNN results have similar structure
                # You may need to adjust based on actual CRNN output format

                comparison_results[array_name] = {
                    "srp_excellent_count": len(excellent_df),
                    "srp_mean_error": srp_data["mean_error"],
                    "cases_analyzed": len(case_indices),
                    "array_config": srp_data
                }

        return comparison_results

    def generate_report(self, srp_results, excellent_cases, comparison_results):
        """Generate comprehensive analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "SRP Strengths Analysis - Option 3",
            "excellence_threshold": self.excellence_threshold,
            "summary": {
                "total_test_cases": len(pd.read_csv(self.test_csv)),
                "arrays_tested": list(self.array_configs.keys()),
                "excellent_cases_by_array": {
                    array: data["count"] for array, data in excellent_cases.items()
                }
            },
            "detailed_results": {
                array: {
                    "excellent_count": data["count"],
                    "excellent_percentage": data["percentage"],
                    "mean_error": data["mean_error"],
                    "std_error": data["std_error"]
                } for array, data in excellent_cases.items()
            }
        }

        # Save report
        report_file = self.results_dir / f"srp_strengths_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📊 Analysis report saved to: {report_file}")

        # Print summary
        print(f"\n🎯 SRP Strengths Analysis Summary:")
        print(f"Excellence threshold: <{self.excellence_threshold}° error")
        print(f"Total test cases: {report['summary']['total_test_cases']}")

        for array, data in excellent_cases.items():
            print(f"  {array}: {data['count']} excellent cases ({data['percentage']:.1f}%) - μ={data['mean_error']:.3f}°")

        return report

    def run_full_analysis(self, use_novel_noise=True):
        """Run the complete SRP strengths analysis"""
        print("🚀 Starting SRP Strengths Analysis (Option 3)")
        print("=" * 60)

        # Step 1: Run SRP on all arrays
        srp_results = self.run_srp_all_arrays(use_novel_noise)

        if not srp_results:
            print("❌ No SRP results obtained. Aborting analysis.")
            return None

        # Step 2: Identify excellent cases
        excellent_cases = self.analyze_excellent_cases(srp_results)

        # Step 3: Compare with CRNN
        comparison_results = self.compare_srp_vs_crnn(excellent_cases)

        # Step 4: Generate report
        report = self.generate_report(srp_results, excellent_cases, comparison_results)

        print(f"\n✅ SRP Strengths Analysis completed!")
        print(f"📁 Results saved in: {self.results_dir}")

        return {
            "srp_results": srp_results,
            "excellent_cases": excellent_cases,
            "comparison_results": comparison_results,
            "report": report
        }

def main():
    """Main execution function"""
    analyzer = SRPStrengthAnalyzer()

    # Run the full analysis
    results = analyzer.run_full_analysis(use_novel_noise=True)

    if results:
        print("\n🎉 Analysis completed successfully!")

        # Print key findings
        excellent_cases = results["excellent_cases"]
        if excellent_cases:
            print("\n🏆 Key Findings:")
            for array, data in excellent_cases.items():
                print(f"  • {array} array: {data['count']} cases with <1° error ({data['percentage']:.1f}%)")

        print("\n💡 Next step: Analyze CRNN performance on these SRP-excellent cases")
        print("    to identify genuine classical method advantages!")

if __name__ == "__main__":
    main()