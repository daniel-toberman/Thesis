#!/usr/bin/env python3
"""
Complete SRP Strengths Analysis - Option 3 Implementation

This script implements the "Classical Strengths First" approach:
1. Run SRP on entire test dataset with all 3 array configurations
2. Find cases where SRP performs excellently (<1° error)
3. Check CRNN performance on those same cases
4. Identify genuine scenarios where classical methods excel

Goal: Find where SRP succeeds and CRNN struggles/fails
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time
import shutil

class CompleteSRPAnalysis:
    def __init__(self, base_dir="/Users/danieltoberman/Documents/git/Thesis"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "hybrid_system" / "analysis" / "srp_strengths_complete"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Test dataset paths
        self.test_csv = "/Users/danieltoberman/Documents/RealMAN_9_channels/test/test_static_source_location.csv"
        self.data_base_dir = "/Users/danieltoberman/Documents/RealMAN_9_channels/extracted"

        # SRP automatically saves to this location
        self.srp_default_output = Path("/Users/danieltoberman/Documents/RealMAN_9_channels/test/test_static_source_location_srp_results.csv")

        # Array configurations
        self.arrays = ["6cm", "12cm", "18cm"]

        # Excellence threshold
        self.excellence_threshold = 1.0  # <1° error

    def run_srp_for_array(self, array_diameter, use_novel_noise=True):
        """Run SRP for a specific array configuration"""
        print(f"\n📡 Running SRP with {array_diameter} array...")

        # Build command
        cmd = [
            "python", "-m", "xsrpMain.xsrp.run_SRP",
            "--csv", self.test_csv,
            "--base-dir", self.data_base_dir,
            "--array_diameter", array_diameter
        ]

        if use_novel_noise:
            cmd.extend([
                "--use_novel_noise",
                "--novel_noise_scene", "Cafeteria2",
                "--novel_noise_snr", "5.0"
            ])

        print(f"Command: {' '.join(cmd)}")

        try:
            # Clean up any existing output file
            if self.srp_default_output.exists():
                self.srp_default_output.unlink()

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ {array_diameter} completed in {duration:.1f}s")

                # Move the results file to our organized location
                array_output_file = self.results_dir / f"srp_results_{array_diameter}.csv"

                if self.srp_default_output.exists():
                    shutil.move(str(self.srp_default_output), str(array_output_file))
                    print(f"📊 Results saved to: {array_output_file}")

                    # Load and validate results
                    df = pd.read_csv(array_output_file)
                    print(f"📈 Processed {len(df)} test cases")

                    return {
                        "success": True,
                        "duration": duration,
                        "results_file": array_output_file,
                        "num_cases": len(df),
                        "df": df
                    }
                else:
                    print(f"⚠️ No output file generated")
                    return {"success": False, "error": "No output file generated"}

            else:
                print(f"❌ {array_diameter} failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "returncode": result.returncode
                }

        except Exception as e:
            print(f"💥 Exception running {array_diameter}: {e}")
            return {"success": False, "error": str(e)}

    def find_excellent_cases(self, array_results):
        """Find cases where each array performs excellently (<1° error)"""
        print(f"\n🔍 Finding excellent cases (<{self.excellence_threshold}° error)...")

        excellent_analysis = {}

        for array, result_data in array_results.items():
            if not result_data.get("success", False):
                print(f"⚠️ Skipping {array} - no successful results")
                continue

            df = result_data["df"]

            # Find excellent cases
            excellent_mask = df["abs_err_deg"] < self.excellence_threshold
            excellent_df = df[excellent_mask].copy()

            # Save excellent cases
            excellent_file = self.results_dir / f"excellent_cases_{array}.csv"
            excellent_df.to_csv(excellent_file, index=False)

            # Calculate statistics
            stats = {
                "total_cases": len(df),
                "excellent_count": len(excellent_df),
                "excellent_percentage": len(excellent_df) / len(df) * 100,
                "mean_error": df["abs_err_deg"].mean(),
                "median_error": df["abs_err_deg"].median(),
                "excellent_mean_error": excellent_df["abs_err_deg"].mean() if len(excellent_df) > 0 else None,
                "excellent_file": excellent_file,
                "excellent_cases": excellent_df
            }

            excellent_analysis[array] = stats

            print(f"📊 {array}: {stats['excellent_count']}/{stats['total_cases']} excellent cases ({stats['excellent_percentage']:.1f}%)")
            if stats['excellent_count'] > 0:
                print(f"    Mean error in excellent cases: {stats['excellent_mean_error']:.3f}°")

        return excellent_analysis

    def analyze_best_array_performance(self, excellent_analysis):
        """Find which array performs best and analyze the overlap"""
        print(f"\n🏆 Analyzing best array performance...")

        # Find best array by percentage of excellent cases
        best_array = max(excellent_analysis.keys(),
                        key=lambda a: excellent_analysis[a]['excellent_percentage'])

        best_stats = excellent_analysis[best_array]
        print(f"🥇 Best array: {best_array} with {best_stats['excellent_percentage']:.1f}% excellent cases")

        # Analyze overlap between arrays (cases that are excellent for multiple arrays)
        if len(excellent_analysis) > 1:
            print(f"\n🔗 Analyzing case overlaps...")

            # Get filenames of excellent cases for each array
            excellent_files = {}
            for array, stats in excellent_analysis.items():
                if stats['excellent_count'] > 0:
                    excellent_files[array] = set(stats['excellent_cases']['filename'].tolist())

            # Find common excellent cases
            if len(excellent_files) > 1:
                arrays = list(excellent_files.keys())
                common_cases = excellent_files[arrays[0]]
                for array in arrays[1:]:
                    common_cases = common_cases.intersection(excellent_files[array])

                print(f"🤝 {len(common_cases)} cases are excellent for all arrays")

                # Find array-specific excellent cases
                for array in arrays:
                    unique_cases = excellent_files[array]
                    for other_array in arrays:
                        if other_array != array:
                            unique_cases = unique_cases - excellent_files[other_array]
                    print(f"🎯 {array} has {len(unique_cases)} unique excellent cases")

        return {
            "best_array": best_array,
            "best_stats": best_stats,
            "excellent_files": excellent_files if len(excellent_analysis) > 1 else {},
            "common_cases": common_cases if len(excellent_analysis) > 1 else set()
        }

    def get_crnn_results_for_comparison(self):
        """Get CRNN results on the same test dataset for comparison"""
        print(f"\n🧠 Loading CRNN results for comparison...")

        # Look for existing CRNN results - prioritize novel noise results
        possible_crnn_files = [
            self.base_dir / "hybrid_system" / "analysis" / "crnn_predictions" / "crnn_predictions_with_confidence_NOVEL_NOISE.csv",
            self.base_dir / "crnn_predictions" / "crnn_predictions_with_confidence_NOVEL_NOISE.csv",
            self.base_dir / "hybrid_system" / "analysis" / "crnn_predictions" / "crnn_predictions_with_confidence_clean.csv",
            self.base_dir / "hybrid_system" / "results" / "crnn_full_dataset_results.csv",
            self.base_dir / "test_detailed_results.csv",
        ]

        crnn_results = None
        for crnn_file in possible_crnn_files:
            if crnn_file.exists():
                print(f"📁 Found CRNN results at: {crnn_file}")
                try:
                    crnn_df = pd.read_csv(crnn_file)
                    print(f"📊 Loaded {len(crnn_df)} CRNN results")
                    crnn_results = {
                        "file": crnn_file,
                        "df": crnn_df
                    }
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {crnn_file}: {e}")
                    continue

        if crnn_results is None:
            print(f"⚠️ No CRNN results found. You may need to run CRNN on the test set first.")
            print(f"💡 Suggested command:")
            print(f"python SSL/run_CRNN.py test --ckpt_path '08_CRNN/checkpoints/best_valid_loss0.0220.ckpt' --use_novel_noise --novel_noise_scene Cafeteria2")

        return crnn_results

    def compare_srp_vs_crnn_on_excellent_cases(self, excellent_analysis, crnn_results):
        """Compare SRP vs CRNN performance on SRP's excellent cases"""
        if crnn_results is None:
            print(f"⚠️ Cannot compare - no CRNN results available")
            return None

        print(f"\n⚖️ Comparing SRP vs CRNN on SRP excellent cases...")

        comparisons = {}
        crnn_df = crnn_results["df"]

        for array, stats in excellent_analysis.items():
            if stats['excellent_count'] == 0:
                continue

            print(f"\n🔍 Analyzing {array} excellent cases...")

            excellent_cases = stats['excellent_cases']
            comparison_data = []

            # For each excellent SRP case, find corresponding CRNN performance
            for idx, row in excellent_cases.iterrows():
                filename = row['filename']
                srp_error = row['abs_err_deg']
                gt_angle = row['gt_angle_deg']
                srp_index = row['idx']  # Use SRP index to match with CRNN

                # Find matching CRNN result by index position
                crnn_match = None
                if srp_index < len(crnn_df):
                    crnn_match = crnn_df.iloc[srp_index]
                elif 'global_idx' in crnn_df.columns:
                    # Try matching by global_idx
                    crnn_matches = crnn_df[crnn_df['global_idx'] == srp_index]
                    if len(crnn_matches) > 0:
                        crnn_match = crnn_matches.iloc[0]

                if crnn_match is not None:
                    # Get CRNN predictions using the correct column names
                    crnn_predicted = crnn_match.get('pred_angle', None)
                    crnn_gt = crnn_match.get('gt_angle', None)

                    if crnn_predicted is not None and crnn_gt is not None:
                        crnn_error = abs(crnn_predicted - crnn_gt)

                        comparison_data.append({
                            'idx': srp_index,
                            'filename': filename,
                            'gt_angle': gt_angle,
                            'srp_predicted': row['srp_angle_deg'],
                            'srp_error': srp_error,
                            'crnn_predicted': crnn_predicted,
                            'crnn_error': crnn_error,
                            'srp_better': srp_error < crnn_error,
                            'crnn_failed': crnn_error > 30.0,  # Using 30° as failure threshold
                            'both_excellent': srp_error < 1.0 and crnn_error < 1.0,
                            'srp_excellent_crnn_poor': srp_error < 1.0 and crnn_error > 30.0
                        })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)

                # Save comparison
                comparison_file = self.results_dir / f"srp_vs_crnn_comparison_{array}.csv"
                comparison_df.to_csv(comparison_file, index=False)

                # Calculate summary statistics
                summary = {
                    "total_comparisons": len(comparison_df),
                    "srp_better_count": comparison_df['srp_better'].sum(),
                    "srp_better_percentage": comparison_df['srp_better'].mean() * 100,
                    "crnn_failures_on_srp_excellent": comparison_df['crnn_failed'].sum(),
                    "crnn_failure_rate_on_srp_excellent": comparison_df['crnn_failed'].mean() * 100,
                    "both_excellent_count": comparison_df['both_excellent'].sum(),
                    "both_excellent_percentage": comparison_df['both_excellent'].mean() * 100,
                    "srp_excellent_crnn_poor_count": comparison_df['srp_excellent_crnn_poor'].sum(),
                    "srp_excellent_crnn_poor_percentage": comparison_df['srp_excellent_crnn_poor'].mean() * 100,
                    "mean_srp_error": comparison_df['srp_error'].mean(),
                    "mean_crnn_error": comparison_df['crnn_error'].mean(),
                    "comparison_file": comparison_file
                }

                comparisons[array] = summary

                print(f"📊 {array} comparison:")
                print(f"    Cases compared: {summary['total_comparisons']}")
                print(f"    SRP better: {summary['srp_better_count']} ({summary['srp_better_percentage']:.1f}%)")
                print(f"    Both methods excellent: {summary['both_excellent_count']} ({summary['both_excellent_percentage']:.1f}%)")
                print(f"    SRP excellent, CRNN poor (>30°): {summary['srp_excellent_crnn_poor_count']} ({summary['srp_excellent_crnn_poor_percentage']:.1f}%)")
                print(f"    CRNN failures on SRP excellent cases: {summary['crnn_failures_on_srp_excellent']} ({summary['crnn_failure_rate_on_srp_excellent']:.1f}%)")

        return comparisons

    def generate_comprehensive_report(self, array_results, excellent_analysis, best_analysis, comparisons):
        """Generate comprehensive analysis report"""
        print(f"\n📄 Generating comprehensive report...")

        report = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "Complete SRP Strengths Analysis - Option 3",
                "excellence_threshold_degrees": self.excellence_threshold,
                "test_dataset": str(self.test_csv),
                "arrays_tested": self.arrays
            },
            "dataset_info": {
                "total_test_cases": excellent_analysis[list(excellent_analysis.keys())[0]]['total_cases'] if excellent_analysis else 0,
                "novel_noise_scene": "Cafeteria2",
                "novel_noise_snr": 5.0
            },
            "srp_performance": {
                array: {
                    "success": data.get("success", False),
                    "total_cases": data.get("num_cases", 0) if data.get("success") else 0,
                    "processing_duration_seconds": data.get("duration", 0) if data.get("success") else 0
                } for array, data in array_results.items()
            },
            "excellent_cases_analysis": {
                array: {
                    "excellent_count": stats["excellent_count"],
                    "excellent_percentage": stats["excellent_percentage"],
                    "mean_error_degrees": stats["mean_error"],
                    "excellent_mean_error_degrees": stats["excellent_mean_error"]
                } for array, stats in excellent_analysis.items()
            },
            "best_performance": {
                "best_array": best_analysis.get("best_array", "N/A"),
                "best_percentage": best_analysis.get("best_stats", {}).get("excellent_percentage", 0)
            },
            "srp_vs_crnn_comparison": comparisons if comparisons else "No CRNN data available",
            "key_findings": self.generate_key_findings(excellent_analysis, comparisons)
        }

        # Save report
        report_file = self.results_dir / f"complete_srp_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"💾 Report saved to: {report_file}")

        # Print key findings
        print(f"\n🎯 Key Findings Summary:")
        for finding in report["key_findings"]:
            print(f"  • {finding}")

        return report

    def generate_key_findings(self, excellent_analysis, comparisons):
        """Generate key findings from the analysis"""
        findings = []

        # SRP Performance findings
        if excellent_analysis:
            total_excellent = sum(stats["excellent_count"] for stats in excellent_analysis.values())
            total_cases = list(excellent_analysis.values())[0]["total_cases"]

            findings.append(f"SRP found {total_excellent} excellent cases (<1° error) across all arrays out of {total_cases} total cases")

            # Best array finding
            best_array = max(excellent_analysis.keys(),
                           key=lambda a: excellent_analysis[a]['excellent_percentage'])
            best_percentage = excellent_analysis[best_array]['excellent_percentage']
            findings.append(f"Best performing array: {best_array} with {best_percentage:.1f}% excellent cases")

            # Array comparison
            if len(excellent_analysis) > 1:
                percentages = [stats['excellent_percentage'] for stats in excellent_analysis.values()]
                findings.append(f"Array performance range: {min(percentages):.1f}% - {max(percentages):.1f}% excellent cases")

        # CRNN comparison findings
        if comparisons:
            for array, comp in comparisons.items():
                if comp["crnn_failure_rate_on_srp_excellent"] > 0:
                    findings.append(f"CRNN fails on {comp['crnn_failure_rate_on_srp_excellent']:.1f}% of {array} SRP excellent cases")

        return findings

    def run_complete_analysis(self):
        """Run the complete SRP strengths analysis"""
        print("🚀 Starting Complete SRP Strengths Analysis")
        print("=" * 60)
        print(f"Excellence threshold: <{self.excellence_threshold}° error")
        print(f"Arrays to test: {', '.join(self.arrays)}")
        print(f"Results directory: {self.results_dir}")

        # Step 1: Run SRP for all arrays
        print(f"\n📡 Step 1: Running SRP for all array configurations...")
        array_results = {}
        total_start_time = time.time()

        for array in self.arrays:
            result = self.run_srp_for_array(array, use_novel_noise=True)
            array_results[array] = result
            time.sleep(1)  # Brief pause between runs

        total_srp_time = time.time() - total_start_time

        # Check if we have any successful results
        successful_arrays = [a for a, r in array_results.items() if r.get("success", False)]
        if not successful_arrays:
            print(f"❌ No successful SRP runs. Cannot proceed with analysis.")
            return None

        print(f"✅ SRP runs completed in {total_srp_time/60:.1f} minutes")
        print(f"Successful arrays: {successful_arrays}")

        # Step 2: Find excellent cases
        print(f"\n🔍 Step 2: Finding excellent cases...")
        excellent_analysis = self.find_excellent_cases(array_results)

        # Step 3: Analyze best performance
        print(f"\n🏆 Step 3: Analyzing best array performance...")
        best_analysis = self.analyze_best_array_performance(excellent_analysis)

        # Step 4: Get CRNN results for comparison
        print(f"\n🧠 Step 4: Getting CRNN results...")
        crnn_results = self.get_crnn_results_for_comparison()

        # Step 5: Compare SRP vs CRNN
        print(f"\n⚖️ Step 5: Comparing SRP vs CRNN...")
        comparisons = self.compare_srp_vs_crnn_on_excellent_cases(excellent_analysis, crnn_results)

        # Step 6: Generate comprehensive report
        print(f"\n📄 Step 6: Generating report...")
        report = self.generate_comprehensive_report(array_results, excellent_analysis, best_analysis, comparisons)

        print(f"\n🎉 Complete SRP Strengths Analysis finished!")
        print(f"📁 All results saved in: {self.results_dir}")

        return {
            "array_results": array_results,
            "excellent_analysis": excellent_analysis,
            "best_analysis": best_analysis,
            "comparisons": comparisons,
            "report": report
        }

def main():
    """Main execution function"""
    # Change to thesis directory
    os.chdir("/Users/danieltoberman/Documents/git/Thesis")

    analyzer = CompleteSRPAnalysis()
    results = analyzer.run_complete_analysis()

    if results:
        print(f"\n💡 Next Steps:")
        print(f"1. Review excellent cases files in the results directory")
        print(f"2. Analyze acoustic properties of SRP-excellent cases")
        print(f"3. If CRNN struggles on these cases, we found genuine classical advantages!")
    else:
        print(f"\n❌ Analysis failed. Check the error messages above.")

if __name__ == "__main__":
    main()