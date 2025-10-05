#!/usr/bin/env python3
"""
CRNN Geometry Robustness Analysis

This script investigates CRNN's resistance to microphone array geometry changes.
Based on finding that CRNN gets 63° MAE on 12cm array vs ~15° on 6cm array.

Key Research Questions:
1. Is CRNN's 63° MAE on 12cm uniform degradation or systematic failures?
2. Does SRP maintain performance across 6cm→12cm better than CRNN?
3. Can we predict geometry-induced failures using confidence metrics?
4. Do we have a genuine hybrid opportunity based on geometry robustness?
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
import matplotlib.pyplot as plt
import seaborn as sns

class GeometryRobustnessAnalyzer:
    def __init__(self, base_dir="/Users/danieltoberman/Documents/git/Thesis"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "hybrid_system" / "analysis" / "geometry_robustness"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Test dataset - clean data without novel noise
        self.test_csv = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
        self.data_base_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"

        # Array configurations to test
        self.arrays = ["6cm", "12cm"]

        print(f"🔬 Geometry Robustness Analyzer initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📊 Test dataset: {self.test_csv}")

    def run_srp_clean_test(self, array_diameter):
        """Run SRP on clean test data (no novel noise) for specified array"""
        print(f"\n📡 Running SRP with {array_diameter} array on clean data...")

        output_file = self.results_dir / f"srp_clean_{array_diameter}_results.csv"

        # Build SRP command - NO novel noise
        cmd = [
            "python", "-m", "xsrpMain.xsrp.run_SRP",
            "--csv", self.test_csv,
            "--base-dir", self.data_base_dir,
            "--array_diameter", array_diameter
            # NOTE: No --use_novel_noise flag = clean test
        ]

        print(f"Command: {' '.join(cmd)}")

        try:
            # Clean up any existing output file first
            default_output = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08_srp_results.csv")
            if default_output.exists():
                default_output.unlink()

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ {array_diameter} SRP completed in {duration:.1f}s")

                # Move results to organized location
                if default_output.exists():
                    default_output.rename(output_file)

                    # Load and verify
                    df = pd.read_csv(output_file)
                    print(f"📊 {array_diameter}: {len(df)} test cases processed")
                    print(f"    Mean error: {df['abs_err_deg'].mean():.2f}°")
                    print(f"    Success rate (≤30°): {(df['abs_err_deg'] <= 30).mean() * 100:.1f}%")

                    return {
                        "success": True,
                        "file": output_file,
                        "duration": duration,
                        "num_cases": len(df),
                        "mean_error": df['abs_err_deg'].mean(),
                        "success_rate": (df['abs_err_deg'] <= 30).mean() * 100
                    }
                else:
                    return {"success": False, "error": "No output file generated"}
            else:
                print(f"❌ {array_diameter} SRP failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

        except Exception as e:
            print(f"💥 Exception running {array_diameter} SRP: {e}")
            return {"success": False, "error": str(e)}

    def check_crnn_12cm_setup(self):
        """Check how CRNN can be run with 12cm array configuration"""
        print(f"\n🧠 Checking CRNN 12cm array setup...")

        # Look for CRNN configuration
        ssl_dir = self.base_dir / "SSL"

        # Check if there are any microphone configuration options
        config_files = [
            ssl_dir / "CRNN.py",
            ssl_dir / "Module.py",
            ssl_dir / "RecordData.py",
            ssl_dir / "utils_.py"
        ]

        found_configs = []
        for config_file in config_files:
            if config_file.exists():
                # Read file and look for microphone selection
                with open(config_file, 'r') as f:
                    content = f.read()
                    if any(keyword in content.lower() for keyword in ['mic_id', 'microphone', 'use_mic', 'channel']):
                        found_configs.append(config_file)

        print(f"📋 Found {len(found_configs)} files with microphone configuration")

        return {
            "config_files": found_configs,
            "needs_modification": len(found_configs) > 0
        }

    def run_crnn_with_array(self, array_diameter):
        """Run CRNN with specified array diameter"""
        print(f"\n🧠 Running CRNN with {array_diameter} array...")

        # This will likely need modification of CRNN code to use different microphone channels
        # For now, document what needs to be done

        if array_diameter == "6cm":
            mic_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Default 6cm
        elif array_diameter == "12cm":
            mic_channels = [0] + list(range(9, 17))      # 12cm: center + outer ring
        else:
            raise ValueError(f"Unsupported array diameter: {array_diameter}")

        print(f"📡 Array {array_diameter} uses microphone channels: {mic_channels}")

        # Generate CRNN results filename
        output_file = self.results_dir / f"crnn_clean_{array_diameter}_results.csv"

        print(f"⚠️ CRNN modification required:")
        print(f"    1. Modify SSL/RecordData.py or utils_.py to use channels: {mic_channels}")
        print(f"    2. Run: python SSL/run_CRNN.py test --ckpt_path '08_CRNN/checkpoints/best_valid_loss0.0220.ckpt'")
        print(f"    3. Save results to: {output_file}")

        if output_file.exists():
            print(f"📊 Found existing CRNN {array_diameter} results: {output_file}")
            df = pd.read_csv(output_file)
            return {
                "success": True,
                "file": output_file,
                "num_cases": len(df),
                "mean_error": df['abs_error'].mean() if 'abs_error' in df.columns else None
            }
        else:
            print(f"❌ No existing CRNN {array_diameter} results found")
            return {
                "success": False,
                "needs_generation": True,
                "mic_channels": mic_channels,
                "output_file": output_file
            }

    def analyze_geometry_effects(self, srp_results, crnn_results):
        """Analyze the effects of geometry changes on both methods"""
        print(f"\n📊 Analyzing geometry effects...")

        analysis = {
            "srp_geometry_effect": {},
            "crnn_geometry_effect": {},
            "comparative_analysis": {}
        }

        # SRP geometry analysis
        if srp_results["6cm"]["success"] and srp_results["12cm"]["success"]:
            srp_6cm_error = srp_results["6cm"]["mean_error"]
            srp_12cm_error = srp_results["12cm"]["mean_error"]
            srp_degradation = srp_12cm_error - srp_6cm_error

            analysis["srp_geometry_effect"] = {
                "6cm_mae": srp_6cm_error,
                "12cm_mae": srp_12cm_error,
                "degradation_degrees": srp_degradation,
                "degradation_percentage": (srp_degradation / srp_6cm_error) * 100,
                "robust_to_geometry": abs(srp_degradation) < 10  # Less than 10° degradation
            }

            print(f"📡 SRP Geometry Effect:")
            print(f"    6cm MAE: {srp_6cm_error:.2f}°")
            print(f"    12cm MAE: {srp_12cm_error:.2f}°")
            print(f"    Degradation: {srp_degradation:.2f}° ({srp_degradation/srp_6cm_error*100:.1f}%)")

        # CRNN geometry analysis
        if crnn_results["6cm"]["success"] and crnn_results["12cm"]["success"]:
            crnn_6cm_error = crnn_results["6cm"]["mean_error"]
            crnn_12cm_error = crnn_results["12cm"]["mean_error"]
            crnn_degradation = crnn_12cm_error - crnn_6cm_error

            analysis["crnn_geometry_effect"] = {
                "6cm_mae": crnn_6cm_error,
                "12cm_mae": crnn_12cm_error,
                "degradation_degrees": crnn_degradation,
                "degradation_percentage": (crnn_degradation / crnn_6cm_error) * 100,
                "robust_to_geometry": abs(crnn_degradation) < 10  # Less than 10° degradation
            }

            print(f"🧠 CRNN Geometry Effect:")
            print(f"    6cm MAE: {crnn_6cm_error:.2f}°")
            print(f"    12cm MAE: {crnn_12cm_error:.2f}°")
            print(f"    Degradation: {crnn_degradation:.2f}° ({crnn_degradation/crnn_6cm_error*100:.1f}%)")

        # Comparative analysis
        if all(results["6cm"]["success"] and results["12cm"]["success"]
               for results in [srp_results, crnn_results]):

            srp_robustness = abs(analysis["srp_geometry_effect"]["degradation_degrees"])
            crnn_robustness = abs(analysis["crnn_geometry_effect"]["degradation_degrees"])

            analysis["comparative_analysis"] = {
                "more_robust_method": "SRP" if srp_robustness < crnn_robustness else "CRNN",
                "robustness_difference": abs(crnn_robustness - srp_robustness),
                "srp_robustness_score": srp_robustness,
                "crnn_robustness_score": crnn_robustness,
                "hybrid_opportunity": crnn_robustness > 20 and srp_robustness < 20  # CRNN degrades >20°, SRP <20°
            }

            print(f"⚖️ Comparative Analysis:")
            print(f"    More robust method: {analysis['comparative_analysis']['more_robust_method']}")
            print(f"    Hybrid opportunity: {analysis['comparative_analysis']['hybrid_opportunity']}")

        return analysis

    def generate_geometry_report(self, srp_results, crnn_results, analysis):
        """Generate comprehensive geometry robustness report"""
        print(f"\n📄 Generating geometry robustness report...")

        report = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "CRNN Geometry Robustness Analysis",
                "test_dataset": str(self.test_csv),
                "arrays_tested": self.arrays
            },
            "srp_results": srp_results,
            "crnn_results": crnn_results,
            "geometry_analysis": analysis,
            "key_findings": [],
            "research_implications": []
        }

        # Generate key findings
        if "srp_geometry_effect" in analysis and analysis["srp_geometry_effect"]:
            srp_deg = analysis["srp_geometry_effect"]["degradation_degrees"]
            report["key_findings"].append(f"SRP geometry degradation: {srp_deg:.1f}° (6cm→12cm)")

        if "crnn_geometry_effect" in analysis and analysis["crnn_geometry_effect"]:
            crnn_deg = analysis["crnn_geometry_effect"]["degradation_degrees"]
            report["key_findings"].append(f"CRNN geometry degradation: {crnn_deg:.1f}° (6cm→12cm)")

        if "comparative_analysis" in analysis and analysis["comparative_analysis"]:
            more_robust = analysis["comparative_analysis"]["more_robust_method"]
            hybrid_opp = analysis["comparative_analysis"]["hybrid_opportunity"]
            report["key_findings"].append(f"More geometry-robust method: {more_robust}")
            if hybrid_opp:
                report["key_findings"].append("Hybrid opportunity detected: Classical methods more geometry-robust")

        # Research implications
        if analysis.get("comparative_analysis", {}).get("hybrid_opportunity", False):
            report["research_implications"].extend([
                "CRNN robustness has limits - significant degradation with array geometry changes",
                "Classical SRP methods may be more robust to geometry variations",
                "Hybrid systems could use geometry-aware switching between methods",
                "Array geometry considerations crucial for neural SSL deployment"
            ])

        # Save report
        report_file = self.results_dir / f"geometry_robustness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"💾 Report saved to: {report_file}")

        # Print summary
        print(f"\n🎯 Key Findings:")
        for finding in report["key_findings"]:
            print(f"  • {finding}")

        if report["research_implications"]:
            print(f"\n🔬 Research Implications:")
            for implication in report["research_implications"]:
                print(f"  • {implication}")

        return report

    def run_complete_analysis(self):
        """Run the complete geometry robustness analysis"""
        print("🚀 Starting Complete Geometry Robustness Analysis")
        print("=" * 60)

        # Step 1: Run SRP on clean data for both arrays
        print(f"\n📡 Step 1: Running SRP on clean test data...")
        srp_results = {}
        for array in self.arrays:
            srp_results[array] = self.run_srp_clean_test(array)

        # Step 2: Check CRNN array configuration
        print(f"\n🧠 Step 2: Checking CRNN array configuration...")
        crnn_config = self.check_crnn_12cm_setup()

        # Step 3: Get CRNN results (or document what's needed)
        print(f"\n🧠 Step 3: Getting CRNN results...")
        crnn_results = {}
        for array in self.arrays:
            crnn_results[array] = self.run_crnn_with_array(array)

        # Step 4: Analyze geometry effects
        print(f"\n📊 Step 4: Analyzing geometry effects...")
        analysis = self.analyze_geometry_effects(srp_results, crnn_results)

        # Step 5: Generate comprehensive report
        print(f"\n📄 Step 5: Generating report...")
        report = self.generate_geometry_report(srp_results, crnn_results, analysis)

        print(f"\n🎉 Geometry Robustness Analysis completed!")
        print(f"📁 Results saved in: {self.results_dir}")

        return {
            "srp_results": srp_results,
            "crnn_results": crnn_results,
            "analysis": analysis,
            "report": report
        }

def main():
    """Main execution function"""
    os.chdir("/Users/danieltoberman/Documents/git/Thesis")

    analyzer = GeometryRobustnessAnalyzer()
    results = analyzer.run_complete_analysis()

    if results:
        print(f"\n💡 Next Steps:")

        # Check what data we have vs what we need
        srp_success = all(r["success"] for r in results["srp_results"].values())
        crnn_success = all(r["success"] for r in results["crnn_results"].values())

        if not srp_success:
            print(f"1. ❌ Complete SRP data collection - some runs failed")
        else:
            print(f"1. ✅ SRP data collection complete")

        if not crnn_success:
            print(f"2. ❌ Generate CRNN 12cm results:")
            print(f"   - Modify CRNN to use microphone channels [0, 9-16]")
            print(f"   - Run CRNN test and save results")
        else:
            print(f"2. ✅ CRNN data collection complete")

        if srp_success and crnn_success:
            print(f"3. 🎯 Analyze failure patterns and prediction opportunities")
            print(f"4. 📊 Generate visualizations of geometry effects")

if __name__ == "__main__":
    main()