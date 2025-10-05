#!/usr/bin/env python3
"""
Run SRP on Full Dataset with All Array Configurations

This script runs SRP-PHAT on the entire test dataset using all three
microphone array configurations (6cm, 12cm, 18cm) to identify cases
where SRP performs excellently (<1° error).

This implements Option 3: Find where SRP succeeds and check CRNN performance there.
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

def run_srp_single_array(array_diameter, test_csv, base_dir, output_dir, use_novel_noise=True):
    """Run SRP for a single array configuration"""

    print(f"\n📡 Running SRP with {array_diameter} array...")

    # Create output filename
    output_file = output_dir / f"srp_full_dataset_{array_diameter}_results.csv"

    # Build command
    cmd = [
        "python", "-m", "xsrpMain.xsrp.run_SRP",
        "--csv", test_csv,
        "--base-dir", base_dir,
        "--array_diameter", array_diameter
    ]

    if use_novel_noise:
        cmd.extend([
            "--use_novel_noise",
            "--novel_noise_scene", "Cafeteria2",
            "--novel_noise_snr", "5.0"
        ])

    print(f"Command: {' '.join(cmd)}")
    print(f"Output will be captured to: {output_file}")

    try:
        # Run command and capture both stdout and stderr
        start_time = time.time()

        with open(output_file, 'w') as f:
            # Write CSV header
            f.write("file_path,predicted_angle,true_angle,angular_error,environment\n")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {array_diameter} completed in {duration:.1f}s")

            # Parse the output to extract results
            output_lines = result.stdout.split('\n')

            # Look for result lines in the output
            results = []
            for line in output_lines:
                if 'predicted angle:' in line.lower() or 'mae:' in line.lower():
                    # This would need to be adapted based on actual SRP output format
                    print(f"  📊 {line.strip()}")

            # For now, let's save the raw output to analyze the format
            raw_output_file = output_dir / f"srp_raw_output_{array_diameter}.txt"
            with open(raw_output_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            print(f"📝 Raw output saved to: {raw_output_file}")

            return {
                "success": True,
                "duration": duration,
                "output_file": output_file,
                "raw_output_file": raw_output_file,
                "stdout_lines": len(output_lines),
                "returncode": result.returncode
            }
        else:
            print(f"❌ {array_diameter} failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {"success": False, "error": result.stderr, "returncode": result.returncode}

    except Exception as e:
        print(f"💥 Exception running {array_diameter}: {e}")
        return {"success": False, "error": str(e)}

def analyze_srp_outputs(results_dir):
    """Analyze the SRP outputs to extract performance data"""
    print(f"\n🔍 Analyzing SRP outputs in {results_dir}...")

    array_configs = ["6cm", "12cm", "18cm"]
    analysis_results = {}

    for array in array_configs:
        raw_output_file = results_dir / f"srp_raw_output_{array}.txt"

        if not raw_output_file.exists():
            print(f"⚠️ No raw output file found for {array}")
            continue

        print(f"📖 Analyzing {array} results...")

        with open(raw_output_file, 'r') as f:
            content = f.read()

        # Extract key information from the output
        # This depends on the actual format of the SRP script output
        lines = content.split('\n')

        # Look for patterns that indicate results
        predicted_angles = []
        true_angles = []
        errors = []

        for line in lines:
            # Try to extract angle information
            # You'll need to adapt this based on actual SRP output format
            if 'angle' in line.lower() and 'pred' in line.lower():
                # Extract numbers from the line
                import re
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if len(numbers) >= 2:
                    try:
                        predicted_angles.append(float(numbers[0]))
                        true_angles.append(float(numbers[1]))
                    except ValueError:
                        continue

        analysis_results[array] = {
            "total_cases": len(predicted_angles),
            "raw_output_lines": len(lines),
            "has_results": len(predicted_angles) > 0
        }

        print(f"  {array}: {len(predicted_angles)} results found")

    return analysis_results

def main():
    """Main execution function"""
    print("🚀 Starting SRP Full Dataset Analysis - All Arrays")
    print("=" * 60)

    # Configuration
    base_dir = "/Users/danieltoberman/Documents/git/Thesis"
    test_csv = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
    data_base_dir = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"

    # Create output directory
    output_dir = Path(base_dir) / "hybrid_system" / "analysis" / "srp_strengths_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Test dataset: {test_csv}")

    # Check if test CSV exists
    if not Path(test_csv).exists():
        print(f"❌ Test CSV not found: {test_csv}")
        return

    # Load test dataset to check size
    test_df = pd.read_csv(test_csv)
    print(f"📈 Test dataset size: {len(test_df)} examples")

    # Array configurations to test
    arrays = ["6cm", "12cm", "18cm"]

    # Run SRP for each array
    results = {}
    total_start_time = time.time()

    for array in arrays:
        result = run_srp_single_array(
            array_diameter=array,
            test_csv=test_csv,
            base_dir=data_base_dir,
            output_dir=output_dir,
            use_novel_noise=True
        )
        results[array] = result

        # Brief pause between runs
        time.sleep(2)

    total_duration = time.time() - total_start_time

    # Analyze results
    print(f"\n📊 Analysis Summary:")
    print(f"Total runtime: {total_duration/60:.1f} minutes")

    successful_runs = [array for array, result in results.items() if result.get("success", False)]
    failed_runs = [array for array, result in results.items() if not result.get("success", False)]

    print(f"✅ Successful runs: {len(successful_runs)} - {successful_runs}")
    print(f"❌ Failed runs: {len(failed_runs)} - {failed_runs}")

    if successful_runs:
        # Analyze outputs
        analysis = analyze_srp_outputs(output_dir)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "test_dataset": test_csv,
            "test_dataset_size": len(test_df),
            "arrays_tested": arrays,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_runtime_minutes": total_duration / 60,
            "results": results,
            "analysis": analysis
        }

        summary_file = output_dir / f"srp_all_arrays_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📄 Summary saved to: {summary_file}")

        # Next steps
        print(f"\n🎯 Next Steps:")
        print(f"1. Check raw output files in {output_dir}")
        print(f"2. Parse SRP results to extract angle predictions")
        print(f"3. Identify cases with <1° error")
        print(f"4. Compare with CRNN performance on same cases")

        if successful_runs:
            print(f"\n✅ SRP runs completed! Check the output files to proceed with analysis.")
        else:
            print(f"\n⚠️ No successful runs. Check the error messages above.")
    else:
        print(f"\n❌ No successful runs. Cannot proceed with analysis.")

if __name__ == "__main__":
    # Change to the thesis directory
    os.chdir("/Users/danieltoberman/Documents/git/Thesis")
    main()