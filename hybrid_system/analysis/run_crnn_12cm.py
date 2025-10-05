#!/usr/bin/env python3
"""
Run CRNN with 12cm Array Configuration

This script modifies CRNN to use the 12cm microphone array and runs it on
the clean test dataset to compare geometry robustness with SRP.

12cm array uses microphones: [0, 9, 10, 11, 12, 13, 14, 15, 16]
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd

def backup_and_modify_recorddata():
    """Backup original RecordData.py and modify for 12cm array"""
    print("🔧 Modifying CRNN for 12cm array...")

    recorddata_path = Path("/Users/danieltoberman/Documents/git/Thesis/SSL/RecordData.py")
    backup_path = recorddata_path.with_suffix(".py.backup")

    # Create backup if it doesn't exist
    if not backup_path.exists():
        shutil.copy2(recorddata_path, backup_path)
        print(f"📁 Backup created: {backup_path}")

    # Read the original file
    with open(recorddata_path, 'r') as f:
        content = f.read()

    # Replace the use_mic_id default for 12cm array
    # Original: use_mic_id=[1, 2, 3, 4, 5, 6, 7, 8, 0]
    # 12cm:     use_mic_id=[0, 9, 10, 11, 12, 13, 14, 15, 16]
    original_line = "use_mic_id=[1, 2, 3, 4, 5, 6, 7, 8, 0]"
    modified_line = "use_mic_id=[0, 9, 10, 11, 12, 13, 14, 15, 16]"

    if original_line in content:
        modified_content = content.replace(original_line, modified_line)

        # Write the modified content
        with open(recorddata_path, 'w') as f:
            f.write(modified_content)

        print(f"✅ Modified CRNN to use 12cm array: [0, 9, 10, 11, 12, 13, 14, 15, 16]")
        return True
    else:
        print(f"❌ Could not find original microphone configuration to modify")
        return False

def restore_recorddata():
    """Restore original RecordData.py from backup"""
    print("🔄 Restoring original CRNN configuration...")

    recorddata_path = Path("/Users/danieltoberman/Documents/git/Thesis/SSL/RecordData.py")
    backup_path = recorddata_path.with_suffix(".py.backup")

    if backup_path.exists():
        shutil.copy2(backup_path, recorddata_path)
        print(f"✅ Restored original configuration from {backup_path}")
        return True
    else:
        print(f"❌ No backup found at {backup_path}")
        return False

def run_crnn_test():
    """Run CRNN test with the modified configuration"""
    print("🧠 Running CRNN test with 12cm array...")

    cmd = [
        "python", "SSL/run_CRNN.py", "test",
        "--ckpt_path", "08_CRNN/checkpoints/best_valid_loss0.0220.ckpt",
        "--trainer.accelerator=auto",
        "--trainer.devices=1",
        "--trainer.precision=16-mixed",
        "--trainer.strategy=auto",
        "--trainer.num_nodes=1"
    ]

    try:
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/danieltoberman/Documents/git/Thesis")

        if result.returncode == 0:
            print("✅ CRNN 12cm test completed successfully")
            print("📊 Output summary:")

            # Look for MAE or error information in the output
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # Check last 20 lines for results
                if any(keyword in line.lower() for keyword in ['mae', 'error', 'test', 'accuracy']):
                    print(f"    {line}")

            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ CRNN 12cm test failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except Exception as e:
        print(f"💥 Exception running CRNN: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def save_results():
    """Look for and save CRNN 12cm results"""
    print("💾 Looking for CRNN 12cm results...")

    # Common locations where CRNN results might be saved
    possible_locations = [
        "/Users/danieltoberman/Documents/git/Thesis/crnn_predictions/crnn_predictions_with_confidence_clean.csv",
        "/Users/danieltoberman/Documents/git/Thesis/test_detailed_results.csv",
        "/Users/danieltoberman/Documents/git/Thesis/SSL_logs/",
    ]

    results_dir = Path("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/analysis/geometry_robustness")
    results_dir.mkdir(parents=True, exist_ok=True)

    for location in possible_locations:
        path = Path(location)
        if path.exists():
            if path.is_file() and path.suffix == '.csv':
                # Copy CSV results
                dest = results_dir / f"crnn_clean_12cm_results.csv"
                shutil.copy2(path, dest)
                print(f"📊 Copied CRNN 12cm results to: {dest}")

                # Analyze the results
                try:
                    df = pd.read_csv(dest)
                    if 'abs_error' in df.columns:
                        mae = df['abs_error'].mean()
                        print(f"🎯 CRNN 12cm MAE: {mae:.2f}°")
                    elif 'error' in df.columns:
                        mae = df['error'].mean()
                        print(f"🎯 CRNN 12cm MAE: {mae:.2f}°")
                    else:
                        print(f"📊 Results saved, columns: {list(df.columns)}")
                except Exception as e:
                    print(f"⚠️ Could not analyze results: {e}")

                return dest

    print("⚠️ No CRNN results found in expected locations")
    return None

def main():
    """Main execution function"""
    print("🚀 Running CRNN with 12cm Array Configuration")
    print("=" * 60)

    try:
        # Step 1: Modify CRNN configuration
        if not backup_and_modify_recorddata():
            print("❌ Failed to modify CRNN configuration")
            return

        # Step 2: Run CRNN test
        result = run_crnn_test()

        # Step 3: Save results
        results_file = save_results()

        # Step 4: Restore original configuration
        restore_recorddata()

        # Step 5: Summary
        print(f"\n🎯 CRNN 12cm Test Summary:")
        if result["success"]:
            print(f"✅ Test completed successfully")
            if results_file:
                print(f"📊 Results saved to: {results_file}")
            else:
                print(f"⚠️ Could not locate/save results file")
        else:
            print(f"❌ Test failed")

        print(f"\n💡 Next Steps:")
        print(f"1. Check results file for detailed analysis")
        print(f"2. Compare with SRP 12cm performance")
        print(f"3. Analyze failure patterns vs 6cm baseline")

    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        # Always try to restore original configuration
        restore_recorddata()

if __name__ == "__main__":
    main()