#!/usr/bin/env python3
"""
Run CRNN with Partial Microphone Replacement Configurations

Tests CRNN with mostly 6cm microphones but 1-2 mics replaced with 12cm or 18cm positions.
This tests gradual degradation rather than complete geometry shift.

Mic 0 (center/reference) ALWAYS stays at the END (training order: [1,2,3,4,5,6,7,8,0]).
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import json

# Define all partial replacement configurations
# CRITICAL: Training used [1, 2, 3, 4, 5, 6, 7, 8, 0] with mic 0 at the END!
CONFIGURATIONS = [
    # Baseline - pure 6cm (for reference)
    {
        'name': '6cm_baseline',
        'mics': [1, 2, 3, 4, 5, 6, 7, 8, 0],
        'description': 'Baseline 6cm array (training configuration)',
        'n_replaced': 0,
        'replacement_type': 'none'
    },

    # Replace 1 mic with 12cm position
    {
        'name': '6cm_1x12cm_pos1',
        'mics': [9, 2, 3, 4, 5, 6, 7, 8, 0],
        'description': 'Replace mic 1 with mic 9 (12cm)',
        'n_replaced': 1,
        'replacement_type': '12cm'
    },
    {
        'name': '6cm_1x12cm_pos3',
        'mics': [1, 2, 11, 4, 5, 6, 7, 8, 0],
        'description': 'Replace mic 3 with mic 11 (12cm)',
        'n_replaced': 1,
        'replacement_type': '12cm'
    },
    {
        'name': '6cm_1x12cm_pos5',
        'mics': [1, 2, 3, 4, 13, 6, 7, 8, 0],
        'description': 'Replace mic 5 with mic 13 (12cm)',
        'n_replaced': 1,
        'replacement_type': '12cm'
    },

    # Replace 2 mics with 12cm positions
    {
        'name': '6cm_2x12cm_opposite',
        'mics': [9, 2, 3, 4, 13, 6, 7, 8, 0],
        'description': 'Replace mics 1,5 with 9,13 (12cm, opposite)',
        'n_replaced': 2,
        'replacement_type': '12cm'
    },
    {
        'name': '6cm_2x12cm_adjacent',
        'mics': [9, 10, 3, 4, 5, 6, 7, 8, 0],
        'description': 'Replace mics 1,2 with 9,10 (12cm, adjacent)',
        'n_replaced': 2,
        'replacement_type': '12cm'
    },

    # Replace 1 mic with 18cm position
    {
        'name': '6cm_1x18cm_pos1',
        'mics': [17, 2, 3, 4, 5, 6, 7, 8, 0],
        'description': 'Replace mic 1 with mic 17 (18cm)',
        'n_replaced': 1,
        'replacement_type': '18cm'
    },
    {
        'name': '6cm_1x18cm_pos5',
        'mics': [1, 2, 3, 4, 21, 6, 7, 8, 0],
        'description': 'Replace mic 5 with mic 21 (18cm)',
        'n_replaced': 1,
        'replacement_type': '18cm'
    },

    # Replace 2 mics with 18cm positions
    {
        'name': '6cm_2x18cm_opposite',
        'mics': [17, 2, 3, 4, 21, 6, 7, 8, 0],
        'description': 'Replace mics 1,5 with 17,21 (18cm, opposite)',
        'n_replaced': 2,
        'replacement_type': '18cm'
    },

    # Replace 3 mics with 12cm positions
    {
        'name': '6cm_3x12cm_alternating',
        'mics': [9, 2, 11, 4, 13, 6, 7, 8, 0],
        'description': 'Replace mics 1,3,5 with 9,11,13 (12cm, alternating)',
        'n_replaced': 3,
        'replacement_type': '12cm'
    },
    {
        'name': '6cm_3x12cm_consecutive',
        'mics': [9, 10, 11, 4, 5, 6, 7, 8, 0],
        'description': 'Replace mics 1,2,3 with 9,10,11 (12cm, consecutive)',
        'n_replaced': 3,
        'replacement_type': '12cm'
    },

    # Replace 4 mics with 12cm positions
    {
        'name': '6cm_4x12cm_opposite_pairs',
        'mics': [9, 2, 11, 4, 13, 6, 15, 8, 0],
        'description': 'Replace mics 1,3,5,7 with 9,11,13,15 (12cm, opposite pairs)',
        'n_replaced': 4,
        'replacement_type': '12cm'
    },
    {
        'name': '6cm_4x12cm_half_consecutive',
        'mics': [9, 10, 11, 12, 5, 6, 7, 8, 0],
        'description': 'Replace mics 1,2,3,4 with 9,10,11,12 (12cm, half array)',
        'n_replaced': 4,
        'replacement_type': '12cm'
    },

    # Replace 3-4 mics with 18cm positions
    {
        'name': '6cm_3x18cm_alternating',
        'mics': [17, 2, 19, 4, 21, 6, 7, 8, 0],
        'description': 'Replace mics 1,3,5 with 17,19,21 (18cm, alternating)',
        'n_replaced': 3,
        'replacement_type': '18cm'
    },
    {
        'name': '6cm_4x18cm_opposite_pairs',
        'mics': [17, 2, 19, 4, 21, 6, 23, 8, 0],
        'description': 'Replace mics 1,3,5,7 with 17,19,21,23 (18cm, opposite pairs)',
        'n_replaced': 4,
        'replacement_type': '18cm'
    },

    # Full array replacements for comparison
    {
        'name': '12cm_full',
        'mics': [9, 10, 11, 12, 13, 14, 15, 16, 0],
        'description': 'Full 12cm array (all 8 outer mics at 12cm)',
        'n_replaced': 8,
        'replacement_type': '12cm'
    },
    {
        'name': '18cm_full',
        'mics': [17, 18, 19, 20, 21, 22, 23, 24, 0],
        'description': 'Full 18cm array (all 8 outer mics at 18cm)',
        'n_replaced': 8,
        'replacement_type': '18cm'
    },
]

def backup_files():
    """Backup original files if not already backed up."""
    recorddata_path = Path("/Users/danieltoberman/Documents/git/Thesis/SSL/RecordData.py")
    recorddata_backup = recorddata_path.with_suffix(".py.backup")

    runcrnn_path = Path("/Users/danieltoberman/Documents/git/Thesis/SSL/run_CRNN.py")
    runcrnn_backup = runcrnn_path.with_suffix(".py.backup_partial")

    # Backup RecordData.py
    if not recorddata_backup.exists():
        shutil.copy2(recorddata_path, recorddata_backup)
        print(f"✅ Backup created: {recorddata_backup}")
    else:
        print(f"📁 Backup already exists: {recorddata_backup}")

    # Backup run_CRNN.py (use different extension to not conflict with other backups)
    if not runcrnn_backup.exists():
        shutil.copy2(runcrnn_path, runcrnn_backup)
        print(f"✅ Backup created: {runcrnn_backup}")
    else:
        print(f"📁 Backup already exists: {runcrnn_backup}")

    return recorddata_path, recorddata_backup, runcrnn_path, runcrnn_backup

def modify_recorddata(recorddata_path, mic_config):
    """Modify RecordData.py to use specified microphone configuration."""
    import re

    with open(recorddata_path, 'r') as f:
        content = f.read()

    # Find and replace the use_mic_id default value in the __init__ signature
    # Pattern: use_mic_id=[...] in the function definition
    # We need to preserve everything else in the line

    mic_str = str(mic_config)

    # Use regex to replace only the default value, preserving indentation and rest of line
    # Match: use_mic_id=[ any content ]
    pattern = r'(use_mic_id=)\[[^\]]*\]'
    replacement = r'\1' + mic_str

    new_content, n_replacements = re.subn(pattern, replacement, content, count=1)

    if n_replacements == 0:
        print(f"❌ Could not find use_mic_id pattern to modify")
        return False

    # Write modified content
    with open(recorddata_path, 'w') as f:
        f.write(new_content)

    print(f"✅ Modified RecordData.py: use_mic_id={mic_str}")
    return True

def modify_run_crnn(runcrnn_path, mic_config):
    """Modify run_CRNN.py to use specified microphone configuration in test dataset."""
    import re

    with open(runcrnn_path, 'r') as f:
        content = f.read()

    mic_str = str(mic_config)

    # Find and replace the hardcoded use_mic_id in dataset_test creation
    # Need to replace entire expression like: use_mic_id=[0] + list(range(17, 25))
    # This pattern matches everything from use_mic_id= up to the next comma (after closing parens/brackets)
    # It handles: [list] or [list] + list(...) or [list] + list(range(...))
    pattern = r'use_mic_id=\[0\]\s*\+\s*list\(range\([^)]+\)\)'

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'dataset_test = RealData(' in line:
            # Found the start of dataset_test creation
            # Look for use_mic_id in the next ~10 lines
            for j in range(i, min(i+10, len(lines))):
                if 'use_mic_id=' in lines[j]:
                    # Replace the entire use_mic_id expression
                    lines[j] = re.sub(pattern, f'use_mic_id={mic_str}', lines[j])
                    print(f"✅ Modified run_CRNN.py test dataset: use_mic_id={mic_str}")

                    # Write back
                    with open(runcrnn_path, 'w') as f:
                        f.write('\n'.join(lines))
                    return True

    print(f"❌ Could not find use_mic_id in dataset_test creation")
    return False

def restore_files(recorddata_path, recorddata_backup, runcrnn_path, runcrnn_backup):
    """Restore original files from backup."""
    success = True

    if recorddata_backup.exists():
        shutil.copy2(recorddata_backup, recorddata_path)
        print(f"✅ Restored original RecordData.py")
    else:
        print(f"❌ No backup found at {recorddata_backup}")
        success = False

    if runcrnn_backup.exists():
        shutil.copy2(runcrnn_backup, runcrnn_path)
        print(f"✅ Restored original run_CRNN.py")
    else:
        print(f"❌ No backup found at {runcrnn_backup}")
        success = False

    return success

def run_crnn_test(config_name, mic_config):
    """Run CRNN test with the current configuration."""
    print(f"🧠 Running CRNN test for {config_name}...")

    # Convert mic config to comma-separated string
    mic_str = ",".join(map(str, mic_config))

    cmd = [
        "python", "SSL/run_CRNN.py", "test",
        "--ckpt_path", "08_CRNN/checkpoints/best_valid_loss0.0220.ckpt",
        "--use_mic_id", mic_str,  # Pass mic IDs as argument
        "--trainer.accelerator=mps",
        "--trainer.devices=1",
        "--trainer.precision=16-mixed",
        "--trainer.strategy=auto",
        "--trainer.num_nodes=1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/Users/danieltoberman/Documents/git/Thesis"
        )

        if result.returncode == 0:
            print(f"✅ CRNN test completed successfully")
            return True
        else:
            print(f"❌ CRNN test failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[-1000:]}")  # Last 1000 chars of error
            return False

    except Exception as e:
        print(f"💥 Exception running CRNN: {e}")
        return False

def save_results(config):
    """Save CRNN results with configuration info."""
    # Source prediction file
    pred_file = Path("/Users/danieltoberman/Documents/git/Thesis/crnn_predictions/crnn_predictions_with_confidence_clean.csv")

    if not pred_file.exists():
        print(f"⚠️ No prediction file found at {pred_file}")
        return None

    # Load predictions
    df = pd.read_csv(pred_file)

    # Calculate metrics
    mae = df['abs_error'].mean()
    median_error = df['abs_error'].median()
    success_rate = (df['abs_error'] <= 5).mean() * 100
    n_samples = len(df)

    print(f"📊 Results: MAE={mae:.2f}°, Median={median_error:.2f}°, Success(≤5°)={success_rate:.1f}%, N={n_samples}")

    # Create results directory
    results_dir = Path("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/analysis/geometry_robustness/partial_replacement")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save with config name
    output_file = results_dir / f"crnn_{config['name']}_results.csv"
    shutil.copy2(pred_file, output_file)
    print(f"💾 Saved results to: {output_file}")

    # Return metrics
    return {
        'config_name': config['name'],
        'description': config['description'],
        'mics': config['mics'],
        'n_replaced': config['n_replaced'],
        'replacement_type': config['replacement_type'],
        'mae': mae,
        'median_error': median_error,
        'success_rate': success_rate,
        'n_samples': n_samples,
        'output_file': str(output_file)
    }

def main():
    """Main execution function."""
    print("="*80)
    print("CRNN Partial Microphone Replacement Testing")
    print("="*80)
    print(f"Testing {len(CONFIGURATIONS)} configurations")
    print("Mic 0 (center/reference) always stays at end of array")
    print("="*80)
    print()

    # Results storage
    all_results = []

    try:
        for i, config in enumerate(CONFIGURATIONS, 1):
            print(f"\n{'='*80}")
            print(f"Configuration {i}/{len(CONFIGURATIONS)}: {config['name']}")
            print(f"{'='*80}")
            print(f"Description: {config['description']}")
            print(f"Microphones: {config['mics']}")
            print(f"Replaced: {config['n_replaced']} mics ({config['replacement_type']})")
            print()

            # Run CRNN test with mic configuration as argument
            if not run_crnn_test(config['name'], config['mics']):
                print(f"⚠️ CRNN test failed for {config['name']}")
                continue

            # Save and analyze results
            metrics = save_results(config)
            if metrics:
                all_results.append(metrics)

            print(f"✅ Completed {config['name']}")

        print(f"\n✅ All configurations tested!")

        # Save summary results
        if all_results:
            results_dir = Path("/Users/danieltoberman/Documents/git/Thesis/hybrid_system/analysis/geometry_robustness/partial_replacement")

            # Save as CSV
            df_summary = pd.DataFrame(all_results)
            summary_csv = results_dir / "partial_replacement_summary.csv"
            df_summary.to_csv(summary_csv, index=False)
            print(f"\n📊 Summary saved to: {summary_csv}")

            # Save as JSON for detailed info
            summary_json = results_dir / "partial_replacement_summary.json"
            with open(summary_json, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"📊 Detailed summary saved to: {summary_json}")

            # Print summary table
            print("\n" + "="*80)
            print("RESULTS SUMMARY")
            print("="*80)
            print(f"{'Configuration':<30} {'N_Replaced':<12} {'Type':<8} {'MAE':<8} {'Median':<8} {'Success%':<10}")
            print("-"*80)
            for result in all_results:
                print(f"{result['config_name']:<30} {result['n_replaced']:<12} {result['replacement_type']:<8} "
                      f"{result['mae']:>6.2f}°  {result['median_error']:>6.2f}°  {result['success_rate']:>6.1f}%")
            print("="*80)

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
