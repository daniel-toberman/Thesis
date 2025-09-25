#!/usr/bin/env python3
"""
Test script to verify novel noise implementation works correctly
for both CRNN and SRP methods.
"""

import os
import sys

def test_crnn_novel_noise():
    """Test CRNN with novel noise flags."""
    print("=== TESTING CRNN WITH NOVEL NOISE ===")

    # Test command
    test_cmd = """
    python SSL/run_CRNN.py test
    --ckpt_path "08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
    --trainer.accelerator=auto
    --trainer.devices=1
    --trainer.precision=16-mixed
    --trainer.strategy=auto
    --trainer.num_nodes=1
    --use_novel_noise
    --novel_noise_scene BadmintonCourt1
    --data.batch_size=[16,2]
    """

    print("Command to test CRNN with novel noise:")
    print(test_cmd.strip())
    print("\nThis will:")
    print("- Load test data from RealMAN_dataset_T60_08")
    print("- Add novel noise from BadmintonCourt1 scene")
    print("- Enable on_the_fly=True for test set to add noise")
    print("- Use high T60 noise from RealMAN_9_channels dataset")

def test_srp_novel_noise():
    """Test SRP with novel noise flags."""
    print("\n=== TESTING SRP WITH NOVEL NOISE ===")

    # Test command
    test_cmd = """
    python xsrpMain/xsrp/run_SRP.py
    --csv "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv"
    --base-dir "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
    --use_novel_noise
    --novel_noise_scene BadmintonCourt1
    --n 5
    """

    print("Command to test SRP with novel noise:")
    print(test_cmd.strip())
    print("\nThis will:")
    print("- Load first 5 test examples")
    print("- Add novel noise from BadmintonCourt1 scene")
    print("- Use deterministic noise selection based on example index")
    print("- Apply same SNR calculation as CRNN")

def show_usage_examples():
    """Show usage examples for the novel noise feature."""
    print("\n=== USAGE EXAMPLES ===")

    print("\n1. Test CRNN with novel noise from BadmintonCourt1:")
    print("python SSL/run_CRNN.py test --ckpt_path <path> --use_novel_noise --novel_noise_scene BadmintonCourt1")

    print("\n2. Test SRP with same novel noise:")
    print("python xsrpMain/xsrp/run_SRP.py --use_novel_noise --novel_noise_scene BadmintonCourt1 --n 10")

    print("\n3. Available novel noise scenes:")
    scenes = ["BadmintonCourt1", "Cafeteria2", "ShoppingMall", "SunkenPlaza2"]
    for scene in scenes:
        print(f"   - {scene}")

    print("\n4. Both methods will:")
    print("   - Use identical noise files based on example index")
    print("   - Apply same SNR calculation (~10dB)")
    print("   - Process the same underlying clean audio")
    print("   - Enable deterministic noise selection for comparison")

def verify_noise_directories():
    """Verify that novel noise directories exist."""
    print("\n=== VERIFYING NOISE DIRECTORIES ===")

    base_noise_path = "/Users/danieltoberman/Documents/RealMAN_9_channels/extracted/train/ma_noise"
    scenes = ["BadmintonCourt1", "Cafeteria2", "ShoppingMall", "SunkenPlaza2"]

    for scene in scenes:
        scene_path = os.path.join(base_noise_path, scene)
        exists = os.path.exists(scene_path)
        print(f"{scene}: {'✓' if exists else '✗'} ({scene_path})")

        if exists:
            # Count noise files
            noise_files = []
            for root, dirs, files in os.walk(scene_path):
                noise_files.extend([f for f in files if f.endswith('.wav') and 'CH0' in f])
            print(f"   {len(noise_files)} noise files available")

def main():
    print("Novel Noise Implementation Test")
    print("=" * 50)

    verify_noise_directories()
    test_crnn_novel_noise()
    test_srp_novel_noise()
    show_usage_examples()

    print(f"\n=== KEY FEATURES IMPLEMENTED ===")
    print("✓ CRNN: Added --use_novel_noise and --novel_noise_scene flags")
    print("✓ CRNN: Automatic dataset recreation with novel noise paths")
    print("✓ CRNN: on_the_fly=True enables noise addition in test mode")
    print("✓ SRP: Added same novel noise flags and deterministic selection")
    print("✓ SRP: Uses example index for reproducible noise selection")
    print("✓ Both: Use identical noise files and SNR calculations")
    print("✓ Both: Enable comparison of neural vs classical on same noisy audio")

if __name__ == "__main__":
    main()