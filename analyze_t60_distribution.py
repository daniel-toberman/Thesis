#!/usr/bin/env python3
"""
Analyze T60 distribution in test sets to determine if the model
truly generalized to higher reverb or if test set lacks high T60 cases.
"""

import pandas as pd
import json
import numpy as np

def load_scene_info():
    """Load scene info with T60 values."""
    with open("/Users/danieltoberman/Documents/RealMAN_9_channels/scene_info.json", 'r') as f:
        scene_info = json.load(f)

    return scene_info

def analyze_t60_in_test_sets():
    """Compare T60 distribution between limited (T60<0.8) and full test sets."""

    scene_info = load_scene_info()

    # Load both CSV files
    try:
        test_08_df = pd.read_csv("/Users/danieltoberman/Documents/RealMAN_9_channels/test/test_static_source_location_08.csv")
        test_full_df = pd.read_csv("/Users/danieltoberman/Documents/RealMAN_9_channels/test/test_static_source_location.csv")
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return

    print("=== T60 DISTRIBUTION ANALYSIS ===")
    print(f"T60 < 0.8 test set: {len(test_08_df)} examples")
    print(f"Full test set: {len(test_full_df)} examples")
    print(f"Additional examples in full set: {len(test_full_df) - len(test_08_df)}")

    # Extract environment names from filenames in both sets
    def extract_environments(df):
        environments = []
        for filename in df['filename']:
            # Extract environment from path like "test/ma_speech/Cafeteria1/static/..."
            parts = str(filename).split('/')
            if len(parts) >= 3:
                env = parts[2]  # Environment name
                environments.append(env)
            else:
                environments.append('Unknown')
        return environments

    test_08_envs = extract_environments(test_08_df)
    test_full_envs = extract_environments(test_full_df)

    # Count environments and their T60 values
    env_counts_08 = {}
    env_counts_full = {}

    for env in test_08_envs:
        env_counts_08[env] = env_counts_08.get(env, 0) + 1

    for env in test_full_envs:
        env_counts_full[env] = env_counts_full.get(env, 0) + 1

    # Analyze T60 values
    t60_08 = []
    t60_full = []
    high_t60_scenes = []

    print(f"\n=== ENVIRONMENT ANALYSIS ===")
    print(f"{'Environment':<20} {'T60':<6} {'Limited':<8} {'Full':<8} {'Added':<8}")
    print("-" * 60)

    for env_name, env_data in scene_info.items():
        if 'T60' in env_data and env_data['T60'] != 'None':
            try:
                t60_val = float(env_data['T60'])

                count_08 = env_counts_08.get(env_name, 0)
                count_full = env_counts_full.get(env_name, 0)
                added = count_full - count_08

                if count_full > 0:  # Only show environments in test set
                    print(f"{env_name:<20} {t60_val:<6.3f} {count_08:<8} {count_full:<8} {added:<8}")

                    # Add to T60 lists weighted by sample count
                    t60_08.extend([t60_val] * count_08)
                    t60_full.extend([t60_val] * count_full)

                    if t60_val >= 0.8:
                        high_t60_scenes.append({
                            'env': env_name,
                            't60': t60_val,
                            'count_08': count_08,
                            'count_full': count_full,
                            'added': added
                        })

            except ValueError:
                continue

    # Statistical analysis
    if len(t60_08) > 0 and len(t60_full) > 0:
        print(f"\n=== T60 STATISTICAL SUMMARY ===")
        print(f"Limited test set (T60 < 0.8):")
        print(f"  Mean T60: {np.mean(t60_08):.3f}")
        print(f"  Max T60: {np.max(t60_08):.3f}")
        print(f"  Samples: {len(t60_08)}")

        print(f"\nFull test set:")
        print(f"  Mean T60: {np.mean(t60_full):.3f}")
        print(f"  Max T60: {np.max(t60_full):.3f}")
        print(f"  Samples: {len(t60_full)}")

        # High T60 analysis
        high_t60_samples = sum([s['count_full'] for s in high_t60_scenes])
        high_t60_added = sum([s['added'] for s in high_t60_scenes])

        print(f"\n=== HIGH T60 (≥0.8) ANALYSIS ===")
        print(f"High T60 scenes in full test set: {len(high_t60_scenes)}")
        print(f"High T60 samples in full test set: {high_t60_samples} ({high_t60_samples/len(t60_full)*100:.1f}%)")
        print(f"High T60 samples added to full set: {high_t60_added}")

        if len(high_t60_scenes) > 0:
            print(f"\nHigh T60 environments:")
            for scene in high_t60_scenes:
                print(f"  {scene['env']}: T60={scene['t60']:.3f}, added {scene['added']} samples")

        # Conclusion
        print(f"\n=== CONCLUSION ===")
        if high_t60_samples > len(t60_full) * 0.1:  # >10% of test set
            print(f"✓ Full test set contains significant high T60 data ({high_t60_samples/len(t60_full)*100:.1f}%)")
            print(f"✓ Model performance difference (4.76° vs 4.65°) suggests good T60 generalization")
            print(f"→ RECOMMENDATION: Focus on different noise types rather than T60")
        else:
            print(f"✗ Full test set lacks sufficient high T60 data ({high_t60_samples/len(t60_full)*100:.1f}%)")
            print(f"→ RECOMMENDATION: Create dedicated high T60 test set for proper evaluation")

    return high_t60_scenes, t60_08, t60_full

def main():
    analyze_t60_in_test_sets()

if __name__ == "__main__":
    main()