import numpy as np
from pathlib import Path

# Path to the .npz file
npz_path = Path(r'C:\daniel\Thesis\hybrid_system\advanced_failure_detection\srp_features_end_result\train_combined_features.npz')

def inspect_npz(file_path):
    """Loads an .npz file and prints its keys."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return
        
    try:
        print(f"Inspecting file: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        print("Keys found in the file:")
        for key in data.files:
            print(f"- {key}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    inspect_npz(npz_path)