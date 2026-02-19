import numpy as np
from pathlib import Path

def inspect_npz(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return
        
    try:
        print(f"Inspecting file: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        print("Keys and shapes:")
        for key in data.files:
            val = data[key]
            shape = val.shape if hasattr(val, 'shape') else 'no shape'
            print(f"- {key}: {shape}")
            if key == 'logits_pre_sig' or key == 'predicted_angles' or key == 'gt_angles':
                print(f"  Sample from {key}: {val[0] if len(val) > 0 else 'empty'}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    inspect_npz(r'C:\daniel\Thesis\train_combined_features-002.npz')
    inspect_npz(r'C:\daniel\Thesis\crnn features\test_6cm_features.npz')
