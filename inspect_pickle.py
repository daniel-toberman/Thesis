import pickle
import sys
import numpy as np

def inspect_pickle(file_path):
    """Loads and inspects the contents of a pickle file."""
    print(f"--- Inspecting: {file_path} ---")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Type of loaded data: {type(data)}")

    if isinstance(data, dict):
        print("\nData is a dictionary. Keys:")
        for key, value in data.items():
            print(f"  - Key: '{key}', Type: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"    Shape: {value.shape}, Dtype: {value.dtype}")
            elif isinstance(value, list):
                print(f"    Length: {len(value)}")
                if value:
                    print(f"    Type of first element: {type(value[0])}")


    elif isinstance(data, list):
        print(f"\nData is a list with {len(data)} elements.")
        if data:
            first_element = data[0]
            print(f"Type of first element: {type(first_element)}")
            if isinstance(first_element, dict):
                print("Keys of the first dictionary element:")
                for key, value in first_element.items():
                    print(f"  - Key: '{key}', Type: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"    Shape: {value.shape}, Dtype: {value.dtype}")
                    elif isinstance(value, list):
                        print(f"    Length: {len(value)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pickle.py <path_to_pickle_file>")
    else:
        inspect_pickle(sys.argv[1])
