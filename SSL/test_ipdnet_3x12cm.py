"""
Test IPDnet baseline performance on 3x12cm consecutive array.

This script:
1. Loads the trained IPDnet model (trained on 6cm array)
2. Tests on 3x12cm consecutive array [9,10,11,4,5,6,7,8,0]
3. Converts IPD predictions to DOA using PredDOA module
4. Computes error metrics and success rate
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Import model and data
import sys
sys.path.append('/Users/danieltoberman/Documents/git/Thesis/SSL')
from SingleTinyIPDnet import SingleTinyIPDnet
from RecordData import RealData
import Module as at_module


def load_model(checkpoint_path, device='cpu'):
    """Load trained IPDnet model from checkpoint."""
    model = SingleTinyIPDnet(input_size=16, hidden_size=128)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle Lightning checkpoint format
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'arch.' prefix from keys if present
        state_dict = {k.replace('arch.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Loaded model from {checkpoint_path}")
    return model


def load_test_data():
    """Load test dataset with 3x12cm consecutive array."""
    dataset = RealData(
        data_dir='/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted/',
        target_dir=['/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv'],
        noise_dir='/Users/danieltoberman/Documents/RealMAN_9_channels/extracted/test/ma_noise/',
        use_mic_id=[9,10,11,4,5,6,7,8,0],  # 3x12cm consecutive array
        on_the_fly=False  # No noise augmentation for testing
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x  # Keep original format
    )

    print(f"✓ Loaded test dataset: {len(dataset)} samples")
    return dataloader


def convert_ipd_to_doa(ipd_predictions, mic_array_geometry, device='cpu'):
    """
    Convert IPD predictions to DOA angles using PredDOA module.

    Args:
        ipd_predictions: (batch, time//5, 1, 512, features) IPD predictions
        mic_array_geometry: Array geometry tensor
        device: Device for computation

    Returns:
        doa_predictions: (batch,) predicted DOA angles
    """
    # Initialize PredDOA module
    pred_doa_module = at_module.PredDOA(mic_location=mic_array_geometry)

    # Prepare batch for PredDOA (expects 5D tensor)
    # ipd_predictions is already (batch, time, 1, freq*2, features)

    # Use predgt2DOA_ipd to convert IPD to DOA
    # This handles the 5D format and processes each mic pair
    batch_doas = []

    for i in range(ipd_predictions.shape[0]):
        # Get single sample: (time, 1, freq*2, features)
        sample = ipd_predictions[i:i+1]

        # Convert to DOA - take mean over time
        # PredDOA expects (batch, time, freq*2, n_mics, sources)
        # Need to reshape our predictions to match

        # For now, take the mean prediction over time
        mean_pred = sample.mean(dim=1)  # (1, 1, 512, features)

        # Extract single dominant direction (simplified)
        # In practice, PredDOA.predgt2DOA does spatial spectrum matching
        # For baseline, we'll use the module's functionality

        batch_doas.append(0.0)  # Placeholder - will implement proper conversion

    return torch.tensor(batch_doas)


def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluate IPDnet on test set.

    Returns:
        results: Dict with MAE, median error, success rate
    """
    model.eval()

    all_predictions = []
    all_gt_angles = []
    all_errors = []

    print("\nEvaluating IPDnet on test set...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Unpack batch (format from RealData)
            for item in batch:
                mic_signal, target, vad, array_topo = item

                # Convert to tensor and move to device
                mic_signal = torch.from_numpy(mic_signal).unsqueeze(0).to(device)  # (1, channels, freq, time)

                # Get model prediction
                ipd_pred = model(mic_signal)

                # Convert IPD to DOA using PredDOA
                # For now, use simplified conversion
                # TODO: Implement proper IPD->DOA conversion with PredDOA module

                # Get ground truth angle
                gt_angle = target[0, 0].item() if torch.is_tensor(target) else target

                # Store for analysis
                all_gt_angles.append(gt_angle)
                # all_predictions.append(pred_angle)  # Will add after proper conversion

    # Compute metrics
    all_gt_angles = np.array(all_gt_angles)

    # Placeholder results until IPD->DOA conversion is properly implemented
    results = {
        'n_samples': len(all_gt_angles),
        'mae': 0.0,  # Placeholder
        'median_error': 0.0,  # Placeholder
        'success_rate': 0.0,  # Placeholder (<= 5 degrees)
        'note': 'IPD to DOA conversion needs to be implemented using PredDOA module'
    }

    return results


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("IPDnet Baseline Evaluation on 3x12cm Consecutive Array")
    print("=" * 80)

    # Configuration
    checkpoint_path = '/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/version_*/checkpoints/best*.ckpt'  # Update with actual path
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    output_dir = Path('/Users/danieltoberman/Documents/git/Thesis/SSL/results/ipdnet_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    # Load model
    # model = load_model(checkpoint_path, device=device)

    # Load test data
    test_loader = load_test_data()

    # Evaluate
    # results = evaluate_model(model, test_loader, device=device)

    # Print results
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    # print(f"Samples: {results['n_samples']}")
    # print(f"MAE: {results['mae']:.2f}°")
    # print(f"Median Error: {results['median_error']:.2f}°")
    # print(f"Success Rate (≤5°): {results['success_rate']:.1f}%")
    print("\nNote: This script template is ready. Need to:")
    print("1. Update checkpoint_path after training completes")
    print("2. Implement proper IPD→DOA conversion using PredDOA module")
    print("3. The conversion is already available in Module.py's PredDOA class")

    # Save results
    # output_file = output_dir / 'ipdnet_baseline_results.pkl'
    # with open(output_file, 'wb') as f:
    #     pickle.dump(results, f)
    # print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
