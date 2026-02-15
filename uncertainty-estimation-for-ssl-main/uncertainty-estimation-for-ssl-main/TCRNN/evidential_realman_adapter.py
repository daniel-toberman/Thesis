"""
Adapter DataModule for training the TCRNN (Evidential) model with the
RealMAN dataset pipeline from the SSL/run_CRNN.py project.

This file bridges the gap between the two projects:
1.  Our `RealData` dataset returns raw audio.
2.  The `TCRNN` model expects pre-processed spectrograms (STFT).

This adapter wraps `RealData`, performs the STFT, and formats the output
tuple to exactly match what the `TCRNN` trainer requires.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import os
import sys

# Add the parent directories to the path to import from SSL and TCRNN projects
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# --- Imports from our existing SSL project ---
from SSL.RecordData import RealData
from SSL.sampler import MyDistributedSampler
import SSL.Module as at_module # For the STFT module

# --- TCRNN project imports ---
# No longer importing from main_crnn to prevent circular dependency
# from main_crnn import DataModule # To subclass and override


class SpectrogramWrapper(Dataset):
    """
    A Dataset wrapper that takes a raw audio dataset (like RealData) and
    transforms its output into spectrograms suitable for the TCRNN model.
    """

    def __init__(self, raw_dataset: RealData, win_len=512, nfft=512, win_shift_ratio=0.625, fre_range_used=None, device='cpu'):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.dev = device
        
        # Initialize the same STFT module used in our original run_CRNN.py
        self.dostft = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        
        if fre_range_used is None:
            self.fre_range_used = range(1, int(nfft / 2) + 1, 1)
        else:
            self.fre_range_used = fre_range_used

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx_seed):
        # 1. Get raw data from our existing dataset
        # Returns: (raw_audio_signal, ground_truth_angles, vad_mask, mic_positions)
        raw_audio, targets, vad, mics = self.raw_dataset[idx_seed]
        
        # Convert to tensor and move to device
        raw_audio_tensor = torch.from_numpy(raw_audio).to(self.dev, non_blocking=True)
        # Add a batch dimension as the STFT module expects it
        raw_audio_tensor = raw_audio_tensor.unsqueeze(0)

        # 2. Apply the STFT pre-processing (logic copied from MyModel.data_preprocess)
        # The STFT module expects shape [nb, nc, ns] but RealData gives [ns, nc]
        stft_out = self.dostft(signal=raw_audio_tensor.permute(0, 2, 1))
        
        nb, nf, nt, nc = stft_out.shape
        stft_permuted = stft_out.permute(0, 3, 1, 2)

        # Normalization and real/imaginary concatenation
        eps = 1e-6
        mag = torch.abs(stft_permuted)
        mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1, keepdim=True)
        mean_value = mean_value.unsqueeze(-1).unsqueeze(-1).expand(mag.shape)
        
        stft_real = torch.real(stft_permuted) / (mean_value + eps)
        stft_imag = torch.imag(stft_permuted) / (mean_value + eps)

        # Concatenate real and imaginary parts along the channel dimension
        # This creates the input format required by the paper and TCRNN model
        real_image_batch = torch.cat((stft_real, stft_imag), dim=1)
        
        # Select frequency range and remove the batch dimension we added
        processed_spectrogram = real_image_batch[:, :, self.fre_range_used, :].squeeze(0)
        
        # 3. Format the ground truth as expected by TCRNN
        # TCRNN expects a dictionary {'doa': tensor}
        # Our `targets` is already a tensor of shape [num_frames, 1]
        # The TCRNN loss function expects angles in RADIANS for its internal conversion
        # Our RealData provides angles in DEGREES, so we convert them here.
        gt_dict = {'doa': torch.deg2rad(targets)}
        
        return processed_spectrogram, gt_dict


class EvidentialRealmanDataModule(pl.LightningDataModule):
    """
    A LightningDataModule that uses our SpectrogramWrapper to feed
    RealMAN data into the TCRNN model.
    """
    def __init__(self, data_dir: str, batch_size: tuple = (16, 8), num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Hardcode other paths to simplify the CLI interface
        self.csv_root = 'C:/daniel/Thesis/SSL/partition'
        self.noise_root = 'C:/daniel/Thesis/SSL'
        print(f"DataModule initialized to use RealMAN dataset.")
        print(f"  - Audio Root: {self.data_dir}")
        print(f"  - CSV Root (hardcoded): {self.csv_root}")
        print(f"  - Noise Root (hardcoded): {self.noise_root}")

    def setup(self, stage: str):
        # This method is called by PyTorch Lightning to set up datasets.
        # We will create instances of our SpectrogramWrapper here.
        
        if stage == "fit":
            # Create the raw RealData dataset for training
            raw_train_dataset = RealData(
                data_dir=self.data_dir,
                target_dir=[os.path.join(self.csv_root, "train/train_static_source_location_08.csv")],
                noise_dir=os.path.join(self.noise_root, "train/ma_noise/"),
                on_the_fly=True # Enable on-the-fly noise for training
            )
            # Wrap it for STFT processing
            self.dataset_train = SpectrogramWrapper(raw_train_dataset)

            # Create the raw RealData dataset for validation
            raw_val_dataset = RealData(
                data_dir=self.data_dir,
                target_dir=[os.path.join(self.csv_root, "val/val_static_source_location_08.csv")],
                noise_dir=os.path.join(self.noise_root, "val/ma_noise/"),
                on_the_fly=False # Disable noise for validation
            )
            self.dataset_val = SpectrogramWrapper(raw_val_dataset)
            
            print(f"Fit stage: Loaded {len(self.dataset_train)} training samples and {len(self.dataset_val)} validation samples.")

        elif stage == "test":
            # Create the raw RealData dataset for testing
            raw_test_dataset = RealData(
                data_dir=self.data_dir,
                target_dir=[os.path.join(self.csv_root, "test/test_static_source_location_08.csv")],
                noise_dir=os.path.join(self.noise_root, "test/ma_noise/"),
                on_the_fly=False
            )
            self.dataset_test = SpectrogramWrapper(raw_test_dataset)
            print(f"Test stage: Loaded {len(self.dataset_test)} test samples.")

    # The train/val/test_dataloader methods are inherited from the parent DataModule
    # but will now use the `self.dataset_*` instances we created above.
    # We will add the distributed sampler from our project for consistency.
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            sampler=MyDistributedSampler(dataset=self.dataset_train, seed=2, shuffle=True),
            batch_size=self.batch_size[0],
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            sampler=MyDistributedSampler(dataset=self.dataset_val, seed=2, shuffle=False),
            batch_size=self.batch_size[1],
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            sampler=MyDistributedSampler(dataset=self.dataset_test, seed=2, shuffle=False),
            batch_size=self.batch_size[1],
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
