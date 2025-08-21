# srp_fallback.py
import numpy as np
import torch

# Import your existing SRP routine:
# NOTE: adjust the import path if your working dir is Thesis\SSL
from xsrpMain.xsrp.run_SRP import run_srp_return_details  # uses your MIC_POSITIONS & settings

def srp_predict_batch(mic_sig_batch: torch.Tensor, fs: int) -> torch.Tensor:
    """
    Args:
      mic_sig_batch: shape [B, T, C] or [B, C, T] (float tensor on any device)
      fs: sample rate (int), same as used to train/evaluate the dataset.

    Returns:
      angles_deg: [B] float32 tensor of azimuth in degrees (0..360)
    """
    x = mic_sig_batch.detach().cpu()
    if x.dim() != 3:
        raise ValueError(f"Expected 3D batch, got {x.shape}")

    # Accept either [B,T,C] or [B,C,T]
    if x.shape[1] < x.shape[2]:  # assume [B, T, C]
        x = x.permute(0, 2, 1)   # -> [B, C, T]

    B, C, T = x.shape
    out = torch.empty((B,), dtype=torch.float32)
    for b in range(B):
        # SRP expects (C, T) float64 numpy
        xb = x[b].numpy().astype(np.float64)
        az, _, _, _ = run_srp_return_details(fs=fs, mic_signals=xb)
        out[b] = float(az)
    return out
