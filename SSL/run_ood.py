# run_ood.py
from __future__ import annotations
import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --- Import your Lightning model & datamodule ---
from run_CRNN import MyModel, MyDataModule  # make sure PYTHONPATH includes this folder


# =============== SRP fallback import (robust) ===============
def _import_srp():
    """
    Tries a few reasonable import paths to locate your SRP function.
    Expected signature: run_srp_return_details(fs: int, mic_signals: np.ndarray) -> (az_deg, srp_obj, srp_map, grid)
    """
    # 1) If user runs from Thesis\SSL, the sibling path Thesis\xsrpMain\xsrp might not be on sys.path.
    # Try to add common parent "Thesis" automatically if visible from CWD.
    candidates = []

    # Current working directory heuristic
    cwd = os.path.abspath(os.getcwd())
    parts = cwd.split(os.sep)
    if "Thesis" in parts:
        thesis_root = os.sep.join(parts[: parts.index("Thesis") + 1])
        candidates.append(os.path.join(thesis_root, "xsrpMain", "xsrp"))
        candidates.append(os.path.join(thesis_root, "xsrpMain"))
        candidates.append(os.path.join(thesis_root))

    # Script directory heuristic
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "..", "xsrpMain", "xsrp"))
    candidates.append(os.path.join(here, "xsrpMain", "xsrp"))

    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Try canonical import
    try:
        from run_SRP import run_srp_return_details  # type: ignore
        return run_srp_return_details
    except Exception:
        pass

    # Try fully qualified path if package layout exists
    try:
        from xsrpMain.xsrp.run_SRP import run_srp_return_details  # type: ignore
        return run_srp_return_details
    except Exception:
        pass

    # Last resort: run_SRP.py in CWD
    try:
        import importlib.util
        for guess in ["run_SRP.py", os.path.join(here, "run_SRP.py")]:
            if os.path.isfile(guess):
                spec = importlib.util.spec_from_file_location("srp_local", guess)
                mod = importlib.util.module_from_spec(spec)  # type: ignore
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
                return getattr(mod, "run_srp_return_details")
    except Exception:
        pass

    raise ImportError(
        "Could not import run_srp_return_details. "
        "Ensure Thesis\\xsrpMain\\xsrp is on PYTHONPATH or run from a folder where run_SRP.py is importable."
    )


RUN_SRP_RETURN_DETAILS = _import_srp()


# =============== OOD utilities ===============
def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    logits: [..., C]
    Returns: energy-based ID score (we return -E so that larger = more ID)
    """
    E = T * torch.logsumexp(logits / T, dim=-1)
    return -E


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).amax(dim=-1)


@dataclass
class MahalanobisModel:
    mu: torch.Tensor       # [1, D]
    precision: torch.Tensor  # [D, D]

    def score(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [..., D]
        Returns: negative squared Mahalanobis distance (larger = more ID)
        """
        orig_shape = feats.shape[:-1]
        f = feats.reshape(-1, feats.shape[-1])
        diff = f - self.mu[0]
        d2 = torch.einsum("nd,dd,nd->n", diff, self.precision, diff)
        s = -d2
        return s.view(*orig_shape)


def fit_mahalanobis_single(feats: torch.Tensor) -> MahalanobisModel:
    """
    feats: [N, D] or [B, T, D]
    Uses Ledoit-Wolf shrinkage (via numpy) for stable covariance.
    """
    f = feats.detach().cpu().numpy().reshape(-1, feats.shape[-1])
    mu = f.mean(axis=0, keepdims=True)
    centered = f - mu
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(centered).covariance_
    except Exception:
        cov = np.cov(centered, rowvar=False) + 1e-6 * np.eye(centered.shape[1])
    prec = np.linalg.pinv(cov)
    return MahalanobisModel(
        mu=torch.from_numpy(mu).float(),
        precision=torch.from_numpy(prec).float()
    )


# =============== Feature extraction (no code change required) ===============
class _Hook:
    def __init__(self, module: torch.nn.Module):
        self.fmap = None
        self.h = module.register_forward_hook(self._cb)

    def _cb(self, m, i, o):
        self.fmap = o.detach()

    def close(self):
        self.h.remove()


def forward_intermediates(model: MyModel, data_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits_pre_sig: [B, T, C]
      penult_feats : [B, T, D]
    Prefers model.arch.forward_with_intermediates if present; otherwise uses hooks
    on arch.relu (penultimate) and arch.ipd2xyz2 (pre-sigmoid logits).
    """
    arch = model.arch

    # If user added the helper method, use it
    if hasattr(arch, "forward_with_intermediates"):
        return arch.forward_with_intermediates(data_tensor)

    # Fallback: hook into penultimate and final linear
    if not (hasattr(arch, "relu") and hasattr(arch, "ipd2xyz2")):
        raise AttributeError(
            "Expected attributes 'relu' and 'ipd2xyz2' on model.arch. "
            "Add forward_with_intermediates to CRNN for a guaranteed path."
        )

    h_pen = _Hook(arch.relu)
    h_log = _Hook(arch.ipd2xyz2)
    # Run a normal forward; model.forward returns post-sigmoid,
    # but the hook on ipd2xyz2 captures pre-sigmoid output.
    _ = arch(data_tensor)
    penult = h_pen.fmap        # [B, T, D]
    logits_pre = h_log.fmap    # [B, T, C]
    h_pen.close(); h_log.close()

    if penult is None or logits_pre is None:
        raise RuntimeError("Hooks did not capture intermediates; ensure modules are reached in forward.")
    return logits_pre, penult


# =============== SRP wrapper ===============
def srp_predict_batch(mic_sig_batch: torch.Tensor, fs: int) -> torch.Tensor:
    """
    mic_sig_batch: [B, T, C] or [B, C, T] (float)
    Returns angles (deg) as [B]
    """
    x = mic_sig_batch.detach().cpu()
    if x.dim() != 3:
        raise ValueError(f"mic_sig_batch must be 3D, got {tuple(x.shape)}")
    if x.shape[1] < x.shape[2]:  # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
    B, C, T = x.shape
    out = torch.empty((B,), dtype=torch.float32)
    for b in range(B):
        xb = x[b].numpy().astype(np.float64)  # (C, T)
        az, _, _, _ = RUN_SRP_RETURN_DETAILS(fs=fs, mic_signals=xb)
        out[b] = float(az)
    return out


# =============== OOD Detector wrapper ===============
@dataclass
class Thresholds:
    energy: float
    msp: float
    maha: float


class OODDetector:
    def __init__(self, model: MyModel, device: Optional[str] = None, energy_T: float = 1.0):
        self.model = model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.T = float(energy_T)
        self.maha: Optional[MahalanobisModel] = None
        self.thr: Optional[Thresholds] = None

    @torch.no_grad()
    def _prep(self, mic_sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.model.data_preprocess(mic_sig_batch=mic_sig)[0]  # your feature tensor
        logits_pre, feats = forward_intermediates(self.model, data)
        return logits_pre, feats

    @torch.no_grad()
    def calibrate(self, id_loader: Iterable, quantile: float = 0.05) -> Thresholds:
        Es, Msps, Feat_list = [], [], []
        for batch in id_loader:
            mic_sig = batch[0].to(self.device)
            logits_pre, feats = self._prep(mic_sig)
            # time-average to clip-level
            Es.append(energy_score(logits_pre, self.T).mean(dim=1))
            Msps.append(msp_score(logits_pre).mean(dim=1))
            Feat_list.append(feats)
        E = torch.cat(Es, 0)
        MSP = torch.cat(Msps, 0)
        F = torch.cat(Feat_list, 0)

        self.maha = fit_mahalanobis_single(F)
        maha_clip = self.maha.score(F).mean(dim=1)

        thr = Thresholds(
            energy=float(torch.quantile(E, quantile)),
            msp=float(torch.quantile(MSP, quantile)),
            maha=float(torch.quantile(maha_clip, quantile)),
        )
        self.thr = thr
        return thr

    @torch.no_grad()
    def score_clip(self, mic_sig: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits_pre, feats = self._prep(mic_sig)
        return {
            "energy": energy_score(logits_pre, self.T).mean(dim=1),
            "msp": msp_score(logits_pre).mean(dim=1),
            "maha": self.maha.score(feats).mean(dim=1) if self.maha else None,
        }

    @torch.no_grad()
    def is_ood(self, scores: Dict[str, torch.Tensor], rule: str = "any") -> torch.Tensor:
        if self.thr is None:
            raise RuntimeError("Call calibrate() first.")
        checks = []
        if scores.get("energy") is not None:
            checks.append(scores["energy"] < self.thr.energy)
        if scores.get("msp") is not None:
            checks.append(scores["msp"] < self.thr.msp)
        if scores.get("maha") is not None:
            checks.append(scores["maha"] < self.thr.maha)
        V = torch.stack(checks, dim=0)  # [K, B]
        return V.any(0) if rule == "any" else V.all(0)


# =============== Main ===============
def parse_args():
    ap = argparse.ArgumentParser(description="CRNN OOD routing to SRP")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted)")
    ap.add_argument("--energy_T", type=float, default=1.0, help="Temperature for Energy score")
    ap.add_argument("--quantile", type=float, default=0.05, help="ID quantile threshold (lower is stricter)")
    ap.add_argument("--rule", type=str, default="any", choices=["any", "all"], help="OOD vote rule across detectors")
    ap.add_argument("--batch_size", type=int, default=32, help="Eval batch size for DataModule")
    ap.add_argument("--num_workers", type=int, default=5, help="Dataloader workers")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV to save results")
    ap.add_argument("--sample_rate", type=int, default=None, help="Override SRP sample rate; else take from model.hparams.sample_rate or 48000")
    return ap.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MyModel.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    # DataModule (use your constructor signature)
    dm = MyDataModule(num_workers=args.num_workers, batch_size=(args.batch_size, args.batch_size))
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # SRP sample rate
    fs = args.sample_rate or getattr(model.hparams, "sample_rate", 48000)

    # OOD detector
    det = OODDetector(model=model, device=device, energy_T=args.energy_T)
    thr = det.calibrate(val_loader, quantile=args.quantile)
    print(f"[Calibrated thresholds] energy={thr.energy:.4f}, msp={thr.msp:.4f}, maha={thr.maha:.4f}")

    # Optional CSV
    writer = None
    if args.out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        csv_f = open(args.out, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)
        writer.writerow(["idx", "energy", "msp", "maha", "OOD", "method", "final_angle_deg"])

    # Inference loop
    idx_base = 0
    with torch.no_grad():
        for batch in test_loader:
            mic_sig = batch[0].to(device)  # raw multichannel audio batch
            B = mic_sig.shape[0]

            # OOD scores & decision
            scores = det.score_clip(mic_sig)
            ood_mask = det.is_ood(scores, rule=args.rule)  # True => route to SRP

            # CRNN prediction (ID path)
            data = model.data_preprocess(mic_sig_batch=mic_sig)[0]
            logits_pre, _ = forward_intermediates(model, data)
            crnn_prob = torch.sigmoid(logits_pre).mean(dim=1)  # [B, C]
            crnn_pred_bin = crnn_prob.argmax(dim=1).float()    # [B]

            # Allocate output vector
            y_hat = torch.empty_like(crnn_pred_bin)

            # Route
            if (~ood_mask).any():
                y_hat[~ood_mask] = crnn_pred_bin[~ood_mask]
            if ood_mask.any():
                y_hat[ood_mask] = srp_predict_batch(mic_sig[ood_mask], fs=fs).to(device)

            # Emit / save
            if writer:
                e = scores["energy"].detach().cpu().numpy()
                msp = scores["msp"].detach().cpu().numpy()
                maha = scores["maha"].detach().cpu().numpy()
                ood_np = ood_mask.detach().cpu().numpy()
                angles = y_hat.detach().cpu().numpy()
                for i in range(B):
                    method = "SRP" if ood_np[i] else "CRNN"
                    writer.writerow([idx_base + i, e[i], msp[i], maha[i], int(ood_np[i]), method, angles[i]])
            else:
                print(f"[Batch {idx_base}-{idx_base+B-1}] "
                      f"OOD={ood_mask.int().tolist()}  "
                      f"final_angles(deg)={y_hat.detach().cpu().tolist()}")

            idx_base += B

    if args.out is not None:
        csv_f.close()
        print(f"Saved results to: {args.out}")


if __name__ == "__main__":
    main()
