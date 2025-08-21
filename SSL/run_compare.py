# run_compare.py
from __future__ import annotations
import argparse, csv, os, sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --- Your codebase imports ---
from run_CRNN import MyModel, MyDataModule  # must be importable from CWD / PYTHONPATH

# ---------- SRP import (same robust loader as before) ----------
def _import_srp():
    candidates = []
    cwd = os.path.abspath(os.getcwd())
    parts = cwd.split(os.sep)
    if "Thesis" in parts:
        thesis_root = os.sep.join(parts[: parts.index("Thesis") + 1])
        candidates += [
            os.path.join(thesis_root, "xsrpMain", "xsrp"),
            os.path.join(thesis_root, "xsrpMain"),
            thesis_root,
        ]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "..", "xsrpMain", "xsrp"),
        os.path.join(here, "xsrpMain", "xsrp"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from run_SRP import run_srp_return_details
        return run_srp_return_details
    except Exception:
        pass
    try:
        from xsrpMain.xsrp.run_SRP import run_srp_return_details
        return run_srp_return_details
    except Exception:
        pass
    # last resort: file next to this script
    for guess in ["run_SRP.py", os.path.join(here, "run_SRP.py")]:
        if os.path.isfile(guess):
            import importlib.util
            spec = importlib.util.spec_from_file_location("srp_local", guess)
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            return getattr(mod, "run_srp_return_details")
    raise ImportError("Cannot import run_srp_return_details (SRP). Please add Thesis\\xsrpMain\\xsrp to PYTHONPATH.")

RUN_SRP_RETURN_DETAILS = _import_srp()

# ---------- OOD pieces (Energy + single-Gaussian Mahalanobis) ----------
def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return -(T * torch.logsumexp(logits / T, dim=-1))  # larger = more ID

@dataclass
class MahalanobisModel:
    mu: torch.Tensor       # [1, D]
    precision: torch.Tensor  # [D, D]
    def score(self, feats: torch.Tensor) -> torch.Tensor:
        orig = feats.shape[:-1]
        f = feats.reshape(-1, feats.shape[-1])
        diff = f - self.mu[0]
        d2 = torch.einsum("nd,dd,nd->n", diff, self.precision, diff)
        return (-d2).view(*orig)  # larger = more ID

def fit_mahalanobis_single(feats: torch.Tensor) -> MahalanobisModel:
    X = feats.detach().cpu().numpy().reshape(-1, feats.shape[-1])
    mu = X.mean(axis=0, keepdims=True)
    centered = X - mu
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(centered).covariance_
    except Exception:
        cov = np.cov(centered, rowvar=False) + 1e-6*np.eye(centered.shape[1])
    prec = np.linalg.pinv(cov)
    return MahalanobisModel(torch.from_numpy(mu).float(), torch.from_numpy(prec).float())

class _Hook:
    def __init__(self, module: torch.nn.Module):
        self.fmap = None
        self.h = module.register_forward_hook(self._cb)
    def _cb(self, m, i, o): self.fmap = o.detach()
    def close(self): self.h.remove()

def forward_intermediates(model: MyModel, data_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    arch = model.arch
    if hasattr(arch, "forward_with_intermediates"):
        return arch.forward_with_intermediates(data_tensor)  # (pre_sig_logits [B,T,360], penult [B,T,256])
    # Fallback: hook penultimate and final linear
    if not (hasattr(arch, "relu") and hasattr(arch, "ipd2xyz2")):
        raise AttributeError("Model.arch missing relu/ipd2xyz2; add forward_with_intermediates to CRNN.")
    h_pen, h_log = _Hook(arch.relu), _Hook(arch.ipd2xyz2)
    _ = arch(data_tensor)  # forward calls sigmoid internally; hooks grab pre-sig and penult
    penult, logits_pre = h_pen.fmap, h_log.fmap
    h_pen.close(); h_log.close()
    if penult is None or logits_pre is None:
        raise RuntimeError("Failed to capture intermediates via hooks.")
    return logits_pre, penult

# ---------- SRP wrapper for batches ----------
def srp_predict_batch(mic_sig_batch: torch.Tensor, fs: int) -> torch.Tensor:
    x = mic_sig_batch.detach().cpu()
    if x.dim() != 3: raise ValueError(f"mic_sig_batch must be [B,T,C] or [B,C,T], got {tuple(x.shape)}")
    if x.shape[1] < x.shape[2]:  # [B,T,C] -> [B,C,T]
        x = x.permute(0,2,1)
    B, C, T = x.shape
    out = torch.empty((B,), dtype=torch.float32)
    for b in range(B):
        az, *_ = RUN_SRP_RETURN_DETAILS(fs=fs, mic_signals=x[b].numpy().astype(np.float64))
        out[b] = float(az)
    return out

# ---------- Angles utils ----------
def angle_from_logits_pre_sig(logits_pre: torch.Tensor) -> torch.Tensor:
    # logits_pre: [B,T,360] -> time-avg -> sigmoid -> argmax
    prob = torch.sigmoid(logits_pre).mean(dim=1)  # [B,360]
    return prob.argmax(dim=1).float()             # 0..359

def circ_abs_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # minimal absolute difference on circle (degrees)
    d = torch.abs(a - b) % 360.0
    return torch.minimum(d, 360.0 - d)

def target_angle_from_batch(batch: Tuple) -> torch.Tensor:
    """
    Try a couple of likely formats:
      - one-hot/soft 360-vector per frame -> time-avg -> argmax
      - direct scalar degrees in batch[1]
    Falls back to argmax over last dim if tensor rank>1.
    """
    tgt = batch[1]
    if isinstance(tgt, (list, tuple)):
        tgt = tgt[0]
    if torch.is_tensor(tgt):
        if tgt.ndim >= 2 and tgt.shape[-1] == 360:
            return tgt.float().mean(dim=1).argmax(dim=1).float()
        if tgt.ndim == 1:  # already [B] degrees
            return tgt.float()
        # generic: last-dim argmax after time-avg if present
        if tgt.ndim >= 2:
            tavg = tgt.float().mean(dim=1) if tgt.ndim > 2 else tgt.float()
            return tavg.argmax(dim=1).float()
    raise ValueError("Could not parse target angles from batch[1]. Adjust target_angle_from_batch() for your dataset.")

# ---------- Metrics ----------
@dataclass
class Metrics:
    mae: float
    medae: float
    acc5: float
    acc10: float
    acc15: float
    n: int

def compute_metrics(pred_deg: torch.Tensor, true_deg: torch.Tensor) -> Metrics:
    diff = circ_abs_diff_deg(pred_deg, true_deg).cpu().numpy()
    mae = float(np.mean(diff))
    medae = float(np.median(diff))
    acc5 = float(np.mean(diff <= 5.0))
    acc10 = float(np.mean(diff <= 10.0))
    acc15 = float(np.mean(diff <= 15.0))
    return Metrics(mae, medae, acc5, acc10, acc15, len(diff))

def print_metrics(title: str, m: Metrics):
    print(f"\n[{title}] N={m.n}")
    print(f"  MAE:   {m.mae:6.3f}°")
    print(f"  MedAE: {m.medae:6.3f}°")
    print(f"  Acc@5°:{m.acc5*100:6.2f}%   Acc@10°:{m.acc10*100:6.2f}%   Acc@15°:{m.acc15*100:6.2f}%")

# ---------- Main compare runner ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Compare CRNN vs SRP vs CRNN+OOD→SRP")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (auto)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=5)
    ap.add_argument("--energy_T", type=float, default=1.0)
    ap.add_argument("--quantile", type=float, default=0.05, help="ID quantile for thresholds")
    ap.add_argument("--rule", type=str, default="any", choices=["any","all"], help="OOD vote rule")
    ap.add_argument("--sample_rate", type=int, default=None, help="Override SRP fs")
    ap.add_argument("--out_csv", type=str, default=None, help="Write per-clip CSV with predictions from all 3 methods")
    return ap.parse_args()

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + data
    model = MyModel.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    dm = MyDataModule(num_workers=args.num_workers, batch_size=(args.batch_size, args.batch_size))
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Sample rate for SRP
    fs = args.sample_rate or getattr(model.hparams, "sample_rate", 48000)

    # --- Calibrate OOD on val (Energy + Mahalanobis) ---
    Es, Feats = [], []
    with torch.no_grad():
        for batch in val_loader:
            mic_sig = batch[0].to(device)
            data = model.data_preprocess(mic_sig_batch=mic_sig)[0]
            logits_pre, feats = forward_intermediates(model, data)
            Es.append(energy_score(logits_pre, args.energy_T).mean(dim=1))  # [B]
            Feats.append(feats)                                            # [B,T,D]
    E_id = torch.cat(Es, 0)
    F_id = torch.cat(Feats, 0)
    maha = fit_mahalanobis_single(F_id)
    maha_id = maha.score(F_id).mean(dim=1)

    thr_energy = float(torch.quantile(E_id, args.quantile))
    thr_maha   = float(torch.quantile(maha_id, args.quantile))
    print(f"[Calibrated] energy_thr={thr_energy:.4f}  maha_thr={thr_maha:.4f}  (T={args.energy_T}, q={args.quantile})")

    # --- Evaluate all 3 methods on test ---
    y_true_all = []
    y_crnn_all = []
    y_srp_all  = []
    y_combo_all = []

    writer = None
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        f = open(args.out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["idx","target_deg","crnn_deg","srp_deg","combo_deg","energy","maha","ood"])

    idx = 0
    with torch.no_grad():
        for batch in test_loader:
            mic_sig = batch[0].to(device)
            tgt_deg = target_angle_from_batch(batch).to(device)  # [B]
            B = mic_sig.shape[0]

            # CRNN prediction
            data = model.data_preprocess(mic_sig_batch=mic_sig)[0]
            logits_pre, feats = forward_intermediates(model, data)
            crnn_deg = angle_from_logits_pre_sig(logits_pre)  # [B]

            # SRP prediction
            srp_deg = srp_predict_batch(mic_sig, fs=fs).to(device)  # [B]

            # OOD scores and decision
            energy_clip = energy_score(logits_pre, args.energy_T).mean(dim=1)    # [B]
            maha_clip   = maha.score(feats).mean(dim=1)                          # [B]
            ood_mask = (energy_clip < thr_energy) | (maha_clip < thr_maha) if args.rule == "any" \
                       else (energy_clip < thr_energy) & (maha_clip < thr_maha)

            # Combination
            combo_deg = torch.where(ood_mask, srp_deg, crnn_deg)

            # Accumulate
            y_true_all.append(tgt_deg.cpu())
            y_crnn_all.append(crnn_deg.cpu())
            y_srp_all.append(srp_deg.cpu())
            y_combo_all.append(combo_deg.cpu())

            if writer:
                for i in range(B):
                    writer.writerow([
                        idx+i,
                        float(tgt_deg[i].cpu()),
                        float(crnn_deg[i].cpu()),
                        float(srp_deg[i].cpu()),
                        float(combo_deg[i].cpu()),
                        float(energy_clip[i].cpu()),
                        float(maha_clip[i].cpu()),
                        int(bool(ood_mask[i].cpu()))
                    ])
            idx += B

    if writer:
        f.close()
        print(f"Saved per-clip results to {args.out_csv}")

    # --- Metrics ---
    y_true  = torch.cat(y_true_all, 0)
    y_crnn  = torch.cat(y_crnn_all, 0)
    y_srp   = torch.cat(y_srp_all, 0)
    y_combo = torch.cat(y_combo_all, 0)

    m_crnn  = compute_metrics(y_crnn, y_true)
    m_srp   = compute_metrics(y_srp,  y_true)
    m_combo = compute_metrics(y_combo, y_true)

    # Summary
    print_metrics("CRNN only", m_crnn)
    print_metrics("SRP only",  m_srp)
    print_metrics("CRNN + OOD→SRP", m_combo)

if __name__ == "__main__":
    main()
