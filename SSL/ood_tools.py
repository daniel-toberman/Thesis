# ood_tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Callable, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf

Tensor = torch.Tensor

@torch.no_grad()
def _gather_id_scores(forward_intermediates: Callable[[Tensor], Tuple[Tensor,Tensor]],
                      loader: Iterable,
                      device: torch.device):
    """Collect pre-sigmoid logits and penultimate features from an ID loader."""
    all_logits, all_feats = [], []
    for batch in loader:
        # your DataModule returns (mic, targets, vad); we only need the model input
        if isinstance(batch, (list, tuple)):
            # In your Lightning steps you do data_preprocess first.
            # Here we rely on MyModel.data_preprocess to keep identical features.
            mic_sig = batch[0].to(device)
            yield mic_sig  # caller will preprocess & forward
        else:
            yield batch.to(device)

def energy_score(logits: Tensor, T: float = 1.0) -> Tensor:
    # logits: [B,T,C] or [B,C]; we reduce over C → return [B,T] (if 3D) or [B]
    l = logits / T
    E = T * torch.logsumexp(l, dim=-1)
    return -E  # larger = more ID (we’ll threshold the quantile)

def msp_score(logits: Tensor) -> Tensor:
    return F.softmax(logits, dim=-1).amax(dim=-1)

@dataclass
class MahalanobisModel:
    mu: Tensor      # [K,D] or [1,D]
    precision: Tensor  # [D,D]

    def score(self, feats: Tensor) -> Tensor:
        # feats: [B,T,D] or [B,D]
        orig_shape = feats.shape[:-1]
        f = feats.reshape(-1, feats.shape[-1])  # [N,D]
        if self.mu.shape[0] == 1:
            diff = f - self.mu[0]
            d2 = torch.einsum("nd,dd,nd->n", diff, self.precision, diff)
            s = -d2
        else:
            diffs = f[:, None, :] - self.mu[None, :, :]          # [N,K,D]
            left = torch.einsum("nkd,dd->nkd", diffs, self.precision)
            d2 = torch.einsum("nkd,nkd->nk", left, diffs)         # [N,K]
            s = -d2.min(dim=1).values
        return s.view(*orig_shape)

def fit_mahalanobis(feats: Tensor, labels: Optional[Tensor] = None, K: Optional[int] = None) -> MahalanobisModel:
    # feats: [B,T,D] → [N,D]
    f = feats.detach().cpu().numpy().reshape(-1, feats.shape[-1])
    if labels is None or K in (None, 1):
        mu = f.mean(axis=0, keepdims=True)
        centered = f - mu
        cov = LedoitWolf().fit(centered).covariance_
        prec = np.linalg.pinv(cov)
        return MahalanobisModel(mu=torch.from_numpy(mu).float(),
                                precision=torch.from_numpy(prec).float())
    # class-conditional (if you have labels per frame/bin; often unnecessary here)
    raise NotImplementedError("Class-conditional labels per frame not wired here.")

class OODDetector:
    def __init__(self,
                 model,                       # your MyModel (LightningModule) or its .arch
                 preprocess_fn: Callable[[Tensor], Tensor],  # MyModel.data_preprocess → returns model input
                 device: Optional[str] = None):
        self.model = model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.preprocess_fn = preprocess_fn
        self.T = 1.0
        self.maha: Optional[MahalanobisModel] = None
        self.thresholds: Dict[str, float] = {}

    @torch.no_grad()
    def _forward_logits_feats(self, mic_sig: Tensor) -> Tuple[Tensor, Tensor]:
        # replicate your Lightning preprocessing for exact parity
        # MyModel.data_preprocess returns [real_imag_stft, targets]; we only need index 0
        data = self.preprocess_fn(mic_sig_batch=mic_sig)[0]
        logits, feats = self.model.arch.forward_with_intermediates(data)  # uses your new method
        return logits, feats

    def calibrate(self, id_loader: Iterable, quantile: float = 0.05, energy_T: float = 1.0):
        self.T = float(energy_T)
        E_scores, MSP_scores, Feats = [], [], []
        for mic_sig in _gather_id_scores(self._forward_logits_feats, id_loader, self.device):
            logits, feats = self._forward_logits_feats(mic_sig)
            # reduce time dimension with mean (or median) to get per-clip scores
            E_scores.append(energy_score(logits, self.T).mean(dim=1))   # [B]
            MSP_scores.append(msp_score(logits).mean(dim=1))             # [B]
            Feats.append(feats)                                          # [B,T,256]
        E = torch.cat(E_scores, 0)
        MSP = torch.cat(MSP_scores, 0)
        F = torch.cat(Feats, 0)
        self.maha = fit_mahalanobis(F)   # single Gaussian over penultimate features

        # Calibrate thresholds on ID (lower tail → OOD)
        self.thresholds = {
            "energy": float(torch.quantile(E, quantile)),
            "msp":    float(torch.quantile(MSP, quantile)),
            "maha":   float(torch.quantile(self.maha.score(F).mean(dim=1), quantile)),
        }
        return self.thresholds

    @torch.no_grad()
    def score_batch(self, mic_sig: Tensor) -> Dict[str, Tensor]:
        logits, feats = self._forward_logits_feats(mic_sig.to(self.device))
        s = {
            "energy": energy_score(logits, self.T),      # [B,T]
            "msp":    msp_score(logits),                 # [B,T]
            "maha":   self.maha.score(feats) if self.maha else None,  # [B,T]
        }
        # return time-averaged clip scores too
        return {k: (v if v is None else v.mean(dim=1)) for k, v in s.items()}

    @torch.no_grad()
    def is_ood(self, scores: Dict[str, Tensor], rule: str = "any") -> Tensor:
        votes = []
        for name, thr in self.thresholds.items():
            if scores.get(name) is not None:
                votes.append(scores[name] < thr)   # True → OOD
        V = torch.stack(votes, 0)  # [K,B]
        return V.any(0) if rule == "any" else V.all(0)
