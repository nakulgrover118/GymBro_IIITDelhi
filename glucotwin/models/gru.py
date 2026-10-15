"""Deep-learning fusion model (PyTorch) used as a comparison to LightGBM.

Architecture
  sequence branch : last 3 h of dynamic signals -> 2-layer GRU (64 units)
  static branch   : EHR + twin state at time t -> MLP (64 units)
  fusion head     : concat -> MLP -> [glucose change at 30/60/120 min, spike logit]
Trained multi-task (MSE on scaled glucose change + BCE on the spike label).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

SEQ_CANDIDATES = ["cgm_now", "cob", "iob", "steps_15", "hr_now", "hrv_mean_30", "tod_sin", "tod_cos"]
SCALE = 50.0


class FusionGRU(nn.Module):
    def __init__(self, n_seq: int, n_static: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_seq, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.static = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, seq, static):
        _, h = self.gru(seq)
        z = torch.cat([h[-1], self.static(static)], dim=1)
        return self.head(z)


class SequenceData:
    """Builds (sequence, static, targets) windows from the feature table."""

    def __init__(self, df: pd.DataFrame, seq_cols, static_cols, step: int, hist_min: int = 180,
                 stats: dict | None = None):
        self.L = hist_min // step
        df = df.sort_values(["patient_id", "minute"]).reset_index(drop=True)
        self.df = df
        seq = df[seq_cols].to_numpy(np.float32)
        st = df[static_cols].to_numpy(np.float32)
        if stats is None:
            stats = {"seq_mu": np.nanmean(seq, 0), "seq_sd": np.nanstd(seq, 0) + 1e-3,
                     "st_mu": np.nanmean(st, 0), "st_sd": np.nanstd(st, 0) + 1e-3}
        self.stats = stats
        self.seq = np.nan_to_num((seq - stats["seq_mu"]) / stats["seq_sd"])
        self.static = np.nan_to_num((st - stats["st_mu"]) / stats["st_sd"])
        now = df["cgm_now"].to_numpy(np.float32)
        self.y = np.stack([(df[f"y_g{h}"].to_numpy(np.float32) - now) / SCALE for h in (30, 60, 120)]
                          + [df["y_spike"].to_numpy(np.float32)], 1)
        # a window is valid if the L previous rows belong to the same patient
        pid = df["patient_id"].to_numpy()
        same = np.ones(len(df), bool)
        same[: self.L] = False
        same[self.L:] = pid[self.L:] == pid[: -self.L]
        self.valid = np.where(same & ~np.isnan(now))[0]

    def batch(self, rows):
        idx = rows[:, None] + np.arange(-self.L + 1, 1)[None, :]
        return (torch.from_numpy(self.seq[idx]), torch.from_numpy(self.static[rows]),
                torch.from_numpy(self.y[rows]))


def _loss(out, y):
    reg_mask = ~torch.isnan(y[:, :3])
    reg = ((out[:, :3] - torch.nan_to_num(y[:, :3])) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(min=1)
    cm = ~torch.isnan(y[:, 3])
    clf = nn.functional.binary_cross_entropy_with_logits(out[cm, 3], y[cm, 3]) if cm.any() else out.sum() * 0
    return reg + clf


def train_gru(tr: SequenceData, va: SequenceData, epochs=6, batch=512, per_epoch=150_000, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FusionGRU(tr.seq.shape[1], tr.static.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        rows = rng.choice(tr.valid, size=min(per_epoch, len(tr.valid)), replace=False)
        for i in range(0, len(rows), batch):
            s, st, y = tr.batch(rows[i:i + batch])
            opt.zero_grad()
            loss = _loss(model(s, st), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vl = evaluate_loss(model, va)
        log(f"    GRU epoch {ep + 1}/{epochs}  val loss {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_loss(model, ds: SequenceData, n=40_000, seed=1):
    model.eval()
    rows = np.random.default_rng(seed).choice(ds.valid, size=min(n, len(ds.valid)), replace=False)
    tot = 0.0
    for i in range(0, len(rows), 2048):
        s, st, y = ds.batch(rows[i:i + 2048])
        tot += _loss(model(s, st), y).item() * len(s)
    return tot / len(rows)


@torch.no_grad()
def predict(model, ds: SequenceData) -> pd.DataFrame:
    model.eval()
    outs = []
    for i in range(0, len(ds.valid), 4096):
        rows = ds.valid[i:i + 4096]
        s, st, _ = ds.batch(rows)
        outs.append(model(s, st).numpy())
    o = np.concatenate(outs) if outs else np.zeros((0, 4))
    base = ds.df.iloc[ds.valid][["patient_id", "minute", "cgm_now"]].reset_index(drop=True)
    for j, h in enumerate((30, 60, 120)):
        base[f"gru_g{h}"] = base.cgm_now.values + o[:, j] * SCALE
    base["gru_spike"] = 1 / (1 + np.exp(-o[:, 3]))
    return base
