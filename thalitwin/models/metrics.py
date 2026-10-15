"""Evaluation metrics: forecast accuracy, Clarke error grid, alert quality."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from thalitwin.config import ALERT_WINDOW_MIN


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mard(y, p):
    y = np.asarray(y, float)
    return float(np.mean(np.abs(y - np.asarray(p)) / y) * 100)


def clarke_zones(ref, pred) -> np.ndarray:
    """Clarke error grid zone (A-E) for each pair (Clarke et al., 1987)."""
    r, p = np.asarray(ref, float), np.asarray(pred, float)
    z = np.full(len(r), "B", dtype="<U1")
    a = ((r <= 70) & (p <= 70)) | (np.abs(p - r) <= 0.2 * r)
    e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    c = ((r >= 70) & (r <= 290) & (p >= r + 110)) | ((r >= 130) & (r <= 180) & (p <= (7 / 5) * r - 182))
    d = ((r >= 240) & (p >= 70) & (p <= 180)) | ((r <= 175 / 3) & (p >= 70) & (p <= 180)) | \
        ((r >= 175 / 3) & (r <= 70) & (p >= (6 / 5) * r))
    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"
    return z


def clarke_summary(ref, pred) -> dict:
    z = clarke_zones(ref, pred)
    return {f"clarke_{k}": float((z == k).mean() * 100) for k in "ABCDE"}


def regression_report(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    out = {"n": int(ok.sum()), "rmse": rmse(y, p), "mae": mae(y, p), "mard": mard(y, p)}
    cz = clarke_summary(y, p)
    out["clarke_A"], out["clarke_AB"] = cz["clarke_A"], cz["clarke_A"] + cz["clarke_B"]
    return out


def classification_report(y, prob, thr) -> dict:
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    ok = ~np.isnan(y) & ~np.isnan(prob)
    y, prob = y[ok], prob[ok]
    pred = prob >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"n": int(ok.sum()), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, prob)), "auprc": float(average_precision_score(y, prob)),
            "threshold": float(thr), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def pick_threshold(y, prob, target_sensitivity=0.80) -> float:
    """Highest threshold that still catches `target_sensitivity` of positive windows."""
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    ok = ~np.isnan(y) & ~np.isnan(prob)
    pos = np.sort(prob[ok][y[ok] == 1])
    if len(pos) == 0:
        return 0.5
    return float(pos[int(np.floor((1 - target_sensitivity) * len(pos)))])


def event_report(df: pd.DataFrame, prob_col: str, thr: float, step: int, min_lead: int = 15,
                 snooze_min: int = 60) -> dict:
    """Event-level alert quality.

    An event is a spike onset (first reading >= 180 after >= 30 min below).
    It counts as *predicted* if an alert fired between 120 and `min_lead`
    minutes before onset. Lead time = onset - first alert in that window.
    Notifications: while the alert is on, a notification is sent unless one
    was sent in the last `snooze_min` minutes. False alarms: notifications
    with no spike onset in the following 2 hours, reported per patient-day.
    """
    detected, leads, n_events, false_eps, n_eps, days = 0, [], 0, 0, 0, 0.0
    for _, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("minute")
        t = g.minute.values
        alert = (g[prob_col].values >= thr) & ~np.isnan(g[prob_col].values)
        onsets = t[g.spike_onset.values == 1]
        days += (t.max() - t.min() + step) / 1440.0
        for o in onsets:
            n_events += 1
            w = (t >= o - ALERT_WINDOW_MIN) & (t <= o - min_lead) & alert
            if w.any():
                detected += 1
                leads.append(o - t[w][0])
        # notifications: sent whenever the alert is on and none was sent in the previous
        # `snooze_min` minutes (so a persistent alert re-notifies hourly, a flickering one does not)
        starts, last = [], -np.inf
        for s in np.where(alert)[0]:
            if t[s] - last >= snooze_min:
                starts.append(s)
                last = t[s]
        n_eps += len(starts)
        for s in starts:
            if not ((onsets > t[s]) & (onsets <= t[s] + ALERT_WINDOW_MIN)).any():
                false_eps += 1
    return {"events": n_events, "event_recall": detected / max(n_events, 1),
            "median_lead_min": float(np.median(leads)) if leads else 0.0,
            "mean_lead_min": float(np.mean(leads)) if leads else 0.0,
            "alerts_per_day": n_eps / max(days, 1e-9),
            "false_alerts_per_day": false_eps / max(days, 1e-9),
            "events_per_day": n_events / max(days, 1e-9)}
