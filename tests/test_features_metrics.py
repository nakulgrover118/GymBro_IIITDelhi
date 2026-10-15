"""Feature leakage, labels and metric checks."""
import numpy as np
import pandas as pd

from thalitwin.features import patient_features
from thalitwin.models.metrics import clarke_zones, event_report, pick_threshold


def _grid(vals, step=5):
    return pd.DataFrame({"minute": np.arange(len(vals)) * step, "cgm": np.asarray(vals, float)})


EMPTY_M = pd.DataFrame(columns=["minute", "carbs", "k", "slot"])
EMPTY_I = pd.DataFrame(columns=["minute", "dose", "ka"])


def test_labels_look_ahead_features_do_not():
    vals = np.r_[np.full(60, 120.0), np.linspace(120, 220, 30), np.full(60, 220.0)]
    f = patient_features(_grid(vals), 5, EMPTY_M, EMPTY_I)
    i = 55
    assert f.cgm_now[i] == 120 and f.cgm_lag30[i] == 120
    assert f.y_g120[i] == vals[i + 24]
    assert f.y_spike[i] == 1                                     # crosses 180 within 2 h
    assert np.isnan(f.y_spike[100])                              # already above 180 -> not a "new spike" row
    # changing the future must not change today's features
    vals2 = vals.copy(); vals2[i + 1:] = 100
    f2 = patient_features(_grid(vals2), 5, EMPTY_M, EMPTY_I)
    feat_cols = [c for c in f.columns if not c.startswith("y_") and c != "spike_onset"]
    pd.testing.assert_series_equal(f.loc[i, feat_cols], f2.loc[i, feat_cols], check_names=False)


def test_spike_onset_detected_once():
    vals = np.r_[np.full(30, 120.0), np.full(10, 200.0), np.full(30, 120.0)]
    f = patient_features(_grid(vals), 5, EMPTY_M, EMPTY_I)
    assert f.spike_onset.sum() == 1 and f.spike_onset[30] == 1


def test_clarke_zones():
    z = clarke_zones([100, 100, 300, 50], [105, 160, 120, 250])
    assert list(z) == ["A", "B", "D", "E"]


def test_threshold_hits_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    thr = pick_threshold(y, p, 0.8)
    assert abs((p[y == 1] >= thr).mean() - 0.8) < 0.02


def test_event_report_counts_lead_time():
    t = np.arange(0, 600, 5)
    onset = np.zeros(len(t), int); onset[60] = 1                 # spike at minute 300
    prob = np.where((t >= 240) & (t < 300), 0.9, 0.1)            # alert from minute 240
    d = pd.DataFrame({"patient_id": "x", "minute": t, "spike_onset": onset, "p": prob})
    r = event_report(d, "p", 0.5, 5)
    assert r["event_recall"] == 1.0 and r["median_lead_min"] == 60 and r["false_alerts_per_day"] == 0
