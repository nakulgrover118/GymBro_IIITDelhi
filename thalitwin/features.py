"""Feature engineering and labels on a regular sampling grid.

Each row is "the patient at time t". Every feature uses only data available
at t; labels look into (t, t+120 min]. Works for 5-min (synthetic) and
15-min (ShanghaiT2DM) grids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thalitwin.config import ALERT_WINDOW_MIN, HORIZONS, SPIKE_MGDL

EHR_FEATURES = ["age", "sex_m", "bmi", "waist_cm", "hba1c", "fpg", "fasting_insulin", "homa_ir",
                "diabetes_duration_y", "egfr", "triglycerides", "hdl", "ldl", "sbp", "hypertension",
                "metformin", "sulfonylurea", "dpp4", "sglt2", "insulin", "tcf7l2_rs7903146_risk_alleles",
                "kcnq1_rs2237892_risk_alleles", "polygenic_risk_z", "family_history_t2d"]
CGM_FEATURES = ["cgm_now", "cgm_lag15", "cgm_lag30", "cgm_lag60", "cgm_lag90", "cgm_lag120", "d15", "d30", "d60",
                "accel", "roll60_mean", "roll60_std", "roll180_mean", "roll180_std", "roll180_max", "roll180_min",
                "mean_24h", "tar_24h", "max_24h", "mins_since_high", "missing_60", "tod_sin", "tod_cos"]
LOG_FEATURES = ["carbs_30", "carbs_60", "carbs_120", "carbs_240", "cob", "mins_since_meal", "last_meal_carbs",
                "last_meal_k", "iob", "mins_since_insulin", "missed_metformin_12h", "weekend"]
WEAR_FEATURES = ["steps_15", "steps_30", "steps_60", "steps_120", "steps_today", "active_min_120",
                 "hr_now", "hr_mean_30", "hr_vs_rest", "hrv_mean_30", "hrv_night", "sleep_hours",
                 "sleep_efficiency", "deep_pct", "rem_pct", "sleep_debt", "mins_since_wake"]
TWIN_FEATURES = ["twin_now", "twin_g30", "twin_g60", "twin_g120", "twin_max120", "twin_known_g60",
                 "twin_known_max120", "twin_exp_carbs", "twin_si_rel", "twin_meal_gain", "twin_gb", "twin_X",
                 "twin_I_excess", "twin_si_today", "twin_disturbance", "twin_d60", "twin_d120", "twin_dmax"]


def _shift(a, k):
    out = np.full_like(a, np.nan, dtype=float)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _window_sum(event_min, event_val, t, width):
    """Sum of event values with minute in (t - width, t]."""
    if len(event_min) == 0:
        return np.zeros(len(t))
    o = np.argsort(event_min)
    em, ev = np.asarray(event_min)[o], np.asarray(event_val, float)[o]
    cs = np.concatenate([[0.0], np.cumsum(ev)])
    hi = np.searchsorted(em, t, side="right")
    lo = np.searchsorted(em, t - width, side="right")
    return cs[hi] - cs[lo]


def _since(event_min, t, cap=600.0):
    if len(event_min) == 0:
        return np.full(len(t), cap), np.full(len(t), -1, int)
    em = np.sort(np.asarray(event_min))
    i = np.searchsorted(em, t, side="right") - 1
    out = np.where(i >= 0, t - em[np.maximum(i, 0)], cap)
    return np.minimum(out, cap), i


def patient_features(grid: pd.DataFrame, step: int, meals: pd.DataFrame, insulin: pd.DataFrame,
                     missed_metformin: np.ndarray | None = None, sleep: pd.DataFrame | None = None,
                     start_weekday: int | None = None) -> pd.DataFrame:
    """grid: minute, cgm and optional steps, heart_rate, hrv_rmssd, sleep_stage (regular step)."""
    t = grid.minute.values.astype(float)
    n = len(t)
    raw = grid.cgm.values.astype(float)
    cg = pd.Series(raw).ffill(limit=max(15 // step, 1)).values
    S = lambda minutes: max(int(round(minutes / step)), 1)
    f = {"minute": grid.minute.values}

    # ---------------- CGM ----------------
    f["cgm_now"] = cg
    for L in (15, 30, 60, 90, 120):
        f[f"cgm_lag{L}"] = _shift(cg, S(L))
    f["d15"] = cg - f["cgm_lag15"]
    f["d30"] = cg - f["cgm_lag30"]
    f["d60"] = cg - f["cgm_lag60"]
    f["accel"] = f["d15"] - (f["cgm_lag15"] - f["cgm_lag30"])
    s = pd.Series(cg)
    for w in (60, 180):
        r = s.rolling(S(w) + 1, min_periods=max(S(w) // 2, 1))
        f[f"roll{w}_mean"], f[f"roll{w}_std"] = r.mean().values, r.std().values
        if w == 180:
            f["roll180_max"], f["roll180_min"] = r.max().values, r.min().values
    r24 = s.rolling(S(1440), min_periods=S(240))
    f["mean_24h"] = r24.mean().values
    f["max_24h"] = r24.max().values
    f["tar_24h"] = (s > SPIKE_MGDL).astype(float).where(s.notna()).rolling(S(1440), min_periods=S(240)).mean().values
    high_min = t[cg > SPIKE_MGDL]
    f["mins_since_high"], _ = _since(high_min, t, cap=1440.0)
    f["missing_60"] = pd.Series(np.isnan(raw).astype(float)).rolling(S(60) + 1, min_periods=1).mean().values
    tod = (t % 1440) / 1440.0
    f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)

    # ---------------- logs: meals, insulin, medicines ----------------
    mm = meals.minute.values.astype(float) if len(meals) else np.array([])
    mc = meals.carbs.values.astype(float) if len(meals) else np.array([])
    mk = meals.k.values.astype(float) if len(meals) else np.array([])
    for w in (30, 60, 120, 240):
        f[f"carbs_{w}"] = _window_sum(mm, mc, t, w)
    cob = np.zeros(n)
    for m0, c, k in zip(mm, mc, mk):
        dt = t - m0
        ok = (dt >= 0) & (dt < 600)
        cob[ok] += c * np.exp(-k * dt[ok]) * (1 + k * dt[ok])
    f["cob"] = cob
    f["mins_since_meal"], idx = _since(mm, t)
    order = np.argsort(mm) if len(mm) else np.array([], int)
    f["last_meal_carbs"] = np.where(idx >= 0, mc[order][np.maximum(idx, 0)] if len(mm) else 0, 0.0)
    f["last_meal_k"] = np.where(idx >= 0, mk[order][np.maximum(idx, 0)] if len(mm) else 0, 0.0)
    iob = np.zeros(n)
    im = insulin.minute.values.astype(float) if len(insulin) else np.array([])
    for m0, dose, ka in zip(im, insulin.dose.values if len(insulin) else [], insulin.ka.values if len(insulin) else []):
        dt = t - m0
        ok = (dt >= 0) & (dt < 720)
        iob[ok] += dose * np.exp(-ka * dt[ok]) * (1 + ka * dt[ok])
    f["iob"] = iob
    f["mins_since_insulin"], _ = _since(im, t)
    mmiss = np.asarray(missed_metformin if missed_metformin is not None else [], float)
    f["missed_metformin_12h"] = _window_sum(mmiss, np.ones(len(mmiss)), t, 720)
    if start_weekday is not None:
        f["weekend"] = (((t // 1440) + start_weekday) % 7 >= 5).astype(float)

    # ---------------- wearables ----------------
    if "steps" in grid:
        st = pd.Series(grid.steps.values.astype(float))
        for w in (15, 30, 60, 120):
            f[f"steps_{w}"] = st.rolling(S(w), min_periods=1).sum().values
        day = (t // 1440).astype(int)
        f["steps_today"] = st.groupby(day).cumsum().values
        f["active_min_120"] = (st > 60 * step).astype(float).rolling(S(120), min_periods=1).sum().values * step
        hr = pd.Series(grid.heart_rate.values.astype(float))
        f["hr_now"] = hr.values
        f["hr_mean_30"] = hr.rolling(S(30), min_periods=1).mean().values
        asleep = grid.sleep_stage.values > 0
        rest = hr.where(asleep).rolling(S(1440), min_periods=S(120)).median().ffill().values
        f["hr_vs_rest"] = f["hr_mean_30"] - rest
        hv = pd.Series(grid.hrv_rmssd.values.astype(float))
        f["hrv_mean_30"] = hv.rolling(S(30), min_periods=1).mean().values
        f["hrv_night"] = hv.where(asleep).rolling(S(1440), min_periods=S(120)).mean().ffill().values
    if sleep is not None and len(sleep):
        sl = sleep.set_index("day")
        # attribute each time point to the most recent completed night
        wake = sl.wake_minute.values
        k = np.searchsorted(wake, t, side="right") - 1
        valid = k >= 0
        for col in ("sleep_hours", "sleep_efficiency", "deep_pct", "rem_pct"):
            v = sl[col].values
            f[col] = np.where(valid, v[np.maximum(k, 0)], np.nan)
        f["sleep_debt"] = np.clip(7.0 - f["sleep_hours"], 0, None)
        f["mins_since_wake"] = np.where(valid, t - wake[np.maximum(k, 0)], np.nan)

    # ---------------- labels ----------------
    for h in HORIZONS:
        f[f"y_g{h}"] = _shift(raw, -S(h))
    W = S(ALERT_WINDOW_MIN)
    fut = np.stack([_shift(raw, -j) for j in range(1, W + 1)], axis=1)
    cov = (~np.isnan(fut)).mean(1)
    fmax = np.nanmax(np.where(np.isnan(fut), -np.inf, fut), axis=1)
    spike = (fmax >= SPIKE_MGDL).astype(float)
    spike[(cov < 0.5) | np.isnan(cg)] = np.nan
    spike[cg >= SPIKE_MGDL] = np.nan          # only predict NEW spikes from below threshold
    f["y_spike"] = spike
    f["y_max120"] = np.where(cov >= 0.5, fmax, np.nan)
    # event onsets: first reading >= 180 after >= 30 min below
    above = np.nan_to_num(raw, nan=0.0) >= SPIKE_MGDL
    below_prev = np.ones(n, bool)
    for j in range(1, S(30) + 1):
        below_prev &= ~_shift(above.astype(float), j).astype(bool) & ~np.isnan(_shift(raw, j))
    f["spike_onset"] = (above & below_prev).astype(np.int8)
    return pd.DataFrame(f)


def add_twin_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df["twin_d60"] = df["twin_g60"] - df["cgm_now"]
    df["twin_d120"] = df["twin_g120"] - df["cgm_now"]
    df["twin_dmax"] = df["twin_max120"] - df["cgm_now"]
    return df


def ehr_static(ehr: pd.DataFrame) -> pd.DataFrame:
    e = ehr.copy()
    e["sex_m"] = (e["sex"] == "M").astype(float)
    cols = ["patient_id"] + [c for c in EHR_FEATURES if c in e]
    out = e[cols].copy()
    for c in cols[1:]:
        out[c] = out[c].astype(float)
    return out
