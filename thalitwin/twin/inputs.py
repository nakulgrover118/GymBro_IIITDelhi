"""Turn what the twin can actually observe into minute-level model inputs.

The twin never sees the simulator's ground truth. It only sees what a real
deployment would have:
  * CGM readings (every 5 or 15 min)
  * wearable steps -> physical activity
  * wearable sleep stages -> last night's sleep -> insulin sensitivity today
  * meals the patient logged in the app (with their own carb estimate)
  * medicine doses logged (taken / missed)
  * the EHR (for the prior physiology)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thalitwin.data.meals import MEAL_BY_KEY, absorption_rate
from thalitwin.data.simulator import STEPS_PER_MIN_AT_FULL, sleep_si_multiplier
from thalitwin.twin.physiology import (KA_PREMIX, PhysioParams, activity_filter, exercise_si_boost,
                                       insulin_appearance, meal_appearance, params_from_clinical)


@dataclass
class Observed:
    patient_id: str
    T: int                              # minutes (whole days from local midnight)
    cgm: np.ndarray                     # (T,) NaN except at sample minutes
    Ra: np.ndarray                      # logged-meal glucose appearance, mg/min
    Ri: np.ndarray                      # logged insulin appearance, uU/min
    A: np.ndarray                       # exercise drive (steps-derived activity through a 30-min lag)
    si: np.ndarray                      # sleep x exercise insulin-sensitivity multiplier
    egp: np.ndarray                     # missed-medicine hepatic output multiplier
    meals: pd.DataFrame                 # minute, carbs, k, slot (logged meals)
    insulin: pd.DataFrame               # minute, dose, ka (taken insulin doses)
    prior: PhysioParams                 # batch of 1
    step: int = 5
    sleep_daily: pd.DataFrame = field(default_factory=pd.DataFrame)


def sleep_by_day(stage: np.ndarray, n_days: int) -> pd.DataFrame:
    """Per-day sleep summary from a 5-min hypnogram (-1 = not in bed).

    Night 'd' is the in-bed period ending on the morning of day d.
    """
    rows = []
    for d in range(n_days):
        lo, hi = max((d * 1440 - 360) // 5, 0), min((d * 1440 + 720) // 5, len(stage))
        seg = stage[lo:hi]
        inbed = np.where(seg >= 0)[0]
        if len(inbed) == 0:
            rows.append(dict(day=d, sleep_hours=np.nan, sleep_efficiency=np.nan, wake_minute=d * 1440 + 420,
                             deep_pct=np.nan, rem_pct=np.nan))
            continue
        s = seg[inbed[0]:inbed[-1] + 1]
        rows.append(dict(day=d, sleep_hours=(s > 0).sum() * 5 / 60, sleep_efficiency=float((s > 0).mean()),
                         wake_minute=int((lo + inbed[-1] + 1) * 5), deep_pct=float((s == 2).mean()),
                         rem_pct=float((s == 3).mean())))
    return pd.DataFrame(rows)


def si_from_sleep(sleep: pd.DataFrame, T: int) -> np.ndarray:
    si = np.ones(T)
    for r in sleep.itertuples():
        if np.isnan(r.sleep_hours):
            continue
        v = float(sleep_si_multiplier(r.sleep_hours, r.sleep_efficiency))
        lo = int(r.wake_minute)
        si[lo:min(lo + 1440, T)] = v
        if r.day == 0:
            si[:lo] = v
    return si


def build_observed(pid: str, ts: pd.DataFrame, meals: pd.DataFrame, doses: pd.DataFrame, ehr: dict,
                   step: int = 5) -> Observed:
    """Observed inputs for one synthetic patient (frames already filtered to pid)."""
    T = int(np.ceil((ts.minute.max() + step) / 1440) * 1440)
    cgm = np.full(T, np.nan)
    cgm[ts.minute.values] = ts.cgm.values
    A = np.zeros(T)
    if "steps" in ts:
        a5 = np.clip(ts.steps.values / (step * STEPS_PER_MIN_AT_FULL), 0, 1)
        A[: len(a5) * step] = np.repeat(a5, step)[:T]
    if "sleep_stage" in ts:
        sleep = sleep_by_day(ts.sleep_stage.values, T // 1440)
        si = si_from_sleep(sleep, T)
    else:
        sleep, si = pd.DataFrame(), np.ones(T)
    si = si * exercise_si_boost(A[None, :])[0]
    A = activity_filter(A)                 # exercise drive seen by the physiology

    lm = meals[meals.logged].copy() if "logged" in meals else meals.copy()
    if "carbs_logged" in lm:
        lm["carbs"] = lm["carbs_logged"]
    if "k" not in lm:
        lm["k"] = [absorption_rate(g, s) for g, s in zip(lm.gi, lm.slow)]
    lm = lm[["minute", "carbs", "k", "slot"]].reset_index(drop=True)
    Ra = meal_appearance(T, lm.minute, lm.carbs, lm.k)

    egp = np.ones(T)
    ins = pd.DataFrame(columns=["minute", "dose", "ka"])
    if len(doses):
        for r in doses.itertuples():
            if r.drug == "metformin" and not r.taken:
                egp[int(r.minute):int(r.minute) + 720] += 0.12
            if r.drug == "insulin_glargine" and not r.taken:
                egp[int(r.minute):int(r.minute) + 1440] += 0.15
        pm = doses[(doses.drug == "insulin_premix") & doses.taken]
        ins = pd.DataFrame({"minute": pm.minute.values, "dose": pm.dose.values.astype(float), "ka": KA_PREMIX})
    Ri = insulin_appearance(T, ins.minute, ins.dose, ins.ka.values if len(ins) else KA_PREMIX)

    prior = params_from_clinical(ehr["fpg"], ehr["fasting_insulin"], ehr["weight_kg"], ehr["hba1c"],
                                 ehr.get("sulfonylurea"), ehr.get("tcf7l2_rs7903146_risk_alleles"))
    return Observed(pid, T, cgm, Ra, Ri, A, si, egp, lm, ins, prior, step, sleep)


def stack_inputs(obs: list[Observed]):
    """Pad patients to a common length and stack into (B, T) arrays."""
    T = max(o.T for o in obs)
    def pad(a, fill):
        out = np.full(T, fill, float)
        out[: len(a)] = a
        return out
    keys = dict(cgm=np.nan, Ra=0.0, Ri=0.0, A=0.0, si=1.0, egp=1.0)
    arr = {k: np.stack([pad(getattr(o, k), f) for o in obs]) for k, f in keys.items()}
    prior = PhysioParams.stack([o.prior for o in obs])
    return T, arr, prior
