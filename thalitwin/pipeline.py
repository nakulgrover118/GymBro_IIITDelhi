"""End-to-end dataset builders for the synthetic cohort and ShanghaiT2DM.

raw streams -> observed inputs -> twin (calibrate, synchronise, forecast)
            -> features + labels -> one parquet per cohort
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from thalitwin import config as C
from thalitwin.data.ehr_generator import generate_cohort, lab_history, write_fhir
from thalitwin.data.simulator import START_DATE, simulate_cohort
from thalitwin.features import add_twin_deltas, ehr_static, patient_features
from thalitwin.twin.engine import calibrate, forecast_grid, run_observer
from thalitwin.twin.inputs import Observed, build_observed, stack_inputs
from thalitwin.twin.physiology import params_from_clinical
from thalitwin.twin.inputs import sleep_by_day  # noqa: F401  (re-exported for scripts)


def _log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def split_patients(ids, seed, frac=(0.70, 0.15, 0.15), groups=None):
    """Patient-level train/val/test split (no patient appears in two sets)."""
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(set(groups if groups is not None else ids)))
    rng.shuffle(keys)
    n = len(keys)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    lab = {k: ("train" if i < a else "val" if i < b else "test") for i, k in enumerate(keys)}
    g = groups if groups is not None else ids
    return pd.DataFrame({"patient_id": ids, "split": [lab[x] for x in g]})


def twin_frame(obs: list[Observed], step: int, t0, verbose=True):
    T, arr, prior = stack_inputs(obs)
    Z = calibrate(T, arr, prior, verbose=False)
    if verbose:
        _log("twin calibrated (daily personal parameters)", t0)
    states = run_observer(T, arr, prior, Z)
    fc = forecast_grid(T, arr, prior, Z, states, obs, step)
    fc["patient_id"] = np.array([o.patient_id for o in obs])[fc.b.values]
    if verbose:
        _log(f"twin forecasts: {len(fc):,} rows", t0)
    zdf = pd.DataFrame({
        "patient_id": np.repeat([o.patient_id for o in obs], Z.shape[0]),
        "day": np.tile(np.arange(Z.shape[0]), len(obs)),
        "si_rel": np.exp(Z[:, :, 0].T.reshape(-1)), "meal_gain": np.exp(Z[:, :, 1].T.reshape(-1)),
        "gb": (prior.Gb[:, None] * np.exp(Z[:, :, 2].T)).reshape(-1),
        "dawn_amp": (prior.dawn_amp[:, None] * np.exp(Z[:, :, 3].T)).reshape(-1),
        "si_abs": (prior.SI[:, None] * np.exp(Z[:, :, 0].T)).reshape(-1),
    })
    return fc.drop(columns=["b"]), zdf


def build_synthetic(n_patients=C.N_PATIENTS, n_days=C.N_DAYS, seed=C.SEED):
    t0 = time.time()
    ehr = generate_cohort(n_patients, seed)
    sim = simulate_cohort(ehr, n_days, seed)
    _log(f"simulated {n_patients} patients x {n_days} days", t0)
    out = C.SYN_DIR
    ehr.to_csv(out / "ehr_patients.csv", index=False)
    lab_history(ehr, seed, START_DATE).to_csv(out / "lab_history.csv", index=False)
    write_fhir(ehr, out / "fhir", START_DATE)
    sim["timeseries"].to_parquet(out / "timeseries.parquet", index=False)
    for k in ("meals", "doses", "daily", "true_params"):
        sim[k].to_csv(out / f"{k}.csv", index=False)
    split_patients(ehr.patient_id.values, seed).to_csv(out / "splits.csv", index=False)
    _log("saved EHR (CSV + FHIR), streams, events", t0)

    ts, meals, doses = sim["timeseries"], sim["meals"], sim["doses"]
    by = lambda df: dict(tuple(df.groupby("patient_id")))
    ts_g, m_g, d_g = by(ts), by(meals), by(doses)
    recs = ehr.to_dict("records")
    obs = [build_observed(r["patient_id"], ts_g[r["patient_id"]], m_g[r["patient_id"]],
                          d_g.get(r["patient_id"], doses.iloc[:0]), r) for r in recs]
    _log("built observed inputs", t0)
    fc, zdf = twin_frame(obs, C.STEP_MIN, t0)
    zdf.to_csv(C.PROC_DIR / "twin_params_synthetic.csv", index=False)

    feats = []
    wd = START_DATE.weekday()
    for r, o in zip(recs, obs):
        pid = r["patient_id"]
        g = ts_g[pid]
        dz = d_g.get(pid, doses.iloc[:0])
        missed = dz[(dz.drug == "metformin") & ~dz.taken].minute.values if len(dz) else []
        f = patient_features(g[["minute", "cgm", "steps", "heart_rate", "hrv_rmssd", "sleep_stage"]].reset_index(drop=True),
                             C.STEP_MIN, o.meals, o.insulin, missed, o.sleep_daily, wd)
        f.insert(0, "patient_id", pid)
        feats.append(f)
    df = pd.concat(feats, ignore_index=True)
    df = df.merge(fc, on=["patient_id", "minute"], how="left")
    df = df.merge(ehr_static(ehr), on="patient_id", how="left")
    df = add_twin_deltas(df)
    df["day"] = df.minute // 1440
    df = df[df.day >= C.WARMUP_DAYS].reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df.to_parquet(C.PROC_DIR / "features_synthetic.parquet", index=False)
    _log(f"features: {df.shape}", t0)
    return df


def build_shanghai(step=15, seed=C.SEED):
    from thalitwin.data.shanghai import load_all
    t0 = time.time()
    summ, recs = load_all(step=step)
    _log(f"loaded ShanghaiT2DM: {len(recs)} recordings / {summ.person.nunique()} people", t0)
    obs = []
    for r in summ.to_dict("records"):
        grid, meals, ins, _ = recs[r["patient_id"]]
        T = int(grid.minute.max() + step)
        T = int(np.ceil(T / 1440) * 1440)
        cgm = np.full(T, np.nan)
        cgm[grid.minute.values] = grid.cgm.values
        from thalitwin.twin.physiology import insulin_appearance, meal_appearance
        Ra = meal_appearance(T, meals.minute, meals.carbs, meals.k)
        Ri = insulin_appearance(T, ins.minute, ins.dose, ins.ka.values) if len(ins) else np.zeros(T)
        prior = params_from_clinical(r["fpg"], r["fasting_insulin"], r["weight_kg"], r["hba1c"],
                                     r["sulfonylurea"], None)
        obs.append(Observed(r["patient_id"], T, cgm, Ra, Ri, np.zeros(T), np.ones(T), np.ones(T),
                            meals[["minute", "carbs", "k", "slot"]], ins, prior, step))
    fc, zdf = twin_frame(obs, step, t0)
    zdf.to_csv(C.PROC_DIR / "twin_params_shanghai.csv", index=False)
    feats = []
    for r, o in zip(summ.to_dict("records"), obs):
        grid = recs[r["patient_id"]][0]
        f = patient_features(grid, step, o.meals, o.insulin)
        f.insert(0, "patient_id", r["patient_id"])
        feats.append(f)
    df = pd.concat(feats, ignore_index=True).merge(fc, on=["patient_id", "minute"], how="left")
    st = summ.copy()
    st["sex_m"] = (st.sex == "M").astype(float)
    keep = ["patient_id", "person", "age", "sex_m", "bmi", "hba1c", "fpg", "fasting_insulin", "homa_ir",
            "diabetes_duration_y", "egfr", "triglycerides", "hdl", "ldl", "hypertension", "metformin",
            "sulfonylurea", "dpp4", "sglt2", "insulin", "acarbose"]
    st = st[keep]
    for c in keep[2:]:
        st[c] = st[c].astype(float)
    df = df.merge(st, on="patient_id", how="left")
    df = add_twin_deltas(df)
    df["day"] = df.minute // 1440
    df = df[df.day >= 1].reset_index(drop=True)       # twin needs one day of history
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df.to_parquet(C.PROC_DIR / "features_shanghai.parquet", index=False)
    split_patients(summ.patient_id.values, seed, groups=summ.person.values).to_csv(
        C.PROC_DIR / "splits_shanghai.csv", index=False)
    _log(f"features: {df.shape}", t0)
    return df
