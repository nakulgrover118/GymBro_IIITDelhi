"""Virtual-cohort simulator: daily life -> physiology -> wearable + CGM streams.

For each synthetic patient we generate a realistic Indian daily routine
(sleep, meals from the regional menu, walks, work stress, festival sweets,
medicine adherence) and push it through the "true" physiology. The patient's
true parameters are hidden perturbations of what their EHR implies, so the
digital twin has to learn them from data rather than reading them off.

Outputs on a 5-minute grid: CGM (with sensor lag, noise and dropouts),
steps, heart rate, HRV (RMSSD) and sleep stage, plus an event log (meals as
the patient logged them, medicine doses) and a daily summary.

Effects deliberately present in the simulator but NOT in the twin
(stress-driven glucose release, slow metabolic drift, unlogged meals and
carb-counting errors) keep the evaluation honest: the twin can never be a
perfect copy, which is exactly the gap the ML layer has to close.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from thalitwin.data.meals import MEAL_BY_KEY, absorption_rate, meals_for
from thalitwin.twin.physiology import (KA_PREMIX, PhysioParams, activity_filter, dawn_profile, exercise_si_boost,
                                       insulin_appearance, meal_appearance, params_from_clinical, simulate)

START_DATE = date(2026, 9, 1)
SNACKS = ["sweet_chai", "chai_biscuits", "samosa", "banana", "mango", "sprouts_chaat", "roasted_chana"]
SWEETS = ["gulab_jamun", "jalebi", "mithai_box"]
STEPS_PER_MIN_AT_FULL = 150.0


def sleep_si_multiplier(sleep_hours, efficiency):
    """Insulin sensitivity the day after a night of sleep.

    One night of ~4-5 h sleep lowers insulin sensitivity by ~15-25%
    (Donga et al., 2010; Spiegel et al., 1999). Used by both the simulator
    and, as a literature prior, by the twin.
    """
    debt = np.clip(7.0 - np.asarray(sleep_hours, float), 0.0, 4.0)
    return np.clip(1.0 - 0.06 * debt - 0.25 * np.clip(0.88 - np.asarray(efficiency, float), 0, 0.4), 0.6, 1.05)


def _choose(rng, options):
    return options[int(rng.integers(len(options)))]


def _sleep_night(rng, persona, weekend, bad):
    """Returns (bed_start, wake) in minutes relative to the morning's day start."""
    bed = persona["persona_bedtime_min"] - 1440 + rng.normal(0, 25) + (rng.uniform(60, 150) if bad else 0)
    wake = persona["persona_wake_min"] + rng.normal(0, 18) + (45 if weekend else 0) - (rng.uniform(0, 40) if bad else 0)
    return int(bed), int(max(wake, bed + 180))


def _hypnogram(rng, n_epochs, bad):
    """Sleep stages for consecutive 5-min epochs: 0 wake, 1 light, 2 deep, 3 REM."""
    st = np.ones(n_epochs, dtype=np.int8)
    lat = int(rng.integers(2, 7 if not bad else 10))
    st[:lat] = 0
    cyc = 18  # 90-minute cycles
    for e in range(lat, n_epochs):
        k = (e - lat) // cyc
        pos = (e - lat) % cyc / cyc
        deep_share = max(0.35 - 0.09 * k, 0.03) * (0.55 if bad else 1.0)
        rem_share = min(0.10 + 0.07 * k, 0.35)
        if pos < deep_share + 0.15 and pos > 0.15:
            st[e] = 2
        elif pos > 1 - rem_share:
            st[e] = 3
    wake_p = 0.13 if bad else 0.04
    st[lat:][rng.random(n_epochs - lat) < wake_p] = 0
    return st


def build_schedule(rng: np.random.Generator, persona: dict, ehr: dict, n_days: int, start: date):
    """Minute-level ground-truth inputs and logs for one patient."""
    T = n_days * 1440
    activity = np.zeros(T)
    asleep = np.zeros(T, dtype=bool)
    stage = np.full(T // 5, -1, dtype=np.int8)
    stress = np.zeros(T)
    egp_extra = np.zeros(T)
    si_sleep = np.ones(T)
    meals, doses, daily = [], [], []
    veg = ehr["vegetarian"]
    region = ehr["region"]

    prev_si = 1.0
    for d in range(n_days):
        day0 = d * 1440
        weekday = (start + timedelta(days=d)).weekday()
        weekend = weekday >= 5
        bad = rng.random() < persona["persona_bad_sleep_prob"]
        is_stress = (not weekend) and rng.random() < persona["persona_stress_prob"]
        sweets_day = rng.random() < persona["persona_sweets_prob"] * 0.35

        # --- sleep (night ending this morning) ---
        bed, wake = _sleep_night(rng, persona, weekend, bad)
        e0, e1 = (day0 + bed) // 5, (day0 + wake) // 5
        hyp = _hypnogram(rng, e1 - e0, bad)
        for e, s in zip(range(e0, e1), hyp):
            if 0 <= e < len(stage):
                stage[e] = s
                if s > 0:
                    asleep[e * 5:(e + 1) * 5] = True
        sleep_h = float((hyp > 0).sum() * 5 / 60)
        eff = float((hyp > 0).mean())
        si_day = float(sleep_si_multiplier(sleep_h, eff))
        lo = max(day0 + wake, 0)
        si_sleep[lo:min(lo + 1440, T)] = si_day
        if d == 0:
            si_sleep[:lo] = prev_si

        # --- meals ---
        port = persona["persona_portion"] * (1.08 if weekend else 1.0)
        plan = [("breakfast", wake + rng.uniform(35, 100))]
        plan.append(("lunch", 13 * 60 + rng.normal(0, 35) + (30 if weekend else 0)))
        if rng.random() < persona["persona_snack_prob"]:
            plan.append(("snack", 17 * 60 + rng.normal(0, 40)))
        plan.append(("dinner", 20.75 * 60 + rng.normal(0, 40) + (30 if weekend else 0)))
        if sweets_day:
            plan.append(("sweet", plan[-1][1] + rng.uniform(20, 60)))
        for slot, t in plan:
            if slot == "snack":
                m = MEAL_BY_KEY[_choose(rng, SNACKS)]
            elif slot == "sweet":
                m = MEAL_BY_KEY[_choose(rng, SWEETS)]
            else:
                opts = [x for x in meals_for(slot, region) if x.veg or not veg]
                m = _choose(rng, opts)
            carbs = m.carbs_g * port * float(np.exp(rng.normal(0, 0.12)))
            logged = rng.random() < persona["persona_logging_adherence"]
            meals.append(dict(minute=int(day0 + t), slot="snack" if slot == "sweet" else slot, meal_key=m.key,
                              meal_name=m.name, carbs_true=round(carbs, 1),
                              carbs_logged=round(carbs * float(np.exp(rng.normal(0, 0.18))), 0) if logged else np.nan,
                              gi=m.gi, slow=m.slow, logged=logged))

        # --- activity ---
        awake_lo, awake_hi = day0 + wake, day0 + 1440 + persona["persona_bedtime_min"] - 1440
        lvl = persona["persona_activity_level"]
        for blk in range(max(awake_lo, 0) // 5, min(awake_hi, T) // 5):
            hour = (blk * 5 % 1440) / 60
            desk = ehr["occupation"] == "desk job" and not weekend and 10 <= hour < 18
            p_move = (0.12 if desk else 0.26) * (0.6 + lvl)
            if rng.random() < p_move:
                activity[blk * 5:(blk + 1) * 5] = rng.choice([0.04, 0.1, 0.2, 0.35], p=[0.45, 0.3, 0.17, 0.08])
        walk_scale = 0.6 if is_stress else 1.0
        if rng.random() < persona["persona_morning_walk_prob"] * walk_scale * (1.2 if weekend else 1.0):
            s = int(day0 + wake + rng.uniform(10, 60))
            activity[s:s + int(rng.uniform(20, 45))] = rng.uniform(0.55, 0.75)
        for mm in meals[-len(plan):]:
            if mm["slot"] in ("lunch", "dinner") and rng.random() < persona["persona_post_meal_walk_prob"] * walk_scale:
                s = int(mm["minute"] + rng.uniform(10, 25))
                activity[s:s + int(rng.uniform(10, 25))] = 0.5
        if ehr["occupation"] == "field work" and not weekend:
            for _ in range(3):
                s = int(day0 + rng.uniform(10, 17) * 60)
                activity[s:s + int(rng.uniform(20, 50))] = rng.uniform(0.3, 0.5)

        # --- stress (hepatic glucose output up, HRV down) ---
        if is_stress:
            stress[max(day0 + wake, 0):min(day0 + 1380, T)] = 1.0

        # --- medicines ---
        missed = 0
        b_t, d_t = plan[0][1], [t for s, t in plan if s == "dinner"][0]
        if ehr["metformin"]:
            for t in (b_t, d_t):
                ok = rng.random() < persona["persona_med_adherence"]
                doses.append(dict(minute=int(day0 + t), drug="metformin", dose=500, unit="mg", taken=ok))
                if not ok:
                    missed += 1
                    s = int(day0 + t)
                    egp_extra[s:s + 720] += 0.12
        if ehr["insulin"]:
            tdd = 0.4 * ehr["weight_kg"]
            if ehr["insulin_regimen"] == "premix_30_70":
                for t, frac in ((b_t - 15, 0.6), (d_t - 15, 0.4)):
                    ok = rng.random() < persona["persona_med_adherence"]
                    doses.append(dict(minute=int(day0 + t), drug="insulin_premix", dose=round(tdd * frac),
                                      unit="IU", taken=ok))
                    missed += (not ok)
            else:
                ok = rng.random() < persona["persona_med_adherence"]
                doses.append(dict(minute=int(day0 + 1320), drug="insulin_glargine", dose=round(tdd * 0.5),
                                  unit="IU", taken=ok))
                if not ok:
                    missed += 1
                    egp_extra[day0 + 1320:day0 + 1320 + 1440] += 0.15

        daily.append(dict(day=d, date=(start + timedelta(days=d)).isoformat(), weekend=weekend, sleep_hours=sleep_h,
                          sleep_efficiency=eff, deep_pct=float((hyp == 2).mean()), rem_pct=float((hyp == 3).mean()),
                          awakenings=int(((hyp[1:] == 0) & (hyp[:-1] > 0)).sum()), bad_sleep=bad,
                          stress_day=is_stress, sweets_day=sweets_day, missed_doses=missed, si_sleep_true=si_day))
        prev_si = si_day

    return dict(activity=np.clip(activity, 0, 1), asleep=asleep, stage=stage, stress=stress,
                egp_extra=egp_extra, si_sleep=si_sleep, meals=pd.DataFrame(meals), doses=pd.DataFrame(doses),
                daily=pd.DataFrame(daily))


def true_params(ehr: pd.DataFrame, rng: np.random.Generator) -> PhysioParams:
    """Hidden 'true' physiology: EHR-implied values with individual deviations."""
    p = params_from_clinical(ehr.fpg.values, ehr.fasting_insulin.values, ehr.weight_kg.values,
                             ehr.hba1c.values, ehr.sulfonylurea.values, ehr.tcf7l2_rs7903146_risk_alleles.values)
    n = len(ehr)
    ln = lambda s: np.exp(rng.normal(0, s, n))
    return p.with_(SI=p.SI * ln(0.28), gamma=p.gamma * ln(0.22), meal_gain=ln(0.10),
                   dawn_amp=p.dawn_amp * ln(0.3), Gb=p.Gb * np.exp(rng.normal(0, 0.06, n)),
                   ex_up=p.ex_up * ln(0.25))


def cgm_sensor(G: np.ndarray, rng: np.random.Generator, step: int = 5) -> np.ndarray:
    """Interstitial lag (tau ~8 min), calibration bias, AR(1) noise, dropouts."""
    B, T = G.shape
    isf = np.empty_like(G)
    x = G[:, 0].copy()
    for t in range(T):
        x = x + (G[:, t] - x) / 8.0
        isf[:, t] = x
    s = isf[:, ::step]
    bias = np.exp(rng.normal(0, 0.04, (B, 1)))
    noise = np.zeros_like(s)
    e = np.zeros(B)
    for k in range(s.shape[1]):
        e = 0.75 * e + rng.normal(0, 3.5, B)
        noise[:, k] = e
    out = np.clip(s * bias + noise + rng.normal(0, 1.5, s.shape), 40, 400)
    for b in range(B):                     # 0-3 signal-loss gaps per sensor
        for _ in range(rng.integers(0, 4)):
            a = rng.integers(0, s.shape[1] - 20)
            out[b, a:a + rng.integers(6, 18)] = np.nan
    return np.round(out, 0)


def simulate_cohort(ehr: pd.DataFrame, n_days: int, seed: int, start: date = START_DATE):
    rng = np.random.default_rng(seed + 7)
    B, T = len(ehr), n_days * 1440
    params = true_params(ehr, rng)
    recs = ehr.to_dict("records")

    Ra, Ri = np.zeros((B, T)), np.zeros((B, T))
    A, si, egp = np.zeros((B, T)), np.ones((B, T)), np.ones((B, T))
    stress = np.zeros((B, T))
    asleep = np.zeros((B, T), dtype=bool)
    stages = np.zeros((B, T // 5), dtype=np.int8)
    all_meals, all_doses, all_daily = [], [], []
    for b, r in enumerate(recs):
        sch = build_schedule(rng, r, r, n_days, start)
        m = sch["meals"]
        Ra[b] = meal_appearance(T, m.minute, m.carbs_true, [absorption_rate(g, s) for g, s in zip(m.gi, m.slow)])
        dz = sch["doses"]
        if len(dz):
            ins = dz[(dz.drug == "insulin_premix") & dz.taken]
            Ri[b] = insulin_appearance(T, ins.minute, ins.dose, KA_PREMIX)
        A[b], asleep[b], stages[b], stress[b] = sch["activity"], sch["asleep"], sch["stage"], sch["stress"]
        # slow unmodelled metabolic drift (hourly AR(1))
        drift = np.zeros(T // 60 + 1)
        for h in range(1, len(drift)):
            drift[h] = 0.92 * drift[h - 1] + rng.normal(0, 0.012)
        egp[b] = 1.0 + 0.18 * sch["stress"] + sch["egp_extra"] + np.repeat(drift, 60)[:T]
        si[b] = sch["si_sleep"]
        for name, df in (("meals", m), ("doses", dz), ("daily", sch["daily"])):
            df = df.copy()
            df.insert(0, "patient_id", r["patient_id"])
            {"meals": all_meals, "doses": all_doses, "daily": all_daily}[name].append(df)

    si = si * exercise_si_boost(A)
    G = simulate(params, T, Ra=Ra, Ri=Ri, activity=activity_filter(A), si_mult=si, egp_mult=egp, dawn=dawn_profile(T))
    cgm = cgm_sensor(G, rng)

    # ---- wearables on the 5-minute grid ----
    n5 = T // 5
    A5 = A.reshape(B, n5, 5)
    steps = np.round((A5 * STEPS_PER_MIN_AT_FULL * rng.uniform(0.85, 1.15, (B, n5, 5))).sum(-1))
    act5 = A5.mean(-1)
    st5 = stress.reshape(B, n5, 5).mean(-1)
    slp5 = asleep.reshape(B, n5, 5).mean(-1) > 0.5
    rest = ehr.persona_resting_hr.values[:, None]
    tod = (np.arange(n5) * 5 % 1440) / 1440
    circ = 2.5 * np.sin(2 * np.pi * (tod - 0.35))[None, :]
    ra5 = Ra.reshape(B, n5, 5).mean(-1)
    hr = rest + circ + 55 * act5 + 6 * st5 - 7 * slp5 + 4 * (stages == 3) + 0.0008 * ra5 + rng.normal(0, 2.2, (B, n5))
    rm = ehr.persona_rmssd.values[:, None]
    stage_f = np.select([stages == 2, stages == 3, stages == 0], [1.4, 0.95, 0.8], 1.15)
    hrv = rm * np.where(slp5 | (stages >= 0), stage_f, 1.0) * (1 - 0.3 * st5) * (1 - 0.45 * np.clip(act5 * 2, 0, 1))
    hrv = hrv * np.exp(rng.normal(0, 0.15, (B, n5)))

    start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time()))
    minute = np.arange(n5) * 5
    ts = pd.DataFrame({
        "patient_id": np.repeat(ehr.patient_id.values, n5),
        "minute": np.tile(minute, B),
        "timestamp": np.tile(start_ts + pd.to_timedelta(minute, unit="min"), B),
        "cgm": cgm.reshape(-1).astype("float32"),
        "glucose_true": G[:, ::5].reshape(-1).round(1).astype("float32"),
        "steps": steps.reshape(-1).astype("int16"),
        "heart_rate": hr.reshape(-1).round(0).astype("float32"),
        "hrv_rmssd": hrv.reshape(-1).round(1).astype("float32"),
        "sleep_stage": stages.reshape(-1),
    })
    meals = pd.concat(all_meals, ignore_index=True)
    doses = pd.concat(all_doses, ignore_index=True) if all_doses else pd.DataFrame()
    daily = pd.concat(all_daily, ignore_index=True)
    tp = pd.DataFrame({f: getattr(params, f) for f in ("Gb", "Ib", "SI", "gamma", "meal_gain", "dawn_amp", "ex_up")})
    tp.insert(0, "patient_id", ehr.patient_id.values)
    return dict(timeseries=ts, meals=meals, doses=doses, daily=daily, true_params=tp)
