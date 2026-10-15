"""The digital-twin engine: calibrate -> synchronise -> forecast -> what-if.

1. Calibrate (every midnight): re-estimate the patient's personal parameters
   (insulin sensitivity, meal response, basal glucose, dawn effect) from the
   last day of CGM using a cross-entropy search, regularised towards the
   EHR-derived prior. Vectorised over patients x candidates.
2. Synchronise (continuously): run the model alongside the patient and nudge
   its glucose state towards every new CGM reading (observer).
3. Forecast: from the synchronised state at any time t, simulate the next
   2 hours using what is already eaten/dosed plus the patient's *learned
   routine* (expected meals from their own history).
4. What-if: the same simulation with doctor-chosen changes (meal, walk,
   sleep, missed metformin, insulin dose).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin.twin.physiology import F_BIO, TAU_EXERCISE, PhysioParams, activity_filter, dawn_profile, simulate

CAL_KEYS = ("SI", "meal_gain", "Gb", "dawn_amp")
PRIOR_SD = np.array([0.35, 0.20, 0.08, 0.50])     # log-scale prior spread around the EHR prior
MEAS_SD = 12.0                                    # CGM + model error, mg/dL
SLOTS = ("breakfast", "lunch", "snack", "dinner")
DEFAULT_HABITS = {"breakfast": (0.9, 480, 45, 1 / 60), "lunch": (0.9, 795, 65, 1 / 60),
                  "snack": (0.5, 1020, 20, 1 / 50), "dinner": (0.9, 1260, 65, 1 / 60)}


def apply_z(prior: PhysioParams, z: np.ndarray) -> PhysioParams:
    """Personalise a prior with log-multipliers z[:, (SI, meal_gain, Gb, dawn)]."""
    e = np.exp(z)
    return prior.with_(SI=prior.SI * e[:, 0], meal_gain=prior.meal_gain * e[:, 1],
                       Gb=prior.Gb * e[:, 2], dawn_amp=prior.dawn_amp * e[:, 3])


def _nearest(cgm: np.ndarray, m: int, tol: int = 15) -> np.ndarray:
    lo, hi = max(m - tol, 0), min(m + tol + 1, cgm.shape[1])
    seg = cgm[:, lo:hi]
    out = np.full(cgm.shape[0], np.nan)
    for j in np.argsort(np.abs(np.arange(lo, hi) - m)):
        v = seg[:, j]
        fill = np.isnan(out) & ~np.isnan(v)
        out[fill] = v[fill]
    return out


def _cem(prior, z0, win, rng, n_cand=24, iters=5, reset_every=360):
    b, W = win["cgm"].shape
    cgm = win["cgm"]
    obs_reset = np.full((b, W), np.nan)
    for r in range(0, W, reset_every):
        obs_reset[:, r] = _nearest(cgm, r)
    G0 = np.where(np.isnan(obs_reset[:, 0]), prior.Gb, obs_reset[:, 0])
    valid = ~np.isnan(cgm)
    n_valid = valid.sum(1)
    has_data = n_valid >= 12
    step = max(int(np.median(np.diff(np.where(valid[np.argmax(n_valid)])[0]))) if n_valid.max() > 1 else 5, 1)
    n_eff = n_valid * step / 60.0                       # ~one independent error per hour
    mu, sd = z0.copy(), np.full_like(z0, 0.25)
    n_elite = max(n_cand // 4, 2)
    P_rep = prior.repeat(n_cand)
    for _ in range(iters):
        z = mu[:, None, :] + sd[:, None, :] * rng.standard_normal((b, n_cand, 4))
        z[:, 0, :] = mu
        z = np.clip(z, -1.6, 1.6)
        G = simulate(apply_z(P_rep, z.reshape(-1, 4)), W, Ra=win["Ra"], Ri=win["Ri"], activity=win["A"],
                     si_mult=win["si"], egp_mult=win["egp"], dawn=win["dawn"], obs=obs_reset, K=1.0,
                     G0=np.repeat(G0, n_cand), rep=n_cand).reshape(b, n_cand, W)
        se = np.where(valid[:, None, :], (G - np.nan_to_num(cgm)[:, None, :]) ** 2, 0.0).sum(-1)
        mse = se / np.maximum(n_valid, 1)[:, None]
        loss = 0.5 * mse / MEAS_SD ** 2 * n_eff[:, None] + 0.5 * ((z / PRIOR_SD) ** 2).sum(-1)
        elite = np.argsort(loss, axis=1)[:, :n_elite]
        ez = np.take_along_axis(z, elite[:, :, None], axis=1)
        mu = np.where(has_data[:, None], ez.mean(1), z0)
        sd = ez.std(1) + 0.02
    return mu


def calibrate(T: int, arr: dict, prior: PhysioParams, seed: int = 0, chunk: int = 50,
              n_cand: int = 24, iters: int = 5, verbose: bool = False) -> np.ndarray:
    """Daily parameter estimates. Z[d] is what the twin believes on day d,
    learned only from data before day d (no look-ahead)."""
    rng = np.random.default_rng(seed)
    B, n_days = prior.batch, T // 1440
    Z = np.zeros((n_days, B, 4))
    for d in range(1, n_days):
        lo, hi = (d - 1) * 1440, d * 1440
        dawn = dawn_profile(hi - lo)
        for c0 in range(0, B, chunk):
            idx = np.arange(c0, min(B, c0 + chunk))
            win = {k: arr[k][idx, lo:hi] for k in ("cgm", "Ra", "Ri", "A", "si", "egp")}
            win["dawn"] = dawn
            mu = _cem(prior.take(idx), Z[d - 1, idx], win, rng, n_cand, iters)
            # exponential smoothing: the twin's beliefs evolve gradually day to day
            Z[d, idx] = mu if d == 1 else 0.5 * Z[d - 1, idx] + 0.5 * mu
        if verbose:
            print(f"  calibrated day {d}/{n_days - 1}", flush=True)
    return Z


OBS_K, OBS_KD, D_TAU, D_TAU_FORECAST = 0.5, 0.03, 240.0, 120.0


def run_observer(T: int, arr: dict, prior: PhysioParams, Z: np.ndarray, K: float = OBS_K, Kd: float = OBS_KD):
    """Synchronised twin states (G, X, I, D) at every minute."""
    B, n_days = prior.batch, T // 1440
    G, X, I, D = np.empty((B, T)), np.empty((B, T)), np.empty((B, T)), np.empty((B, T))
    first = np.array([row[~np.isnan(row)][0] if (~np.isnan(row)).any() else np.nan for row in arr["cgm"]])
    g0 = np.where(np.isnan(first), prior.Gb, first)
    x0, i0, d0 = np.zeros(B), prior.Ib.copy(), np.zeros(B)
    dawn = dawn_profile(1440)
    for d in range(n_days):
        sl = slice(d * 1440, (d + 1) * 1440)
        g, x, i, dd = simulate(apply_z(prior, Z[d]), 1440, Ra=arr["Ra"][:, sl], Ri=arr["Ri"][:, sl],
                               activity=arr["A"][:, sl], si_mult=arr["si"][:, sl], egp_mult=arr["egp"][:, sl],
                               dawn=dawn, obs=arr["cgm"][:, sl], K=K, G0=g0, X0=x0, I0=i0, return_states=True,
                               D0=d0, Kd=Kd, d_tau=D_TAU)
        G[:, sl], X[:, sl], I[:, sl], D[:, sl] = g, x, i, dd
        g0, x0, i0, d0 = g[:, -1], x[:, -1], i[:, -1], dd[:, -1]
    return G, X, I, D


# ---------------------------------------------------------------------------
# Routine learning (expected meals)
# ---------------------------------------------------------------------------

def slot_of(minute_of_day) -> np.ndarray:
    m = np.asarray(minute_of_day) / 60.0
    return np.select([m < 10.5, m < 15.5, m < 18.5], ["breakfast", "lunch", "snack"], "dinner")


def meal_habits(meals: pd.DataFrame, day: int) -> dict:
    """Patient's usual meal times/sizes learned from logged meals before `day`."""
    past = meals[meals.minute < day * 1440]
    n_past = day
    if n_past < 1 or past.empty:
        return dict(DEFAULT_HABITS)
    past = past.assign(day=past.minute // 1440, mod=past.minute % 1440)
    out = {}
    for s in SLOTS:
        g = past[past.slot == s]
        if g.empty:
            out[s] = (0.0, DEFAULT_HABITS[s][1], DEFAULT_HABITS[s][2], DEFAULT_HABITS[s][3])
            continue
        first = g.sort_values("minute").groupby("day").first()
        out[s] = (min(len(first) / n_past, 1.0), float(first["mod"].median()),
                  float(first["carbs"].median()), float(first["k"].median()))
    return out


def _kernel(c, k, dt):
    dt = np.maximum(dt, 0.0)
    return F_BIO * c * 1000.0 * k * k * dt * np.exp(-k * dt)


def future_inputs(starts: np.ndarray, H: int, meals: pd.DataFrame, insulin: pd.DataFrame,
                  habits: dict | None, day_start: int):
    """Ra/Ri over [s, s+H) for every start s, using only information known at s
    (+ expected meals from the learned routine when `habits` is given)."""
    tau = np.arange(H)[None, :]
    s = starts[:, None]
    Ra = np.zeros((len(starts), H))
    exp_carbs = np.zeros(len(starts))
    rel = meals[(meals.minute > starts.min() - 480) & (meals.minute <= starts.max())]
    for m in rel.itertuples():
        known = (m.minute <= s)
        Ra += np.where(known, _kernel(m.carbs, m.k, s + tau - m.minute), 0.0)
    Ri = np.zeros((len(starts), H))
    reli = insulin[(insulin.minute > starts.min() - 600) & (insulin.minute <= starts.max())]
    for r in reli.itertuples():
        known = (r.minute <= s)
        Ri += np.where(known, r.dose * 1e6 * r.ka ** 2 * np.maximum(s + tau - r.minute, 0)
                       * np.exp(-r.ka * np.maximum(s + tau - r.minute, 0)), 0.0)
    if habits:
        today = meals[(meals.minute >= day_start) & (meals.minute < day_start + 1440)]
        for slot, (p, mod, carbs, k) in habits.items():
            if p < 0.4 or carbs <= 0:
                continue
            e = day_start + mod
            eaten_at = today[today.slot == slot].minute.min() if (today.slot == slot).any() else np.inf
            st = starts.astype(float)
            not_eaten = st < eaten_at
            t_e = np.where(e > st, e, np.where(st - e < 120, st + 15, np.inf))
            off = t_e - st
            use = not_eaten & (off < H)
            Ra += np.where(use[:, None], _kernel(p * carbs, k, tau - off[:, None]), 0.0)
            exp_carbs += np.where(use, p * carbs, 0.0)
    return Ra, Ri, exp_carbs


def forecast_grid(T: int, arr: dict, prior: PhysioParams, Z: np.ndarray, states, obs_list, step: int,
                  H: int = 120, days=None, chunk: int = 40):
    """Twin forecasts from every sample time. Returns a long DataFrame."""
    G, X, I, D = states
    B, n_days = prior.batch, T // 1440
    days = range(n_days) if days is None else days
    tau = np.arange(H)
    out = []
    for d in days:
        starts = d * 1440 + np.arange(0, 1440, step)
        S = len(starts)
        dawn = np.exp(-0.5 * ((((starts[:, None] + tau[None, :]) % 1440) - 360.0) / 75.0) ** 2)
        P_day = apply_z(prior, Z[d])
        for c0 in range(0, B, chunk):
            idx = np.arange(c0, min(B, c0 + chunk))
            nb = len(idx)
            Ra_e, Ra_k, Ri_all, expc = [], [], [], []
            for b in idx:
                o = obs_list[b]
                hab = meal_habits(o.meals, d)
                ra_e, ri, ec = future_inputs(starts, H, o.meals, o.insulin, hab, d * 1440)
                ra_k, _, _ = future_inputs(starts, H, o.meals, o.insulin, None, d * 1440)
                Ra_e.append(ra_e), Ra_k.append(ra_k), Ri_all.append(ri), expc.append(ec)
            Ri_all = np.concatenate(Ri_all)
            P = P_day.take(idx).repeat(S)
            # start from the latest CGM reading when there is one (hard re-sync)
            g_obs = arr["cgm"][idx][:, starts].reshape(-1)
            g0 = np.where(np.isnan(g_obs), G[idx][:, starts].reshape(-1), g_obs)
            x0 = X[idx][:, starts].reshape(-1)
            i0 = I[idx][:, starts].reshape(-1)
            d0 = D[idx][:, starts].reshape(-1)
            si = np.broadcast_to(arr["si"][idx][:, starts].reshape(-1, 1), (nb * S, H))
            egp = np.broadcast_to(arr["egp"][idx][:, starts].reshape(-1, 1), (nb * S, H))
            dw = np.tile(dawn, (nb, 1))
            a_now = arr["A"][idx][:, starts].reshape(-1, 1)
            act = 0.03 + (a_now - 0.03) * np.exp(-tau[None, :] / TAU_EXERCISE)   # exercise drive fades
            res = {}
            for tag, Ra in (("exp", np.concatenate(Ra_e)), ("known", np.concatenate(Ra_k))):
                g = simulate(P, H, Ra=Ra, Ri=Ri_all, activity=act, si_mult=si, egp_mult=egp, dawn=dw,
                             G0=g0, X0=x0, I0=i0, D0=d0, d_tau=D_TAU_FORECAST)
                res[tag] = g
            ge, gk = res["exp"], res["known"]
            si_rel = np.exp(Z[d][idx, 0])
            out.append(pd.DataFrame({
                "b": np.repeat(idx, S), "minute": np.tile(starts, nb),
                "twin_now": G[idx][:, starts].reshape(-1), "twin_X": x0, "twin_I_excess": i0 - np.repeat(P_day.Ib[idx], S),
                "twin_g30": ge[:, 29], "twin_g60": ge[:, 59], "twin_g120": ge[:, 119],
                "twin_max120": ge.max(1), "twin_known_g60": gk[:, 59], "twin_known_max120": gk.max(1),
                "twin_exp_carbs": np.concatenate(expc), "twin_disturbance": d0,
                "twin_si_rel": np.repeat(si_rel, S), "twin_meal_gain": np.repeat(np.exp(Z[d][idx, 1]), S),
                "twin_gb": np.repeat(P_day.Gb[idx], S),
                "twin_si_today": si[:, 0] * np.repeat(si_rel, S),
            }))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# What-if simulation (single patient, many scenarios)
# ---------------------------------------------------------------------------

def what_if(params: PhysioParams, state: tuple, start_minute: int, H: int, base: dict, scenarios: list[dict],
            d_tau: float = D_TAU_FORECAST, a0: float = 0.0):
    """Simulate H minutes from `state` for each scenario.

    base: dict with 1-D arrays of length H: Ra, Ri, A (raw activity), si, egp.
    a0: current exercise drive (filtered activity) at the start.
    scenario keys (all optional):
      meals: list of (offset_min, carbs_g, k)    extra meals
      walks: list of (offset_min, duration_min, intensity)
      si_scale: float (e.g. sleep change)        egp_add: float (e.g. missed metformin)
      insulin: list of (offset_min, dose_iu, ka) carbs_scale: float (smaller portions)
    """
    n = len(scenarios)
    tau = np.arange(H)
    Ra = np.tile(base["Ra"], (n, 1)).astype(float)
    Ri = np.tile(base["Ri"], (n, 1)).astype(float)
    A = np.tile(base["A"], (n, 1)).astype(float)
    si = np.tile(base["si"], (n, 1)).astype(float)
    egp = np.tile(base["egp"], (n, 1)).astype(float)
    for j, sc in enumerate(scenarios):
        Ra[j] *= sc.get("carbs_scale", 1.0)
        for off, c, k in sc.get("meals", []):
            Ra[j] += _kernel(c, k, tau - off)
        for off, dur, inten in sc.get("walks", []):
            A[j, int(off):int(off + dur)] = np.maximum(A[j, int(off):int(off + dur)], inten)
            boost = 1.0 + 0.25 * min(dur * inten / 30.0, 1.0) * np.exp(-np.maximum(tau - off - dur, 0) / 360.0)
            si[j] *= np.where(tau >= off, boost, 1.0)
        for off, dose, ka in sc.get("insulin", []):
            dt = np.maximum(tau - off, 0)
            Ri[j] += dose * 1e6 * ka ** 2 * dt * np.exp(-ka * dt)
        si[j] *= sc.get("si_scale", 1.0)
        egp[j] += sc.get("egp_add", 0.0)
    dawn = np.exp(-0.5 * ((((start_minute + tau) % 1440) - 360.0) / 75.0) ** 2)
    A = activity_filter(A, a0=a0)
    P = params.repeat(n)
    g0, x0, i0, d0 = (np.full(n, float(v)) for v in (tuple(state) + (0.0,))[:4])
    return simulate(P, H, Ra=Ra, Ri=Ri, activity=A, si_mult=si, egp_mult=egp, dawn=dawn, G0=g0, X0=x0, I0=i0,
                    D0=d0, d_tau=d_tau)
