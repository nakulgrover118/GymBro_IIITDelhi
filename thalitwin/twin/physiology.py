"""Glucose-insulin physiology used by both the cohort simulator and the twin.

The core is an extended Bergman minimal model (Bergman et al., 1979) with
additions commonly used in type-2 diabetes simulators:

    dG/dt = -(SG + X + ex_up*E(t)) * G + SG*Gb*egp(t) + gain*Ra(t)/V
    dX/dt = -p2 * X + p2 * SI * si(t) * (I - Ib)
    dI/dt = -n * (I - Ib) + gamma * max(G - Gb, 0) + Ri(t)/VI

G  plasma glucose (mg/dL)          X  remote insulin action (1/min)
I  plasma insulin (uU/mL)          Ra meal glucose appearance (mg/min)
Ri exogenous insulin appearance    E  exercise drive = activity (0..1) through a 30-min lag
si(t)  insulin-sensitivity multiplier (sleep debt, recent exercise)
egp(t) hepatic glucose output multiplier (dawn phenomenon, stress, missed
       metformin, slow drift)

Everything is vectorised over a batch dimension B so that many patients,
parameter candidates (calibration) or start times (forecasting) can be
simulated in a single time loop.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

# Fixed structural constants (literature values for adults with T2D)
SG_DEFAULT = 0.010      # glucose effectiveness, 1/min
P2_DEFAULT = 0.025      # insulin action rate, 1/min
N_DEFAULT = 0.14        # insulin clearance, 1/min
F_BIO = 0.9             # carbohydrate bioavailability
KA_RAPID = 0.018        # s.c. regular/rapid insulin absorption, 1/min
KA_PREMIX = 0.012       # premixed insulin absorption, 1/min
TAU_EXERCISE = 30.0     # muscle glucose uptake persists ~30 min after activity stops, min


@dataclass
class PhysioParams:
    """Per-patient parameters. Every field is an array of shape (B,)."""
    Gb: np.ndarray          # basal (fasting) glucose, mg/dL
    Ib: np.ndarray          # basal insulin, uU/mL
    SI: np.ndarray          # insulin sensitivity, mL/uU/min
    gamma: np.ndarray       # beta-cell responsiveness, uU/mL/min per mg/dL
    V: np.ndarray           # glucose distribution volume, dL
    VI: np.ndarray          # insulin distribution volume, mL
    SG: np.ndarray
    p2: np.ndarray
    n: np.ndarray
    ex_up: np.ndarray       # extra glucose uptake at full activity, 1/min
    meal_gain: np.ndarray   # scales meal carbs (bioavailability / carb-count error)
    dawn_amp: np.ndarray    # dawn phenomenon amplitude (fraction of EGP)

    @property
    def batch(self) -> int:
        return int(np.asarray(self.Gb).shape[0])

    def take(self, idx) -> "PhysioParams":
        return PhysioParams(**{f.name: np.asarray(getattr(self, f.name))[idx] for f in fields(self)})

    def repeat(self, k: int) -> "PhysioParams":
        """Repeat each patient k times (patient-major order)."""
        return PhysioParams(**{f.name: np.repeat(np.asarray(getattr(self, f.name)), k) for f in fields(self)})

    def with_(self, **kw) -> "PhysioParams":
        return replace(self, **kw)

    def to_dict(self, i: int = 0) -> dict:
        return {f.name: float(np.asarray(getattr(self, f.name))[i]) for f in fields(self)}

    @staticmethod
    def stack(items: list["PhysioParams"]) -> "PhysioParams":
        return PhysioParams(**{f.name: np.concatenate([np.atleast_1d(getattr(p, f.name)) for p in items])
                               for f in fields(PhysioParams)})


def params_from_clinical(fpg, fasting_insulin, weight_kg, hba1c_pct=None,
                         on_sulfonylurea=None, tcf7l2_alleles=None) -> PhysioParams:
    """Initial twin parameters from EHR values (arrays or scalars).

    SI is scaled from HOMA-IR and gamma from HOMA-B (Matthews et al., 1985),
    so a patient's lab report directly personalises the physiology.
    """
    fpg = np.atleast_1d(np.asarray(fpg, float))
    fi = np.atleast_1d(np.asarray(fasting_insulin, float))
    w = np.atleast_1d(np.asarray(weight_kg, float))
    B = fpg.shape[0]
    homa_ir = np.clip(fpg * fi / 405.0, 0.5, 20.0)
    homa_b = np.clip(360.0 * fi / np.maximum(fpg - 63.0, 5.0), 5.0, 250.0)
    SI = 1.1e-3 / homa_ir
    gamma = 0.0013 * homa_b
    if on_sulfonylurea is not None:
        gamma = gamma * np.where(np.atleast_1d(on_sulfonylurea), 1.25, 1.0)
    if tcf7l2_alleles is not None:
        gamma = gamma * (1.0 - 0.10 * np.atleast_1d(tcf7l2_alleles))
    dawn = np.full(B, 0.12)
    if hba1c_pct is not None:
        dawn = 0.06 + 0.02 * np.clip(np.atleast_1d(hba1c_pct) - 6.0, 0, 6)
    return PhysioParams(
        Gb=fpg.copy(), Ib=fi.copy(), SI=SI, gamma=gamma,
        V=1.7 * w, VI=120.0 * w,
        SG=np.full(B, SG_DEFAULT), p2=np.full(B, P2_DEFAULT), n=np.full(B, N_DEFAULT),
        ex_up=np.full(B, 0.05), meal_gain=np.ones(B), dawn_amp=dawn,
    )


# ---------------------------------------------------------------------------
# Input signals (all on a 1-minute grid)
# ---------------------------------------------------------------------------

def gamma2_kernel(k: float, length: int) -> np.ndarray:
    """Impulse response of two equal first-order compartments (area = 1)."""
    t = np.arange(length, dtype=float)
    return k * k * t * np.exp(-k * t)


def meal_appearance(T: int, minutes, carbs_g, kabs, length: int = 480) -> np.ndarray:
    """Glucose appearance rate Ra (mg/min) from a list of meals."""
    Ra = np.zeros(T + length)
    for m, c, k in zip(minutes, carbs_g, kabs):
        m = int(m)
        if m < 0 or m >= T or c <= 0:
            continue
        Ra[m:m + length] += F_BIO * c * 1000.0 * gamma2_kernel(k, length)
    return Ra[:T]


def insulin_appearance(T: int, minutes, doses_iu, ka, length: int = 600) -> np.ndarray:
    """Exogenous insulin appearance in plasma (uU/min) from s.c. boluses."""
    Ri = np.zeros(T + length)
    for m, d, k in zip(minutes, doses_iu, np.broadcast_to(ka, len(doses_iu))):
        m = int(m)
        if m < 0 or m >= T or d <= 0:
            continue
        Ri[m:m + length] += d * 1e6 * gamma2_kernel(float(k), length)
    return Ri[:T]


def activity_filter(activity: np.ndarray, tau: float = TAU_EXERCISE, a0=0.0) -> np.ndarray:
    """Effective exercise drive: activity passed through a first-order lag, so
    insulin-independent muscle uptake builds during a walk and fades after it
    (post-meal walks cut the glucose peak by ~20-30%, e.g. Reynolds 2016)."""
    act = np.asarray(activity, float)
    out = np.empty_like(act)
    e = np.zeros(act.shape[:-1]) + a0
    for t in range(act.shape[-1]):
        e = e + (act[..., t] - e) / tau
        out[..., t] = e
    return out


def exercise_si_boost(activity: np.ndarray, tau_min: float = 360.0, gain: float = 0.25) -> np.ndarray:
    """Insulin sensitivity rises for hours after activity (activity memory)."""
    mem = np.zeros_like(activity, dtype=float)
    a = 1.0 / tau_min
    acc = np.zeros(activity.shape[:-1])
    for t in range(activity.shape[-1]):
        acc = acc + activity[..., t] - a * acc
        mem[..., t] = acc
    return 1.0 + gain * np.clip(mem / 30.0, 0.0, 1.0)


def dawn_profile(T: int, start_minute_of_day: int = 0) -> np.ndarray:
    """Hepatic glucose output bump between ~03:00 and ~09:00 (peaks 06:00)."""
    mod = (np.arange(T) + start_minute_of_day) % 1440
    return np.exp(-0.5 * ((mod - 360.0) / 75.0) ** 2)


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

def _row(x, B, T):
    x = np.asarray(x, float)
    if x.ndim == 0:
        return np.full((B, T), float(x))
    if x.ndim == 1:
        return np.broadcast_to(x[None, :], (B, T))
    return x


def simulate(p: PhysioParams, T: int, Ra=0.0, Ri=0.0, activity=0.0, si_mult=1.0,
             egp_mult=1.0, dawn=0.0, G0=None, X0=None, I0=None,
             obs=None, K: float = 0.0, return_states: bool = False, rep: int = 1,
             D0=None, Kd: float = 0.0, d_tau: float = 120.0):
    """Euler-integrate the model at 1-minute steps for T minutes.

    Ra, Ri, activity, si_mult, egp_mult, dawn: scalars, (T,) or (B/rep, T).
    obs: optional (B/rep, T) glucose observations (NaN where missing). When
    given, the state is nudged towards them with gain K - a simple Luenberger
    observer that keeps the twin synchronised with the real patient.
    rep: each input row is shared by `rep` consecutive batch members (used to
    evaluate many parameter candidates per patient without copying inputs).
    D0, Kd, d_tau: disturbance observer. D (mg/dL/min) is an extra glucose
    input the model cannot explain (unlogged snacks, stress hormones,
    carb-count errors, hepatic output drift). Each observation updates it by
    Kd * innovation; it relaxes to zero with time constant d_tau. In a
    forecast (Kd = 0) the last estimate simply fades out.
    """
    B = p.batch
    Bi = B // rep
    Ra, Ri, A = _row(Ra, Bi, T), _row(Ri, Bi, T), _row(activity, Bi, T)
    sim, egp, dwn = _row(si_mult, Bi, T), _row(egp_mult, Bi, T), _row(dawn, Bi, T)
    col = (lambda a, t: a[:, t]) if rep == 1 else (lambda a, t: np.repeat(a[:, t], rep))
    G = np.array(p.Gb, float) if G0 is None else np.array(G0, float)
    X = np.zeros(B) if X0 is None else np.array(X0, float)
    I = np.array(p.Ib, float) if I0 is None else np.array(I0, float)
    D = np.zeros(B) if D0 is None else np.array(D0, float)
    d_keep = 1.0 - 1.0 / d_tau
    Gs = np.empty((B, T))
    if return_states:
        Xs, Is, Ds = np.empty((B, T)), np.empty((B, T)), np.empty((B, T))
    SGGb = p.SG * p.Gb
    rV = p.meal_gain / p.V
    rVI = 1.0 / p.VI
    for t in range(T):
        uptake = (p.SG + X + p.ex_up * col(A, t)) * G
        dG = -uptake + SGGb * (col(egp, t) + p.dawn_amp * col(dwn, t)) + rV * col(Ra, t)
        dX = -p.p2 * X + p.p2 * p.SI * col(sim, t) * (I - p.Ib)
        dI = -p.n * (I - p.Ib) + p.gamma * np.maximum(G - p.Gb, 0.0) + rVI * col(Ri, t)
        G = np.maximum(G + dG + D, 20.0)
        X = X + dX
        I = np.maximum(I + dI, 0.0)
        D = D * d_keep
        if obs is not None and K > 0:
            o = col(obs, t)
            m = ~np.isnan(o)
            if m.any():
                innov = np.where(m, o, G) - G
                G = G + K * innov
                if Kd > 0:
                    D = np.clip(D + Kd * innov, -3.0, 3.0)
        Gs[:, t] = G
        if return_states:
            Xs[:, t], Is[:, t], Ds[:, t] = X, I, D
    if return_states:
        return Gs, Xs, Is, Ds
    return Gs
