"""Runtime service shared by the API and the doctor dashboard.

Loads the deployed models and the virtual patients, and answers:
  * what is this patient's state right now (forecast, spike risk, reasons)?
  * what does the ward look like (patients ranked by risk)?
  * what would happen if ... (twin what-if simulation)?
"""
from __future__ import annotations

import json
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from glucotwin import config as C
from glucotwin.data.meals import MEAL_BY_KEY, absorption_rate
from glucotwin.data.simulator import START_DATE, sleep_si_multiplier
from glucotwin.models.explain import reasons
from glucotwin.twin.engine import future_inputs, meal_habits, run_observer, what_if
from glucotwin.twin.inputs import build_observed, stack_inputs

DASH_DIR = C.ARTIFACTS_DIR / "dashboard"


class GlucoTwinService:
    def __init__(self, cohort: str = "synthetic"):
        mdir = C.MODELS_DIR / cohort
        self.meta = json.loads((mdir / "meta.json").read_text())
        self.cols = self.meta["features"]
        self.clf = lgb.Booster(model_file=str(mdir / "spike_classifier.txt"))
        self.regs = {h: lgb.Booster(model_file=str(mdir / f"glucose_{h}min.txt")) for h in C.HORIZONS}
        self.thr = float(self.meta["threshold"])
        self.feats = pd.read_parquet(DASH_DIR / "features.parquet")
        self.ehr = pd.read_csv(C.SYN_DIR / "ehr_patients.csv").set_index("patient_id")
        self.labs = pd.read_csv(C.SYN_DIR / "lab_history.csv")
        self.ts = pd.read_parquet(C.SYN_DIR / "timeseries.parquet",
                                  filters=[("patient_id", "in", list(self.feats.patient_id.unique()))])
        self.meals = pd.read_csv(C.SYN_DIR / "meals.csv")
        self.doses = pd.read_csv(C.SYN_DIR / "doses.csv")
        self.daily = pd.read_csv(C.SYN_DIR / "daily.csv")
        self.zpar = pd.read_csv(C.PROC_DIR / "twin_params_synthetic.csv") if (C.PROC_DIR / "twin_params_synthetic.csv").exists() \
            else pd.read_csv(DASH_DIR / "twin_params.csv")
        self.preds = self._predict_all()

    # ------------------------------------------------------------------ models
    def _predict_all(self) -> pd.DataFrame:
        f = self.feats
        X = f[self.cols]
        out = f[["patient_id", "minute", "cgm_now"]].copy()
        p = self.clf.predict(X)
        p[(f.cgm_now.values >= C.SPIKE_MGDL) | f.cgm_now.isna().values] = np.nan
        out["prob_spike"] = p
        for h, r in self.regs.items():
            out[f"pred_g{h}"] = f.cgm_now.values + r.predict(X)
        return out

    @property
    def patient_ids(self) -> list[str]:
        return sorted(self.feats.patient_id.unique())

    def time_range(self, pid: str):
        m = self.feats[self.feats.patient_id == pid].minute
        return int(m.min()), int(m.max())

    @staticmethod
    def minute_to_ts(minute: int) -> pd.Timestamp:
        return pd.Timestamp(START_DATE) + pd.Timedelta(minutes=int(minute))

    def _row(self, pid, minute):
        f = self.feats[(self.feats.patient_id == pid) & (self.feats.minute <= minute)]
        return f.iloc[-1] if len(f) else None

    def state(self, pid: str, minute: int) -> dict:
        """Everything the dashboard shows for one patient at one moment."""
        row = self._row(pid, minute)
        if row is None:
            return {}
        m = int(row.minute)
        pr = self.preds[(self.preds.patient_id == pid) & (self.preds.minute == m)].iloc[0]
        x = row[self.cols].astype(float).to_frame().T
        contrib = self.clf.predict(x, pred_contrib=True)[0][:-1]
        prob = float(pr.prob_spike) if not np.isnan(pr.prob_spike) else None
        cur = float(row.cgm_now) if not np.isnan(row.cgm_now) else None
        status = ("high now" if cur is not None and cur >= C.SPIKE_MGDL else
                  "low now" if cur is not None and cur < C.HYPO_MGDL else
                  "alert" if prob is not None and prob >= self.thr else "stable")
        conf = self.meta["conformal"]
        fc = [{"horizon": h, "glucose": float(pr[f"pred_g{h}"]),
               "lo80": float(pr[f"pred_g{h}"] - conf[str(h)]["q80"]),
               "hi80": float(pr[f"pred_g{h}"] + conf[str(h)]["q80"])} for h in C.HORIZONS]
        return {
            "patient_id": pid, "minute": m, "timestamp": str(self.minute_to_ts(m)),
            "glucose_now": cur, "trend_30min": float(row.d30) if not np.isnan(row.d30) else None,
            "spike_probability": prob, "alert_threshold": self.thr, "status": status,
            "forecast": fc, "reasons": reasons(contrib, row, self.cols, k=5),
            "twin": {k: float(row[k]) for k in ("twin_g30", "twin_g60", "twin_g120", "twin_max120",
                                                "twin_si_rel", "twin_si_today", "twin_meal_gain", "twin_gb",
                                                "twin_exp_carbs", "twin_disturbance") if k in row},
            "lifestyle": {k: (float(row[k]) if not pd.isna(row[k]) else None) for k in
                          ("sleep_hours", "sleep_efficiency", "hrv_night", "steps_today", "steps_60",
                           "carbs_120", "cob", "iob", "mins_since_meal", "missed_metformin_12h") if k in row},
        }

    def ward(self, minute: int) -> pd.DataFrame:
        rows = []
        for pid in self.patient_ids:
            s = self.state(pid, minute)
            if not s:
                continue
            e = self.ehr.loc[pid]
            hist = self.feats[(self.feats.patient_id == pid) & (self.feats.minute <= minute)].tail(288 * 7)
            g = hist.cgm_now.dropna()
            rows.append({"patient_id": pid, "name": e["name"], "age": int(e.age), "sex": e.sex, "city": e.city,
                         "hba1c": e.hba1c, "glucose_now": s["glucose_now"], "trend_30min": s["trend_30min"],
                         "spike_risk": s["spike_probability"], "status": s["status"],
                         "forecast_2h": s["forecast"][-1]["glucose"],
                         "tir_7d": float(((g >= 70) & (g <= 180)).mean()) if len(g) else None,
                         "top_reason": s["reasons"][0]["text"] if s["reasons"] else ""})
        df = pd.DataFrame(rows)
        order = df.status.map({"alert": 0, "high now": 1, "low now": 1, "stable": 2})
        return df.assign(_o=order, _r=-df.spike_risk.fillna(-1)).sort_values(["_o", "_r"]).drop(columns=["_o", "_r"])

    def history(self, pid: str, minute: int, hours: int = 24) -> pd.DataFrame:
        t = self.ts[(self.ts.patient_id == pid) & (self.ts.minute <= minute) & (self.ts.minute > minute - hours * 60)]
        p = self.preds[self.preds.patient_id == pid][["minute", "prob_spike", "pred_g60", "pred_g120"]]
        out = t.merge(p, on="minute", how="left")
        out["time"] = [self.minute_to_ts(m) for m in out.minute]
        return out

    def meals_between(self, pid, lo, hi):
        m = self.meals[(self.meals.patient_id == pid) & (self.meals.minute > lo) & (self.meals.minute <= hi)]
        return m[m.logged]

    # ------------------------------------------------------------------ twin
    @lru_cache(maxsize=64)
    def _twin(self, pid: str):
        e = self.ehr.loc[pid].to_dict()
        e["patient_id"] = pid
        ts = self.ts[self.ts.patient_id == pid]
        obs = build_observed(pid, ts, self.meals[self.meals.patient_id == pid],
                             self.doses[self.doses.patient_id == pid], e)
        T, arr, prior = stack_inputs([obs])
        zp = self.zpar[self.zpar.patient_id == pid].sort_values("day")
        n_days = T // 1440
        Z = np.zeros((n_days, 1, 4))
        for r in zp.itertuples():
            if r.day < n_days:
                Z[r.day, 0] = [np.log(r.si_rel), np.log(r.meal_gain), np.log(r.gb / prior.Gb[0]),
                               np.log(r.dawn_amp / prior.dawn_amp[0])]
        states = run_observer(T, arr, prior, Z)
        return obs, arr, prior, Z, states

    def twin_params_today(self, pid: str, minute: int) -> dict:
        obs, arr, prior, Z, _ = self._twin(pid)
        d = min(minute // 1440, Z.shape[0] - 1)
        return {"insulin_sensitivity_vs_ehr": float(np.exp(Z[d, 0, 0])),
                "meal_response_gain": float(np.exp(Z[d, 0, 1])),
                "basal_glucose": float(prior.Gb[0] * np.exp(Z[d, 0, 2])),
                "dawn_effect": float(prior.dawn_amp[0] * np.exp(Z[d, 0, 3])),
                "ehr_prior_basal_glucose": float(prior.Gb[0]), "ehr_prior_si": float(prior.SI[0])}

    def simulate_scenarios(self, pid: str, minute: int, scenarios: list[dict], H: int = 360,
                           include_routine: bool = True) -> np.ndarray:
        """Each scenario dict uses engine.what_if keys; offsets are minutes from now."""
        from glucotwin.twin.engine import apply_z
        obs, arr, prior, Z, (G, X, I, D) = self._twin(pid)
        s = int(minute)
        d = min(s // 1440, Z.shape[0] - 1)
        hab = meal_habits(obs.meals, d) if include_routine else None
        Ra, Ri, _ = future_inputs(np.array([s]), H, obs.meals, obs.insulin, hab, d * 1440)
        base = {"Ra": Ra[0], "Ri": Ri[0], "A": np.full(H, 0.03), "si": np.full(H, arr["si"][0, s]),
                "egp": np.full(H, arr["egp"][0, s])}
        g0 = arr["cgm"][0, s] if not np.isnan(arr["cgm"][0, s]) else G[0, s]
        P = apply_z(prior, Z[d])
        return what_if(P, (g0, X[0, s], I[0, s], D[0, s]), s, H, base, scenarios, a0=float(arr["A"][0, s]))


def meal_scenario(meal_key: str, offset: int, portion: float = 1.0):
    m = MEAL_BY_KEY[meal_key]
    return (offset, m.carbs_g * portion, absorption_rate(m.gi, m.slow))


def sleep_scale(current_hours: float, new_hours: float, eff: float = 0.9) -> float:
    return float(sleep_si_multiplier(new_hours, eff) / sleep_si_multiplier(current_hours, eff))
