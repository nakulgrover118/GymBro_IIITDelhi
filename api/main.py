"""ThaliTwin REST API.

Run:  uvicorn api.main:app --reload       (docs at http://127.0.0.1:8000/docs)

The API exposes the same service the dashboard uses, so a hospital EHR, a
wearable companion app or a CGM vendor could integrate with the twin.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from thalitwin.data.meals import MEAL_BY_KEY, MEALS  # noqa: E402
from thalitwin.inference import ThaliTwinService, meal_scenario, sleep_scale  # noqa: E402

app = FastAPI(title="ThaliTwin API", version="1.0.0",
              description="Digital twin for type-2 diabetes: 2-hour glucose spike prediction from EHR + wearables. "
                          "All patients are synthetic.")
_svc: Optional[ThaliTwinService] = None


def svc() -> ThaliTwinService:
    global _svc
    if _svc is None:
        _svc = ThaliTwinService()
    return _svc


def _check(pid: str):
    if pid not in svc().patient_ids:
        raise HTTPException(404, f"unknown patient {pid}")


@app.get("/health")
def health():
    return {"status": "ok", "patients": len(svc().patient_ids), "alert_threshold": svc().thr}


@app.get("/patients")
def patients(minute: Optional[int] = None):
    s = svc()
    m = minute if minute is not None else min(s.time_range(p)[1] for p in s.patient_ids)
    return {"minute": m, "timestamp": str(s.minute_to_ts(m)),
            "patients": s.ward(m).fillna("").to_dict("records")}


@app.get("/patients/{pid}")
def patient(pid: str):
    _check(pid)
    e = svc().ehr.loc[pid].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in e.items() if not k.startswith("persona_")}


@app.get("/patients/{pid}/state")
def patient_state(pid: str, minute: Optional[int] = None):
    """Live twin state: current glucose, 30/60/120-min forecast with 80% interval,
    spike probability, alert flag and the top reasons behind it."""
    _check(pid)
    s = svc()
    m = minute if minute is not None else s.time_range(pid)[1]
    return s.state(pid, m)


@app.get("/patients/{pid}/twin")
def twin_parameters(pid: str, minute: Optional[int] = None):
    """The twin's current personal physiology estimates (re-learned every day)."""
    _check(pid)
    s = svc()
    m = minute if minute is not None else s.time_range(pid)[1]
    return s.twin_params_today(pid, m)


class Meal(BaseModel):
    meal_key: str = Field(..., examples=["rice_dal_sabzi"])
    offset_min: int = Field(0, ge=0, le=600)
    portion: float = Field(1.0, gt=0, le=3)


class Walk(BaseModel):
    offset_min: int = Field(30, ge=0, le=600)
    duration_min: int = Field(20, gt=0, le=180)
    intensity: float = Field(0.5, gt=0, le=1.0)


class Scenario(BaseModel):
    name: str = "scenario"
    meals: list[Meal] = []
    walks: list[Walk] = []
    sleep_hours: Optional[float] = Field(None, ge=2, le=12)
    skip_metformin: bool = False
    carbs_scale: float = Field(1.0, gt=0, le=2)


class WhatIfRequest(BaseModel):
    minute: Optional[int] = None
    horizon_min: int = Field(360, ge=60, le=720)
    scenarios: list[Scenario]


@app.post("/patients/{pid}/what-if")
def what_if(pid: str, req: WhatIfRequest):
    """Simulate alternative futures with the patient's twin (meals, walks, sleep, missed metformin)."""
    _check(pid)
    s = svc()
    m = req.minute if req.minute is not None else s.time_range(pid)[1]
    st = s.state(pid, m)
    cur_sleep = st["lifestyle"].get("sleep_hours") or 7.0
    scs = []
    for sc in req.scenarios:
        for ml in sc.meals:
            if ml.meal_key not in MEAL_BY_KEY:
                raise HTTPException(400, f"unknown meal {ml.meal_key}")
        scs.append({"meals": [meal_scenario(x.meal_key, x.offset_min, x.portion) for x in sc.meals],
                    "walks": [(w.offset_min, w.duration_min, w.intensity) for w in sc.walks],
                    "si_scale": sleep_scale(cur_sleep, sc.sleep_hours) if sc.sleep_hours else 1.0,
                    "egp_add": 0.12 if sc.skip_metformin else 0.0, "carbs_scale": sc.carbs_scale})
    G = s.simulate_scenarios(pid, m, scs, H=req.horizon_min)
    return {"minute": m, "scenarios": [
        {"name": sc.name, "peak": float(g.max()), "time_above_180_min": int((g > 180).sum()),
         "glucose_every_15min": [round(float(v), 1) for v in g[::15]]} for sc, g in zip(req.scenarios, G)]}


@app.get("/meals")
def meals():
    return [{"key": m.key, "name": m.name, "carbs_g": m.carbs_g, "gi": m.gi, "slot": m.slot, "region": m.region}
            for m in MEALS]
