"""Loader for the real-world ShanghaiT2DM dataset.

Zhao Q. et al. "Chinese diabetes datasets for data-driven machine learning",
Scientific Data 10, 35 (2023). doi:10.1038/s41597-023-01940-7
Data: figshare doi:10.6084/m9.figshare.20444397 (CC BY 4.0).

100 people with type-2 diabetes, 3-14 days of 15-minute CGM each, plus
dietary records (free text), insulin / oral drug logs and a clinical
summary (labs, complications, medicines). This is the "anonymised real
data" validation track: the same twin + model pipeline runs on it.

Diet records are free text ("Rice 150 g\\nVegetable 100 g"), so carbs are
estimated with a small food -> carbohydrate-density table (g carbs per
100 g, China Food Composition Tables / USDA, rounded).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from glucotwin.config import SHANGHAI_DIR
from glucotwin.data.meals import absorption_rate
from glucotwin.twin.engine import slot_of

# keyword -> (carbs per 100 g, glycaemic index). First match wins, so order matters.
FOOD_TABLE = [
    ("fried rice", 30, 72), ("glutinous", 30, 85), ("congee", 9, 78), ("porridge", 9, 78), ("gruel", 9, 78),
    ("coarse grain steamed", 45, 60), ("coarse grain", 25, 55), ("multigrain", 25, 55), ("brown rice", 24, 55),
    ("rice noodle", 25, 60), ("rice", 28, 73), ("noodle", 25, 55), ("pasta", 25, 50), ("vermicelli", 25, 55),
    ("steamed bread", 47, 80), ("steamed bun", 45, 75), ("bun", 45, 75), ("mantou", 47, 80), ("bread", 50, 70),
    ("dumpling", 25, 60), ("wonton", 25, 60), ("pancake", 40, 70), ("cake", 50, 70), ("biscuit", 70, 70),
    ("cracker", 70, 70), ("cereal", 70, 70), ("oat", 12, 55), ("corn", 20, 55), ("sweet potato", 20, 63),
    ("potato", 17, 78), ("taro", 18, 55), ("yam", 12, 50), ("pumpkin", 7, 65), ("lotus root", 15, 50),
    ("milk", 5, 35), ("soy milk", 3, 30), ("yogurt", 12, 35), ("banana", 22, 52), ("apple", 13, 38),
    ("pear", 13, 38), ("orange", 10, 43), ("grape", 17, 50), ("watermelon", 8, 72), ("kiwi", 14, 50),
    ("peach", 10, 42), ("dragon fruit", 13, 50), ("fruit", 12, 50), ("juice", 10, 50), ("sugar", 100, 65),
    ("candy", 90, 70), ("honey", 80, 60), ("nut", 20, 20), ("bean", 10, 30), ("tofu", 3, 15),
    ("egg", 1, 0), ("pork", 0, 0), ("beef", 0, 0), ("chicken", 0, 0), ("duck", 0, 0), ("fish", 0, 0),
    ("shrimp", 0, 0), ("meat", 0, 0), ("vegetable", 4, 15), ("greens", 4, 15), ("cabbage", 4, 15),
    ("spinach", 3, 15), ("salad", 4, 15),
]
DEFAULT_DENSITY, DEFAULT_GI = 10, 55
DEFAULT_MEAL_CARBS = 45.0


def estimate_carbs(text) -> tuple[float, float] | None:
    """(carbs_g, carb-weighted GI) from a free-text diet record, or None if not recorded."""
    if not isinstance(text, str) or not text.strip() or "not available" in text.lower():
        return None
    total, gi_acc = 0.0, 0.0
    found = False
    for line in re.split(r"[\n;,]+", text):
        line = line.strip().lower()
        if not line:
            continue
        g = re.search(r"(\d+(?:\.\d+)?)\s*(g|ml)\b", line)
        grams = float(g.group(1)) if g else None
        dens, gi = DEFAULT_DENSITY, DEFAULT_GI
        for kw, d, gidx in FOOD_TABLE:
            if kw in line:
                dens, gi = d, gidx
                break
        if grams is None:
            continue
        found = True
        c = grams * dens / 100.0
        total += c
        gi_acc += c * gi
    if not found:
        return DEFAULT_MEAL_CARBS, DEFAULT_GI
    return total, (gi_acc / total if total > 0 else DEFAULT_GI)


def _insulin_ka(name: str) -> float | None:
    n = name.lower()
    if any(k in n for k in ("glargine", "lantus", "detemir", "levemir", "degludec", "tresiba", "toujeo", "basalin")):
        return None                                   # basal insulin: already reflected in fasting state
    if any(k in n for k in ("30r", "40r", "50r", "70/30", "mix", "25r", "premix")):
        return 0.012
    if any(k in n for k in ("aspart", "novorapid", "lispro", "humalog", "glulisine", "apidra")):
        return 0.025
    return 0.018                                      # regular human insulin


def _parse_doses(text) -> list[tuple[float, float]]:
    out = []
    if not isinstance(text, str):
        return out
    for part in re.split(r"[\n;]+", text):
        m = re.search(r"(\d+(?:\.\d+)?)\s*iu", part.lower())
        if m:
            ka = _insulin_ka(part)
            if ka is not None:
                out.append((float(m.group(1)), ka))
    return out


def _hba1c_pct(mmol_mol):
    return mmol_mol / 10.929 + 2.15


def load_summary(base: Path = SHANGHAI_DIR) -> pd.DataFrame:
    s = pd.read_excel(base / "Shanghai_T2DM_Summary.xlsx")
    s.columns = [c.strip() for c in s.columns]
    col = lambda key: [c for c in s.columns if c.lower().startswith(key.lower())][0]
    agents = s[col("Hypoglycemic Agents")].fillna("").str.lower()
    comorb = s[col("Comorbidities")].fillna("").str.lower()
    fi = pd.to_numeric(s[col("Fasting Insulin")], errors="coerce") / 6.0    # pmol/L -> uU/mL
    fpg = pd.to_numeric(s[col("Fasting Plasma Glucose")], errors="coerce")
    out = pd.DataFrame({
        "patient_id": s[col("Patient Number")].astype(str),
        "person": s[col("Patient Number")].astype(str).str.slice(0, 4),
        "sex": np.where(s[col("Gender")] == 2, "M", "F"),
        "age": pd.to_numeric(s[col("Age")], errors="coerce"),
        "weight_kg": pd.to_numeric(s[col("Weight")], errors="coerce"),
        "bmi": pd.to_numeric(s[col("BMI")], errors="coerce"),
        "diabetes_duration_y": pd.to_numeric(s[col("Duration of diabetes")], errors="coerce"),
        "hba1c": _hba1c_pct(pd.to_numeric(s[col("HbA1c")], errors="coerce")),
        "fpg": fpg, "fasting_insulin": fi, "homa_ir": fpg * fi / 405.0,
        "egfr": pd.to_numeric(s[col("Estimated Glomerular")], errors="coerce"),
        "triglycerides": pd.to_numeric(s[col("Triglyceride")], errors="coerce") * 88.57,
        "hdl": pd.to_numeric(s[col("High-Density")], errors="coerce") * 38.67,
        "ldl": pd.to_numeric(s[col("Low-Density")], errors="coerce") * 38.67,
        "hypertension": comorb.str.contains("hypertension"),
        "metformin": agents.str.contains("metformin"),
        "sulfonylurea": agents.str.contains("glimepiride|gliclazide|glipizide|gliquidone|glibenclamide"),
        "dpp4": agents.str.contains("gliptin"),
        "sglt2": agents.str.contains("gliflozin"),
        "insulin": agents.str.contains("insulin|humulin|novolin|gansulin|lantus|novorapid|glargine|aspart|detemir|degludec|lispro|mix"),
        "acarbose": agents.str.contains("acarbose|voglibose"),
        "hypoglycemia_hx": s[col("Hypoglycemia")].astype(str).str.lower().eq("yes"),
    })
    out["fasting_insulin"] = out["fasting_insulin"].fillna(out["fasting_insulin"].median())
    out["fpg"] = out["fpg"].fillna(out["fpg"].median())
    out["weight_kg"] = out["weight_kg"].fillna(out["weight_kg"].median())
    out["hba1c"] = out["hba1c"].fillna(out["hba1c"].median())
    return out


def load_recording(path: Path, step: int = 15):
    """One CGM recording -> (grid, meals, insulin, start_timestamp)."""
    d = pd.read_excel(path)
    d.columns = [str(c).strip() for c in d.columns]
    tcol = d.columns[0]
    cgm_col = [c for c in d.columns if c.lower().startswith("cgm")][0]
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna(subset=[tcol]).sort_values(tcol)
    t0 = d[tcol].iloc[0].normalize()
    minute = ((d[tcol] - t0).dt.total_seconds() / 60.0).values
    d["minute"] = minute
    g = d.dropna(subset=[cgm_col])
    gm = (np.round(g.minute.values / step) * step).astype(int)
    cg = pd.Series(pd.to_numeric(g[cgm_col], errors="coerce").values).groupby(gm).mean()
    T = int(np.ceil((gm.max() + step) / 1440) * 1440)
    grid = pd.DataFrame({"minute": np.arange(0, T, step)})
    grid["cgm"] = grid.minute.map(cg).astype(float)

    diet_col = [c for c in d.columns if c.lower().startswith("dietary intake")]
    meals = []
    if diet_col:
        for r in d[d[diet_col[0]].notna()].itertuples(index=False):
            est = estimate_carbs(r[d.columns.get_loc(diet_col[0])])
            if est is None:
                continue
            m = int(r[d.columns.get_loc("minute")])
            meals.append(dict(minute=m, carbs=round(est[0], 1), gi=est[1], k=absorption_rate(est[1], 0.3),
                              slot=str(slot_of(m % 1440))))
    meals = pd.DataFrame(meals, columns=["minute", "carbs", "gi", "k", "slot"])

    ins = []
    sc = [c for c in d.columns if c.lower().startswith("insulin dose - s.c")]
    bol = [c for c in d.columns if "bolus" in c.lower()]
    iv = [c for c in d.columns if c.lower().startswith("insulin dose - i.v")]
    for r in d.itertuples(index=False):
        m = int(r[d.columns.get_loc("minute")])
        if sc:
            for dose, ka in _parse_doses(r[d.columns.get_loc(sc[0])]):
                ins.append(dict(minute=m, dose=dose, ka=ka))
        if bol:
            v = pd.to_numeric(r[d.columns.get_loc(bol[0])], errors="coerce")
            if pd.notna(v) and v > 0:
                ins.append(dict(minute=m, dose=float(v), ka=0.018))
        if iv:
            for dose, _ in _parse_doses(r[d.columns.get_loc(iv[0])]):
                ins.append(dict(minute=m, dose=dose, ka=0.05))
    ins = pd.DataFrame(ins, columns=["minute", "dose", "ka"])
    return grid, meals, ins, t0


def load_all(base: Path = SHANGHAI_DIR, step: int = 15):
    summ = load_summary(base)
    recs = {}
    for p in sorted((base / "Shanghai_T2DM").glob("*.xls*")):
        if p.name.startswith("~$"):
            continue
        pid = p.stem
        if pid not in set(summ.patient_id):
            continue
        recs[pid] = load_recording(p, step)
    return summ[summ.patient_id.isin(recs)].reset_index(drop=True), recs
