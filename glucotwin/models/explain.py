"""Turn SHAP values into reasons a doctor can read."""
from __future__ import annotations

import numpy as np
import pandas as pd

# feature -> (template when it RAISES risk, template when it LOWERS risk)
TEMPLATES = {
    "cgm_now": ("glucose already at {v:.0f} mg/dL", "glucose comfortably low at {v:.0f} mg/dL"),
    "d15": ("rising {v:+.0f} mg/dL in the last 15 min", "falling {v:+.0f} mg/dL in the last 15 min"),
    "d30": ("rising {v:+.0f} mg/dL in the last 30 min", "falling {v:+.0f} mg/dL in the last 30 min"),
    "d60": ("up {v:+.0f} mg/dL over the last hour", "down {v:+.0f} mg/dL over the last hour"),
    "carbs_30": ("{v:.0f} g carbs eaten in the last 30 min", "no recent carbs"),
    "carbs_60": ("{v:.0f} g carbs eaten in the last hour", "no carbs in the last hour"),
    "carbs_120": ("{v:.0f} g carbs in the last 2 h", "light carb intake in the last 2 h"),
    "cob": ("{v:.0f} g carbs still being absorbed", "little food left to absorb"),
    "mins_since_meal": ("last meal {v:.0f} min ago", "last meal {v:.0f} min ago"),
    "last_meal_k": ("last meal was fast-absorbing (high GI)", "last meal was slow-absorbing (low GI)"),
    "iob": ("insulin on board is low ({v:.1f} IU)", "{v:.1f} IU insulin still active"),
    "missed_metformin_12h": ("missed metformin dose", "metformin taken on time"),
    "steps_30": ("inactive in the last 30 min ({v:.0f} steps)", "active in the last 30 min ({v:.0f} steps)"),
    "steps_60": ("inactive in the last hour ({v:.0f} steps)", "walked {v:.0f} steps in the last hour"),
    "steps_120": ("only {v:.0f} steps in the last 2 h", "{v:.0f} steps in the last 2 h"),
    "active_min_120": ("no brisk activity recently", "{v:.0f} active minutes in the last 2 h"),
    "sleep_hours": ("short sleep last night ({v:.1f} h)", "good sleep last night ({v:.1f} h)"),
    "sleep_debt": ("sleep debt of {v:.1f} h", "well rested"),
    "sleep_efficiency": ("restless sleep (efficiency {v:.0%})", "efficient sleep ({v:.0%})"),
    "hrv_night": ("low overnight HRV ({v:.0f} ms) - stress/poor recovery", "healthy overnight HRV ({v:.0f} ms)"),
    "hrv_mean_30": ("low HRV now ({v:.0f} ms)", "HRV normal ({v:.0f} ms)"),
    "hr_vs_rest": ("heart rate {v:+.0f} bpm above resting", "heart rate near resting"),
    "hba1c": ("HbA1c {v:.1f}% (poor long-term control)", "HbA1c {v:.1f}%"),
    "homa_ir": ("high insulin resistance (HOMA-IR {v:.1f})", "lower insulin resistance (HOMA-IR {v:.1f})"),
    "fpg": ("high fasting glucose ({v:.0f} mg/dL)", "fasting glucose {v:.0f} mg/dL"),
    "bmi": ("BMI {v:.1f}", "BMI {v:.1f}"),
    "tar_24h": ("{v:.0%} of last 24 h above 180", "mostly in range over the last 24 h"),
    "mean_24h": ("24-h average {v:.0f} mg/dL", "24-h average {v:.0f} mg/dL"),
    "twin_max120": ("twin simulates a peak of {v:.0f} mg/dL", "twin simulates a peak of only {v:.0f} mg/dL"),
    "twin_dmax": ("twin expects a rise of {v:+.0f} mg/dL", "twin expects only {v:+.0f} mg/dL"),
    "twin_g60": ("twin forecasts {v:.0f} mg/dL in 1 h", "twin forecasts {v:.0f} mg/dL in 1 h"),
    "twin_g120": ("twin forecasts {v:.0f} mg/dL in 2 h", "twin forecasts {v:.0f} mg/dL in 2 h"),
    "twin_known_max120": ("even without another meal the twin reaches {v:.0f} mg/dL", "twin sees no rise from food already eaten"),
    "twin_d60": ("twin expects {v:+.0f} mg/dL within 1 h", "twin expects {v:+.0f} mg/dL within 1 h"),
    "twin_d120": ("twin expects {v:+.0f} mg/dL within 2 h", "twin expects {v:+.0f} mg/dL within 2 h"),
    "last_meal_carbs": ("last meal had {v:.0f} g carbs", "last meal was light ({v:.0f} g carbs)"),
    "steps_today": ("only {v:.0f} steps so far today", "{v:,.0f} steps today"),
    "twin_exp_carbs": ("usual meal due soon (~{v:.0f} g carbs)", "no meal expected in the next 2 h"),
    "twin_si_today": ("insulin sensitivity reduced today (x{v:.2f})", "insulin sensitivity good today (x{v:.2f})"),
    "twin_si_rel": ("twin learned low insulin sensitivity (x{v:.2f})", "twin learned better insulin sensitivity (x{v:.2f})"),
    "twin_disturbance": ("unexplained upward glucose drift", "unexplained downward drift"),
    "mins_since_wake": ("{v:.0f} min since waking (usual meal times)", "{v:.0f} min since waking"),
    "tod_sin": ("time of day", "time of day"),
    "tod_cos": ("time of day", "time of day"),
}


# related features are explained once, by their strongest member
GROUPS = {**{f: "twin_forecast" for f in ("twin_max120", "twin_g120", "twin_g60", "twin_g30", "twin_dmax", "twin_d60",
                                          "twin_d120", "twin_known_max120", "twin_known_g60", "twin_now")},
          **{f: "carbs" for f in ("carbs_30", "carbs_60", "carbs_120", "carbs_240", "cob", "last_meal_carbs")},
          **{f: "activity" for f in ("steps_15", "steps_30", "steps_60", "steps_120", "active_min_120", "steps_today")},
          **{f: "trend" for f in ("d15", "d30", "d60", "accel")},
          **{f: "sleep" for f in ("sleep_hours", "sleep_debt", "sleep_efficiency", "deep_pct", "rem_pct")},
          **{f: "hrv" for f in ("hrv_night", "hrv_mean_30")},
          **{f: "sensitivity" for f in ("twin_si_today", "twin_si_rel")}}
SKIP = {"tod_sin", "tod_cos"}          # not actionable for a clinician


def reasons(shap_row: np.ndarray, x_row: pd.Series, cols: list[str], k: int = 4) -> list[dict]:
    """Top-k plain-language drivers for one prediction (sign = direction of effect)."""
    shap_row = np.asarray(shap_row, float)
    groups: dict[str, list[int]] = {}
    for j, f in enumerate(cols):
        if f in SKIP:
            continue
        groups.setdefault(GROUPS.get(f, f), []).append(j)
    scored = []
    for g, idx in groups.items():
        tot = float(shap_row[idx].sum())
        cand = [j for j in sorted(idx, key=lambda j: -abs(shap_row[j]))
                if cols[j] in TEMPLATES and not pd.isna(x_row[cols[j]])]
        if cand:
            scored.append((abs(tot), tot, cols[cand[0]]))
    out = []
    for _, tot, f in sorted(scored, reverse=True):
        up, down = TEMPLATES[f]
        text = (up if tot > 0 else down).format(v=float(x_row[f]))
        out.append({"feature": f, "text": text, "impact": tot,
                    "direction": "raises risk" if tot > 0 else "lowers risk"})
        if len(out) >= k:
            break
    return out
