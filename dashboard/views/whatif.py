"""What-if simulator: ask the patient's digital twin about alternative futures."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BLUE, GREY, RED, TEAL, card, select_patient, setup
from glucotwin.data.meals import MEAL_BY_KEY, MEALS, absorption_rate
from glucotwin.inference import meal_scenario, sleep_scale

svc = setup("What-if")
minute = st.session_state.minute

st.title("🧪 What-if simulator")
st.caption("The twin is the patient's personalised physiology (re-learned nightly). Change the plan and it "
           "simulates the next hours. For counselling and education - not a dosing tool.")
pid = select_patient(svc, key="pid_whatif")
e = svc.ehr.loc[pid]
s = svc.state(pid, minute)
cur_sleep = s["lifestyle"].get("sleep_hours") or 7.0

veg = bool(e.vegetarian)
menu = [m for m in MEALS if m.veg or not veg]
H = 360

c1, c2, c3 = st.columns([1.1, 1.1, 1])
with c1:
    st.markdown("##### 🍽 Next meal")
    add_meal = st.toggle("Add a meal", value=True)
    mk = st.selectbox("Dish", [m.key for m in menu], index=[m.key for m in menu].index("rice_dal_sabzi"),
                      format_func=lambda k: f"{MEAL_BY_KEY[k].name} · {MEAL_BY_KEY[k].carbs_g:.0f} g carbs · GI {MEAL_BY_KEY[k].gi}",
                      disabled=not add_meal)
    off = st.slider("Eaten in (min)", 0, 180, 30, 15, disabled=not add_meal)
    portion = st.slider("Portion", 0.5, 1.5, 1.0, 0.25, disabled=not add_meal)
with c2:
    st.markdown("##### 🚶 Activity & sleep")
    walk = st.toggle("Walk after the meal")
    wdur = st.slider("Walk duration (min)", 5, 45, 15, 5, disabled=not walk)
    wdelay = st.slider("Start after eating (min)", 0, 60, 15, 5, disabled=not walk)
    sleep_default = float(np.clip(round(cur_sleep * 4) / 4, 4, 9))
    sleep_h = st.slider("Last night's sleep (h)", 4.0, 9.0, sleep_default, 0.25,
                        help=f"Wearable recorded {cur_sleep:.1f} h. Sleep debt lowers insulin sensitivity.")
with c3:
    st.markdown("##### 💊 Medication")
    skip_met = st.toggle("Skip next metformin dose", disabled=not bool(e.metformin))
    routine = st.toggle("Include usual routine (learned meals)", value=False,
                        help="Adds the patient's typical meals the twin has learned, at their usual times.")
    st.caption(f"Patient: {e['name']}, {e.age}{e.sex}, HbA1c {e.hba1c}%, "
               f"{'vegetarian' if veg else 'non-veg'}, {e.region} Indian diet")

base_sc = {}
plan = {"meals": [], "walks": []}
if add_meal:
    plan["meals"].append(meal_scenario(mk, off, portion))
if walk and add_meal:
    plan["walks"].append((off + wdelay, wdur, 0.55))
elif walk:
    plan["walks"].append((wdelay, wdur, 0.55))
plan["si_scale"] = sleep_scale(sleep_default, sleep_h)
plan["egp_add"] = 0.12 if skip_met else 0.0

# automatic alternatives the twin evaluates for comparison
alts = []
if add_meal:
    alts.append(("Same meal + 15-min walk after", {**plan, "walks": plan["walks"] + [(off + 15, 15, 0.55)]}))
    alts.append(("Half portion", {**plan, "meals": [meal_scenario(mk, off, portion * 0.5)]}))
    low_gi = "millet_thali" if MEAL_BY_KEY[mk].slot != "breakfast" else "besan_chilla"
    alts.append((f"Swap to {MEAL_BY_KEY[low_gi].name}", {**plan, "meals": [meal_scenario(low_gi, off, portion)]}))

names = ["Without this plan (baseline)", "Your plan"] + [a[0] for a in alts]
scs = [{"si_scale": 1.0}, plan] + [a[1] for a in alts]
G = svc.simulate_scenarios(pid, minute, scs, H=H, include_routine=routine)
t0 = svc.minute_to_ts(s["minute"])
tt = [t0 + pd.Timedelta(minutes=i) for i in range(H)]

fig = go.Figure()
fig.add_hrect(y0=70, y1=180, fillcolor=TEAL, opacity=0.07, line_width=0)
fig.add_hline(y=180, line=dict(color=AMBER, dash="dot", width=1))
hist = svc.history(pid, minute, 6)
fig.add_trace(go.Scatter(x=hist.time, y=hist.cgm, name="CGM (past 6 h)", line=dict(color=BLUE, width=2)))
styles = [dict(color=GREY, dash="dot"), dict(color=RED, width=3), dict(color=TEAL), dict(color="#7c3aed"),
          dict(color=AMBER)]
for n, g, st_ in zip(names, G, styles):
    fig.add_trace(go.Scatter(x=tt, y=g, name=n, line=st_))
fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="glucose mg/dL",
                  legend=dict(orientation="h", y=-0.15), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

rows = []
for n, g in zip(names, G):
    rows.append({"Scenario": n, "Peak (mg/dL)": round(float(g.max())), "Time of peak": tt[int(g.argmax())].strftime("%H:%M"),
                 "Minutes above 180": int((g > 180).sum()), "Glucose in 6 h": round(float(g[-1]))})
tab = pd.DataFrame(rows)
best = tab.iloc[2:].sort_values("Minutes above 180").head(1) if len(tab) > 2 else tab.iloc[1:2]
k = st.columns(3)
with k[0]:
    card("Your plan: peak", f"{tab.iloc[1]['Peak (mg/dL)']} mg/dL", f"at {tab.iloc[1]['Time of peak']}", RED)
with k[1]:
    card("Your plan: time above 180", f"{tab.iloc[1]['Minutes above 180']} min", "next 6 hours", AMBER)
with k[2]:
    if len(tab) > 2:
        b = best.iloc[0]
        card("Best alternative", b["Scenario"], f"peak {b['Peak (mg/dL)']} · {b['Minutes above 180']} min above 180", TEAL)
st.dataframe(tab, hide_index=True, use_container_width=True)
st.caption("Simulated with the patient's calibrated twin: Bergman-type glucose-insulin model + gut absorption by "
           "glycaemic index + exercise uptake + sleep-dependent insulin sensitivity + disturbance observer.")
