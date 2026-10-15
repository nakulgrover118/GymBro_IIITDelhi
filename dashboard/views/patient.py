"""Patient twin: the virtual patient a doctor interacts with."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import (AMBER, BLUE, GREY, RED, STATUS_COLOR, STATUS_LABEL, TEAL, card, chips, fmt, risk_txt,
                    select_patient, setup, trend_arrow)
from thalitwin import config as C
from thalitwin.data.ehr_generator import CONDITIONS, LOINC

svc = setup("Patient twin")
minute = st.session_state.minute

top = st.columns([3, 2])
with top[0]:
    pid = select_patient(svc)
e = svc.ehr.loc[pid]
s = svc.state(pid, minute)
if not s:
    st.warning("No data for this patient at the selected time.")
    st.stop()

# ---------------------------------------------------------------- header
st.markdown(f"## 🧬 {e['name']}  <span class='gt-chip gt-synth'>SYNTHETIC PATIENT</span>", unsafe_allow_html=True)
st.caption(f"{pid} · {e.age} y · {'Male' if e.sex == 'M' else 'Female'} · {e.city} · {e.occupation} · "
           f"{'vegetarian' if e.vegetarian else 'non-vegetarian'} · T2D for {e.diabetes_duration_y:.0f} y · "
           f"{svc.minute_to_ts(s['minute']).strftime('%a %d %b, %H:%M')}")

status = s["status"]
col = STATUS_COLOR[status]
prob = s["spike_probability"]
if status == "alert":
    why = "; ".join([r["text"] for r in s["reasons"] if r["impact"] > 0][:3]) or "combined pattern of recent signals"
    st.markdown(f"<div class='gt-banner' style='border-color:{AMBER}; background:rgba(217,119,6,.10)'>"
                f"<b>⚠️ Glucose spike likely within 2 hours</b> &nbsp;·&nbsp; risk {risk_txt(prob)} "
                f"(alert threshold {s['alert_threshold']:.0%})<br><span style='opacity:.85'>Why: {why}</span><br>"
                f"<span style='opacity:.75'>Try the <b>What-if simulator</b> to compare a post-meal walk or a "
                f"smaller portion before acting.</span></div>", unsafe_allow_html=True)
elif status in ("high now", "low now"):
    st.markdown(f"<div class='gt-banner' style='border-color:{RED}; background:rgba(220,38,38,.08)'>"
                f"<b>{STATUS_LABEL[status]}</b> — glucose {fmt(s['glucose_now'])} mg/dL</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- KPI cards
f120 = s["forecast"][-1]
tw = svc.twin_params_today(pid, minute)
life = s["lifestyle"]
k = st.columns(6)
with k[0]:
    card("Glucose now", f"{fmt(s['glucose_now'])} {trend_arrow(s['trend_30min'])}",
         f"{fmt(s['trend_30min'], '{:+.0f}')} mg/dL in 30 min", col)
with k[1]:
    card("2-h spike risk", risk_txt(prob) if prob is not None else "–",
         STATUS_LABEL[status], col)
with k[2]:
    card("Forecast in 2 h", f"{f120['glucose']:.0f}", f"80% range {f120['lo80']:.0f}–{f120['hi80']:.0f} mg/dL")
with k[3]:
    card("Insulin sensitivity (twin)", f"×{tw['insulin_sensitivity_vs_ehr']:.2f}",
         f"vs EHR estimate · today ×{s['twin'].get('twin_si_today', np.nan):.2f} incl. sleep/exercise")
with k[4]:
    card("Last night's sleep", f"{fmt(life.get('sleep_hours'), '{:.1f}')} h",
         f"efficiency {fmt((life.get('sleep_efficiency') or np.nan) * 100, '{:.0f}%')} · HRV {fmt(life.get('hrv_night'))} ms")
with k[5]:
    card("Steps today", fmt(life.get("steps_today"), "{:,.0f}"),
         f"{fmt(life.get('steps_60'), '{:,.0f}')} in the last hour")

st.write("")
left, right = st.columns([2.3, 1])

# ---------------------------------------------------------------- main timeline
with left:
    hours = st.radio("Window", [6, 12, 24, 48], index=1, horizontal=True, format_func=lambda h: f"last {h} h",
                     label_visibility="collapsed")
    h = svc.history(pid, minute, hours)
    meals = svc.meals_between(pid, minute - hours * 60, minute)
    dz = svc.doses[(svc.doses.patient_id == pid) & (svc.doses.minute > minute - hours * 60) & (svc.doses.minute <= minute)]
    now_t = svc.minute_to_ts(s["minute"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.52, 0.16, 0.16, 0.16],
                        subplot_titles=("Glucose (CGM) · forecast with 80% interval · digital twin simulation",
                                        "Predicted 2-h spike risk", "Steps per 5 min", "Heart rate · HRV"))
    fig.add_hrect(y0=70, y1=180, fillcolor=TEAL, opacity=0.07, line_width=0, row=1, col=1)
    fig.add_hline(y=180, line=dict(color=AMBER, dash="dot", width=1), row=1, col=1)
    fig.add_hline(y=70, line=dict(color=RED, dash="dot", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.time, y=h.cgm, mode="lines", name="CGM", line=dict(color=BLUE, width=2.2)), 1, 1)
    ft = [now_t] + [now_t + pd.Timedelta(minutes=x["horizon"]) for x in s["forecast"]]
    g0 = s["glucose_now"] or s["forecast"][0]["glucose"]
    fig.add_trace(go.Scatter(x=ft + ft[::-1], y=[g0] + [x["hi80"] for x in s["forecast"]] +
                             [x["lo80"] for x in s["forecast"]][::-1] + [g0], fill="toself",
                             fillcolor="rgba(217,119,6,.18)", line=dict(width=0), name="80% interval", mode="lines",
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=ft, y=[g0] + [x["glucose"] for x in s["forecast"]], mode="lines+markers",
                             name="ThaliTwin forecast", line=dict(color=AMBER, width=2.5)), 1, 1)
    tw_ = s["twin"]
    fig.add_trace(go.Scatter(x=ft, y=[g0, tw_["twin_g30"], tw_["twin_g60"], tw_["twin_g120"]], mode="lines",
                             name="Twin physiology only", line=dict(color=GREY, dash="dash")), 1, 1)
    ymax = max(260, np.nanmax(h.cgm.values) + 30 if len(h) else 260, max(x["hi80"] for x in s["forecast"]) + 20)
    for m in meals.itertuples():
        t = svc.minute_to_ts(m.minute)
        fig.add_trace(go.Scatter(x=[t], y=[ymax - 15], mode="markers+text", text=[f"🍽 {m.carbs_logged:.0f}g"],
                                 textposition="bottom center", marker=dict(symbol="triangle-down", size=10, color=TEAL),
                                 name=m.meal_name, showlegend=False, hovertext=f"{m.meal_name} · {m.carbs_logged:.0f} g carbs logged"), 1, 1)
    for d in dz.itertuples():
        t = svc.minute_to_ts(d.minute)
        sym, c_ = ("circle", TEAL) if d.taken else ("x", RED)
        fig.add_trace(go.Scatter(x=[t], y=[45], mode="markers", marker=dict(symbol=sym, size=8, color=c_),
                                 showlegend=False, hovertext=f"{d.drug} {d.dose:.0f} {d.unit} · {'taken' if d.taken else 'MISSED'}"), 1, 1)
    fig.add_vline(x=now_t, line=dict(color=GREY, width=1))
    fig.add_trace(go.Scatter(x=h.time, y=h.prob_spike * 100, mode="lines", fill="tozeroy", name="Spike risk %",
                             line=dict(color=AMBER, width=1.5), fillcolor="rgba(217,119,6,.15)"), 2, 1)
    fig.add_hline(y=svc.thr * 100, line=dict(color=AMBER, dash="dot", width=1), row=2, col=1)
    fig.add_trace(go.Bar(x=h.time, y=h.steps, name="Steps", marker_color=TEAL, opacity=0.8), 3, 1)
    fig.add_trace(go.Scatter(x=h.time, y=h.heart_rate, name="Heart rate (bpm)", line=dict(color=RED, width=1.2)), 4, 1)
    fig.add_trace(go.Scatter(x=h.time, y=h.hrv_rmssd, name="HRV RMSSD (ms)", line=dict(color=BLUE, width=1),
                             opacity=0.6), 4, 1)
    asleep = h[h.sleep_stage > 0]
    if len(asleep):
        fig.add_trace(go.Scatter(x=asleep.time, y=np.full(len(asleep), 40), mode="markers", name="Asleep",
                                 marker=dict(color="#7c3aed", size=4, symbol="square")), 4, 1)
    fig.update_yaxes(title_text="mg/dL", range=[40, ymax], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(height=780, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.05),
                      hovermode="x unified", bargap=0)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- explanation + twin
with right:
    st.markdown("#### Why this prediction")
    rs = s["reasons"]
    if rs:
        rdf = pd.DataFrame(rs)[::-1]
        bar = go.Figure(go.Bar(x=rdf.impact, y=rdf.text, orientation="h",
                               marker_color=[AMBER if v > 0 else TEAL for v in rdf.impact]))
        bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="← lowers risk · raises risk →")
        st.plotly_chart(bar, use_container_width=True)
        st.caption("Exact per-prediction contributions (TreeSHAP) translated into plain language.")
    st.markdown("#### Twin physiology (learned)")
    zp = svc.zpar[svc.zpar.patient_id == pid].sort_values("day")
    zp = zp[zp.day <= minute // 1440]
    tf = go.Figure()
    tf.add_trace(go.Scatter(x=zp.day, y=zp.si_rel, mode="lines+markers", name="Insulin sensitivity ×EHR",
                            line=dict(color=BLUE)))
    tf.add_trace(go.Scatter(x=zp.day, y=zp.meal_gain, mode="lines+markers", name="Meal response ×",
                            line=dict(color=AMBER)))
    tf.add_hline(y=1.0, line=dict(color=GREY, dash="dot"))
    tf.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Monitoring day",
                     legend=dict(orientation="h", y=1.15))
    st.plotly_chart(tf, use_container_width=True)
    st.caption(f"Re-learned every midnight from the previous day's CGM. Basal glucose now {tw['basal_glucose']:.0f} "
               f"mg/dL (EHR fasting value {tw['ehr_prior_basal_glucose']:.0f}).")

# ---------------------------------------------------------------- EHR + daily patterns
st.divider()
a, b, c = st.columns([1.2, 1.2, 1])
with a:
    st.markdown("#### Electronic health record")
    dx = [f"{CONDITIONS[k][1]} ({CONDITIONS[k][0]})" for k in CONDITIONS if k == "t2d" or bool(e.get(k))]
    chips(dx)
    meds = {"metformin": "Metformin 500 mg BD", "sulfonylurea": "Glimepiride 2 mg OD", "dpp4": "Teneligliptin 20 mg",
            "sglt2": "Dapagliflozin 10 mg", "statin": "Atorvastatin 20 mg", "arb": "Telmisartan 40 mg"}
    m_list = [v for k_, v in meds.items() if bool(e[k_])]
    if bool(e.insulin):
        m_list.append("Premixed insulin 30/70 BD" if e.insulin_regimen == "premix_30_70" else "Insulin glargine OD")
    st.markdown("**Medications:** " + ", ".join(m_list))
    st.markdown(f"**Genetics:** TCF7L2 rs7903146 risk alleles {e.tcf7l2_rs7903146_risk_alleles} · "
                f"KCNQ1 rs2237892 {e.kcnq1_rs2237892_risk_alleles} · polygenic risk z {e.polygenic_risk_z:+.2f} · "
                f"family history {'yes' if e.family_history_t2d else 'no'}")
    short = {"hba1c": "HbA1c", "fpg": "Fasting glucose", "fasting_insulin": "Fasting insulin", "ldl": "LDL-C",
             "hdl": "HDL-C", "triglycerides": "Triglycerides", "egfr": "eGFR", "uacr": "Urine ACR", "sbp": "Systolic BP",
             "dbp": "Diastolic BP", "bmi": "BMI", "waist_cm": "Waist"}
    st.dataframe(pd.DataFrame({"Test": list(short.values()), "Value": [f"{e[x]} {LOINC[x][2]}" for x in short],
                               "LOINC": [LOINC[x][0] for x in short]}),
                 hide_index=True, use_container_width=True, height=300)
with b:
    st.markdown("#### HbA1c trend (2 years)")
    lh = svc.labs[svc.labs.patient_id == pid]
    lf = go.Figure(go.Scatter(x=pd.to_datetime(lh.date), y=lh.hba1c, mode="lines+markers", line=dict(color=BLUE)))
    lf.add_hline(y=7.0, line=dict(color=TEAL, dash="dot"), annotation_text="target 7%")
    lf.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
    st.plotly_chart(lf, use_container_width=True)
    st.markdown("#### 7-day glucose profile")
    wk = svc.ts[(svc.ts.patient_id == pid) & (svc.ts.minute <= minute) & (svc.ts.minute > minute - 7 * 1440)].copy()
    wk["hod"] = (wk.minute % 1440) // 30 * 0.5
    q = wk.groupby("hod").cgm.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    pf = go.Figure()
    pf.add_trace(go.Scatter(x=q.index, y=q[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    pf.add_trace(go.Scatter(x=q.index, y=q[0.1], fill="tonexty", fillcolor="rgba(37,99,235,.12)", line=dict(width=0),
                            name="10–90%"))
    pf.add_trace(go.Scatter(x=q.index, y=q[0.75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    pf.add_trace(go.Scatter(x=q.index, y=q[0.25], fill="tonexty", fillcolor="rgba(37,99,235,.28)", line=dict(width=0),
                            name="25–75%"))
    pf.add_trace(go.Scatter(x=q.index, y=q[0.5], line=dict(color=BLUE, width=2), name="median"))
    pf.add_hrect(y0=70, y1=180, fillcolor=TEAL, opacity=0.06, line_width=0)
    pf.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="hour of day",
                     legend=dict(orientation="h", y=1.15))
    st.plotly_chart(pf, use_container_width=True)
with c:
    st.markdown("#### Time in ranges (7 d)")
    g = wk.cgm.dropna()
    bins = {"Very high >250": (g > 250).mean(), "High 181–250": ((g > 180) & (g <= 250)).mean(),
            "In range 70–180": ((g >= 70) & (g <= 180)).mean(), "Low <70": (g < 70).mean()}
    colors = [RED, AMBER, TEAL, "#7c3aed"]
    tf2 = go.Figure()
    for (lab, v), cc in zip(bins.items(), colors):
        tf2.add_trace(go.Bar(x=["7 days"], y=[v * 100], name=f"{lab}: {v:.0%}", marker_color=cc))
    tf2.update_layout(barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%",
                      legend=dict(y=0.5))
    st.plotly_chart(tf2, use_container_width=True)
    gmi = 3.31 + 0.02392 * g.mean() if len(g) else np.nan
    st.markdown(f"**Mean glucose** {g.mean():.0f} mg/dL · **GMI** {gmi:.1f}% · **CV** {g.std() / g.mean():.0%}")
    st.caption("International consensus targets: TIR > 70%, TBR < 4%, CV ≤ 36%.")
