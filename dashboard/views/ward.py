"""Ward overview: every monitored patient, ranked by 2-hour spike risk."""
import pandas as pd
import streamlit as st

from common import STATUS_COLOR, STATUS_LABEL, card, fmt, setup, trend_arrow

svc = setup("Ward overview")
minute = st.session_state.minute
ward = svc.ward(minute)

st.title("🏥 Ward overview")
st.caption(f"Remote monitoring panel · {svc.minute_to_ts(minute).strftime('%A %d %b %Y, %H:%M')} · "
           "every patient has a digital twin that re-calibrates nightly")

c = st.columns(5)
with c[0]:
    card("Patients monitored", len(ward), "CGM + wearable + EHR")
with c[1]:
    card("Spike alerts", int((ward.status == "alert").sum()), "predicted > 180 mg/dL within 2 h", "#d97706")
with c[2]:
    card("Above 180 now", int((ward.status == "high now").sum()), "already hyperglycaemic", "#dc2626")
with c[3]:
    card("Below 70 now", int((ward.status == "low now").sum()), "hypoglycaemia", "#dc2626")
with c[4]:
    card("Median time in range", fmt(ward.tir_7d.median() * 100, "{:.0f}%"), "70–180 mg/dL, last 7 days", "#0f766e")

st.write("")
st.subheader("Patients by priority")
st.caption("Sorted by status then predicted risk. Pick a patient below to open their twin.")

view = pd.DataFrame({
    "Status": [STATUS_LABEL[s] for s in ward.status],
    "Patient": [f"{r.patient_id} · {r.name}" for r in ward.itertuples()],
    "Age/Sex": [f"{r.age}{r.sex}" for r in ward.itertuples()],
    "City": ward.city,
    "Glucose": [f"{fmt(g)} {trend_arrow(d)}" for g, d in zip(ward.glucose_now, ward.trend_30min)],
    "2-h spike risk": (ward.spike_risk.fillna(0).values * 100).round(0),
    "Forecast 2 h": ward.forecast_2h.round(0),
    "HbA1c %": ward.hba1c,
    "TIR 7 d": (ward.tir_7d.values * 100).round(0),
    "Main driver": ward.top_reason,
})


def _color(v):
    for k, lab in STATUS_LABEL.items():
        if v == lab:
            return f"color: {STATUS_COLOR[k]}; font-weight: 600"
    return ""


st.dataframe(
    view.style.map(_color, subset=["Status"]),
    hide_index=True, use_container_width=True, height=560,
    column_config={
        "2-h spike risk": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
        "TIR 7 d": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
        "Forecast 2 h": st.column_config.NumberColumn(format="%d mg/dL"),
        "HbA1c %": st.column_config.NumberColumn(format="%.1f"),
    },
)
st.caption("Risk and TIR bars run from 0 to 100%. Alert threshold is tuned to catch 80% of spikes on validation patients.")

c1, c2 = st.columns([3, 1])
with c1:
    pick = st.selectbox("Open patient twin", ward.patient_id,
                        format_func=lambda p: f"{p} · {ward.set_index('patient_id').loc[p, 'name']}")
with c2:
    st.write("")
    st.write("")
    if st.button("Open twin →", type="primary", use_container_width=True):
        st.session_state.pid = pick
        st.switch_page("views/patient.py")
