"""Shared state, styling and helpers for the GlucoTwin doctor dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glucotwin import config as C  # noqa: E402
from glucotwin.inference import GlucoTwinService  # noqa: E402

TEAL, AMBER, RED, BLUE, GREY = "#0f766e", "#d97706", "#dc2626", "#2563eb", "#64748b"
STATUS_COLOR = {"alert": AMBER, "high now": RED, "low now": RED, "stable": TEAL}
STATUS_LABEL = {"alert": "Spike predicted", "high now": "Above 180 now", "low now": "Below 70 now",
                "stable": "Stable"}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.gt-card {border: 1px solid rgba(100,116,139,.25); border-radius: 12px; padding: 14px 16px; height: 100%;}
.gt-card .lbl {font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7;}
.gt-card .val {font-size: 1.7rem; font-weight: 650; line-height: 1.2; margin-top: 2px;}
.gt-card .sub {font-size: .85rem; opacity: .75;}
.gt-chip {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; margin: 2px 4px 2px 0;
          border: 1px solid rgba(100,116,139,.35);}
.gt-banner {border-radius: 12px; padding: 14px 18px; margin: 6px 0 12px 0; border-left: 6px solid;}
.gt-synth {background: rgba(37,99,235,.12); color: #2563eb; border: 1px solid rgba(37,99,235,.35);}
</style>
"""


@st.cache_resource(show_spinner="Loading digital twins and models...")
def service() -> GlucoTwinService:
    return GlucoTwinService()


def setup(title: str):
    st.markdown(CSS, unsafe_allow_html=True)
    svc = service()
    if "minute" not in st.session_state:
        lo = min(svc.time_range(p)[0] for p in svc.patient_ids)
        st.session_state.minute = lo + 2 * 1440 + 14 * 60 + 30      # start the demo at 14:30 on day 5
    if "pid" not in st.session_state:
        st.session_state.pid = svc.ward(st.session_state.minute).patient_id.iloc[0]
    sidebar_clock(svc)
    return svc


def sidebar_clock(svc):
    lo = min(svc.time_range(p)[0] for p in svc.patient_ids)
    hi = min(svc.time_range(p)[1] for p in svc.patient_ids)
    with st.sidebar:
        st.markdown("### 🕒 Virtual clock")
        st.caption("All patients are synthetic. Move the clock or press play to replay their live data.")
        day0 = lo // 1440
        n_days = hi // 1440 - day0 + 1
        cur = st.session_state.minute
        day = st.select_slider("Day", options=list(range(day0, day0 + n_days)), value=cur // 1440,
                               format_func=lambda d: svc.minute_to_ts(d * 1440).strftime("%a %d %b"))
        times = list(range(0, 1440, 15))
        tod = st.select_slider("Time", options=times, value=(cur % 1440) // 15 * 15,
                               format_func=lambda m: f"{m // 60:02d}:{m % 60:02d}")
        st.session_state.minute = int(np.clip(day * 1440 + tod, lo, hi))
        st.markdown(f"**{svc.minute_to_ts(st.session_state.minute).strftime('%A %d %b %Y, %H:%M')}**")
        play = st.toggle("▶ Live replay (15 min / 2 s)", key="play")
    if play:
        _ticker(hi)


@st.fragment(run_every=2.0)
def _ticker(hi):
    if st.session_state.get("play") and st.session_state.minute + 15 <= hi:
        st.session_state.minute += 15
        st.rerun()


def card(label, value, sub="", color=None):
    style = f"color:{color};" if color else ""
    st.markdown(f"<div class='gt-card'><div class='lbl'>{label}</div><div class='val' style='{style}'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def chips(items):
    st.markdown("".join(f"<span class='gt-chip'>{i}</span>" for i in items), unsafe_allow_html=True)


def trend_arrow(d30):
    if d30 is None or np.isnan(d30):
        return "–"
    r = d30 / 30.0
    return "⇈" if r > 2 else "↑" if r > 1 else "↗" if r > 0.5 else "⇊" if r < -2 else "↓" if r < -1 else "↘" if r < -0.5 else "→"


def fmt(v, f="{:.0f}", none="–"):
    return none if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def risk_txt(p):
    return ">99%" if p >= 0.995 else "<1%" if p < 0.005 else f"{p:.0%}"


def select_patient(svc, key="pid_select"):
    ward = svc.ward(st.session_state.minute)
    labels = {r.patient_id: f"{r.patient_id} · {r.name} ({r.age}{r.sex})" for r in ward.itertuples()}
    ids = list(labels)
    idx = ids.index(st.session_state.pid) if st.session_state.pid in ids else 0
    pid = st.selectbox("Patient", ids, index=idx, format_func=lambda p: labels[p], key=key)
    st.session_state.pid = pid
    return pid


HORIZONS = C.HORIZONS
