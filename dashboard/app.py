"""ThaliTwin doctor dashboard.

Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="ThaliTwin – Diabetes Digital Twin", page_icon="🩺", layout="wide")

pages = [
    st.Page("views/ward.py", title="Ward overview", icon="🏥", default=True),
    st.Page("views/patient.py", title="Patient twin", icon="🧬"),
    st.Page("views/whatif.py", title="What-if simulator", icon="🧪"),
    st.Page("views/performance.py", title="Model performance", icon="📊"),
    st.Page("views/about.py", title="About & data", icon="ℹ️"),
]
st.navigation(pages).run()
