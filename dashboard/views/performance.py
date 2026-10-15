"""Model performance on held-out patients (synthetic Indian cohort + real ShanghaiT2DM)."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BLUE, GREY, TEAL, card, setup
from glucotwin import config as C
from glucotwin.models.metrics import clarke_zones

setup("Performance")
st.title("📊 Model performance")
st.caption("Every number is measured on patients the models never saw during training (patient-level split).")

COHORTS = {"synthetic": "Synthetic Indian cohort (EHR + CGM + wearables)",
           "shanghai": "Real-world ShanghaiT2DM (EHR + CGM + meal/insulin logs)"}
avail = [c for c in COHORTS if (C.RESULTS_DIR / f"{c}_results.json").exists()]
if not avail:
    st.info("Run `python scripts/train_evaluate.py` to generate results.")
    st.stop()

tabs = st.tabs([COHORTS[c] for c in avail])
for tab, cohort in zip(tabs, avail):
    with tab:
        R = C.RESULTS_DIR
        res = json.loads((R / f"{cohort}_results.json").read_text())
        clf = pd.read_csv(R / f"{cohort}_classification.csv")
        reg = pd.read_csv(R / f"{cohort}_regression.csv")
        ev = pd.read_csv(R / f"{cohort}_events.csv")
        streams = pd.read_csv(R / f"{cohort}_shap_streams.csv")
        feats = pd.read_csv(R / f"{cohort}_shap_features.csv")
        final = "LightGBM: GlucoTwin (fusion + digital twin)"
        fc, fe = clf[clf.model == final].iloc[0], ev[ev.model == final].iloc[0]
        fr = reg[(reg.model == final)].set_index("horizon")
        p = res["patients"]
        who = (f"train {p['train']} · validation {p['val']} · test {p['test']} patients" if "train" in p else
               f"{p['people']} people / {p['total_recordings']} recordings")
        st.caption(f"Evaluation: {res['evaluation']} · {who} · {res['rows_evaluated']:,} time points · "
                   f"sampling every {res['step_min']} min")
        k = st.columns(6)
        with k[0]:
            card("AUROC (spike in 2 h)", f"{fc.auroc:.3f}", f"AUPRC {fc.auprc:.3f} · prevalence {fc.prevalence:.0%}", TEAL)
        with k[1]:
            card("Spikes caught early", f"{fe.event_recall:.0%}", f"≥15 min ahead · {int(fe.events)} spike events", TEAL)
        with k[2]:
            card("Median warning time", f"{fe.median_lead_min:.0f} min", "before glucose crosses 180", AMBER)
        with k[3]:
            card("False alerts", f"{fe.false_alerts_per_day:.1f}/day", f"{fe.alerts_per_day:.1f} alerts/day total", GREY)
        with k[4]:
            card("Forecast error 60 min", f"{fr.loc[60, 'rmse']:.1f}", f"RMSE mg/dL · MARD {fr.loc[60, 'mard']:.1f}%", BLUE)
        with k[5]:
            card("Clarke zones A+B (60 min)", f"{fr.loc[60, 'clarke_AB']:.1f}%", f"zone A {fr.loc[60, 'clarke_A']:.1f}%", BLUE)

        st.write("")
        a, b = st.columns(2)
        with a:
            st.markdown("##### Does fusing the data streams help? (ablation)")
            cc = clf.copy()
            cc["short"] = cc.model.str.replace("LightGBM: ", "")
            fig = go.Figure(go.Bar(x=cc.auroc, y=cc.short, orientation="h", text=cc.auroc.round(3),
                                   marker_color=[TEAL if "GlucoTwin" in m else BLUE if "LightGBM" in m else GREY
                                                 for m in cc.model]))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0.5, 1.0], title="AUROC"),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.markdown("##### Glucose forecast error by horizon (RMSE, mg/dL)")
            pv = reg.pivot_table(index="model", columns="horizon", values="rmse").sort_values(120)
            fig = go.Figure()
            for h, col in zip((30, 60, 120), (BLUE, AMBER, TEAL)):
                fig.add_trace(go.Bar(y=pv.index.str.replace("LightGBM: ", ""), x=pv[h], name=f"{h} min",
                                     orientation="h", marker_color=col))
            fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

        c, d = st.columns(2)
        with c:
            st.markdown("##### Which data stream drives the predictions? (SHAP share)")
            fig = go.Figure(go.Bar(x=streams.share * 100, y=streams.stream, orientation="h",
                                   text=(streams.share * 100).round(1).astype(str) + "%", marker_color=TEAL))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="% of total |SHAP|",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(feats.head(12)[["feature", "stream", "mean_abs_shap"]], hide_index=True, use_container_width=True)
        with d:
            st.markdown("##### Clarke error grid (60-min forecasts, test patients)")
            pr = pd.read_parquet(R / f"{cohort}_test_predictions.parquet", columns=["y_g60", "pred_g60"]).dropna()
            pr = pr.sample(min(6000, len(pr)), random_state=0)
            z = clarke_zones(pr.y_g60.values, pr.pred_g60.values)
            fig = go.Figure()
            for zone, col in zip("ABCDE", (TEAL, BLUE, AMBER, "#dc2626", "#7c3aed")):
                m = z == zone
                if m.any():
                    fig.add_trace(go.Scattergl(x=pr.y_g60[m], y=pr.pred_g60[m], mode="markers", name=f"Zone {zone}",
                                               marker=dict(size=3, color=col, opacity=0.5)))
            fig.add_trace(go.Scatter(x=[40, 400], y=[40, 400], mode="lines", line=dict(color=GREY, dash="dot"),
                                     showlegend=False))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="actual CGM (mg/dL)",
                              yaxis_title="predicted (mg/dL)", xaxis=dict(range=[40, 400]), yaxis=dict(range=[40, 400]))
            st.plotly_chart(fig, use_container_width=True)

        tf = R / f"{cohort}_alert_tradeoff.csv"
        if tf.exists():
            st.markdown("##### Alert burden vs spikes caught (choose the operating point)")
            tr = pd.read_csv(tf).sort_values("false_alerts_per_day")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tr.false_alerts_per_day, y=tr.event_recall * 100, mode="lines+markers+text",
                                     text=[f"{v:.0f} min" for v in tr.median_lead_min], textposition="top left",
                                     line=dict(color=TEAL), name="GlucoTwin"))
            d0 = tr.iloc[(tr.threshold_offset.abs()).argmin()]
            fig.add_trace(go.Scatter(x=[d0.false_alerts_per_day], y=[d0.event_recall * 100], mode="markers",
                                     marker=dict(size=14, color=AMBER, symbol="star"), name="default threshold"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="false alerts per patient-day",
                              yaxis_title="% of spikes flagged ≥15 min ahead", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Labels show the median warning time. Repeat notifications are snoozed for 60 minutes.")

        st.markdown("##### Full tables")
        t1, t2, t3 = st.tabs(["Spike classification", "Event-level alerts", "Glucose forecasting"])
        with t1:
            st.dataframe(clf.round(3), hide_index=True, use_container_width=True)
        with t2:
            st.dataframe(ev.round(3), hide_index=True, use_container_width=True)
        with t3:
            st.dataframe(reg.round(2), hide_index=True, use_container_width=True)
        conf = res.get("conformal", {})
        if conf:
            st.caption("80% prediction intervals (split-conformal, calibrated on validation patients): test coverage " +
                       ", ".join(f"{h} min {v['test_coverage_80']:.0%}" for h, v in conf.items()))
