"""Step 4: figures for the README, deck and architecture PDF (docs/figures/, docs/*.pdf)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from glucotwin import config as C  # noqa: E402

TEAL, AMBER, RED, BLUE, GREY, INK, PURPLE = "#0f766e", "#d97706", "#dc2626", "#2563eb", "#64748b", "#0f172a", "#7c3aed"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})
F = C.FIG_DIR


def box(ax, x, y, w, h, title, lines, color, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=color + "14", ec=color, lw=1.6))
    ax.text(x + 0.012, y + h - 0.022, title, fontsize=fs + 1.4, fontweight="bold", color=color, va="top")
    ax.text(x + 0.012, y + h - 0.058, "\n".join(lines), fontsize=fs, color=INK, va="top", linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color))


def architecture():
    fig = plt.figure(figsize=(17, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "GlucoTwin – Digital Twin for Type-2 Diabetes: system architecture", ha="center",
            fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.932, "Team Gym Bro · IIIT Delhi · predicts glucose spikes > 180 mg/dL up to 2 h ahead by fusing EHR + wearable streams",
            ha="center", fontsize=10.5, color=GREY)
    cols = [(0.015, "1  DATA SOURCES"), (0.215, "2  INGEST & FEATURES"), (0.415, "3  DIGITAL TWIN ENGINE"),
            (0.635, "4  AI / ML LAYER"), (0.83, "5  CLINICIAN DELIVERY")]
    for x, t in cols:
        ax.text(x + 0.005, 0.895, t, fontsize=10, fontweight="bold", color=GREY)
    W = 0.18
    box(ax, 0.015, 0.64, W, 0.235, "Static: EHR (synthetic)",
        ["Synthea-style Python generator", "FHIR R4 bundles · 200 Indian patients", "Demographics, ICD-10 diagnoses",
         "LOINC labs: HbA1c, FPG, insulin,", "  lipids, eGFR, BP · 2-yr history", "Medicines (metformin, SU, insulin)",
         "Genetics: TCF7L2, KCNQ1, PRS"], BLUE)
    box(ax, 0.015, 0.375, W, 0.245, "Dynamic: wearables + CGM",
        ["CGM every 5 min (sensor lag, noise,", "  drop-outs) · steps · heart rate", "HRV (RMSSD) · sleep stages",
         "Simulated from a physiological", "  model + Indian daily routines:", "  regional meals, walks, stress,",
         "  festival sweets, missed doses"], TEAL)
    box(ax, 0.015, 0.215, W, 0.14, "Patient app logs",
        ["Meals from 35-dish Indian menu", "  (carbs, GI) · medicine doses", "Logged imperfectly (as in real life)"], AMBER)
    box(ax, 0.015, 0.06, W, 0.135, "Real-world validation",
        ["ShanghaiT2DM (Sci Data 2023,", "  CC BY 4.0): 100 people, 15-min", "  CGM, diet + insulin logs, labs"], PURPLE)

    box(ax, 0.215, 0.52, W, 0.355, "Ingestion & alignment",
        ["FHIR parser -> EHR feature vector", "Streams aligned on a 5/15-min grid",
         "Sleep -> nightly summary per day", "Steps -> activity intensity", "Logged meals -> carb appearance",
         "  (2-compartment gut model, GI)", "Insulin -> s.c. absorption model", "Missed metformin -> hepatic output"], BLUE)
    box(ax, 0.215, 0.18, W, 0.32, "Feature engineering (t = now)",
        ["23 CGM: lags, slopes, variability,", "  time above range, time of day", "12 logs: carbs on board, IOB, doses",
         "17 wearables: steps, HR vs rest,", "  HRV, sleep debt, efficiency", "24 EHR: HbA1c, HOMA-IR, eGFR, genes",
         "18 twin: simulated 30/60/120 min,", "  peak, insulin sensitivity today"], BLUE)

    box(ax, 0.415, 0.69, 0.2, 0.185, "a. Personal physiology",
        ["Bergman minimal model + insulin", "  secretion, gut absorption, exercise", "  uptake, sleep -> insulin sensitivity,",
         "  dawn phenomenon", "Prior from EHR (HOMA-IR, HOMA-B)"], TEAL)
    box(ax, 0.415, 0.515, 0.2, 0.155, "b. Nightly calibration",
        ["Cross-entropy search re-learns", "  insulin sensitivity, meal gain,", "  basal glucose, dawn effect",
         "  (regularised to EHR prior)"], TEAL)
    box(ax, 0.415, 0.355, 0.2, 0.14, "c. Live synchronisation",
        ["Observer nudges twin to each CGM", "Disturbance state learns unexplained", "  inputs (unlogged snacks, stress)"], TEAL)
    box(ax, 0.415, 0.18, 0.2, 0.155, "d. Forecast + what-if",
        ["Simulates next 2-6 h from now", "Learns routine -> expected meals", "What-if: meal, walk, sleep,",
         "  missed metformin, portion size"], TEAL)

    box(ax, 0.635, 0.64, 0.175, 0.235, "Spike classifier",
        ["LightGBM · P(> 180 mg/dL in 2 h)", "Threshold tuned on validation", "  patients for 80% sensitivity",
         "Event-level: recall, lead time,", "  false alerts / day"], AMBER)
    box(ax, 0.635, 0.42, 0.175, 0.2, "Glucose forecasters",
        ["LightGBM 30 / 60 / 120 min", "Split-conformal 80% intervals", "Clarke error grid evaluation"], AMBER)
    box(ax, 0.635, 0.25, 0.175, 0.15, "Explainability",
        ["Exact TreeSHAP per prediction", "-> plain-language reasons", "Stream-level attribution"], AMBER)
    box(ax, 0.635, 0.06, 0.175, 0.17, "Benchmarks / ablation",
        ["Persistence · trend · twin-only", "CGM / +EHR / +wearables /", "  fusion / fusion + twin",
         "Deep learning: GRU fusion"], AMBER)

    box(ax, 0.83, 0.56, 0.155, 0.315, "Doctor dashboard",
        ["Streamlit + Plotly", "Ward view ranked by risk", "Patient twin: CGM, forecast", "  band, alert + reasons,",
         "  wearables, EHR, labs, AGP", "What-if simulator", "Live replay mode", "Model performance"], RED)
    box(ax, 0.83, 0.33, 0.155, 0.205, "REST API (FastAPI)",
        ["/patients · /state", "/twin · /what-if", "Integrates with EHR (FHIR),", "  CGM / wearable apps"], RED)
    box(ax, 0.83, 0.06, 0.155, 0.245, "Governance",
        ["Synthetic + anonymised data", "  only (DPDP Act 2023)", "Patient-level train/val/test", "No PII in features",
         "Not a medical device:", "  decision support only"], GREY)

    for y in (0.75, 0.58, 0.29, 0.188):                        # sources -> ingest / features
        arrow(ax, 0.197, y, 0.213, y)
    arrow(ax, 0.305, 0.52, 0.305, 0.502, BLUE)                # ingest -> features
    arrow(ax, 0.397, 0.78, 0.413, 0.78)                       # EHR prior -> physiology
    arrow(ax, 0.397, 0.42, 0.413, 0.42)                       # streams -> synchronisation
    for y0, y1 in ((0.69, 0.672), (0.515, 0.497), (0.355, 0.337)):
        arrow(ax, 0.515, y0, 0.515, y1, TEAL)
    arrow(ax, 0.617, 0.78, 0.633, 0.78)                       # twin features -> classifier
    arrow(ax, 0.617, 0.55, 0.633, 0.55)                       # twin features -> forecasters
    arrow(ax, 0.812, 0.75, 0.828, 0.75)                       # -> dashboard
    arrow(ax, 0.812, 0.45, 0.828, 0.45)                       # -> API
    ax.text(0.515, 0.125, "twin simulations become features for the\nML layer and power the what-if tool",
            fontsize=8, color=TEAL, ha="center", style="italic")
    fig.savefig(F / "architecture.png", dpi=170)
    fig.savefig(C.ROOT / "docs" / "architecture.pdf")
    plt.close(fig)


def _res(cohort):
    R = C.RESULTS_DIR
    if not (R / f"{cohort}_results.json").exists():
        return None
    return {k: pd.read_csv(R / f"{cohort}_{k}.csv") for k in ("classification", "regression", "events", "shap_streams",
                                                              "shap_features")}


def results_figures(cohort, label):
    r = _res(cohort)
    if r is None:
        return
    clf = r["classification"].copy()
    clf["short"] = clf.model.str.replace("LightGBM: ", "").str.replace("Deep learning: ", "")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cols = [TEAL if "GlucoTwin" in m else BLUE if "LightGBM" in m else PURPLE if "GRU" in m else GREY for m in clf.model]
    ax.barh(clf.short, clf.auroc, color=cols)
    for i, v in enumerate(clf.auroc):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC – new spike > 180 mg/dL within 2 h (held-out patients)")
    ax.set_title(f"Ablation: value of each data stream · {label}")
    fig.tight_layout()
    fig.savefig(F / f"{cohort}_ablation_auroc.png", dpi=170)
    plt.close(fig)

    reg = r["regression"]
    pv = reg.pivot_table(index="model", columns="horizon", values="rmse").sort_values(120, ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(pv))
    for j, (h, col) in enumerate(zip((30, 60, 120), (BLUE, AMBER, TEAL))):
        ax.barh(y + (j - 1) * 0.27, pv[h], height=0.27, color=col, label=f"{h} min")
    ax.set_yticks(y, pv.index.str.replace("LightGBM: ", "").str.replace("Deep learning: ", ""))
    ax.set_xlabel("RMSE (mg/dL), lower is better")
    ax.set_title(f"Glucose forecast error by horizon · {label}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(F / f"{cohort}_rmse.png", dpi=170)
    plt.close(fig)

    st = r["shap_streams"]
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.barh(st.stream, st.share * 100, color=TEAL)
    for i, v in enumerate(st.share * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("share of total |SHAP| for spike alerts")
    ax.set_title(f"Which stream drives predictions? · {label}")
    fig.tight_layout()
    fig.savefig(F / f"{cohort}_shap_streams.png", dpi=170)
    plt.close(fig)

    pr = pd.read_parquet(C.RESULTS_DIR / f"{cohort}_test_predictions.parquet")
    from glucotwin.models.metrics import clarke_zones
    d = pr[["y_g60", "pred_g60"]].dropna().sample(min(8000, len(pr)), random_state=0)
    z = clarke_zones(d.y_g60.values, d.pred_g60.values)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for zone, col in zip("ABCDE", (TEAL, BLUE, AMBER, RED, PURPLE)):
        m = z == zone
        ax.scatter(d.y_g60[m], d.pred_g60[m], s=3, alpha=0.4, color=col, label=f"{zone}: {m.mean():.1%}")
    ax.plot([40, 400], [40, 400], color=GREY, ls=":")
    ax.set_xlim(40, 400)
    ax.set_ylim(40, 400)
    ax.set_xlabel("actual CGM in 60 min (mg/dL)")
    ax.set_ylabel("GlucoTwin forecast (mg/dL)")
    ax.set_title(f"Clarke error grid, 60-min · {label}")
    ax.legend(frameon=False, markerscale=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(F / f"{cohort}_clarke.png", dpi=170)
    plt.close(fig)

    # lead time distribution for the deployed model
    leads = []
    for _, g in pr.groupby("patient_id"):
        g = g.sort_values("minute")
        p = g.alert_score.values.copy()
        p[(g.cgm_now.values >= C.SPIKE_MGDL) | np.isnan(g.cgm_now.values)] = np.nan
        a = p >= 0
        t = g.minute.values
        for o in t[g.spike_onset.values == 1]:
            w = (t >= o - 120) & (t <= o - 15) & a
            if w.any():
                leads.append(o - t[w][0])
    if leads:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(leads, bins=np.arange(15, 125, 10), color=AMBER, edgecolor="white")
        ax.axvline(np.median(leads), color=INK, ls="--")
        ax.text(np.median(leads) + 2, ax.get_ylim()[1] * 0.9, f"median {np.median(leads):.0f} min", fontsize=9)
        ax.set_xlabel("warning time before glucose crossed 180 mg/dL (min)")
        ax.set_ylabel("spike events")
        ax.set_title(f"How early are spikes flagged? · {label}")
        fig.tight_layout()
        fig.savefig(F / f"{cohort}_lead_time.png", dpi=170)
        plt.close(fig)


def example_patient():
    pr_path = C.RESULTS_DIR / "synthetic_test_predictions.parquet"
    if not pr_path.exists():
        return
    pr = pd.read_parquet(pr_path)
    meta = json.loads((C.MODELS_DIR / "synthetic" / "meta.json").read_text())
    ts = pd.read_parquet(C.SYN_DIR / "timeseries.parquet")
    meals = pd.read_csv(C.SYN_DIR / "meals.csv")
    # pick the test patient-day with the most onsets for an illustrative trace
    pr["day"] = pr.minute // 1440
    best = pr.groupby(["patient_id", "day"]).spike_onset.sum().sort_values().index[-1]
    pid, day = best
    lo, hi = day * 1440 + 360, day * 1440 + 1440
    g = pr[(pr.patient_id == pid) & (pr.minute >= lo) & (pr.minute < hi)].sort_values("minute")
    t = ts[(ts.patient_id == pid) & (ts.minute >= lo) & (ts.minute < hi)]
    m = meals[(meals.patient_id == pid) & (meals.minute >= lo) & (meals.minute < hi)]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2, 1]))
    hrs = lambda x: (np.asarray(x) % 1440) / 60
    a1.axhspan(70, 180, color=TEAL, alpha=0.07)
    a1.axhline(180, color=AMBER, ls=":", lw=1)
    a1.plot(hrs(g.minute), g.cgm_now, color=BLUE, lw=2, label="CGM")
    a1.plot(hrs(g.minute + 60), g.pred_g60, color=AMBER, lw=1.3, alpha=0.9, label="GlucoTwin forecast made 60 min earlier")
    for r in m.itertuples():
        a1.annotate(f"{r.meal_name}\n{r.carbs_true:.0f} g", (hrs(r.minute), 60), fontsize=7.5, ha="center", color=TEAL)
        a1.axvline(hrs(r.minute), color=TEAL, alpha=0.25)
    a1.set_ylabel("mg/dL")
    a1.legend(frameon=False, loc="upper left", fontsize=8.5)
    a1.set_title(f"A day in the life of synthetic test patient {pid}: forecasts and early alerts")
    p = g.prob_spike.where(g.cgm_now < 180)
    a2.fill_between(hrs(g.minute), 0, p * 100, color=AMBER, alpha=0.3)
    a2.plot(hrs(g.minute), p * 100, color=AMBER, lw=1)
    a2.axhline(meta["threshold"] * 100, color=AMBER, ls=":")
    on = g[g.spike_onset == 1]
    a2.scatter(hrs(on.minute), np.full(len(on), 95), marker="v", color=RED, s=30, label="spike onset")
    a2.set_ylabel("2-h risk %")
    a2.set_ylim(0, 100)
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    a3.bar(hrs(t.minute), t.steps, width=5 / 60, color=TEAL)
    a3.set_ylabel("steps/5 min")
    a3.set_xlabel("hour of day")
    fig.tight_layout()
    fig.savefig(F / "example_patient_day.png", dpi=170)
    plt.close(fig)


def twin_learning():
    zp_path, tp_path = C.PROC_DIR / "twin_params_synthetic.csv", C.SYN_DIR / "true_params.csv"
    if not (zp_path.exists() and tp_path.exists()):
        return
    zp, tp = pd.read_csv(zp_path), pd.read_csv(tp_path)
    d = zp.merge(tp[["patient_id", "SI", "Gb"]], on="patient_id")
    rows = []
    for day, g in d.groupby("day"):
        rows.append({"day": day, "si_corr": np.corrcoef(np.log(g.si_abs), np.log(g.SI))[0, 1],
                     "si_err": np.median(np.abs(np.log(g.si_abs / g.SI))) * 100,
                     "gb_err": np.median(np.abs(g.gb - g.Gb))})
    r = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.plot(r.day, r.si_corr, "o-", color=BLUE)
    a1.set_xlabel("monitoring day (day 0 = EHR prior only)")
    a1.set_ylabel("correlation with truth")
    a1.set_title("Twin's insulin-sensitivity estimate vs hidden truth")
    a2.plot(r.day, r.si_err, "o-", color=TEAL)
    a2.set_xlabel("monitoring day (day 0 = EHR prior only)")
    a2.set_ylabel("median absolute error (%)")
    a2.set_title("Error of the personalised estimate")
    fig.tight_layout()
    fig.savefig(F / "twin_learning.png", dpi=170)
    plt.close(fig)
    r.to_csv(C.RESULTS_DIR / "twin_learning.csv", index=False)


def data_streams():
    p = C.SYN_DIR / "timeseries.parquet"
    if not p.exists():
        return
    ts = pd.read_parquet(p, filters=[("patient_id", "==", "GT-0007")])
    g = ts[(ts.minute >= 3 * 1440) & (ts.minute < 5 * 1440)]
    h = g.minute / 60 - 72
    fig, ax = plt.subplots(5, 1, figsize=(11, 7.4), sharex=True)
    ax[0].plot(h, g.cgm, color=BLUE)
    ax[0].axhline(180, color=AMBER, ls=":")
    ax[0].set_ylabel("CGM\nmg/dL")
    ax[1].bar(h, g.steps, width=5 / 60, color=TEAL)
    ax[1].set_ylabel("steps")
    ax[2].plot(h, g.heart_rate, color=RED, lw=0.9)
    ax[2].set_ylabel("HR bpm")
    ax[3].plot(h, g.hrv_rmssd, color=PURPLE, lw=0.8)
    ax[3].set_ylabel("HRV ms")
    st = g.sleep_stage.where(g.sleep_stage >= 0)
    ax[4].step(h, st.map({0: 3, 3: 2, 1: 1, 2: 0}), color=INK, lw=1)
    ax[4].set_yticks([0, 1, 2, 3], ["deep", "light", "REM", "awake"])
    ax[4].set_xlabel("hours")
    ax[0].set_title("Dynamic data streams for one synthetic patient (48 h, 5-min resolution)")
    fig.tight_layout()
    fig.savefig(F / "data_streams.png", dpi=170)
    plt.close(fig)


def tradeoff_figure():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for cohort, label, col in (("synthetic", "synthetic Indian cohort", TEAL), ("shanghai", "real ShanghaiT2DM (5-fold CV)", PURPLE)):
        f = C.RESULTS_DIR / f"{cohort}_alert_tradeoff.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f).sort_values("false_alerts_per_day")
        ax.plot(t.false_alerts_per_day, t.event_recall * 100, "o-", color=col, label=label)
        d0 = t.iloc[t.threshold_offset.abs().argmin()]
        ax.plot(d0.false_alerts_per_day, d0.event_recall * 100, "*", color=AMBER, ms=14)
    ax.set_xlabel("false alerts per patient-day (60-min snooze)")
    ax.set_ylabel("% of spikes flagged >= 15 min ahead")
    ax.set_title("Alert burden vs spikes caught (star = default threshold)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(F / "alert_tradeoff.png", dpi=170)
    plt.close(fig)


def whatif_example():
    try:
        from glucotwin.inference import GlucoTwinService, meal_scenario
    except Exception:
        return None
    svc = GlucoTwinService()
    pid = svc.patient_ids[0]
    lo, hi = svc.time_range(pid)
    minute = (lo // 1440 + 2) * 1440 + 13 * 60          # day 5, 13:00 - just before lunch
    H = 300
    scen = [("Rice + dal + sabzi (usual lunch)", {"meals": [meal_scenario("rice_dal_sabzi", 0)]}),
            ("+ 15-min walk after eating", {"meals": [meal_scenario("rice_dal_sabzi", 0)], "walks": [(15, 15, 0.55)]}),
            ("Half portion", {"meals": [meal_scenario("rice_dal_sabzi", 0, 0.5)]}),
            ("Swap to millet roti + dal + salad", {"meals": [meal_scenario("millet_thali", 0)]}),
            ("Usual lunch after a 4.5-h night", {"meals": [meal_scenario("rice_dal_sabzi", 0)], "si_scale": 0.85})]
    G = svc.simulate_scenarios(pid, minute, [s for _, s in scen], H=H, include_routine=False)
    cols = [RED, TEAL, BLUE, PURPLE, AMBER]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    hrs = 13 + np.arange(H) / 60
    ax.axhspan(70, 180, color=TEAL, alpha=0.07)
    ax.axhline(180, color=AMBER, ls=":", lw=1)
    rows = []
    for (n, _), g, c in zip(scen, G, cols):
        ax.plot(hrs, g, color=c, lw=2.2 if n.startswith("Rice") else 1.6, label=f"{n}: peak {g.max():.0f}")
        rows.append({"scenario": n, "peak": round(float(g.max())), "minutes_above_180": int((g > 180).sum())})
    ax.set_xlabel("hour of day")
    ax.set_ylabel("simulated glucose (mg/dL)")
    ax.set_title(f"What-if: the twin of {pid} compares lunch options")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(F / "whatif_example.png", dpi=170)
    plt.close(fig)
    # an example alert with its reasons, for the deck/README
    ex = None
    for q in svc.patient_ids:
        pr = svc.preds[(svc.preds.patient_id == q) & (svc.preds.minute // 1440 >= lo // 1440 + 2)]
        hit = pr[(pr.prob_spike >= max(svc.thr, 0.8)) & ((pr.minute % 1440) > 11 * 60) & ((pr.minute % 1440) < 14 * 60)]
        if len(hit):
            st = svc.state(q, int(hit.minute.iloc[0]))
            ex = {"patient": q, "time": svc.minute_to_ts(st["minute"]).strftime("%a %d %b, %H:%M"),
                  "glucose": st["glucose_now"], "prob": st["spike_probability"],
                  "forecast_2h": st["forecast"][-1]["glucose"],
                  "reasons": [{"text": r["text"], "direction": r["direction"]} for r in st["reasons"][:4]]}
            break
    return {"patient": pid, "scenarios": rows, "example_alert": ex}


def summary(whatif):
    out = {}
    final = "LightGBM: GlucoTwin (fusion + digital twin)"
    for cohort in ("synthetic", "shanghai"):
        R = C.RESULTS_DIR
        if not (R / f"{cohort}_results.json").exists():
            continue
        res = json.loads((R / f"{cohort}_results.json").read_text())
        clf = pd.read_csv(R / f"{cohort}_classification.csv").set_index("model")
        ev = pd.read_csv(R / f"{cohort}_events.csv").set_index("model")
        reg = pd.read_csv(R / f"{cohort}_regression.csv")
        tr = pd.read_csv(R / f"{cohort}_alert_tradeoff.csv")
        rm = lambda m, h: float(reg[(reg.model == m) & (reg.horizon == h)].rmse.iloc[0])
        r60 = reg[(reg.model == final) & (reg.horizon == 60)].iloc[0]
        quiet = tr.iloc[(tr.false_alerts_per_day - 1.0).abs().argmin()]
        out[cohort] = {
            "evaluation": res["evaluation"], "patients": res["patients"], "rows": res["rows_evaluated"],
            "auroc": clf.loc[final, "auroc"], "auprc": clf.loc[final, "auprc"], "prevalence": clf.loc[final, "prevalence"],
            "sensitivity": clf.loc[final, "sensitivity"], "specificity": clf.loc[final, "specificity"],
            "ppv": clf.loc[final, "ppv"],
            "event_recall": ev.loc[final, "event_recall"], "median_lead": ev.loc[final, "median_lead_min"],
            "false_per_day": ev.loc[final, "false_alerts_per_day"], "alerts_per_day": ev.loc[final, "alerts_per_day"],
            "events": int(ev.loc[final, "events"]), "events_per_day": ev.loc[final, "events_per_day"],
            "quiet_mode": {k: float(quiet[k]) for k in ("event_recall", "median_lead_min", "false_alerts_per_day")},
            "rmse": {h: rm(final, h) for h in (30, 60, 120)},
            "rmse_persistence": {h: rm("Persistence (no change)", h) for h in (30, 60, 120)},
            "rmse_twin": {h: rm("Digital twin alone (physiology)", h) for h in (30, 60, 120)},
            "mard60": float(r60.mard), "clarkeA60": float(r60.clarke_A), "clarkeAB60": float(r60.clarke_AB),
            "coverage80": {h: v["test_coverage_80"] for h, v in res["conformal"].items()},
            "ablation": [[m, float(a)] for m, a in clf.auroc.items()],
            "streams": res["stream_importance"],
        }
    if whatif:
        out["example_alert"] = whatif.pop("example_alert", None)
        out["whatif"] = whatif
    tl = C.RESULTS_DIR / "twin_learning.csv"
    if tl.exists():
        t = pd.read_csv(tl)
        later = t[t.day >= 1]
        out["twin_learning"] = {"si_corr_day0": float(t.si_corr.iloc[0]), "si_corr_last": float(later.si_corr.mean()),
                                "si_err_day0": float(t.si_err.iloc[0]), "si_err_last": float(later.si_err.mean())}
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    architecture()
    data_streams()
    twin_learning()
    results_figures("synthetic", "synthetic Indian cohort")
    results_figures("shanghai", "real-world ShanghaiT2DM")
    example_patient()
    tradeoff_figure()
    summary(whatif_example())
    print("figures written to", F)
