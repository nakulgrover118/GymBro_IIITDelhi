"""Train + evaluate every model variant on one cohort, with patient-level splits.

Synthetic cohort: one fixed train/val/test split (140/30/30 patients).
ShanghaiT2DM: 5-fold cross-validation grouped by person, so every real
patient is scored by a model that never saw them (out-of-fold pooling).

Outputs (artifacts/):
  models/<cohort>/        deployed models (LightGBM boosters + metadata)
  results/<cohort>_*.csv  regression, classification, event-level and alert trade-off tables
  results/<cohort>_results.json
  results/<cohort>_test_predictions.parquet
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import shap

from thalitwin import config as C
from thalitwin.features import EHR_FEATURES, TWIN_FEATURES
from thalitwin.models.gbm import FEATURE_SETS, STREAM_OF, columns_for, fit_classifier, fit_regressor, predict_proba
from thalitwin.models.metrics import classification_report, event_report, pick_threshold, regression_report

FINAL = "ThaliTwin (fusion + digital twin)"
GRU_NAME = "Deep learning: GRU fusion (sequence + EHR + twin)"
TARGET_SENS = 0.80
BASELINES = {"Trend rule (glucose + 2x 30-min change)": lambda d: d.cgm_now + 2 * d.d30,
             "Digital twin alone (simulated peak)": lambda d: d.twin_max120}


def _load(cohort):
    df = pd.read_parquet(C.PROC_DIR / f"features_{cohort}.parquet")
    sp = pd.read_csv(C.SYN_DIR / "splits.csv" if cohort == "synthetic" else C.PROC_DIR / "splits_shanghai.csv")
    return df.merge(sp, on="patient_id")


def _gbm_name(n):
    return f"LightGBM: {n}"


def fit_predict(tr, va, te, step, use_gru, say, train_stride=1):
    """Fit all variants on (tr, va); return test predictions, thresholds and the deployed models."""
    trs = tr.iloc[::train_stride]
    out = te[["patient_id", "minute", "cgm_now", "d30", "spike_onset", "y_spike"] + [f"y_g{h}" for h in C.HORIZONS]
             + [f"twin_g{h}" for h in C.HORIZONS] + ["twin_max120"]].copy()
    thr, deploy = {}, {}
    vac, trc = va[va.y_spike.notna()], trs[trs.y_spike.notna()]
    for name, score in BASELINES.items():
        thr[name] = pick_threshold(vac.y_spike, score(vac).values, TARGET_SENS)
        out[f"prob::{name}"] = score(te).values.astype(float)
    for h in C.HORIZONS:
        out[f"g{h}::Persistence (no change)"] = te.cgm_now.values
        out[f"g{h}::Linear trend extrapolation"] = (te.cgm_now + te.d30 / 30.0 * h).values
        out[f"g{h}::Digital twin alone (physiology)"] = te[f"twin_g{h}"].values
    for name in FEATURE_SETS:
        cols = columns_for(name, te.columns)
        clf = fit_classifier(trc[cols], trc.y_spike, vac[cols], vac.y_spike)
        thr[_gbm_name(name)] = pick_threshold(vac.y_spike, predict_proba(clf, vac[cols]), TARGET_SENS)
        out[f"prob::{_gbm_name(name)}"] = predict_proba(clf, te[cols])
        regs = {}
        for h in C.HORIZONS:
            yc = f"y_g{h}"
            rtr, rva = trs[trs[yc].notna() & trs.cgm_now.notna()], va[va[yc].notna() & va.cgm_now.notna()]
            reg = fit_regressor(rtr[cols], rtr[yc] - rtr.cgm_now, rva[cols], rva[yc] - rva.cgm_now)
            out[f"g{h}::{_gbm_name(name)}"] = te.cgm_now.values + reg.predict(te[cols])
            regs[h] = (reg, rva)
        if name == FINAL:
            conf = {}
            for h, (reg, rva) in regs.items():
                r = np.abs(rva[f"y_g{h}"].values - (rva.cgm_now.values + reg.predict(rva[cols])))
                conf[str(h)] = {"q80": float(np.quantile(r, 0.80)), "q95": float(np.quantile(r, 0.95))}
            sample = te[te.y_spike.notna()].sample(min(4000, int(te.y_spike.notna().sum())), random_state=0)
            sv = shap.TreeExplainer(clf.booster_).shap_values(sample[cols])
            sv = sv[1] if isinstance(sv, list) else sv
            deploy = {"clf": clf, "regs": {h: r for h, (r, _) in regs.items()}, "cols": cols, "conformal": conf,
                      "shap": pd.Series(np.abs(sv).mean(0), index=cols)}
        say(f"trained {name} ({len(cols)} features)")
    if use_gru:
        try:
            from thalitwin.models.gru import SEQ_CANDIDATES, SequenceData, predict, train_gru
            seq_cols = [c for c in SEQ_CANDIDATES if c in te.columns]
            st_cols = [c for c in EHR_FEATURES + TWIN_FEATURES if c in te.columns]
            dtr = SequenceData(tr, seq_cols, st_cols, step)
            dva = SequenceData(va, seq_cols, st_cols, step, stats=dtr.stats)
            dte = SequenceData(te, seq_cols, st_cols, step, stats=dtr.stats)
            model = train_gru(dtr, dva, epochs=6, per_epoch=min(150_000, len(dtr.valid)), log=say)
            g = te[["patient_id", "minute"]].merge(predict(model, dte), on=["patient_id", "minute"], how="left")
            gv = va[["patient_id", "minute", "y_spike"]].merge(predict(model, dva), on=["patient_id", "minute"], how="left")
            thr[GRU_NAME] = pick_threshold(gv.y_spike, gv.gru_spike, TARGET_SENS)
            out[f"prob::{GRU_NAME}"] = g.gru_spike.values
            for h in C.HORIZONS:
                out[f"g{h}::{GRU_NAME}"] = g[f"gru_g{h}"].values
            say("GRU trained")
        except Exception as e:  # keep the pipeline usable without torch
            say(f"GRU skipped: {e}")
    # threshold-adjusted alert scores (alert when score >= 0) so folds with different thresholds can be pooled
    for name, t in thr.items():
        out[f"score::{name}"] = out[f"prob::{name}"] - t
    out["prob_spike"] = out[f"prob::{_gbm_name(FINAL)}"]
    out["alert_score"] = out[f"score::{_gbm_name(FINAL)}"]
    for h in C.HORIZONS:
        out[f"pred_g{h}"] = out[f"g{h}::{_gbm_name(FINAL)}"]
        q = deploy["conformal"][str(h)]["q80"]
        out[f"lo80_g{h}"], out[f"hi80_g{h}"] = out[f"pred_g{h}"] - q, out[f"pred_g{h}"] + q
    return out, thr, deploy


def _events(p: pd.DataFrame, score_col: str, step: int, thr: float = 0.0) -> dict:
    d = p[["patient_id", "minute", "cgm_now", "spike_onset"]].copy()
    s = p[score_col].values.astype(float).copy()
    s[(d.cgm_now.values >= C.SPIKE_MGDL) | np.isnan(d.cgm_now.values)] = np.nan   # alerts only from below 180
    d["s"] = s
    return event_report(d, "s", thr, step)


def evaluate(p: pd.DataFrame, step: int):
    """Tables computed from (pooled) test predictions."""
    clf_rows, reg_rows, ev_rows = [], [], []
    names = [c.split("::", 1)[1] for c in p.columns if c.startswith("prob::")]
    m = p.y_spike.notna().values
    for n in names:
        rep = classification_report(p.y_spike.values[m], p[f"prob::{n}"].values[m], 0.5)
        adj = classification_report(p.y_spike.values[m], p[f"score::{n}"].values[m], 0.0)
        for k in ("sensitivity", "specificity", "ppv", "f1"):
            rep[k] = adj[k]
        rep.pop("threshold")
        clf_rows.append({"model": n, **rep})
        ev_rows.append({"model": n, **_events(p, f"score::{n}", step)})
    rnames = sorted({c.split("::", 1)[1] for c in p.columns if c.startswith("g30::")},
                    key=lambda x: (not x.startswith(("Pers", "Lin", "Dig")), x))
    for n in rnames:
        for h in C.HORIZONS:
            reg_rows.append({"model": n, "horizon": h, **regression_report(p[f"y_g{h}"], p[f"g{h}::{n}"])})
    order = list(BASELINES) + [_gbm_name(n) for n in FEATURE_SETS] + [GRU_NAME]
    key = lambda df: df.assign(_o=df.model.map({n: i for i, n in enumerate(order)}).fillna(99)).sort_values("_o").drop(columns="_o")
    return key(pd.DataFrame(clf_rows)), pd.DataFrame(reg_rows), key(pd.DataFrame(ev_rows))


def run(cohort: str = "synthetic", step: int = 5, train_stride: int = 2, use_gru: bool = True,
        cv_folds: int | None = None, log=print):
    t0 = time.time()
    say = lambda m: log(f"[{time.time() - t0:6.1f}s] {m}")
    df = _load(cohort)
    res = {"cohort": cohort, "step_min": step, "target_sensitivity": TARGET_SENS}
    if cv_folds:
        group = df["person"] if "person" in df else df["patient_id"]
        persons = np.array(sorted(group.unique()))
        rng = np.random.default_rng(C.SEED)
        rng.shuffle(persons)
        folds = np.array_split(persons, cv_folds)
        parts, shap_acc, conf_acc = [], [], []
        for k, test_p in enumerate(folds):
            rest = np.array([x for x in persons if x not in set(test_p)])
            val_p = rest[: max(int(0.18 * len(rest)), 2)]
            te = df[group.isin(test_p)].reset_index(drop=True)
            va = df[group.isin(val_p)].reset_index(drop=True)
            tr = df[~group.isin(test_p) & ~group.isin(val_p)].reset_index(drop=True)
            say(f"fold {k + 1}/{cv_folds}: train {tr.patient_id.nunique()} / val {va.patient_id.nunique()} / "
                f"test {te.patient_id.nunique()} recordings")
            out, thr, dep = fit_predict(tr, va, te, step, use_gru, say, train_stride)
            out["fold"] = k
            parts.append(out)
            shap_acc.append(dep["shap"])
            conf_acc.append(dep["conformal"])
            if k == 0:
                deploy, thr_final = dep, thr
        preds = pd.concat(parts, ignore_index=True)
        shap_mean = pd.concat(shap_acc, axis=1).mean(1)
        res["evaluation"] = f"{cv_folds}-fold cross-validation grouped by person (out-of-fold pooled)"
        res["patients"] = {"total_recordings": int(df.patient_id.nunique()), "people": int(len(persons))}
        conformal = {h: {"q80": float(np.mean([c[h]["q80"] for c in conf_acc])),
                         "q95": float(np.mean([c[h]["q95"] for c in conf_acc]))} for h in conf_acc[0]}
    else:
        tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
        say(f"{cohort}: train {tr.patient_id.nunique()} / val {va.patient_id.nunique()} / test "
            f"{te.patient_id.nunique()} patients")
        preds, thr_final, deploy = fit_predict(tr, va, te, step, use_gru, say, train_stride)
        shap_mean = deploy["shap"]
        conformal = deploy["conformal"]
        res["evaluation"] = "held-out test patients (patient-level split)"
        res["patients"] = {s: int(d.patient_id.nunique()) for s, d in (("train", tr), ("val", va), ("test", te))}
    res["rows_evaluated"] = int(len(preds))

    clf, reg, ev = evaluate(preds, step)
    if cohort == "shanghai":      # no wearables in ShanghaiT2DM: name the variants by what they really use
        ren = lambda x: x.replace("wearables & logs", "meal & insulin logs")
        for t in (clf, reg, ev):
            t["model"] = t["model"].map(ren)
    for h in C.HORIZONS:
        y = preds[f"y_g{h}"]
        ok = y.notna() & preds[f"pred_g{h}"].notna()
        conformal[str(h)]["test_coverage_80"] = float(((y[ok] >= preds[f"lo80_g{h}"][ok]) &
                                                       (y[ok] <= preds[f"hi80_g{h}"][ok])).mean())
    res["conformal"] = conformal
    trade = []
    for t in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
        trade.append({"threshold_offset": t, "threshold": float(thr_final[_gbm_name(FINAL)]) + t,
                      **_events(preds, "alert_score", step, t)})

    imp = pd.DataFrame({"feature": shap_mean.index, "mean_abs_shap": shap_mean.values})
    imp["stream"] = imp.feature.map(STREAM_OF)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    streams = imp.groupby("stream").mean_abs_shap.sum()
    streams = (streams / streams.sum()).sort_values(ascending=False)
    res["stream_importance"] = streams.round(4).to_dict()

    out_m = C.MODELS_DIR / cohort
    out_m.mkdir(parents=True, exist_ok=True)
    deploy["clf"].booster_.save_model(str(out_m / "spike_classifier.txt"))
    for h, reg_m in deploy["regs"].items():
        reg_m.booster_.save_model(str(out_m / f"glucose_{h}min.txt"))
    final_thr = float(thr_final[_gbm_name(FINAL)])
    meta = {"features": deploy["cols"], "threshold": final_thr, "target_sensitivity": TARGET_SENS,
            "conformal": deploy["conformal"], "horizons": list(C.HORIZONS), "step_min": step,
            "snooze_min": 60}
    (out_m / "meta.json").write_text(json.dumps(meta, indent=1))
    res["final_threshold"] = final_thr

    R = C.RESULTS_DIR
    reg.to_csv(R / f"{cohort}_regression.csv", index=False)
    clf.to_csv(R / f"{cohort}_classification.csv", index=False)
    ev.to_csv(R / f"{cohort}_events.csv", index=False)
    pd.DataFrame(trade).to_csv(R / f"{cohort}_alert_tradeoff.csv", index=False)
    imp.to_csv(R / f"{cohort}_shap_features.csv", index=False)
    streams.rename("share").to_csv(R / f"{cohort}_shap_streams.csv")
    keep = [c for c in preds.columns if not c.startswith(("g30::", "g60::", "g120::"))]
    pk = preds[keep].copy()
    for c in pk.columns:
        if pk[c].dtype == np.float64:
            pk[c] = pk[c].astype(np.float32)
    pk.to_parquet(R / f"{cohort}_test_predictions.parquet", index=False, compression="zstd")
    (R / f"{cohort}_results.json").write_text(json.dumps(res, indent=1))
    say("saved models and results")
    return res
