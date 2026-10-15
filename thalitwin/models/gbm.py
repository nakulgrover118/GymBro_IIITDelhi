"""Gradient-boosted models (LightGBM) for glucose forecasts and spike alerts."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

from thalitwin.features import CGM_FEATURES, EHR_FEATURES, LOG_FEATURES, TWIN_FEATURES, WEAR_FEATURES

BASE = dict(n_estimators=3000, learning_rate=0.04, num_leaves=63, min_child_samples=200, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=-1)

FEATURE_SETS = {
    "CGM only": [CGM_FEATURES],
    "CGM + EHR": [CGM_FEATURES, EHR_FEATURES],
    "CGM + wearables & logs": [CGM_FEATURES, LOG_FEATURES, WEAR_FEATURES],
    "Fusion (CGM + EHR + wearables & logs)": [CGM_FEATURES, EHR_FEATURES, LOG_FEATURES, WEAR_FEATURES],
    "ThaliTwin (fusion + digital twin)": [CGM_FEATURES, EHR_FEATURES, LOG_FEATURES, WEAR_FEATURES, TWIN_FEATURES],
}
STREAM_OF = {**{f: "CGM" for f in CGM_FEATURES}, **{f: "EHR" for f in EHR_FEATURES},
             **{f: "Meal & medicine logs" for f in LOG_FEATURES}, **{f: "Wearables" for f in WEAR_FEATURES},
             **{f: "Digital twin" for f in TWIN_FEATURES}}


def columns_for(set_name: str, available) -> list[str]:
    cols = [c for group in FEATURE_SETS[set_name] for c in group]
    av = set(available)
    return [c for c in dict.fromkeys(cols) if c in av]


warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


def fit_regressor(Xtr, ytr, Xva, yva, **kw):
    m = lgb.LGBMRegressor(**{**BASE, **kw})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def fit_classifier(Xtr, ytr, Xva, yva, **kw):
    m = lgb.LGBMClassifier(**{**BASE, **kw})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict_proba(m, X) -> np.ndarray:
    return m.predict_proba(X)[:, 1]
