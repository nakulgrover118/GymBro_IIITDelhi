"""Project-wide paths and clinical constants for ThaliTwin."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SHANGHAI_DIR = RAW_DIR / "shanghai" / "extracted"
SYN_DIR = DATA_DIR / "synthetic"
PROC_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIG_DIR = ROOT / "docs" / "figures"

# Clinical thresholds (ADA Standards of Care / international CGM consensus)
SPIKE_MGDL = 180.0          # upper limit of time-in-range
SEVERE_MGDL = 250.0         # level-2 hyperglycaemia
HYPO_MGDL = 70.0            # level-1 hypoglycaemia

# Prediction set-up
HORIZONS = (30, 60, 120)    # minutes ahead for glucose forecasts
ALERT_WINDOW_MIN = 120      # "spike within the next 2 hours"
ALERT_THRESHOLD = 0.5       # default probability needed to raise an alert

# Synthetic cohort
SEED = 2026
N_PATIENTS = 200
N_DAYS = 14
STEP_MIN = 5                # wearable + CGM sampling interval (minutes)
WARMUP_DAYS = 2             # days the twin observes before it starts predicting

for _d in (SYN_DIR, PROC_DIR, MODELS_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
