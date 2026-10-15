"""Step 2: train all models, run the ablation study and save results.

Usage:  python scripts/train_evaluate.py [--cohort synthetic|shanghai|both] [--no-gru]
"""
import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucotwin import config as C  # noqa: E402
from glucotwin.models.experiment import run  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="both", choices=["synthetic", "shanghai", "both"])
    ap.add_argument("--no-gru", action="store_true")
    a = ap.parse_args()
    if a.cohort in ("synthetic", "both"):
        print("== Synthetic Indian cohort ==")
        run("synthetic", step=C.STEP_MIN, train_stride=2, use_gru=not a.no_gru, log=print)
    if a.cohort in ("shanghai", "both") and (C.PROC_DIR / "features_shanghai.parquet").exists():
        print("== ShanghaiT2DM (real) ==")
        run("shanghai", step=15, train_stride=1, use_gru=not a.no_gru, cv_folds=5, log=print)
