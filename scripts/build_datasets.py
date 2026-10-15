"""Step 1: generate the synthetic cohort, load ShanghaiT2DM, run the twin, build features.

Usage:  python scripts/build_datasets.py [--synthetic-only | --shanghai-only]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thalitwin import config as C  # noqa: E402
from thalitwin.pipeline import build_shanghai, build_synthetic  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--shanghai-only", action="store_true")
    a = ap.parse_args()
    if not a.shanghai_only:
        print("== Synthetic Indian T2D cohort ==")
        build_synthetic()
    if not a.synthetic_only:
        if (C.SHANGHAI_DIR / "Shanghai_T2DM_Summary.xlsx").exists():
            print("== ShanghaiT2DM (real, anonymised) ==")
            build_shanghai()
        else:
            print("ShanghaiT2DM not found - run scripts/download_shanghai.py first (skipping real-data track)")
