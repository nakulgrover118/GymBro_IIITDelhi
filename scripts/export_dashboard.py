"""Step 3: package the held-out test patients for the dashboard/API (artifacts/dashboard/)."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from thalitwin import config as C  # noqa: E402
from thalitwin.features import LOG_FEATURES, TWIN_FEATURES, WEAR_FEATURES  # noqa: E402

if __name__ == "__main__":
    out = C.ARTIFACTS_DIR / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((C.MODELS_DIR / "synthetic" / "meta.json").read_text())
    split = pd.read_csv(C.SYN_DIR / "splits.csv")
    test_ids = split[split.split == "test"].patient_id.tolist()
    keep = ["patient_id", "minute", "cgm_now", "d30", "spike_onset", "y_spike"]
    keep += [c for c in dict.fromkeys(meta["features"] + LOG_FEATURES + WEAR_FEATURES + TWIN_FEATURES) if c not in keep]
    df = pd.read_parquet(C.PROC_DIR / "features_synthetic.parquet",
                         filters=[("patient_id", "in", test_ids)])
    df = df[[c for c in keep if c in df.columns]]
    df.to_parquet(out / "features.parquet", index=False, compression="zstd")
    shutil.copy(C.PROC_DIR / "twin_params_synthetic.csv", out / "twin_params.csv")
    mb = (out / "features.parquet").stat().st_size / 1e6
    print(f"exported {df.patient_id.nunique()} test patients, {len(df):,} rows, {mb:.1f} MB")
