"""One command to rebuild everything: data -> twin -> models -> dashboard package -> figures.

    python run_all.py            (~20-30 min on a laptop CPU)
    python run_all.py --fast     (skip the deep-learning comparison)
"""
import subprocess
import sys

PY = sys.executable
fast = "--fast" in sys.argv
steps = [
    [PY, "scripts/download_shanghai.py"],
    [PY, "scripts/build_datasets.py"],
    [PY, "scripts/train_evaluate.py"] + (["--no-gru"] if fast else []),
    [PY, "scripts/export_dashboard.py"],
    [PY, "scripts/make_figures.py"],
    [PY, "scripts/write_reports.py"],
]
for cmd in steps:
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
print("\nDone. Start the dashboard:  streamlit run dashboard/app.py")
