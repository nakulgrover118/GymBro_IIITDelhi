"""Download the ShanghaiT2DM dataset (CC BY 4.0) from figshare into data/raw/shanghai/extracted/."""
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C  # noqa: E402

URL = "https://ndownloader.figshare.com/files/38259264"   # figshare article 20444397, file data.zip

if __name__ == "__main__":
    out = C.SHANGHAI_DIR
    if (out / "Shanghai_T2DM_Summary.xlsx").exists():
        print("already downloaded:", out)
        sys.exit(0)
    print("downloading ShanghaiT2DM (3.4 MB, CC BY 4.0) ...")
    data = urllib.request.urlopen(URL, timeout=120).read()
    tmp = out.parent / "_tmp"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp)
    inner = next(p for p in tmp.iterdir() if p.is_dir())      # top folder has a non-ASCII name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(inner), str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    for junk in out.rglob("~$*"):
        junk.unlink()
    print("saved to", out)
