"""Build the self-contained LunarSouth release (GitHub repo folder + zip).

    .venv/bin/python tools/build_release.py [out_dir]      (default: ../LunarSouth  and ../LunarSouth.zip)

* Copies only what the app needs at run time (traced) plus code, tests and docs.
* Every file in the output is <= 24 MB (GitHub web upload limit 25 MB, git limit 100 MB): larger rasters are stored
  as gzip parts in data/packed/ and rebuilt bit-exact (SHA-256 checked) by clps_engine.data_bootstrap on first run.
* Leaves out rebuildable / unused bulk (data/processed/polewide/grid200m.npz is only an intermediate of
  tools/build_polewide.py and is never read by the app), the venv, caches, outputs and old app versions.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SRC.parent / "LunarSouth"
PART = 24 * 2**20
EXCLUDE = [".venv/*", "*/__pycache__/*", "__pycache__/*", "*.pyc", "outputs/*", "*.zip", "datakit (1).py",
           "app_v2_pre_polewide.py", "app_legacy.py", "data/discovered_sites.prev.csv", "keep_public.sh",
           "start_public.sh", "setup_named_tunnel.sh", "data/processed/polewide/grid200m.npz",
           "requirements-deploy.txt", "requirements-geo.txt", ".dockerignore", "*.tmp"]
PINNED = """numpy==2.5.4
pandas==3.1.1
plotly==7.1.0
scipy==1.18.1
streamlit==1.67.0
skyfield==1.55
rasterio==1.5.2
matplotlib==3.11.2
tabulate==0.10.0
Jinja2==3.1.6
altair==6.3.0
pydeck==0.9.3
"""

RUN_SH = """#!/usr/bin/env bash
# LunarSouth - run locally (Linux / macOS). Needs Python 3.12 or 3.13.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import clps_engine"
exec .venv/bin/streamlit run app.py "$@"
"""
RUN_BAT = """@echo off
REM LunarSouth - run locally (Windows). Needs Python 3.12 or 3.13 from python.org.
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv || python -m venv .venv )
.venv\\Scripts\\python -m pip install -q --upgrade pip
.venv\\Scripts\\python -m pip install -q -r requirements.txt
.venv\\Scripts\\python -c "import clps_engine"
.venv\\Scripts\\streamlit run app.py
"""
DEPLOY_MD = """# Deploying LunarSouth (free, always on, PC can be off)

## 1. Put this folder on GitHub
* **Browser:** create a public repo on github.com → *Add file → Upload files* → drag **the contents** of this folder
  (not the zip). Every file is ≤ 24 MB, so the browser upload accepts all of them. GitHub's browser upload takes up
  to 100 files at a time, so upload folder by folder if it complains.
* **Command line:** `git init && git add . && git commit -m "LunarSouth" && git branch -M main &&
  git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main`

## 2. Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**. Pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Python version 3.13** (3.12 also works). Deploy.
4. The first boot takes about 5 minutes: installing packages, then unpacking the rasters once. You get
   `https://<name>.streamlit.app`. Apps sleep after a few days without visitors and wake on the next visit (~1 min).

## Run it on any computer
* Linux / macOS: `./run.sh`  ·  Windows: double-click `run.bat`  → opens http://localhost:8501
* Or by hand: `pip install -r requirements.txt && streamlit run app.py`

The large NASA rasters are in `data/packed/` as compressed parts. They are rebuilt bit-exact (SHA-256 checked)
automatically on first start, or explicitly with `python -c "import clps_engine"`.
"""
GITIGNORE = """.venv/
__pycache__/
*.pyc
outputs/
*.tmp
# rebuilt on first run from data/packed/ (keep them out of git)
{packed}
"""


def excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in EXCLUDE)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "packed").mkdir(parents=True)
    manifest, packed_paths, n = [], [], 0
    for f in sorted(SRC.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(SRC).as_posix()
        if excluded(rel) or rel.startswith("data/packed/"):
            continue
        if rel in ("requirements.txt", "Dockerfile"):
            continue
        size = f.stat().st_size
        if size <= PART:
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
            continue
        # big file -> gzip -> parts
        raw = f.read_bytes()
        gz = gzip.compress(raw, compresslevel=9, mtime=0)
        stem = rel.replace("/", "__")
        parts = []
        for i in range(0, len(gz), PART):
            name = f"{stem}.gz.part{i // PART:02d}"
            (OUT / "data" / "packed" / name).write_bytes(gz[i:i + PART])
            parts.append(name)
        manifest.append({"path": rel, "size": size, "sha256": hashlib.sha256(raw).hexdigest(), "parts": parts})
        packed_paths.append(rel)
        print(f"  packed {rel}: {size / 2**20:.0f} MB -> {len(gz) / 2**20:.0f} MB in {len(parts)} part(s)")
    (OUT / "data" / "packed" / "manifest.json").write_text(json.dumps({"files": manifest}, indent=1))
    (OUT / "requirements.txt").write_text(PINNED)
    (OUT / "run.sh").write_text(RUN_SH)
    (OUT / "run.sh").chmod(0o755)
    (OUT / "run.bat").write_text(RUN_BAT.replace("\n", "\r\n"))
    (OUT / "DEPLOY.md").write_text(DEPLOY_MD)
    (OUT / ".gitignore").write_text(GITIGNORE.format(packed="\n".join(packed_paths)))
    (OUT / "Dockerfile").write_text(
        "FROM python:3.13-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . .\nRUN python -c \"import clps_engine\"\nEXPOSE 8501\n"
        'CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]\n')
    big = [p for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > PART]
    assert not big, f"files over 24 MB: {big}"
    zpath = OUT.with_suffix(".zip")
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo.from_file(p, Path(OUT.name) / p.relative_to(OUT))
                info.compress_type = zipfile.ZIP_STORED if "/packed/" in p.as_posix() else zipfile.ZIP_DEFLATED
                with open(p, "rb") as fh:
                    z.writestr(info, fh.read())
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\n{n} files copied, {len(manifest)} packed. Folder {tot / 2**20:.0f} MB -> {zpath} ({zpath.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
