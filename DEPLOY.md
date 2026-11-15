# Deploying LunarSouth (free, always on, PC can be off)

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
