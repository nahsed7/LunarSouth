#!/usr/bin/env bash
# LunarSouth - run locally (Linux / macOS). Needs Python 3.12 or 3.13.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import clps_engine"
exec .venv/bin/streamlit run app.py "$@"
