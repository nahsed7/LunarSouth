@echo off
REM LunarSouth - run locally (Windows). Needs Python 3.12 or 3.13 from python.org.
cd /d "%~dp0"
if not exist .venv ( py -3 -m venv .venv || python -m venv .venv )
.venv\Scripts\python -m pip install -q --upgrade pip
.venv\Scripts\python -m pip install -q -r requirements.txt
.venv\Scripts\python -c "import clps_engine"
.venv\Scripts\streamlit run app.py
