"""
LUNARSOUTH - CLPS Mission Planner (Lunar South Pole Mission Engineering System)
NASA Space Apps Challenge 2026 - Human Exploration / Moon

Pick ANY point on the south pole (80S-90S) -> Sun/Earth geometry -> terrain -> generation (M4)
-> consumption & battery survival (M5) -> comparison -> downloadable report.
All five models run on ONE shared timeline (clps_engine.mission). Legacy UI: app_legacy.py.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from clps_engine import energy as en  # noqa: E402
from clps_engine import power_gen as pg  # noqa: E402
from clps_engine import viz3d, viz_pole, viz_power  # noqa: E402
from clps_engine import rf_link, events as ev_mod, guide  # noqa: E402
from clps_engine import activities as act, pv_electrical as pve  # noqa: E402
from clps_engine import extensions as ext_mod  # noqa: E402
from clps_engine import sim_payload  # noqa: E402
from clps_engine import handoff as ho  # noqa: E402
from clps_engine import equipment as eq  # noqa: E402
from clps_engine import scan as scan_mod  # noqa: E402
from clps_engine import marker as mk_mod  # noqa: E402
from clps_engine.candidates import all_candidates  # noqa: E402
from clps_engine.dte import EARTH_RULES  # noqa: E402
from clps_engine.mission import feasibility_score, run_mission  # noqa: E402
from clps_engine.report import build_html  # noqa: E402
from clps_engine.sites import Site, custom_site  # noqa: E402

PROC = ROOT / "data" / "processed"
TEAM = "LUNARSOUTH"

st.set_page_config(page_title="LunarSouth · Lunar Mission Planner", page_icon="🌑", layout="wide",
                   menu_items={"About": "**LunarSouth** - CLPS mission planner for the lunar south pole (NASA Space Apps 2026). "
                                        "Five physics models (illumination, Earth/DTE link, terrain, power, energy survival) on NASA LOLA/LROC data, "
                                        "a best-point scanner, real-equipment loadouts and a playable mission sim."})

st.markdown(
    """
<style>
:root{--bg:#04080f;--panel:#0a1422;--line:rgba(56,217,255,.22);--cyan:#38d9ff;--txt:#d7f6ff;--dim:#7fa7b8;
--ok:#3ddc97;--warn:#ffc857;--bad:#ff5d73}
.stApp{background:radial-gradient(1200px 600px at 20% -10%,#0b2238 0%,#04080f 55%) fixed;color:var(--txt)}
section[data-testid="stSidebar"]{background:#060d18;border-right:1px solid var(--line)}
h1,h2,h3{font-family:'Segoe UI',ui-monospace,monospace;letter-spacing:.02em;color:#e9fbff}
.brandbar{display:flex;align-items:center;gap:16px;padding:10px 16px;border:1px solid var(--line);border-radius:12px;
background:linear-gradient(90deg,rgba(56,217,255,.10),rgba(4,8,15,.2) 60%);margin-bottom:8px}
.brandbar .t{font-family:ui-monospace,monospace;font-size:22px;letter-spacing:.32em;color:#fff}
.brandbar .s{font-size:12px;color:var(--dim);letter-spacing:.08em}
.kpi{background:linear-gradient(180deg,rgba(13,30,48,.9),rgba(6,14,24,.9));border:1px solid var(--line);border-radius:10px;
padding:10px 12px;margin-bottom:8px;box-shadow:0 0 18px rgba(56,217,255,.06) inset}
.kpi .l{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
.kpi .v{font-family:'Segoe UI',ui-monospace,monospace;font-size:22px;color:#fff;font-variant-numeric:tabular-nums}
.kpi .s{font-size:11px;color:var(--dim)}
.banner{border-radius:10px;padding:12px 16px;font-family:ui-monospace,monospace;font-size:14px;margin:6px 0 12px}
.b-ok{background:rgba(61,220,151,.10);border:1px solid var(--ok);color:#aef5d2}
.b-warn{background:rgba(255,200,87,.10);border:1px solid var(--warn);color:#ffe3a6}
.b-bad{background:rgba(255,93,115,.10);border:1px solid var(--bad);color:#ffc0c9}
.src{font-size:11px;color:var(--dim);font-family:ui-monospace,monospace}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin:2px;font-size:11px;color:var(--cyan)}
.card{background:rgba(10,20,34,.85);border:1px solid var(--line);border-radius:12px;padding:12px 14px;height:100%}
.card h4{margin:0 0 6px 0;color:#fff}.card .p{font-size:11px;color:var(--cyan);letter-spacing:.12em}
.card li{font-size:13px}
.footer{margin-top:30px;padding:12px;border-top:1px solid var(--line);font-size:12px;color:var(--dim);text-align:center;letter-spacing:.18em}
</style>
""",
    unsafe_allow_html=True,
)
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(4,8,15,.6)",
            font=dict(family="Segoe UI, JetBrains Mono, sans-serif", size=12, color="#d7f6ff"), margin=dict(t=40, l=50, r=20, b=40))

LOGO = """<svg width="54" height="54" viewBox="-6 0 60 54"><circle cx="18" cy="27" r="7" fill="#4aa8ff"/><circle cx="40" cy="27" r="3.2" fill="#cfd8dc"/>
<circle cx="18" cy="27" r="22" fill="none" stroke="#38d9ff" stroke-opacity=".25"/>
<g fill="#38d9ff"><circle cx="33" cy="27" r="1.6"/><circle cx="46" cy="27" r="1.6"/><circle cx="-3" cy="27" r="1.6"/>
<circle cx="29" cy="8" r="1.6"/><circle cx="29" cy="46" r="1.6"/></g>
<g font-size="5" fill="#7fa7b8"><text x="31" y="24">L1</text><text x="44" y="24">L2</text><text x="-5" y="24">L3</text>
<text x="27" y="6">L4</text><text x="27" y="53">L5</text></g></svg>"""


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>",
                 unsafe_allow_html=True)


def fmt(v, nd=1, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.{nd}f}{unit}"


# ------------------------------------------------------------------ sites / location
ALL = {s.id: s for s in all_candidates()}
DISC_P = ROOT / "data" / "discovered_sites.csv"
DISC = pd.read_csv(DISC_P) if DISC_P.exists() else pd.DataFrame()
PROJECT = [s for s in ALL.values() if s.source in ("baseline", "research")]
if "pick" not in st.session_state:
    st.session_state["pick"] = (float(DISC.latitude_deg.iloc[0]), float(DISC.longitude_deg.iloc[0])) if len(DISC) else (-89.45, -137.3)
    st.session_state["in_lat"], st.session_state["in_lon"] = st.session_state["pick"]

def sk(key: str, default):
    """Session-state-backed widget key (lets mission profiles set any sidebar control)."""
    if key not in st.session_state:
        st.session_state[key] = default
    return key


PROFILE_KEYS = ["start_d", "start_h", "len_mode", "days", "end_d", "step", "orient", "area", "eff", "tilt", "faz", "dust", "degr", "tco",
                "mppt", "cap", "soc0", "dod", "safe", "ceff", "pmax", "mli", "tint", "scenario", "earth_rule", "band", "ptx", "gtx", "gt",
                "rmax", "relay", "loc_mode", "in_lat", "in_lon", "sel_disc", "sel_cand", "sel_grp", "sel_preset"]
if "_pending_profile" in st.session_state:          # applied BEFORE any widget exists (Streamlit forbids changing live widgets)
    _pp = st.session_state.pop("_pending_profile")
    for _k, _v in _pp.get("settings", {}).items():
        if _k not in PROFILE_KEYS:
            continue
        if _k in ("start_d", "end_d") and isinstance(_v, str):
            _v = date.fromisoformat(_v)
        st.session_state[_k] = _v
    if "subs" in _pp:
        st.session_state["subs"] = pd.DataFrame(_pp["subs"])
    if "plan" in _pp:
        st.session_state["plan"] = _pp["plan"]
    if "equipment" in _pp:
        st.session_state["equip_qty"] = dict(_pp["equipment"].get("qty", {}))
        st.session_state["equip_src"] = dict(_pp["equipment"].get("sources", {}))
    if "comparison_points" in _pp:
        st.session_state["basket"] = list(_pp["comparison_points"])[:16]
    st.session_state["_profile_loaded"] = _pp.get("name", "profile")

if "_pending_pick" in st.session_state:             # map click from the explorer: widgets may only be changed before they exist
    _plat, _plon = st.session_state.pop("_pending_pick")
    st.session_state["pick"] = (_plat, _plon)
    _near = [x for x in ALL.values() if x.source.startswith("preset:") and abs(x.lat - _plat) < 1e-3 and abs(x.lon - _plon) < 1e-3]
    if _plat > -80.0 and _near:                       # global-map marker outside the polar cap -> select that preset mission
        st.session_state["loc_mode"] = "Mission presets (Apollo · CLPS · international)"
        st.session_state["sel_grp"], st.session_state["sel_preset"] = _near[0].source, _near[0].id
    elif _plat <= -80.0:
        st.session_state["in_lat"], st.session_state["in_lon"] = _plat, _plon
        st.session_state["loc_mode"] = "Map pick / coordinates"

st.sidebar.markdown(f"<div class='src' style='letter-spacing:.3em;color:#38d9ff'>{TEAM}</div>", unsafe_allow_html=True)
if "_profile_loaded" in st.session_state:
    st.sidebar.success(f"Loaded mission profile: {st.session_state.pop('_profile_loaded')}")
st.sidebar.markdown("## 1 · Location")
PRESET_SRC = sorted({s.source for s in ALL.values() if s.source.startswith("preset:")})
loc_mode = st.sidebar.radio("Choose by", ["Map pick / coordinates", "LunarSouth discoveries", "Project candidates (16)",
                                          "Mission presets (Apollo · CLPS · international)"],
                            help="Any point 80°S→90°S can be analysed. Click the South Pole Explorer map to pick.", key="loc_mode")
if loc_mode.startswith("Map"):
    plat = st.sidebar.number_input("Latitude (deg)", -90.0, -80.0, format="%.5f", key="in_lat")
    plon = st.sidebar.number_input("Longitude (deg E)", -180.0, 360.0, format="%.5f", key="in_lon")
    st.session_state["pick"] = (plat, plon)
    from clps_engine.rasters import product_registry
    _reg = product_registry()
    _d = _reg.get("LDEM_20M") or _reg.get("LOLA_DEM_80M")
    zz = float(_d.sample_latlon(plat, plon)) if _d is not None and _d.covers(plat, plon) else float("nan")
    site = custom_site(plat, plon, zz if np.isfinite(zz) else 0.0, f"Point {plat:.4f}°, {plon:.4f}°E", "POINT")
    st.sidebar.caption(f"DEM elevation {fmt(zz, 1, ' m')} ({_d.product_id if _d else 'n/a'}).")
elif loc_mode.startswith("LunarSouth"):
    ids = [s.id for s in ALL.values() if s.source == "discovered"]
    si = DISC.set_index("site_id").suitability_index if len(DISC) else pd.Series(dtype=float)
    sid = st.sidebar.selectbox("Discovery", ids, key="sel_disc", format_func=lambda i: f"{i} · index {si.get(i, float('nan')):.1f}")
    site = ALL[sid]
elif loc_mode.startswith("Project"):
    sid = st.sidebar.selectbox("Candidate", [s.id for s in PROJECT], key="sel_cand", format_func=lambda i: f"{i} · {ALL[i].name} ({ALL[i].source})")
    site = ALL[sid]
else:
    grp = st.sidebar.selectbox("Preset library", PRESET_SRC, key="sel_grp", format_func=lambda g: g.replace("preset:", "").replace("_", " "))
    ids = [s.id for s in ALL.values() if s.source == grp]
    if st.session_state.get("sel_preset") not in ids:
        st.session_state.pop("sel_preset", None)
    sid = st.sidebar.selectbox("Mission / site", ids, key="sel_preset", format_func=lambda i: f"{ALL[i].name} · {', '.join(ALL[i].tags[:3])}")
    site = ALL[sid]
    if abs(site.lat) < 80:
        st.sidebar.caption("Outside the 80°S polar cap: horizon from the global LOLA 1.9 km DEM; landing-scale terrain not assessed.")

# ---- 📍 custom coordinate marker, drawn on every map
if "_pending_marker" in st.session_state:          # set before the marker widgets exist
    st.session_state["mk_lat"], st.session_state["mk_lon"] = st.session_state.pop("_pending_marker")
    st.session_state["mk_on"] = True
if "mk_lat" not in st.session_state:
    _qp = st.query_params
    try:
        st.session_state["mk_lat"] = float(_qp.get("mlat", -89.5))
        st.session_state["mk_lon"] = float(_qp.get("mlon", 45.0))
        st.session_state["mk_on"] = "mlat" in _qp
        st.session_state["mk_label"] = str(_qp.get("mlabel", "My point"))[:40]
    except ValueError:
        st.session_state["mk_lat"], st.session_state["mk_lon"], st.session_state["mk_on"] = -89.5, 45.0, False
with st.sidebar.expander("📍 Custom marker (all maps)", expanded=bool(st.session_state.get("mk_on"))):
    st.checkbox("Show marker on all maps", key="mk_on")
    _mc = st.columns(2)
    _mc[0].number_input("Marker latitude", -90.0, 90.0, step=0.01, format="%.5f", key="mk_lat")
    _mc[1].number_input("Marker longitude °E", -180.0, 360.0, step=0.01, format="%.5f", key="mk_lon")
    st.text_input("Label", key="mk_label", max_chars=40)
    _mb = st.columns(2)
    if _mb[0].button("⇱ Put at analysed point", key="mk_here", width="stretch"):
        st.session_state["_pending_marker"] = (round(site.lat, 5), round(site.lon, 5))
        st.rerun()
    if _mb[1].button("▶ Analyse marker", key="mk_go", width="stretch", disabled=st.session_state["mk_lat"] > -80.0,
                     help="Runs all five models at the marker (80°S→90°S only)"):
        st.session_state["_pending_pick"] = (round(st.session_state["mk_lat"], 5), round(st.session_state["mk_lon"], 5))
        st.rerun()
    st.caption("Magenta ⊕ pin on the Explorer (2D, 3D, global), Site 3D, M3 slope & illumination, rank and coverage maps. "
               "Share it with `?mlat=-89.5&mlon=45&mlabel=Name` on the URL.")
MK = ({"lat": float(st.session_state["mk_lat"]), "lon": float(st.session_state["mk_lon"]),
       "label": (st.session_state.get("mk_label") or "My point").strip() or "My point"} if st.session_state.get("mk_on") else None)

if st.sidebar.button("↺ Reset all settings", help="Back to the default site, dates, array, battery, loads and equipment"):
    for _k in list(st.session_state.keys()):
        del st.session_state[_k]
    st.rerun()
st.sidebar.markdown("## 2 · Mission window")
MAX_DAYS, MAX_ROWS = 730, 9000      # up to 2 years; time steps capped so a run stays ~3 s (SPICE DE421 covers 1900-2050)
start_d = st.sidebar.date_input("Start (UTC)", min_value=date(2000, 1, 1), max_value=date(2049, 12, 1), key=sk("start_d", date(2026, 10, 15)))
start_h = st.sidebar.slider("Start hour (UTC)", 0, 23, key=sk("start_h", 0))
len_mode = st.sidebar.radio("Mission length by", ["Days", "End date"], horizontal=True, key=sk("len_mode", "Days"))
if len_mode == "Days":
    days = int(st.sidebar.number_input("Duration (days)", 3, MAX_DAYS, step=1, key=sk("days", 30),
                                       help="29.5 d = one synodic lunar day · 365 d = a full year"))
else:
    end_max = min(start_d + timedelta(days=MAX_DAYS), date(2049, 12, 31))
    sk("end_d", date(2027, 12, 31))
    st.session_state["end_d"] = min(max(st.session_state["end_d"], start_d + timedelta(days=3)), end_max)
    end_d = st.sidebar.date_input("End (UTC)", min_value=start_d + timedelta(days=3), max_value=end_max, key="end_d")
    days = max(3, (end_d - start_d).days)
STEPS = [10, 15, 30, 60, 120, 240]
ok_steps = [s_ for s_ in STEPS if days * 1440 / s_ <= MAX_ROWS]
sk("step", 30)
if st.session_state["step"] not in ok_steps:
    st.session_state["step"] = ok_steps[0]
step = st.sidebar.select_slider("Time step (min)", ok_steps, key="step",
                                help=f"Finer steps are disabled for long missions (≤ {MAX_ROWS:,} steps per run).")
end_dt = datetime.combine(start_d, time(start_h)) + timedelta(days=days)
st.sidebar.caption(f"{days} d · ends {end_dt:%Y-%m-%d %H:%M} UTC · {int(days * 1440 / step):,} steps"
                   + (" · long run: steps coarsened" if ok_steps[0] > 10 else ""))
start_iso = datetime.combine(start_d, time(start_h)).isoformat()

st.sidebar.markdown("## 3 · Solar array (Model 4)")
orient = st.sidebar.selectbox("Orientation", ["vertical_tracking", "vertical_fixed", "fixed_tilt", "horizontal", "dual_axis"],
                              key=sk("orient", "vertical_tracking"), help="South-pole Sun is near the horizon: vertical panels dominate.")
area = st.sidebar.number_input("Active area (m²)", 0.1, 50.0, step=0.1, key=sk("area", 1.2))
eff = st.sidebar.slider("Cell efficiency (AM0, 28 °C)", 0.10, 0.36, step=0.005, key=sk("eff", 0.295))
tilt = st.sidebar.slider("Tilt from horizontal (fixed_tilt)", 0, 90, key=sk("tilt", 75))
faz = st.sidebar.slider("Panel facing azimuth (fixed modes)", 0, 359, key=sk("faz", 0))
dust = st.sidebar.slider("Dust transmission factor", 0.5, 1.0, step=0.01, key=sk("dust", 0.95))
degr = st.sidebar.slider("Aging / radiation factor", 0.6, 1.0, step=0.01, key=sk("degr", 0.97))
tco = st.sidebar.number_input("Temp. coefficient (1/°C)", -0.01, 0.0, step=0.0001, format="%.4f", key=sk("tco", -0.0022))
mppt = st.sidebar.slider("MPPT / PCDU efficiency", 0.80, 1.0, step=0.01, key=sk("mppt", 0.96))

st.sidebar.markdown("## 4 · Battery (Model 5)")
cap = st.sidebar.number_input("Capacity (Wh)", 100.0, 200000.0, step=100.0, key=sk("cap", 4000.0))
soc0 = st.sidebar.slider("Initial SoC", 0.2, 1.0, step=0.05, key=sk("soc0", 0.9))
dod = st.sidebar.slider("Max depth of discharge", 0.3, 1.0, step=0.05, key=sk("dod", 0.8))
safe = st.sidebar.slider("Min safe SoC (shed science below)", 0.05, 0.8, step=0.05, key=sk("safe", 0.30))
ceff = st.sidebar.slider("Charge / discharge efficiency", 0.80, 1.0, step=0.01, key=sk("ceff", 0.95))
pmax = st.sidebar.number_input("Max charge/discharge power (W)", 10.0, 20000.0, step=10.0, key=sk("pmax", 1000.0))
st.sidebar.markdown("## 5 · Thermal & scenario")
mli = st.sidebar.slider("MLI effective emittance", 0.005, 0.10, step=0.005, key=sk("mli", 0.02))
tint = st.sidebar.slider("Internal survival set point (°C)", -40, 20, key=sk("tint", -20))
scenario = st.sidebar.selectbox("Operational scenario", list(en.SCENARIOS), format_func=lambda k: en.SCENARIOS[k], key=sk("scenario", "nominal"))
earth_rule = st.sidebar.selectbox("Earth visible when…", list(EARTH_RULES), key=sk("earth_rule", list(EARTH_RULES)[0]),
                                  format_func=lambda k: {"any_part": "any part of disc clears horizon",
                                                         "centre": "disc centre clears horizon", "whole_disc": "whole disc clears horizon"}[k])
st.sidebar.markdown("## 6 · Comms link (DTE)")
band = st.sidebar.selectbox("Band", ["X (8.45 GHz)", "S (2.25 GHz)", "Ka (32 GHz)"], key=sk("band", "X (8.45 GHz)"))
fghz = {"X": 8.45, "S": 2.25, "Ka": 32.0}[band[0] if not band.startswith("Ka") else "Ka"]
ptx = st.sidebar.slider("Transmit power (W)", 1.0, 100.0, step=1.0, key=sk("ptx", 10.0))
gtx = st.sidebar.slider("Lander antenna gain (dBi)", 0.0, 40.0, step=1.0, key=sk("gtx", 20.0))
gt = st.sidebar.slider("Ground station G/T (dB/K)", 30.0, 60.0, step=0.5, key=sk("gt", 50.0), help="DSN 34 m X-band ≈ 50–53 dB/K")
rmax = st.sidebar.select_slider("Radio max rate", [1e4, 1e5, 5e5, 1e6, 2e6, 1e7, 1e8], key=sk("rmax", 2e6), format_func=lambda v: f"{v/1e6:g} Mbps")
relay = st.sidebar.checkbox("Lunar relay orbiter (assumed)", key=sk("relay", False))
show_calc = st.sidebar.toggle("🧮 Show calculations", value=False, help="Reveal step-by-step equations with this point's numbers in every model tab.")

array = pg.SolarArrayConfig(area_m2=area, cell_efficiency=eff, orientation=orient, tilt_deg=tilt, facing_az_deg=faz,
                            dust_factor=dust, degradation_factor=degr, temp_coeff_per_c=tco, mppt_efficiency=mppt,
                            rps_w=float(st.session_state.get("equip_src", {}).get("rps_w", 0.0)))
battery = en.BatteryConfig(capacity_wh=cap, initial_soc=soc0, max_dod=dod, min_safe_soc=safe, charge_efficiency=ceff,
                           discharge_efficiency=ceff, max_charge_w=pmax, max_discharge_w=pmax)
thermal = en.ThermalConfig(mli_effective_emittance=mli, internal_min_temp_c=tint,
                           waste_heat_w=float(st.session_state.get("equip_src", {}).get("heat_w", 0.0)))
lbc = rf_link.LinkBudgetConfig(band=band[:2].strip(), freq_ghz=fghz, tx_power_w=ptx, tx_antenna_gain_dbi=gtx, gs_g_over_t_dbk=gt,
                               max_rate_bps=float(rmax), relay_enabled=relay)

if "subs" not in st.session_state:
    st.session_state["subs"] = pd.DataFrame([s.to_dict() for s in en.default_subsystems()])
if "plan" not in st.session_state:
    st.session_state["plan"] = act.ActivityPlan().to_json()


def subs_from_df(df: pd.DataFrame) -> list:
    out = []
    for r in df.to_dict("records"):
        if not r.get("name") or pd.isna(r.get("active_w")):
            continue
        out.append(en.Subsystem(**{k: (r[k] if not (isinstance(r[k], float) and np.isnan(r[k])) else en.Subsystem.__dataclass_fields__[k].default)
                                  for k in en.Subsystem.__dataclass_fields__ if k in r}))
    return out


@st.cache_data(show_spinner=False, max_entries=96)
def cached_run(site_json, start_iso, hours, step, array_json, subs_json, batt_json, therm_json, scenario, earth_rule, link_json, plan_json="{}"):
    s = Site(**json.loads(site_json))
    return run_mission(s, start_iso, hours, step, array=pg.SolarArrayConfig(**json.loads(array_json)),
                       subsystems=[en.Subsystem(**d) for d in json.loads(subs_json)], battery=en.BatteryConfig(**json.loads(batt_json)),
                       thermal=en.ThermalConfig(**json.loads(therm_json)), scenario=scenario, earth_rule=earth_rule,
                       link_budget=rf_link.LinkBudgetConfig(**json.loads(link_json)),
                       activities=act.ActivityPlan.from_json(json.loads(plan_json)) if plan_json != "{}" else None)


def run_args(s, sc=None, start=None, hrs=None, stp=None) -> tuple:
    subs = subs_from_df(st.session_state["subs"])
    return (json.dumps(s.to_dict()), start or start_iso, float(hrs or days * 24), float(stp or step),
            json.dumps(array.to_dict()), json.dumps([x.to_dict() for x in subs]), json.dumps(battery.to_dict()),
            json.dumps(asdict(thermal)), sc or scenario, earth_rule, json.dumps(lbc.to_dict()), json.dumps(st.session_state["plan"]))


def run_for(s, sc=None, start=None, hrs=None, stp=None):
    return cached_run(*run_args(s, sc, start, hrs, stp))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_scene(site_json):
    return viz3d.scene_for(Site(**json.loads(site_json)))


SIM_TPL = ROOT / "static" / "mission_sim.html"


@st.cache_data(show_spinner=False, max_entries=24)
def cached_sim_html(mode: str, args: tuple, tpl_mtime: float) -> str:
    """Model 5 Mission Sim / power dashboard page with the shared-timeline payload inlined (rebuilt only when inputs change)."""
    R_ = cached_run(*args)
    s_ = Site(**json.loads(args[0]))
    P_ = sim_payload.build_payload(
        R_, cached_scene(args[0]), s_, array=pg.SolarArrayConfig(**json.loads(args[4])), battery=en.BatteryConfig(**json.loads(args[6])),
        thermal=en.ThermalConfig(**json.loads(args[7])), lbc=rf_link.LinkBudgetConfig(**json.loads(args[10])),
        subsystems=[en.Subsystem(**d) for d in json.loads(args[5])], mode=mode)
    if mode == "dash":
        P_.pop("dem", None)   # the dashboard needs only the timeline
    return SIM_TPL.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/null", sim_payload.payload_json(P_).replace("</", "<\\/"))


def sim_page(mode: str, height: int):
    try:
        st.iframe(cached_sim_html(mode, run_args(site), SIM_TPL.stat().st_mtime), height=height)
    except Exception as e:  # noqa: BLE001
        st.error(f"Mission Sim unavailable: {e}")


# ------------------------------------------------------------------ header + run
st.markdown(
    f"<div class='brandbar'>{LOGO}<div><div class='t'>{TEAM}</div>"
    "<div class='s'>CLPS MISSION PLANNER · LUNAR SOUTH POLE 80°S–90°S · NASA SPACE APPS 2026</div></div></div>",
    unsafe_allow_html=True)
st.markdown(
    "<span class='tag'>M1 Illumination</span><span class='tag'>M2 Earth / DTE</span><span class='tag'>M3 Terrain</span>"
    "<span class='tag'>M4 Solar generation</span><span class='tag'>M5 Energy survival</span>"
    "<span class='src'> &nbsp; ① NASA Moon Trek · ② ASU LROC QuickMap · ③ JPL Horizons · ④ CSA · + NASA PGDA LOLA 20 m / 5 m, Diviner</span>",
    unsafe_allow_html=True)

with st.spinner(f"Running M1→M5 for {site.name} …"):
    try:
        R = run_for(site)
    except Exception as e:
        st.error(f"Mission chain failed: {e}")
        st.exception(e)
        st.stop()
tl = R.timeline
row = R.summary_row()
surv = R.survival
cls = "b-ok" if surv["status"].startswith("NOMINAL") else "b-warn" if surv["status"].startswith("DEGRADED") else "b-bad"
st.markdown(
    f"<div class='banner {cls}'>{site.id} · {site.name} · {site.lat:.5f}°, {site.lon:.5f}°E · {site.elev_m:,.0f} m &nbsp;|&nbsp; "
    f"{start_iso[:16]} UTC + {days} d &nbsp;|&nbsp; {en.SCENARIOS[scenario]}<br>"
    f"ENERGY: <b>{surv['status']}</b> &nbsp;|&nbsp; TERRAIN: <b>{R.terrain.get('status','n/a')}</b> &nbsp;|&nbsp; "
    f"FEASIBILITY INDEX {feasibility_score(row)}/100</div>", unsafe_allow_html=True)
c = st.columns(8)
kpi(c[0], "Sunlit (terrain)", fmt(R.solar["sunlit_pct"], 1, "%"), f"longest dark {fmt(R.solar['longest_darkness_h'],0,' h')}")
kpi(c[1], "Earth visible", fmt(R.dte["earth_visible_pct"], 1, "%"), f"longest blackout {fmt(R.dte['longest_dte_blackout_h'],0,' h')}")
kpi(c[2], "Sun + Earth", fmt(row["sun_and_earth_pct"], 1, "%"), "simultaneous")
kpi(c[3], "Generated", fmt(R.generation["total_gen_kwh"], 1, " kWh"), f"peak {fmt(R.generation['peak_gen_w'],0,' W')}")
kpi(c[4], "Demand", fmt(surv["total_demand_kwh"], 1, " kWh"), f"mean {fmt(surv['mean_demand_w'],0,' W')}")
kpi(c[5], "Min SoC", fmt(surv["min_soc_pct"], 0, "%"), f"floor {100*(1-battery.max_dod):.0f}%")
kpi(c[6], "Ops supported", fmt(surv["operational_time_supported_pct"], 1, "%"), f"deficit {fmt(surv['energy_deficit_kwh'],1,' kWh')}")
ttc = surv["hours_to_critical_soc"]
kpi(c[7], "Time to critical SoC", "never" if ttc is None else fmt(ttc, 0, " h"), f"safe SoC {100*max(safe,1-dod):.0f}%")
st.markdown(f"<div class='src'>Horizon: {R.horizon_source} · Ephemeris: {tl['geometry_source'].iloc[0]} "
            f"(validated against JPL Horizons, max |Δel| 1.3e-5°)</div>", unsafe_allow_html=True)
_lib = ho.libration_stats(tl)
st.markdown(
    "<div class='banner' style='background:rgba(74,168,255,.08);border:1px solid rgba(74,168,255,.45);color:#cfe8ff;font-family:inherit'>"
    "🌍 <b>On Earth</b> the Sun climbs high every 24 h and a nearby hill costs you minutes of light. &nbsp; 🌑 <b>Here</b> one day–night cycle "
    f"lasts <b>29.5 Earth days</b>, the Sun never rises above <b>{tl.sun_alt.max():.1f}°</b> and circles the horizon, so a ridge tens of km away "
    f"can mean <b>{R.solar['longest_darkness_h'] / 24:.1f} days</b> of darkness. Earth hangs almost still (it wobbles <b>{_lib['earth_alt_swing']:.1f}°</b> "
    f"in elevation from libration) and is above this site's horizon <b>{R.dte['earth_visible_pct']:.0f}%</b> of the time. "
    "Nights reach −200 °C, so batteries must keep heaters alive through them. <span class='src'>(More in 📘 Guide.)</span></div>",
    unsafe_allow_html=True)

EXT = ext_mod.discover()
tabs = st.tabs(["🗺 South Pole Explorer", "🌐 Site 3D", "⏱ Mission clock", "☀️ M1 Illumination", "🌍 M2 Earth / DTE & link", "⛰ M3 Terrain",
                "⚡ M4 Power generation", "🔋 M5 Energy survival", "🎮 M5 Mission Sim", "⚖️ Compare · seasons", "🏆 Rank & compare points", "🔀 Diff & handoff", "📘 Guide", "🛰 Data sources", "🧪 Validation",
                "📄 Report"] + [f"🧩 {e.TAB_TITLE}" for e in EXT])
T_EXPL, T_3D, T_CLOCK, T_M1, T_M2, T_M3, T_M4, T_M5, T_SIM, T_CMP, T_RANK, T_DIFF, T_GUIDE, T_DATA, T_VAL, T_REP = range(16)


def calc_box(title: str, md_text: str):
    if show_calc:
        with st.container(border=True):
            st.markdown(f"**🧮 {title}**")
            st.markdown(md_text)


def shade_bands(fig, t, flag, color, row=None, col=None):
    f = np.asarray(flag, bool)
    if not f.any():
        return
    ch = np.flatnonzero(np.diff(np.r_[0, f.astype(int), 0]))
    for a, b in zip(ch[::2], ch[1::2]):
        kw = dict(x0=t.iloc[a], x1=t.iloc[min(b, len(t) - 1)], fillcolor=color, line_width=0, layer="below")
        if row:
            fig.add_vrect(row=row, col=col, **kw)
        else:
            fig.add_vrect(**kw)


def _pick_latlon(pt, kind: str):
    """Lat/lon from a Plotly click. Uses the point's x/y (always present); customdata may arrive as list OR dict."""
    if not isinstance(pt, dict):
        try:
            pt = dict(pt)
        except Exception:
            return None
    x, y = pt.get("x"), pt.get("y")
    try:
        if kind == "global":
            lat, lon = float(y), float(x)
        else:
            from clps_engine.rasters import ps_to_latlon as _p2l
            la, lo = _p2l(float(x) * 1000.0, float(y) * 1000.0)
            lat, lon = float(la), float(lo)
    except (TypeError, ValueError):
        cd = pt.get("customdata")
        if isinstance(cd, dict):
            cd = [cd.get(k) for k in sorted(cd, key=lambda k: int(k) if str(k).isdigit() else 99)]
        if not cd or len(cd) < 2:
            return None
        lat, lon = float(cd[0]), float(cd[1])
    if not (-90 <= lat <= 90) or (kind != "global" and lat > -79.0):
        return None
    return round(lat, 5), round(lon, 5)


def _sel_points(ev):
    try:
        return ev["selection"]["points"]
    except Exception:
        try:
            return ev.selection.points
        except Exception:
            return []


# ------------------------------------------------------------------ explorer
with tabs[T_EXPL]:
    if not viz_pole.stack():
        st.warning("Pole-wide products missing - run `python tools/build_polewide.py all`.")
    else:
        cc = st.columns([2, 1.3, 1, 1])
        layer = cc[0].selectbox("Map layer", list(viz_pole.LAYERS), format_func=lambda k: viz_pole.LAYERS[k][0], key="pole_layer")
        mode3d = cc[1].radio("View", ["2D map (click to pick)", "3D south pole", "Global map (presets)"], key="pole_view", horizontal=True)
        show_proj = cc[2].checkbox("Project candidates", True)
        cc[2].checkbox("Crater labels", True, key="craters")
        vex = cc[3].slider("3D vertical ×", 1.0, 15.0, 6.0, 1.0)
        sel = (site.lat, site.lon)
        pt_i = st.slider("Explorer time (drives the live Sun, Earth and 'Illumination NOW' layer)", 0, len(tl) - 1, 0, key="pole_t",
                         format="step %d")
        t_pole = tl.time_utc.iloc[pt_i]
        st.markdown(f"<div class='src'>⏱ {t_pole:%Y-%m-%d %H:%M} UTC - pick layer <b>Illumination NOW</b> to see real shadows sweep the cap</div>",
                    unsafe_allow_html=True)
        if mode3d.startswith("Global"):
            gfig = viz_pole.global_map([x for x in ALL.values() if x.source.startswith("preset") or x.source in ("baseline", "research")], sel)
            if gfig is not None:
                mk_mod.add_global(gfig, MK)
                evg = st.plotly_chart(gfig, width="stretch", on_select="rerun", selection_mode=("points",), key="globalmap")
                gp = _sel_points(evg)
                new = _pick_latlon(gp[0], "global") if gp else None
                if new is not None and new != tuple(st.session_state["pick"]):
                    st.session_state["_pending_pick"] = new   # applied before the sidebar widgets are drawn
                    st.rerun()
                st.caption("LOLA LDEM_16 global relief (1.9 km). Click a mission marker to analyse it. Non-polar points use the global DEM horizon; "
                           "landing-scale terrain is only assessed inside the 80°S cap.")
        elif mode3d.startswith("2D"):
            fig = viz_pole.pole_map(layer, PROJECT if show_proj else [], DISC, sel, t_utc=t_pole)
            if st.session_state.get("craters", True):
                viz_pole.add_craters(fig)
            mk_mod.add_pole2d(fig, MK)
            ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points",), key="polemap",
                                 config={"displaylogo": False, "scrollZoom": True})
            pts = _sel_points(ev)
            if pts:
                new = _pick_latlon(pts[0], "pole")
                if new is not None:
                    if new != tuple(round(v, 5) for v in st.session_state["pick"]) or not loc_mode.startswith("Map"):
                        st.session_state["_pending_pick"] = new
                        st.rerun()
            st.caption("Click any cell (or a ★ discovery) to run all five models at that exact point. Scroll to zoom, drag to pan. "
                       "Rings: 80, 82.5, 85, 87.5°S; spokes every 30° longitude. Each layer names its product or model.")
        else:
            fig3p = viz_pole.pole_3d(layer, PROJECT if show_proj else [], DISC, sel, vexag=vex, t_utc=t_pole)
            if MK:
                mk_mod.add_pole3d(fig3p, MK, viz_pole.value_at("elevation", MK["lat"], MK["lon"]), vex)
            st.plotly_chart(fig3p, width="stretch")
            st.caption("Rotate / zoom / pan. Geometry: LOLA 80°S 20 m DEM averaged to a 2 km display mesh over the full 608 × 608 km "
                       "polar cap; colour = selected layer; white pin = analysed point.")
        a, b = st.columns([1, 1])
        with a:
            st.markdown("**LunarSouth discoveries** - best composite cells, ≥ 8 km apart, from the whole cap")
            if len(DISC):
                st.dataframe(DISC, width="stretch", hide_index=True, height=330)
        with b:
            lat_ = viz_pole.lattice()
            if lat_ is not None:
                st.markdown(f"**Whole-pole Model 1/2/4 lattice · {len(lat_):,} points · year 2027 · 5 km spacing**")
                figs = go.Figure(go.Scattergl(x=lat_.sunlit_pct, y=lat_.earth_pct, mode="markers",
                                              marker=dict(size=4, color=lat_.mean_specific_power_w_m2, colorscale="YlOrBr", showscale=True,
                                                          colorbar=dict(title="W/m²")),
                                              customdata=np.c_[lat_.lat, lat_.lon],
                                              hovertemplate="%{customdata[0]:.3f}, %{customdata[1]:.3f}<br>sun %{x:.1f}% · earth %{y:.1f}%<extra></extra>"))
                figs.add_trace(go.Scatter(x=[R.solar["sunlit_pct"]], y=[R.dte["earth_visible_pct"]], mode="markers", name="this point (mission window)",
                                          marker=dict(size=14, symbol="circle-open", color="#fff", line=dict(width=3))))
                figs.update_layout(**PLOT, height=330, xaxis_title="sunlit % (2027)", yaxis_title="Earth visible %",
                                   title="Trade space: Sun vs Earth access, coloured by power", showlegend=False)
                st.plotly_chart(figs, width="stretch")
        calc_box("Suitability index (map layer 'suitability')", r"""
$S = 100\,[\,0.40\,\min(V_\odot/0.8,1) + 0.25\,\min(V_\oplus/0.8,1) + 0.30\,f_{\le10^\circ} + 0.05\,(1-f_{PSR})\,]$

- $V_\odot, V_\oplus$: LOLA AVGVISIB Sun/Earth (Moon Trek products) inside 85°S; Model 1/2 lattice (SPICE × LOLA horizon) outside.
- $f_{\le10^\circ}$: share of **20 m** pixels with slope ≤10° in each 1 km cell (LDEM_80S_20MPP, slope at native resolution).
- $f_{PSR}$: LPSR_80S_20MPP permanently-shadowed share. Discoveries = local maxima ≥ 8 km apart.
- Lattice power: Model 4 chain per 1 m² vertical-tracking array for every 2 h of 2027, horizon ray-cast from LOLA at each lattice point.""")

# ------------------------------------------------------------------ site 3D
scene = cached_scene(json.dumps(site.to_dict()))
with tabs[T_3D]:
    cc = st.columns([3, 1, 1])
    idx = cc[0].slider("Mission time", 0, len(tl) - 1, 0, key="t3d")
    layer3 = cc[1].selectbox("Terrain layer", list(viz3d.LAYERS), format_func=lambda k: viz3d.LAYERS[k][0])
    vex3 = cc[2].slider("Vertical exaggeration", 1.0, 6.0, 2.0, 0.5)
    r_ = tl.iloc[idx]
    st.markdown(
        f"<div class='src'>{r_['time_utc']:%Y-%m-%d %H:%M} UTC · Sun el {r_['sun_alt']:.2f}° az {r_['sun_az']:.1f}° "
        f"(horizon {r_['sun_horizon_deg']:.2f}°, disc {100*r_['sun_disc_fraction']:.0f}% visible) · "
        f"Earth el {r_['earth_alt']:.2f}° az {r_['earth_az']:.1f}° (horizon {r_['earth_horizon_deg']:.2f}°) · "
        f"P_gen {r_['gen_w']:.0f} W · load {r_['load_served_w']:.0f} W · SoC {r_['soc_pct']:.0f}%</div>", unsafe_allow_html=True)
    cdn = ev_mod.countdowns(R.events, r_.time_utc).set_index("event")
    hud = " &nbsp;·&nbsp; ".join(f"{cdn.loc[k,'icon']} {k.title()} {cdn.loc[k,'t_minus']}" for k in
                                 ("SUNSET" if bool(r_.sunlit) else "SUNRISE", "EARTHSET" if bool(r_.earth_visible) else "EARTHRISE",
                                  "LINK LOS" if bool(r_.link_up) else "LINK AOS", "SOC < SAFE"))
    st.markdown(f"<div class='banner b-warn' style='font-size:13px'>⏱ {hud}</div>", unsafe_allow_html=True)
    if scene is None:
        st.warning("No DEM covers this point - 3D scene unavailable.")
    else:
        fig = viz3d.scene_figure(site, scene, layer=layer3, sun_alt=float(r_.sun_alt), sun_az=float(r_.sun_az),
                                 earth_alt=float(r_.earth_alt), earth_az=float(r_.earth_az), sunlit_site=bool(r_.sunlit),
                                 earth_visible_site=bool(r_.earth_visible), mask=R.mask,
                                 neighbours=viz3d.neighbours_in_scene(site, ALL.values()), vexag=vex3,
                                 time_label=f"{r_['time_utc']:%Y-%m-%d %H:%M} UTC")
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02, showarrow=False, align="left",
                           text="<br>".join(f"{cdn.loc[k,'icon']} {k}: {cdn.loc[k,'t_minus']}" for k in ("SUNSET", "SUNRISE", "EARTHSET", "EARTHRISE", "LINK LOS")),
                           font=dict(size=12, color="#bfefff", family="JetBrains Mono, monospace"), bgcolor="rgba(4,8,15,.75)",
                           bordercolor="rgba(56,217,255,.4)", borderwidth=1)
        mk_mod.add_scene3d(fig, MK, site, scene, vex3)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption("24 × 24 km local east/north grid from LOLA 80 m. 'Illumination at selected time' = Lambert × cast shadow ray-marched over "
                   "this DEM; terrain beyond the window is the 200 km horizon ring. Solid ray = visible, dashed = blocked.")

# ------------------------------------------------------------------ mission clock (synchronised multi-site / multi-date)
with tabs[T_CLOCK]:
    st.markdown("**One clock, many lanes.** Every lane runs the full M1→M5 chain; the scrubber moves all lanes together "
                "(elapsed mission time from each lane's own start date).")
    lane_pool = {**{k: v for k, v in ALL.items()}, **({"POINT": site} if site.id == "POINT" else {})}
    default_lanes = [site.id] + [k for k in ("LN-001", "LM1", "A15") if k in lane_pool and k != site.id][:2]
    lc = st.columns([3, 2])
    lane_ids = lc[0].multiselect("Lanes (locations)", list(lane_pool), default=default_lanes, max_selections=4)
    shifts_txt = lc[1].text_input("Per-lane start shift (days, comma-separated)", ",".join("0" for _ in lane_ids),
                                  help="e.g. 0,14 compares the same site at two landing dates")
    try:
        lane_shift = [float(x) for x in shifts_txt.split(",")] + [0.0] * 4
    except ValueError:
        lane_shift = [0.0] * 8
    lanes = []
    for i, k in enumerate(lane_ids):
        s0 = (datetime.fromisoformat(start_iso) + timedelta(days=lane_shift[i])).isoformat()
        lanes.append((k, s0, run_for(lane_pool[k], start=s0)))
    if lanes:
        n = min(len(l[2].timeline) for l in lanes)
        st.session_state["clock_i"] = int(min(max(st.session_state.get("clock_i", 0), 0), n - 1))
        b1, b2, b3, b4 = st.columns([1, 1, 1, 6])
        stepn = max(1, int(6 * 60 / step))
        if b1.button("⏮ −6 h"):
            st.session_state["clock_i"] = max(0, st.session_state["clock_i"] - stepn)
        if b2.button("⏭ +6 h"):
            st.session_state["clock_i"] = min(n - 1, st.session_state["clock_i"] + stepn)
        if b3.button("⏩ next event"):
            r0 = lanes[0][2]
            tnow = r0.timeline.time_utc.iloc[st.session_state["clock_i"]]
            fut = r0.events[r0.events.time_utc > tnow]
            if len(fut):
                st.session_state["clock_i"] = int(min(n - 1, np.searchsorted(r0.timeline.time_utc.values, fut.time_utc.iloc[0].to_datetime64())))
        ci_clock = b4.slider("Elapsed mission time (steps)", 0, n - 1, key="clock_i")
        elapsed_h = ci_clock * step / 60
        st.markdown(f"<div class='src'>T+{elapsed_h/24:.0f} d {elapsed_h%24:04.1f} h</div>", unsafe_allow_html=True)
        cols = st.columns(len(lanes))
        for col, (k, s0, rr) in zip(cols, lanes):
            r1 = rr.timeline.iloc[ci_clock]
            cdl = ev_mod.countdowns(rr.events, r1.time_utc)
            chip = lambda ok, on, off: f"<span class='tag' style='color:{'#7dffb0' if ok else '#ff9d93'}'>{on if ok else off}</span>"
            col.markdown(f"<div class='card'><div class='p'>{k} · start {s0[:10]}</div><h4>{lane_pool[k].name[:38]}</h4>"
                         f"<div class='src'>{r1.time_utc:%Y-%m-%d %H:%M} UTC</div>"
                         + chip(bool(r1.sunlit), "☀ SUN", "☾ DARK") + chip(bool(r1.earth_visible), "🌍 EARTH", "🌍 BELOW")
                         + chip(bool(r1.link_up), f"📡 {r1.dsn_station} {r1.link_rate_bps/1e3:.0f} kbps", "📡 NO LINK")
                         + f"<div style='margin-top:6px'>⚡ {r1.gen_w:.0f} W · load {r1.load_served_w:.0f} W · 🔋 {r1.soc_pct:.0f}%</div>"
                         + "<ul>" + "".join(f"<li>{x.icon} {x.event.title()}: <b>{x.t_minus}</b></li>" for x in cdl.itertuples()
                                            if x.event in ("SUNSET", "SUNRISE", "EARTHSET", "EARTHRISE", "LINK LOS", "SOC < SAFE")) + "</ul></div>",
                         unsafe_allow_html=True)
        figc = make_subplots(rows=len(lanes), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                             subplot_titles=[f"{k} (start {s0[:10]})" for k, s0, _ in lanes])
        for i, (k, s0, rr) in enumerate(lanes, 1):
            t_el = np.arange(len(rr.timeline))[:n] * step / 60 / 24
            tt = rr.timeline.iloc[:n]
            z = np.vstack([tt.sunlit.astype(int) * 2, tt.earth_visible.astype(int) * 3, tt.link_up.astype(int) * 4, (~tt.shedding).astype(int) * 1])
            figc.add_trace(go.Heatmap(x=t_el, y=["Sun", "Earth", "Link", "Power OK"], z=z, showscale=False,
                                      colorscale=[[0, "#0b1220"], [0.25, "#3ddc97"], [0.5, "#ffd54a"], [0.75, "#4aa8ff"], [1, "#38d9ff"]]), i, 1)
            figc.add_vline(x=elapsed_h / 24, line_color="#ffffff", line_width=2, row=i, col=1)
        figc.update_layout(**PLOT, height=170 * len(lanes) + 60, title="Synchronised state timeline (x = elapsed days; white line = clock)")
        st.plotly_chart(figc, width="stretch")
        st.markdown("**Event log - first lane** (terrain-corrected Sun/Earth events, link AOS/LOS, DSN handoffs, power events)")
        evs = lanes[0][2].events.copy()
        tnow = lanes[0][2].timeline.time_utc.iloc[ci_clock]
        evs["T-minus h"] = ((evs.time_utc - tnow).dt.total_seconds() / 3600).round(1)
        st.dataframe(evs[["time_utc", "icon", "event", "detail", "T-minus h"]], width="stretch", hide_index=True, height=300)

ci = int(st.session_state.get("t3d", 0))
rc = tl.iloc[min(ci, len(tl) - 1)]

# ------------------------------------------------------------------ M1
with tabs[T_M1]:
    c = st.columns(6)
    kpi(c[0], "Sunlit time", fmt(R.solar["sunlit_pct"], 1, "%"), f"{fmt(R.solar['sunlit_hours'],0)} h of {days*24} h")
    kpi(c[1], "Mean disc visible", fmt(R.solar["mean_disc_fraction_pct"], 1, "%"), f"partial {fmt(R.solar['partial_pct'],1,'%')} of time")
    kpi(c[2], "Longest sunlight", fmt(R.solar["longest_sunlight_h"], 0, " h"))
    kpi(c[3], "Longest darkness", fmt(R.solar["longest_darkness_h"], 0, " h"))
    kpi(c[4], "Max Sun elevation", fmt(R.solar["max_sun_alt_deg"], 2, "°"))
    kpi(c[5], "Terrain sunrises", str(R.solar["n_sunrises"]))
    fig = go.Figure()
    shade_bands(fig, tl.time_utc, tl.sunlit, "rgba(255,213,74,.07)")
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.sun_alt, name="Sun elevation", line=dict(color="#ffd54a")))
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.sun_horizon_deg, name="Terrain horizon toward Sun", line=dict(color="#ff8c42", dash="dot")))
    fig.add_trace(go.Scatter(x=tl.time_utc, y=100 * tl.sun_disc_fraction, name="Disc visible %", yaxis="y2", line=dict(color="#7dffb0", width=1)))
    fig.update_layout(**PLOT, height=380, hovermode="x unified", yaxis_title="deg", title="Solar elevation vs local terrain horizon (shaded = sunlit)",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="%"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns([1, 1])
    with a:
        pol = go.Figure()
        if R.mask is not None:
            pol.add_trace(go.Scatterpolar(theta=R.mask[0], r=90 - np.asarray(R.mask[1]), mode="lines", fill="toself",
                                          fillcolor="rgba(120,130,140,.35)", line=dict(color="#9fb3c8"), name="Terrain horizon"))
        k = max(1, len(tl) // 1500)
        pol.add_trace(go.Scatterpolar(theta=tl.sun_az[::k], r=90 - tl.sun_alt[::k], mode="markers", marker=dict(size=3, color="#ffd54a"), name="Sun track"))
        pol.add_trace(go.Scatterpolar(theta=tl.earth_az[::k], r=90 - tl.earth_alt[::k], mode="markers", marker=dict(size=3, color="#4aa8ff"), name="Earth track"))
        pol.update_layout(**PLOT, height=430, title="Sky plot (zenith centre; r = 90° − elevation)",
                          polar=dict(angularaxis=dict(direction="clockwise", rotation=90), radialaxis=dict(range=[75, 100], tickvals=[80, 85, 90, 95])))
        st.plotly_chart(pol, width="stretch")
    with b:
        st.markdown("**Terrain-corrected sunrise / sunset windows**")
        st.dataframe(R.solar_windows.assign(hours=R.solar_windows.hours.round(1)), width="stretch", height=230)
        st.markdown("**Independent illumination products at this point**")
        st.dataframe(pd.DataFrame([
            {"product": "① Moon Trek · LOLA AVGVISIB_85S_060M (model beyond 85S)", "res": "60 m / 5 km", "period": "18.6-yr avg",
             "value": fmt(100 * viz_pole.value_at("sun", site.lat, site.lon), 1, "%")},
            {"product": "② QuickMap · WAC_POLE_ILL_PCT_SOUTH_100M", "res": "100 m", "period": "1 lunar yr images",
             "value": fmt(viz_pole.value_at("wac", site.lat, site.lon), 1, "%")},
            {"product": "This run (mean disc visible)", "res": "point + 200 km horizon", "period": f"{days} d", "value": fmt(R.solar["mean_disc_fraction_pct"], 1, "%")},
        ]), width="stretch", hide_index=True)
    calc_box(f"Model 1 at {rc['time_utc']:%Y-%m-%d %H:%M} UTC (time index {ci}, set on Site 3D slider)", f"""
1. Sun direction (SPICE DE421 / MOON_ME, validated vs JPL Horizons): elevation $e_\\odot$ = **{rc.sun_alt:.4f}°**, azimuth $A_\\odot$ = **{rc.sun_az:.3f}°**, distance **{rc.sun_dist_au:.5f} AU**
2. Terrain horizon interpolated at $A_\\odot$ from the 200 km LOLA ray-cast: $h(A_\\odot)$ = **{rc.sun_horizon_deg:.4f}°**
3. Disc centre above horizon in radii: $d = (e_\\odot - h)/r_\\odot$ = ({rc.sun_alt:.4f} − {rc.sun_horizon_deg:.4f}) / {rc.sun_radius:.4f} = **{np.clip((rc.sun_alt-rc.sun_horizon_deg)/rc.sun_radius,-1,1):.3f}** (clipped ±1)
4. Visible disc fraction (circular segment): $f = 1 - [\\arccos d - d\\sqrt{{1-d^2}}]/\\pi$ = **{rc.sun_disc_fraction:.4f}**
5. Sunlit flag (centre rule) $e_\\odot > h$ → **{bool(rc.sunlit)}**
6. Horizon ray-cast: $\\epsilon = \\operatorname{{atan2}}((R+h_t)\\cos g - (R+h_o),\\ (R+h_t)\\sin g)$, $g = d/R$, R = 1737.4 km, $h_o$ = ground + 2 m mast.""")

# ------------------------------------------------------------------ M2
with tabs[T_M2]:
    c = st.columns(6)
    kpi(c[0], "Earth visible", fmt(R.dte["earth_visible_pct"], 1, "%"))
    kpi(c[1], "Total DTE time", fmt(R.dte["earth_visible_pct"] * days * 24 / 100, 0, " h"))
    kpi(c[2], "Longest window", fmt(R.dte["longest_dte_h"], 0, " h"))
    kpi(c[3], "Longest blackout", fmt(R.dte["longest_dte_blackout_h"], 0, " h"))
    kpi(c[4], "Earthrises", str(R.dte["n_windows"]))
    kpi(c[5], "RF-geometry 'good'", fmt(R.rf.get("good_pct"), 1, "%"), "≥2° margin")
    fig = go.Figure()
    shade_bands(fig, tl.time_utc, tl.earth_visible, "rgba(74,168,255,.08)")
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.earth_alt, name="Earth centre elevation", line=dict(color="#4aa8ff")))
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.earth_horizon_deg, name="Terrain horizon toward Earth", line=dict(color="#9fb3c8", dash="dot")))
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.visibility_margin_deg, name="Visibility margin", line=dict(color="#7dffb0", width=1)))
    fig.add_hline(y=0, line_color="#ff5d73", line_dash="dash")
    fig.update_layout(**PLOT, height=380, hovermode="x unified", yaxis_title="deg", title="Earth elevation vs terrain horizon (shaded = geometric DTE)",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    a.markdown("**Earthrise / Earthset windows**")
    a.dataframe(R.dte_windows.assign(hours=R.dte_windows.hours.round(1)), width="stretch", height=260)
    b.metric("① Moon Trek · AVGVISIB_85S_060M_201608_EARTH at this point", fmt(100 * viz_pole.value_at("earth", site.lat, site.lon), 1, "%"))
    b.info("Geometric visibility (above) is necessary but not sufficient - the link budget below decides whether data actually flows.")
    L = R.link
    st.markdown("#### DTE link budget · DSN coverage · handoffs" + (" · relay" if relay else ""))
    c = st.columns(6)
    kpi(c[0], "Link closed", fmt(L["link_closed_pct"], 1, "%"), f"geometric {fmt(L['geometric_visible_pct'],1,'%')}")
    kpi(c[1], "Data volume", fmt(L["data_volume_gbit"], 1, " Gbit"), f"{days} d")
    kpi(c[2], "Mean rate (up)", fmt(L["mean_rate_when_up_kbps"], 0, " kbps"))
    kpi(c[3], "Median C/N0", fmt(L["median_cn0_dbhz"], 1, " dB-Hz"), f"FSPL {fmt(L['mean_fspl_db'],1,' dB')}")
    kpi(c[4], "DSN handoffs", str(L["dsn_handoffs"]), " · ".join(f"{k[:3]} {v:.0f}%" for k, v in L["station_share_pct"].items()))
    kpi(c[5], "Longest link outage", fmt(L["longest_link_outage_h"], 0, " h"), f"Earth up, no station {fmt(L['visible_but_no_station_pct'],1,'%')}")
    figl = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.35, 0.25], vertical_spacing=0.05)
    figl.add_trace(go.Scatter(x=tl.time_utc, y=tl.link_rate_bps / 1e3, name="Achievable rate kbps", fill="tozeroy", line=dict(color="#38d9ff")), 1, 1)
    if relay:
        figl.add_trace(go.Scatter(x=tl.time_utc, y=tl.relay_rate_bps / 1e3, name="via relay kbps", line=dict(color="#9b5cff")), 1, 1)
    for k, colr in zip(rf_link.DSN, ("#ffd54a", "#ff8c42", "#3ddc97")):
        figl.add_trace(go.Scatter(x=tl.time_utc, y=tl[f"dsn_el_{k}"], name=f"Moon el @ {k}", line=dict(color=colr, width=1)), 2, 1)
    figl.add_hline(y=lbc.gs_elev_mask_deg, line_dash="dot", line_color="#ff5d73", row=2, col=1)
    code = tl.dsn_station.map({"none": 0, "Goldstone": 1, "Madrid": 2, "Canberra": 3}).fillna(0)
    figl.add_trace(go.Heatmap(x=tl.time_utc, y=["serving DSN"], z=[code.values], showscale=False,
                              colorscale=[[0, "#11182a"], [0.33, "#ffd54a"], [0.66, "#ff8c42"], [1, "#3ddc97"]],
                              hovertemplate="%{x}<br>0 none · 1 Goldstone · 2 Madrid · 3 Canberra: %{z}<extra></extra>"), 3, 1)
    figl.update_layout(**PLOT, height=520, hovermode="x unified", title=f"{band} link: rate, DSN station view of the Moon, serving station & handoffs",
                       legend=dict(orientation="h", y=-0.1))
    figl.update_yaxes(title_text="kbps", row=1, col=1)
    figl.update_yaxes(title_text="deg", row=2, col=1)
    st.plotly_chart(figl, width="stretch")
    st.caption(L["note"] + " DSN complexes: Goldstone, Madrid, Canberra (published locations); 10° elevation mask; "
               "relay is a deterministic availability assumption, not a specific orbiter.")
    calc_box(f"Link budget at {rc['time_utc']:%Y-%m-%d %H:%M} UTC", f"""
| term | expression | value |
|---|---|---|
| Range | SPICE Earth distance | **{rc.earth_dist_km:,.0f} km** |
| FSPL | $20\\log_{{10}}(4\\pi d f/c)$, f = {fghz} GHz | **{rc.fspl_db:.2f} dB** |
| EIRP | $10\\log P_{{tx}} + G_{{tx}} - L_{{tx}} - L_{{point}}$ = 10log({ptx}) + {gtx} − {lbc.tx_losses_db} − L_p | **{rc.eirp_dbw:.2f} dBW** |
| C/N0 | EIRP − FSPL − L_misc({lbc.misc_losses_db}) + G/T({gt}) + 228.6 | **{rc.cn0_dbhz:.2f} dB-Hz** |
| Rate | $10^{{(C/N_0 - E_b/N_0 - M)/10}}$, Eb/N0 {lbc.required_ebn0_db} dB, margin {lbc.required_margin_db} dB, cap {rmax/1e6:g} Mbps | **{rc.link_rate_bps/1e3:,.1f} kbps** |
| Station | highest DSN complex with Moon > {lbc.gs_elev_mask_deg}° | **{rc.dsn_station}** |""")
    calc_box(f"Model 2 at {rc['time_utc']:%Y-%m-%d %H:%M} UTC", f"""
- Earth centre elevation $e_\\oplus$ = **{rc.earth_alt:.4f}°**, azimuth **{rc.earth_az:.3f}°**, range **{rc.earth_dist_km:,.0f} km**; angular radius $r_\\oplus$ = asin(6378.137 / range) = **{rc.earth_radius:.4f}°**
- Horizon toward Earth $h$ = **{rc.earth_horizon_deg:.4f}°**
- Margin (rule '{earth_rule}', k = {EARTH_RULES[earth_rule]:+.0f}): $m = e_\\oplus + k\\,r_\\oplus - h$ = **{rc.visibility_margin_deg:.4f}°** → visible = **{bool(rc.earth_visible)}**""")

# ------------------------------------------------------------------ M3
with tabs[T_M3]:
    T = R.terrain
    c = st.columns(5)
    best = T.get("best", {})
    kpi(c[0], "Terrain index", fmt(T.get("terrain_score"), 1), T.get("status", ""))
    kpi(c[1], "≤10° area (1 km)", fmt(best.get("pct_slope_le_10deg"), 1, "%"), T.get("best_source", ""))
    kpi(c[2], "Slope p90", fmt(best.get("slope_p90_deg"), 1, "°"))
    kpi(c[3], "Relief (1 km)", fmt(best.get("relief_m"), 0, " m"))
    kpi(c[4], "Horizon >2° az", fmt(T.get("horizon_pct_above_2deg"), 0, "%"), f"mean {fmt(T.get('horizon_mean_deg'),2,'°')}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Hazards**")
        for h in T.get("hazards", []) or ["none flagged at this resolution"]:
            st.markdown(f"- ⚠️ {h}")
        st.markdown("**Uncertainty / resolution limits**")
        for u in T.get("uncertainty", []) or ["—"]:
            st.markdown(f"- {u}")
        if "all_resolutions" in T:
            ar = T["all_resolutions"]
            st.markdown("**Same 1 km footprint at every covering DEM (native resolutions, never mixed)**")
            st.dataframe(ar.drop(columns=["site_id"]).round(2), width="stretch", hide_index=True)
            figr = go.Figure(go.Bar(x=ar["product"] + " (" + ar.resolution_m.map(lambda v: f"{v:g}") + " m)", y=ar.pct_slope_le_10deg,
                                    marker_color="#1f9d6b", text=ar.pct_slope_le_10deg.round(1), textposition="outside"))
            figr.update_layout(**PLOT, height=260, title="Resolution effect: % of footprint ≤10°", yaxis_title="%")
            st.plotly_chart(figr, width="stretch")
    with b:
        if scene is not None:
            figs = go.Figure(go.Heatmap(x=scene["x_m"] / 1000, y=scene["y_m"] / 1000, z=scene["slope_deg"], zmin=0, zmax=30,
                                        colorscale=[[0, "#0b3d2e"], [0.33, "#1f9d6b"], [0.5, "#f2c14e"], [0.75, "#f06a3b"], [1, "#b3122e"]],
                                        colorbar=dict(title="deg")))
            figs.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=12, color="#38d9ff"), name=site.id))
            mk_mod.add_local2d(figs, MK, site, float(np.max(np.abs(scene["x_m"]))) / 1000)
            figs.update_layout(**PLOT, height=430, title="Slope, 24 km window (LOLA 80 m)", yaxis_scaleanchor="x")
            st.plotly_chart(figs, width="stretch")
    st.info("Screening only - not landing certification. A low mean slope does not make a site safe: p90 slope, relief, "
            "unresolved boulders/craters below the DEM posting and PSR proximity are reported separately.")
    calc_box("Model 3", r"""
- Slope per pixel: $\theta = \arctan\sqrt{(\partial z/\partial x)^2 + (\partial z/\partial y)^2}$ (central differences at each DEM's **native** posting; nodata never bridged).
- Terrain index $= 0.5\,P_{\le10^\circ} + 0.3\max(0, 100 - 4\max(\theta_{90}-5,0)) + 0.2\max(0, 100 - \text{relief}/5)$ from the finest DEM covering the point
  (NAC 3.5 m → PGDA 5 m → LOLA 20 m → 80 m).""")

# ------------------------------------------------------------------ M4 power generation
@st.cache_data(show_spinner=False, max_entries=12)
def cached_illum(args: tuple, n_samples: int):
    R_ = cached_run(*args)
    s_ = Site(**json.loads(args[0]))
    sc_ = cached_scene(args[0])
    if sc_ is None:
        return None
    far_, src_ = sim_payload.far_field_mask(s_, sc_, R_.mask)
    out = ho.illumination_map(sc_, R_.timeline, far_, n_samples=n_samples)
    out["far_src"] = src_
    return out


with tabs[T_M3]:
    st.markdown("---\n### ☀️ Ray-traced illumination & shadow physics (whole landing area, this mission window)")
    g_ = ho.grazing_stats(tl)
    psr_km = ho.psr_boundary_km(scene) if scene is not None else float("nan")
    c = st.columns(5)
    kpi(c[0], "Nearest PSR boundary", "none in 12 km" if psr_km == float("inf") else fmt(psr_km, 2, " km"), "LOLA LPSR 60 m")
    if g_.get("lit_steps"):
        kpi(c[1], "Sun margin when lit (median)", fmt(g_["median_margin_deg"], 2, "°"), f"p10 {g_['p10_margin_deg']:.2f}°")
        kpi(c[2], "Grazing light", fmt(g_["grazing_pct"], 1, "%"), "of lit time with margin < 0.5°")
        kpi(c[3], "Partial Sun disc", fmt(g_["partial_disc_pct"], 1, "%"), "of lit time")
        kpi(c[4], "Highest Sun", fmt(g_["max_sun_alt_deg"], 2, "°"), "above the ideal horizon")
    st.caption("Grazing light is fragile: with the Sun < 0.5° above the terrain skyline, a 10 m boulder 1 km away or a DEM error flips the "
               "result, so treat those hours as uncertain.")
    if scene is None:
        st.info("No DEM scene covers this point.")
    else:
        nsamp = st.select_slider("Ray-trace samples through the mission", [24, 48, 96], 48, key="illum_n",
                                 help="Each sample casts terrain shadows over the 24 × 24 km scene from 2 m above every 160 m pixel.")
        if st.button("Compute ray-traced illumination map", key="illum_go") or st.session_state.get("illum_key") == (run_args(site), nsamp):
            st.session_state["illum_key"] = (run_args(site), nsamp)
            with st.spinner("Ray-tracing terrain shadows …"):
                im = cached_illum(run_args(site), nsamp)
            if im is not None:
                fig_i = go.Figure(go.Heatmap(z=100 * im["frac"], x=im["x_m"] / 1000, y=im["y_m"] / 1000, colorscale="Inferno", zmin=0, zmax=100,
                                             colorbar=dict(title="% of mission lit")))
                if "lola_psr_60m" in scene:
                    fig_i.add_trace(go.Contour(z=scene["lola_psr_60m"], x=scene["x_m"] / 1000, y=scene["y_m"] / 1000, showscale=False,
                                               contours=dict(start=0.5, end=0.5, coloring="none"), line=dict(color="#6fa8ff", width=1.5),
                                               name="PSR boundary", hoverinfo="skip"))
                fig_i.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=[site.id], textposition="top center",
                                           marker=dict(symbol="x", size=12, color="#38d9ff"), name="site"))
                b1k = im["best_1000"]
                fig_i.add_trace(go.Scatter(x=[b1k["x"] / 1000], y=[b1k["y"] / 1000], mode="markers", marker=dict(symbol="star", size=14, color="#3ddc97"),
                                           name="best spot within 1 km"))
                mk_mod.add_local2d(fig_i, MK, site, float(np.max(np.abs(im["x_m"]))) / 1000)
                fig_i.update_layout(**PLOT, height=560, title=f"Fraction of the mission each spot sees the Sun ({im['n_samples']} ray-traced samples)",
                                    xaxis_title="east (km)", yaxis_title="north (km)", yaxis_scaleanchor="x", legend=dict(orientation="h", y=-0.12))
                st.plotly_chart(fig_i, width="stretch")
                c = st.columns(4)
                kpi(c[0], "Site (ray-traced)", fmt(100 * im["site_frac"], 1, "%"), f"M1 horizon model {R.solar['sunlit_pct']:.1f}%")
                for j, rad in enumerate((500, 1000, 3000), 1):
                    bb = im[f"best_{rad}"]
                    kpi(c[j], f"Best within {rad / 1000:g} km", fmt(100 * bb["frac"], 1, "%"),
                        f"{bb['dist_m']:.0f} m away · {100 * (bb['frac'] - im['site_frac']):+.1f} pts")
                st.caption(f"Near field: rays over the LOLA 80 m scene DEM from 2 m above each pixel; far field: {im['far_src']}. The site pixel "
                           "reproduces the Model 1 horizon result to within sampling noise. Blue line = LOLA permanent-shadow boundary.")

with tabs[T_M4]:
    G = R.generation
    if G.get("rps_gen_kwh"):
        st.info(f"⚛ A radioisotope / fission source adds **{array.rps_w:,.0f} W** day and night: **{G['rps_gen_kwh']:,.1f} kWh** of the "
                f"{G['total_gen_kwh']:,.1f} kWh total; solar supplies **{G['solar_gen_kwh']:,.1f} kWh**. Change it in 🔋 M5 → equipment loadout.")
    c = st.columns(6)
    kpi(c[0], "Nameplate @1 AU", fmt(G["nameplate_w_at_1au"], 0, " W"), f"{area} m² × {eff:.3f}")
    kpi(c[1], "Mean / peak", f"{G['mean_gen_w']:.0f} / {G['peak_gen_w']:.0f} W")
    kpi(c[2], "Total generated", fmt(G["total_gen_kwh"], 1, " kWh"), f"{G['mean_daily_gen_kwh']:.2f} kWh/day")
    kpi(c[3], "Capacity factor", fmt(G["capacity_factor_pct"], 1, "%"))
    kpi(c[4], "Longest zero-gen", fmt(G["longest_zero_gen_h"], 0, " h"))
    kpi(c[5], "Mean cell temp", fmt(G["mean_cell_temp_c"], 0, " °C"))
    st.plotly_chart(viz_power.gauges(G, surv, battery), width="stretch")
    st.plotly_chart(viz_power.power_timeline(tl), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(viz_power.loss_waterfall(tl, R.array, step), width="stretch")
    b.plotly_chart(viz_power.power_rose(tl, step), width="stretch")
    st.plotly_chart(viz_power.calendar_heatmap(tl), width="stretch")
    a, b = st.columns(2)
    with a:
        d = tl.set_index("time_utc")[["gen_wh_step"]].resample("1D").sum() / 1000
        dem_d = (tl.set_index("time_utc")["load_demand_w"] * step / 60).resample("1D").sum() / 1000
        figd = go.Figure([go.Bar(x=d.index, y=d.gen_wh_step, name="Generated kWh", marker_color="#ffd54a"),
                          go.Scatter(x=dem_d.index, y=dem_d.values, name="Demand kWh", line=dict(color="#ff5d73")),
                          go.Scatter(x=d.index, y=d.gen_wh_step.cumsum(), name="Cumulative kWh", yaxis="y2", line=dict(color="#38d9ff", dash="dot"))])
        figd.update_layout(**PLOT, height=330, title=f"Daily energy · best {G['best_day']} ({G['best_day_kwh']:.2f}) · worst {G['worst_day']} ({G['worst_day_kwh']:.2f}) kWh",
                           yaxis2=dict(overlaying="y", side="right", title="cumulative"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figd, width="stretch")
    with b:
        sens = pg.sensitivity(tl, R.array, step)
        sens["label"] = sens.parameter + " = " + sens.value.astype(str).str[:6]
        figt = go.Figure(go.Bar(y=sens.label, x=sens.change_pct, orientation="h",
                                marker_color=np.where(sens.change_pct >= 0, "#3ddc97", "#ff5d73")))
        figt.update_layout(**PLOT, height=330, title="Sensitivity tornado - % change in total generation", xaxis_title="%")
        st.plotly_chart(figt, width="stretch")
    st.markdown("### ⚡ Electrical view - how many amps the sunlight produces")
    E = R.electrical
    cellcfg = pve.CellConfig()
    c = st.columns(6)
    kpi(c[0], "Array layout", f"{E['cells_series']}s × {E['strings_parallel']}p", f"{E['cells_series']*E['strings_parallel']} cells")
    kpi(c[1], "Peak array current", fmt(E["peak_array_current_a"], 2, " A"), f"Isc peak {fmt(E['peak_isc_a'],2,' A')}")
    kpi(c[2], "Mean current (lit)", fmt(E["mean_array_current_lit_a"], 2, " A"), f"@ ~{fmt(E['mean_vmp_string_v'],1,' V')}")
    kpi(c[3], "Peak bus current", fmt(E["peak_bus_current_a"], 2, " A"), f"{cellcfg.bus_v:.0f} V regulated bus")
    kpi(c[4], "Charge into battery", fmt(E["charge_ah"], 0, " Ah"), f"{cellcfg.battery_v} V pack")
    kpi(c[5], "Current now", fmt(float(rc.imp_array_a), 2, " A"), f"{rc['time_utc']:%m-%d %H:%M} (Site 3D slider)")
    st.plotly_chart(viz_power.current_timeline(tl), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(viz_power.irradiance_current_scatter(tl), width="stretch")
    b.plotly_chart(viz_power.iv_figure(tl, ci if ci < len(tl) else 0, cellcfg), width="stretch")
    a, b = st.columns([3, 2])
    a.plotly_chart(viz_power.iv_animation(tl, cellcfg), width="stretch")
    b.plotly_chart(viz_power.sun_to_amps_sankey(tl, cellcfg, R.array, step), width="stretch")
    st.caption("Electrical chain inputs come from Model 1/4 (SPICE/Horizons Sun geometry, LOLA terrain horizon, 1361/r² irradiance, incidence, "
               "radiative cell temperature). Cell parameters (Isc 17 mA/cm², Voc 2.69 V, −6.2 mV/°C …) are typical space triple-junction "
               "assumptions, not a datasheet.")
    calc_box(f"Electrical step at {rc['time_utc']:%Y-%m-%d %H:%M} UTC", f"""
- Effective irradiance $G_{{eff}} = G_{{poa}}\\,IAM\\,f_{{dust}}\\,f_{{age}}$ → fraction of AM0 $G_{{eff}}/1361$
- Cell $I_{{sc}} = 17\\,\\text{{mA/cm}}^2 × {cellcfg.area_cm2}\\,\\text{{cm}}^2 × G_{{eff}}/1361 × [1+{cellcfg.disc_alpha_per_c}(T-28)]$; array $I_{{sc}}$ = × {E['strings_parallel']} strings = **{rc.isc_array_a:.3f} A**
- $V_{{oc}} = 2.69 + ({cellcfg.dvoc_v_per_c})(T-28) + n V_t \\ln(G_{{eff}}/1361)$, × {E['cells_series']} cells = **{rc.voc_string_v:.2f} V**; $V_{{mp}}$ = **{rc.vmp_string_v:.2f} V**
- $I_{{mp}} = P_{{DC}}/V_{{mp}}$ = {rc.gen_dc_w:.1f} / {max(rc.vmp_string_v, 1e-9):.2f} = **{rc.imp_array_a:.3f} A** · bus $I = P_{{bus}}/28$ = **{rc.bus_current_a:.3f} A** · battery **{rc.battery_current_a:+.3f} A**""")

    lat_ = viz_pole.lattice()
    if lat_ is not None:
        p_here = viz_pole.value_at("power", site.lat, site.lon)
        figc = go.Figure(go.Histogram(x=lat_.mean_specific_power_w_m2, nbinsx=60, marker_color="rgba(255,176,46,.6)"))
        if np.isfinite(p_here):
            figc.add_vline(x=p_here, line_color="#ffffff", line_width=3, annotation_text=f"this point ≈ {p_here:.0f} W/m²")
        figc.update_layout(**PLOT, height=240, title="Where this point sits among all south-pole lattice points (2027 mean W/m², vertical tracking)")
        st.plotly_chart(figc, width="stretch")
    calc_box(f"Model 4 at {rc['time_utc']:%Y-%m-%d %H:%M} UTC", f"""
| step | expression | value |
|---|---|---|
| Irradiance | $G = 1361 / r^2$ = 1361 / {rc.sun_dist_au:.5f}² | **{rc.irradiance_w_m2:.1f} W/m²** |
| Visible disc | $f$ (Model 1) | **{rc.sun_disc_fraction:.4f}** |
| Incidence | $\\cos i$ ({orient}) | **{rc.incidence_cos:.4f}** |
| Cover glass | $IAM = 1 - b_0(1/\\cos i - 1)$, $b_0$ = {R.array.iam_b0} | **{rc.iam:.4f}** |
| Plane of array | $G f \\cos i$ | **{rc.panel_irradiance_w_m2:.1f} W/m²** |
| Cell temperature | $T^4 = \\alpha\\,G_{{poa}}\\,IAM\\,(1-\\eta)/(\\sigma(\\varepsilon_f+\\varepsilon_b)) + T_{{sink}}^4$ | **{fmt(rc.cell_temp_c, 1, ' °C')}** |
| Efficiency | $\\eta = \\eta_{{ref}}[1+\\beta(T-28)]$, β = {tco} | **{fmt(rc.cell_eff, 4)}** |
| DC power | $G_{{poa}}\\,IAM\\,A\\,k_{{pack}}\\,\\eta\\,f_{{dust}}\\,f_{{age}}$ (A={area}, k={R.array.packing_factor}, dust={dust}, age={degr}) | **{rc.gen_dc_w:.1f} W** |
| Bus power | $P_{{DC}}\\,\\eta_{{MPPT}}\\,\\eta_{{harness}}$ (×{mppt} ×{R.array.harness_efficiency}) | **{rc.gen_w:.1f} W** |""")

# ------------------------------------------------------------------ M5
with tabs[T_M5]:
    st.markdown("### 🧰 Real-equipment loadout - fly actual lunar hardware")
    st.caption("Pick real instruments, radios, computers, ISRU plants and power sources (VIPER, PRIME-1, Blue Ghost, Mars 2020, Kilopower …). "
               "Each becomes a Model 5 load with its own schedule and priority; RTGs and reactors add constant bus power and RHUs add heat. "
               "Figures are rounded public values for concept budgets - not instrument ICDs.")
    if "equip_qty" not in st.session_state:
        st.session_state["equip_qty"] = {}
    eqc = st.columns([1.3, 1, 1])
    lo_name = eqc[0].selectbox("Start from a loadout", ["(keep current)"] + list(eq.LOADOUTS), key="eq_loadout")
    if eqc[0].button("Load this loadout", key="eq_load") and lo_name in eq.LOADOUTS:
        st.session_state["equip_qty"] = dict(eq.LOADOUTS[lo_name])
        st.session_state.pop("eq_editor", None)
        st.rerun()
    cat_f = eqc[1].multiselect("Show categories", eq.CATEGORIES, default=eq.CATEGORIES, key="eq_cats")
    cdf = eq.catalog_frame()
    cdf.insert(0, "qty", [int(st.session_state["equip_qty"].get(i, 0)) for i in cdf.id])
    view = cdf[cdf.category.isin(cat_f)].reset_index(drop=True)
    eed = st.data_editor(view, width="stretch", hide_index=True, key="eq_editor", height=360,
                         disabled=[c for c in view.columns if c != "qty"],
                         column_config={"qty": st.column_config.NumberColumn("Qty", min_value=0, max_value=20, step=1, help="How many units"),
                                        "active_w": st.column_config.NumberColumn("Active W", format="%.0f"),
                                        "standby_w": st.column_config.NumberColumn("Standby W", format="%.0f"),
                                        "mass_kg": st.column_config.NumberColumn("Mass kg", format="%.1f"),
                                        "rps_w": st.column_config.NumberColumn("Gen We", format="%.0f", help="Electrical output (power sources)"),
                                        "heat_w": st.column_config.NumberColumn("Heat Wt", format="%.0f", help="Heat delivered to the warm box"),
                                        "note": st.column_config.TextColumn("What it does", width="large")})
    qty = {**{k: v for k, v in st.session_state["equip_qty"].items() if k not in set(view.id)},
           **{r.id: int(r.qty) for r in eed.itertuples() if int(r.qty or 0) > 0}}
    sumq = eq.loadout_summary(qty, sunlit_frac=R.solar["sunlit_pct"] / 100.0)
    q = st.columns(7)
    kpi(q[0], "Units", f"{sumq['items']}")
    kpi(q[1], "Mass", fmt(sumq["mass_kg"], 1, " kg"))
    kpi(q[2], "Peak draw", fmt(sumq["peak_w"], 0, " W"), "all on at once")
    kpi(q[3], "Mean draw", fmt(sumq["mean_w"], 0, " W"), "schedule-weighted")
    kpi(q[4], "Night draw", fmt(sumq["night_w"], 0, " W"), "always-on + dark loads")
    kpi(q[5], "Nuclear power", fmt(sumq["rps_w"], 0, " We"), "RTG / fission")
    kpi(q[6], "Warm-box heat", fmt(sumq["heat_w"], 0, " Wt"), "RHU / RTG")
    night_need_wh = max(sumq["night_w"] - sumq["rps_w"], 0) * R.solar["longest_darkness_h"]
    usable_wh = cap * dod
    verdict = ("✅ the battery covers the longest night for this loadout" if night_need_wh <= usable_wh else
               f"⚠️ the longest night ({R.solar['longest_darkness_h']:.0f} h) needs ≈ {night_need_wh / 1000:.1f} kWh but the battery has "
               f"{usable_wh / 1000:.1f} kWh usable - add an RTG, shed loads at night, or pick a sunnier point")
    st.caption(f"Quick check (before heaters): {verdict}.")
    e1, e2, e3 = st.columns([1.2, 1.2, 2])
    base_rows = st.session_state["subs"][~st.session_state["subs"]["name"].astype(str).str.startswith("⚙ ")]
    keep_base = e3.checkbox("Keep the generic bus loads in the table (avionics, sensors …)", True, key="eq_keep_base",
                            help="Untick to fly only the real equipment you picked")
    if e1.button("✅ Apply loadout to all models", type="primary", key="eq_apply"):
        new_rows = pd.DataFrame([s_.to_dict() for s_ in eq.loadout_subsystems(qty)])
        st.session_state["subs"] = pd.concat([base_rows if keep_base else base_rows.iloc[0:0], new_rows], ignore_index=True)
        st.session_state["equip_qty"] = qty
        st.session_state["equip_src"] = eq.loadout_sources(qty)
        st.session_state.pop("sub_editor", None)
        st.rerun()
    if e2.button("Remove all equipment", key="eq_clear"):
        st.session_state["subs"] = base_rows.reset_index(drop=True)
        st.session_state["equip_qty"] = {}
        st.session_state["equip_src"] = {}
        st.session_state.pop("eq_editor", None)
        st.rerun()
    src_now = st.session_state.get("equip_src", {})
    if src_now.get("rps_w") or src_now.get("heat_w"):
        st.info(f"Active power sources: **{src_now.get('rps_w', 0):.0f} We** constant generation and **{src_now.get('heat_w', 0):.0f} Wt** "
                "warm-box heat are included in M4, M5 and the Mission Sim.")
    st.markdown("---")
    st.markdown("**Subsystem load table** - every value is an editable engineering assumption (no CSA/NASA rating is implied).")
    ed = st.data_editor(st.session_state["subs"], num_rows="dynamic", width="stretch",
                        column_config={"schedule": st.column_config.SelectboxColumn(options=list(en.SCHEDULES)),
                                       "priority": st.column_config.NumberColumn(min_value=1, max_value=4, help="1 survival-critical … 4 discretionary")},
                        key="sub_editor")
    if st.button("Apply load table to all models", type="primary"):
        st.session_state["subs"] = ed
        st.rerun()
    st.markdown("### 🛰 Activity planner - operations that drive power usage")
    plan = st.session_state["plan"]
    pa, pb = st.columns([3, 2])
    with pa:
        st.markdown("**Communication sessions** - repeat period, duration and RF amplitude; scheduled only while the DTE link is up")
        ced = st.data_editor(pd.DataFrame(plan["comms"]), num_rows="dynamic", width="stretch", key="comm_editor",
                             column_config={"rf_power_w": st.column_config.NumberColumn("RF power W (amplitude)", min_value=0.5, max_value=200.0),
                                            "duration_min": st.column_config.NumberColumn("duration min", min_value=1.0, max_value=1440.0),
                                            "period_h": st.column_config.NumberColumn("every … h", min_value=0.5, max_value=720.0),
                                            "sspa_eff": st.column_config.NumberColumn("SSPA η", min_value=0.05, max_value=0.9),
                                            "priority": st.column_config.NumberColumn(min_value=1, max_value=4)})
        st.markdown("**Science / drilling bursts**")
        bed = st.data_editor(pd.DataFrame(plan["bursts"]), num_rows="dynamic", width="stretch", key="burst_editor")
    with pb:
        trv = plan["traverse"]
        st.markdown("**Mobility (rover traverse)**")
        t_en = st.checkbox("Enable traverse", trv["enabled"], key="tr_en")
        t_v = st.slider("Speed (m/s)", 0.02, 0.50, float(trv["speed_m_s"]), 0.01, key="tr_v")
        t_m = st.slider("Vehicle mass (kg)", 20.0, 2000.0, float(trv["mass_kg"]), 10.0, key="tr_m")
        t_sl = st.slider("Average slope (°)", 0.0, 25.0, float(trv["slope_deg"]), 0.5, key="tr_sl")
        t_crr = st.slider("Rolling resistance Crr (regolith)", 0.05, 0.40, float(trv["crr"]), 0.01, key="tr_crr")
        t_h = st.slider("Driving hours per day", 0.5, 20.0, float(trv["hours_per_day"]), 0.5, key="tr_h")
        t_d = st.slider("Traverse days", 1.0, 30.0, float(trv["days"]), 1.0, key="tr_d")
        t_s0 = st.slider("Start on mission day", 0.0, 30.0, float(trv["start_day"]), 0.5, key="tr_s0")
        t_sun = st.checkbox("Drive only in sunlight", trv["sunlit_only"], key="tr_sun")
    if st.button("Apply activity plan to all models", type="primary"):
        st.session_state["plan"] = {"comms": [r for r in ced.to_dict("records") if r.get("name")],
                                    "bursts": [r for r in bed.to_dict("records") if r.get("name")],
                                    "traverse": {**trv, "enabled": t_en, "speed_m_s": t_v, "mass_kg": t_m, "slope_deg": t_sl, "crr": t_crr,
                                                 "hours_per_day": t_h, "days": t_d, "start_day": t_s0, "sunlit_only": t_sun}}
        st.rerun()
    st.plotly_chart(viz_power.load_stack(tl), width="stretch")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(viz_power.comm_sessions_fig(R.sessions), width="stretch")
        if len(R.sessions):
            agg = R.sessions.groupby("session").agg(sessions=("minutes", "size"), total_min=("minutes", "sum"), energy_wh=("energy_wh", "sum"),
                                                    data_gb=("data_mb", lambda x: x.sum() / 1000)).round(2).reset_index()
            st.dataframe(agg, width="stretch", hide_index=True)
    with b:
        trv_obj = act.Traverse(**{**plan["traverse"], "speed_m_s": t_v, "mass_kg": t_m, "slope_deg": t_sl, "crr": t_crr})
        hotel = float(tl.load_demand_w.mean())
        curve = act.speed_energy_curve(trv_obj, hotel)
        st.plotly_chart(viz_power.speed_curve_fig(curve, t_v, hotel), width="stretch")
        st.caption(f"Drive power at {t_v:.2f} m/s on {t_sl:.1f}°: **{trv_obj.drive_w():.0f} W** "
                   f"(= {t_m:.0f} kg × 1.62 m/s² × ({t_crr:.2f}·cosθ + sinθ) × v / η + {trv_obj.drive_avionics_w:.0f} W avionics). "
                   "Slower is not always cheaper: hotel loads keep running while you crawl.")
        if len(R.traverse):
            st.metric("Distance covered", fmt(float(R.traverse.distance_km.iloc[-1]), 2, " km"))
    S_ = R.stability
    st.markdown("### 🏕 Outpost energy stability")
    c = st.columns(6)
    kpi(c[0], "Stability index", fmt(S_["stability_index"], 0, "/100"))
    kpi(c[1], "SoC in safe band", fmt(S_["soc_in_band_pct"], 1, "%"))
    kpi(c[2], "24 h windows net-positive", fmt(S_["positive_24h_energy_pct"], 1, "%"))
    kpi(c[3], "Worst 24 h net", fmt(S_["worst_24h_net_kwh"], 2, " kWh"))
    kpi(c[4], "Battery autonomy", fmt(S_["autonomy_h_on_battery"], 0, " h"), f"night load {fmt(S_['night_mean_load_w'],0,' W')}")
    kpi(c[5], "Recovery after min SoC", "n/a" if S_["recovery_h_after_min_soc"] is None else fmt(S_["recovery_h_after_min_soc"], 0, " h"))
    st.plotly_chart(viz_power.stability_fig(tl, step), width="stretch")
    st.markdown("### 📊 Sunlight harvest → generation → usage (interactive)")
    st.caption("Power BI-style cross-filtering: drag a time window, click a usage slice to isolate it, click a day to drill in. "
               "Play the same mission hands-on in **🎮 M5 Mission Sim**.")
    sim_page("dash", 900)
    st.plotly_chart(viz_power.energy_sankey(tl, R.subsystem_summary, step), width="stretch")
    a, b = st.columns([1, 1])
    with a:
        sm = R.subsystem_summary
        figp = go.Figure(go.Pie(labels=sm.subsystem, values=sm.demand_kwh, hole=0.55, textinfo="percent",
                                marker=dict(colors=["#38d9ff", "#4aa8ff", "#ffd54a", "#3ddc97", "#ff8c42", "#9b5cff", "#ff5d73", "#7fa7b8"])))
        figp.update_layout(**PLOT, height=330, title="Demand by subsystem (kWh share)")
        st.plotly_chart(figp, width="stretch")
    with b:
        rows = []
        scen_key = (site.id, start_iso, days, step)
        if days > 120 and st.session_state.get("scen_ok") != scen_key:
            st.info(f"Long mission ({days} d): the six-scenario comparison runs on demand (~{max(1, round(days / 120))} s per scenario).")
            if st.button("Run all six scenarios", key="run_scen"):
                st.session_state["scen_ok"] = scen_key
                st.rerun()
        for sc in (en.SCENARIOS if days <= 120 or st.session_state.get("scen_ok") == scen_key else []):
            rr = run_for(site, sc)
            s_ = rr.survival
            rows.append({"scenario": en.SCENARIOS[sc].split(" (")[0], "status": s_["status"].split(" - ")[0], "gen kWh": round(rr.generation["total_gen_kwh"], 1),
                         "demand kWh": round(s_["total_demand_kwh"], 1), "min SoC %": round(s_["min_soc_pct"], 1),
                         "ops %": round(s_["operational_time_supported_pct"], 1), "h→critical": s_["hours_to_critical_soc"]})
        scd = pd.DataFrame(rows)
        if rows:
            figsc = go.Figure([go.Bar(x=scd.scenario, y=scd["min SoC %"], name="Min SoC %", marker_color="#3ddc97"),
                               go.Bar(x=scd.scenario, y=scd["ops %"], name="Ops supported %", marker_color="#38d9ff")])
            figsc.add_hline(y=100 * (1 - dod), line_dash="dot", line_color="#ff5d73")
            figsc.update_layout(**PLOT, height=330, barmode="group", title="All six scenarios at this point")
            st.plotly_chart(figsc, width="stretch")
            st.dataframe(scd, width="stretch", hide_index=True)
    st.caption("Survival heaters = MLI radiative leak εσA(T_int⁴−T_env⁴) + conduction; dark sink −200 °C from the CSA Lunar Utility "
               "Vehicle requirement. Extended-darkness scenario uses the CSA 'roughly two-week' night.")
    calc_box(f"Model 5 battery step at {rc['time_utc']:%Y-%m-%d %H:%M} UTC", f"""
- Demand $D$ = Σ subsystem loads (schedule × priority policy) + heaters = **{rc.load_demand_w:.1f} W** (heater **{rc.heater_w:.1f} W**)
- Net $= P_{{gen}} - D_{{served}}$ = {rc.gen_w:.1f} − {rc.load_served_w:.1f} = **{rc.net_w:.1f} W**
- Charge $c = \\min(net, P_{{max}}, (C-E)/(\\eta_c\\Delta t))$ → **{rc.batt_charge_w:.1f} W** · discharge $d = \\min(-net, P_{{max}}, (E-E_{{floor}})\\eta_d/\\Delta t)$ → **{rc.batt_discharge_w:.1f} W**
- $E_{{t+1}} = E_t + c\\,\\eta_c\\Delta t - (d/\\eta_d)\\Delta t - E_t k_{{sd}}\\Delta t$ → SoC **{rc.soc_pct:.2f}%**
- Policy: below safe SoC keep priority ≤ 2; at the DoD floor keep priority 1 only → shedding **{bool(rc.shedding)}**, unserved **{rc.energy_deficit_w:.1f} W**""")

# ------------------------------------------------------------------ M5 mission sim (playable)
with tabs[T_SIM]:
    st.markdown("**Fly the mission.** A rover and an outpost share the same Sun, Earth, terrain and link timeline as M1–M5. "
                "Drive the rover across the real LOLA terrain: its sunlight, Earth view and slope come from its own position, not from the site. "
                "Run science, send signals, switch subsystems, and watch harvest, generation, usage, battery and time remaining react live.")
    sim_page("sim", 2400)
    with st.expander("How the simulation works (models & assumptions)"):
        st.markdown(f"""
- **Clock:** the shared M1–M5 timeline ({step:g} min steps, interpolated in 1–5 min sub-steps). Speeds run from 1 min/s up to 1 day/s.
- **Rover illumination:** a 360° horizon is ray-marched over the 80 m LOLA scene DEM **from the rover's current cell**, combined with a
  far-field mask (LOLA terrain 12–200 km from the site). Elevation includes the local-vertical tilt d/R. At the site centre this
  reproduces the Model 1 sunlit flag (validated in `tests/test_mission_sim.py`).
- **Rover power:** mast-mounted sun-tracking panel {1.0} m² × irradiance × cos(elevation) × cell η(T) × dust × ageing × MPPT.
  Driving uses P = m·g·(Crr·cosθ + sinθ)·v/η + avionics, with θ = the signed grade along the heading from the DEM (the same formula as
  `activities.Traverse`). Slopes above 25° are impassable, and speed halves above 15°.
- **Thermal:** MLI radiative leak + conduction (the M5 thermal model). The dark sink is −200 °C, or −230 °C inside permanently shadowed
  regions. Half of the motor and instrument dissipation offsets the heater demand.
- **Batteries:** the outpost uses the sidebar battery and the M5 load table. The game's battery step is a line-for-line port of
  `energy.simulate` (below safe SoC only priority ≤ 2 loads run; at the DoD floor only priority 1). Critical loads unpowered for 2 h
  means the asset is lost.
- **Signals:** DC = P_RF/η_SSPA + exciter. Rate = the M2 link-budget C/N0 scaled by RF power (capped at the radio limit), only while
  Earth is above the outpost horizon and a DSN complex sees the Moon. Rover DTE uses a 6 dBi low-gain antenna at 2 W. The UHF relay to
  the outpost needs ≤ 8 km and a clear terrain line of sight.
- **Science targets** (T1…) are picked from the data layers: LOLA PSR cold traps, high average-Sun-visibility ridges, a high
  Earth-visibility spot and 14–22° crater-wall outcrops.
- Rover mass, panel, battery, instrument powers and data yields are **engineering assumptions**, not mission specifications.
  Progress autosaves in your browser, so changing a sidebar setting restarts the game with the new configuration.
""")

# ------------------------------------------------------------------ compare
with tabs[T_CMP]:
    pool = {s.id: s for s in ALL.values()}
    if loc_mode.startswith("Map"):
        pool["POINT"] = site
    default = [k for k in pool if k.startswith("LN-")][:10] + [k for k in ("LM1", "Site23", "CR07") if k in pool] + (["POINT"] if "POINT" in pool else [])
    sel = st.multiselect("Locations", list(pool), default=default)
    if st.button("Run comparison", type="primary") or st.session_state.get("cmp_done"):
        st.session_state["cmp_done"] = True
        rows = []
        prog = st.progress(0.0)
        for i, k in enumerate(sel):
            rw = run_for(pool[k]).summary_row()
            rw["feasibility"] = feasibility_score(rw)
            rows.append(rw)
            prog.progress((i + 1) / max(len(sel), 1))
        if rows:
            cmp = pd.DataFrame(rows).sort_values("feasibility", ascending=False)
            cols = ["site_id", "site_name", "feasibility", "survival_status", "terrain_status", "sunlit_pct", "longest_dark_h", "earth_visible_pct",
                    "longest_dte_blackout_h", "gen_total_kwh", "min_soc_pct", "ops_supported_pct", "terrain_score", "terrain_resolution_m"]
            st.dataframe(cmp[cols].round(1), width="stretch", hide_index=True)
            st.download_button("Download comparison CSV", cmp.to_csv(index=False).encode(), "site_comparison.csv")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cmp.site_id, y=cmp.sunlit_pct, name="Sunlit %", marker_color="#ffd54a"))
            fig.add_trace(go.Bar(x=cmp.site_id, y=cmp.earth_visible_pct, name="Earth visible %", marker_color="#4aa8ff"))
            fig.add_trace(go.Scatter(x=cmp.site_id, y=cmp.feasibility, name="Feasibility index", mode="markers+lines", marker=dict(size=10, color="#3ddc97")))
            fig.update_layout(**PLOT, height=380, barmode="group", title="Ranking on identical array, loads, scenario and window")
            st.plotly_chart(fig, width="stretch")
    st.markdown("---\n**Seasons & multi-lunation comparison** - the same point over consecutive lunar days or calendar months")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    mode_s = sc1.radio("Split by", ["Lunations (29.53 d)", "Calendar months"], key="season_mode")
    nseg = sc2.slider("Number of segments", 2, 24, 6, help="13 lunations ≈ one year; 24 calendar months = two years")
    if sc2.button("Run seasonal comparison"):
        rows = []
        prog = st.progress(0.0)
        for i in range(nseg):
            if mode_s.startswith("Lun"):
                s0 = datetime.fromisoformat(start_iso) + timedelta(days=29.530589 * i)
                hrs = 29.530589 * 24
            else:
                base = datetime.fromisoformat(start_iso).replace(day=1)
                y, m = base.year + (base.month - 1 + i) // 12, (base.month - 1 + i) % 12 + 1
                s0 = base.replace(year=y, month=m)
                y2, m2 = s0.year + s0.month // 12, s0.month % 12 + 1
                hrs = (s0.replace(year=y2, month=m2) - s0).total_seconds() / 3600
            rr = run_for(site, start=s0.isoformat(), hrs=hrs, stp=60)
            rows.append({"segment": f"L{i+1}" if mode_s.startswith("Lun") else f"{s0:%Y-%m}", "start": f"{s0:%Y-%m-%d}",
                         "sunlit %": rr.solar["sunlit_pct"], "longest dark h": rr.solar["longest_darkness_h"],
                         "Earth %": rr.dte["earth_visible_pct"], "link %": rr.link["link_closed_pct"],
                         "gen kWh": rr.generation["total_gen_kwh"], "min SoC %": rr.survival["min_soc_pct"],
                         "status": rr.survival["status"].split(" - ")[0]})
            prog.progress((i + 1) / nseg)
        st.session_state["season_rows"] = pd.DataFrame(rows)
    if "season_rows" in st.session_state:
        sr = st.session_state["season_rows"]
        sc3.dataframe(sr.round(1), width="stretch", hide_index=True, height=260)
        figs = make_subplots(rows=1, cols=4, subplot_titles=["Sunlit %", "Longest dark (h)", "Generation (kWh)", "Min SoC %"])
        for j, (col_, colr) in enumerate((("sunlit %", "#ffd54a"), ("longest dark h", "#ff5d73"), ("gen kWh", "#ffb02e"), ("min SoC %", "#3ddc97")), 1):
            figs.add_trace(go.Bar(x=sr.segment, y=sr[col_], marker_color=colr, showlegend=False), 1, j)
        figs.update_layout(**PLOT, height=300, title="Small multiples across seasons / lunations (same site, array, loads)")
        st.plotly_chart(figs, width="stretch")
    st.markdown("---\n**Date sensitivity - what if the landing date slips?**")
    c1, c2 = st.columns([1, 3])
    shifts = c1.multiselect("Shift start by (days)", [-21, -14, -7, -3, 0, 3, 7, 14, 21], default=[-14, -7, 0, 7, 14])
    if c1.button("Run date sweep"):
        rows = []
        for dsh in shifts:
            s0 = (datetime.fromisoformat(start_iso) + timedelta(days=dsh)).isoformat()
            rw = run_for(site, start=s0).summary_row()
            rows.append({"shift_d": dsh, "start": s0[:10], "sunlit %": rw["sunlit_pct"], "longest dark h": rw["longest_dark_h"],
                         "Earth %": rw["earth_visible_pct"], "gen kWh": rw["gen_total_kwh"], "min SoC %": rw["min_soc_pct"], "status": rw["survival_status"]})
        ds = pd.DataFrame(rows)
        c2.dataframe(ds.round(1), width="stretch", hide_index=True)
        figd = go.Figure([go.Scatter(x=ds.shift_d, y=ds["sunlit %"], name="Sunlit %", line=dict(color="#ffd54a")),
                          go.Scatter(x=ds.shift_d, y=ds["min SoC %"], name="Min SoC %", line=dict(color="#3ddc97")),
                          go.Scatter(x=ds.shift_d, y=ds["Earth %"], name="Earth %", line=dict(color="#4aa8ff"))])
        figd.update_layout(**PLOT, height=320, xaxis_title="start shift (days)")
        c2.plotly_chart(figd, width="stretch")

# ------------------------------------------------------------------ scenario diff, sensitivity, profiles & handoff
def dem_elev(lat: float, lon: float) -> float:
    from clps_engine.rasters import product_registry
    reg = product_registry()
    for k in ("LDEM_20M", "LOLA_DEM_80M"):
        d = reg.get(k)
        if d is not None and d.covers(lat, lon):
            z = float(d.sample_latlon(lat, lon))
            if np.isfinite(z):
                return z
    return 0.0


def shifted_site(s0, dlat: float, dlon: float):
    la, lo = s0.lat + dlat, s0.lon + dlon
    return custom_site(la, lo, dem_elev(la, lo), f"{s0.name} {dlat:+.1f}°lat {dlon:+.1f}°lon", f"{s0.id}~{dlat:+.1f}/{dlon:+.1f}")


def shift_m(s0, s1) -> float:
    from clps_engine.rasters import latlon_to_ps
    a, b = latlon_to_ps(s0.lat, s0.lon), latlon_to_ps(s1.lat, s1.lon)
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ------------------------------------------------------------------ rank & compare points (best-point scan + comparison basket)
def _pt_site(p: dict):
    return custom_site(float(p["lat"]), float(p["lon"]), float(p.get("elev", dem_elev(float(p["lat"]), float(p["lon"])))), p["name"], p["id"])


def _full_point(p: dict, weights: dict) -> dict:
    s_ = ALL[p["id"]] if p["id"] in ALL else (site if p["id"] == site.id else _pt_site(p))
    R_ = run_for(s_)
    rw = R_.summary_row()
    rw["gen_total_kwh_per_day"] = rw["gen_total_kwh"] / max(days, 1)
    ctx = scan_mod.point_context(s_.lat, s_.lon)
    rw["psr_km"] = ctx.get("psr_km", float("nan"))
    rw["score_pct"], parts = scan_mod.full_score(rw, weights)
    rw.update({f"c_{k}": v for k, v in parts.items()})
    rw["feasibility"] = feasibility_score(rw)
    rw["point_id"], rw["point_name"] = p["id"], p["name"]
    return rw


with tabs[T_RANK]:
    st.markdown("**Rank & compare points.** Scan the whole 80°S cap (≈ 226 000 one-km cells) for the best places with a weighted, "
                "transparent percentage score, then put any points side by side with the full M1–M5 models.")
    if "basket" not in st.session_state:
        st.session_state["basket"] = []
    basket = st.session_state["basket"]

    def _add_pt(pid, name, lat, lon, elev=None):
        if any(abs(b["lat"] - lat) < 1e-4 and abs(b["lon"] - lon) < 1e-4 for b in basket):
            return False
        if len(basket) >= 16:
            st.warning("The comparison basket holds 16 points - remove some first.")
            return False
        basket.append({"id": pid, "name": name, "lat": float(lat), "lon": float(lon), **({"elev": float(elev)} if elev is not None else {})})
        return True

    # ---- weights
    st.markdown("#### 1 · What matters for this mission?")
    wc = st.columns([1.4] + [1] * len(scan_mod.CRITERIA))
    preset_w = wc[0].selectbox("Weight preset", list(scan_mod.PRESET_WEIGHTS), key="rk_preset",
                               help="Pick a mission goal, then fine-tune any weight. Weights are relative (they need not sum to 100).")
    if st.session_state.get("_rk_last_preset") != preset_w:
        for k, v in scan_mod.PRESET_WEIGHTS[preset_w].items():
            st.session_state[f"rk_w_{k}"] = int(v)
        st.session_state["_rk_last_preset"] = preset_w
    weights = {}
    for col_, (k, (lab, hlp)) in zip(wc[1:], scan_mod.CRITERIA.items()):
        weights[k] = col_.number_input(lab, 0, 100, step=5, key=sk(f"rk_w_{k}", scan_mod.DEFAULT_WEIGHTS[k]), help=hlp)
    tw = sum(weights.values()) or 1
    st.caption("Score % = Σ weight × criterion (0–1) ÷ Σ weight × 100. Effective shares: " +
               " · ".join(f"{scan_mod.CRITERIA[k][0]} {100 * v / tw:.0f}%" for k, v in weights.items() if v))

    # ---- scan
    st.markdown("#### 2 · Scan for the best points")
    sc = st.columns([1.3, 1, 1, 1, 1])
    region = sc[0].radio("Search area", ["Whole 80°S cap", "Around the selected point"], key="rk_region")
    rad = sc[1].slider("Radius (km)", 5, 300, 60, 5, key="rk_rad", disabled=region.startswith("Whole"))
    topn = sc[2].slider("How many points", 5, 40, 15, key="rk_top")
    sep = sc[3].slider("Min separation (km)", 1, 30, 8, key="rk_sep", help="Keeps only the best cell in each neighbourhood")
    mslope = sc[4].slider("Max mean slope (°)", 3, 25, 15, key="rk_slope")
    if st.button("🔎 Scan and rank", type="primary", key="rk_go"):
        with st.spinner("Scoring every 1 km cell of the cap ..."):
            res = scan_mod.scan(weights, top_n=topn, min_sep_km=sep, max_slope_deg=mslope,
                                center=(site.lat, site.lon) if region.startswith("Around") else None,
                                radius_km=rad if region.startswith("Around") else None)
        st.session_state["rk_res"] = res
        st.session_state["rk_cells"] = res.attrs.get("cells_considered", 0)
        st.session_state.pop("rk_full", None)
    res = st.session_state.get("rk_res")
    if res is not None and len(res):
        st.success(f"Ranked the best {len(res)} of {st.session_state.get('rk_cells', 0):,} eligible cells "
                   f"(slope ≤ {mslope}°, not inside a permanent shadow, ≥ 50% valid DEM).")
        show = res[["rank", "score_pct", "lat", "lon", "elev_m", "sun_vis_pct", "earth_vis_pct", "flat_le10_pct", "slope_deg",
                    "longest_dark_h", "longest_blackout_h", "psr_km"]]
        a, b = st.columns([1.35, 1])
        a.dataframe(show, width="stretch", hide_index=True, height=420,
                    column_config={"score_pct": st.column_config.ProgressColumn("Score %", min_value=0, max_value=100, format="%.1f%%"),
                                   "sun_vis_pct": "Sun %", "earth_vis_pct": "Earth %", "flat_le10_pct": "≤10° ground %",
                                   "longest_dark_h": "Longest dark h", "longest_blackout_h": "Longest blackout h", "psr_km": "km to PSR"})
        sm = scan_mod.score_map(weights, mslope)
        if sm is not None:
            fx, fy, fz = sm
            fm = go.Figure(go.Heatmap(x=fx, y=fy, z=fz, colorscale="Viridis", zmin=20, zmax=80, colorbar=dict(title="score %"),
                                      hovertemplate="score %{z:.1f}%<extra></extra>"))
            from clps_engine.rasters import latlon_to_ps as _ps
            px_, py_ = _ps(res.lat.to_numpy(), res.lon.to_numpy())
            fm.add_trace(go.Scatter(x=np.asarray(px_) / 1000, y=np.asarray(py_) / 1000, mode="markers+text", text=[f"#{r}" for r in res["rank"]],
                                    textposition="top center", marker=dict(size=10, color="#ff5d73", line=dict(color="white", width=1)),
                                    customdata=np.c_[res.lat, res.lon, res.score_pct], name="ranked",
                                    hovertemplate="#%{text} · %{customdata[2]:.1f}%<br>%{customdata[0]:.4f}°, %{customdata[1]:.4f}°E<extra></extra>"))
            sx, sy = _ps(site.lat, site.lon)
            fm.add_trace(go.Scatter(x=[sx / 1000], y=[sy / 1000], mode="markers", marker=dict(symbol="star", size=16, color="#ffd54a"), name="selected"))
            mk_mod.add_pole2d(fm, MK, gl=False)
            fm.update_layout(**PLOT, height=420, title="Score map (2 km) and ranked points", showlegend=False,
                             xaxis=dict(title="x km", scaleanchor="y"), yaxis=dict(title="y km"))
            b.plotly_chart(fm, width="stretch")
        figc = go.Figure()
        for k, colr in zip(scan_mod.CRITERIA, ["#ffd54a", "#4aa8ff", "#3ddc97", "#9b5cff", "#ff8c42", "#38d9ff", "#7fa7b8"]):
            if weights.get(k):
                figc.add_trace(go.Bar(x=[f"#{r}" for r in res["rank"]], y=res[f"c_{k}"].fillna(0) * weights[k] / tw, name=scan_mod.CRITERIA[k][0],
                                      marker_color=colr))
        figc.update_layout(**PLOT, height=330, barmode="stack", title="Why each point scores what it does (weighted contributions, %)",
                           yaxis_title="score %")
        st.plotly_chart(figc, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("➕ Add all ranked points to comparison", key="rk_add_all"):
            n0 = len(basket)
            for _, r in res.iterrows():
                _add_pt(f"SCAN-{int(r['rank']):02d}", f"Scan #{int(r['rank'])} ({r.score_pct:.1f}%)", r.lat, r.lon, r.elev_m)
            st.toast(f"Added {len(basket) - n0} points")
        kver = c2.number_input("Verify top K with full models", 1, min(15, len(res)), min(5, len(res)), key="rk_k")
        if c3.button("🧪 Run full M1–M5 on top K", key="rk_full_go"):
            rows = []
            prog = st.progress(0.0)
            for i, (_, r) in enumerate(res.head(int(kver)).iterrows()):
                p = {"id": f"SCAN-{int(r['rank']):02d}", "name": f"Scan #{int(r['rank'])}", "lat": r.lat, "lon": r.lon, "elev": r.elev_m}
                rw = _full_point(p, weights)
                rw["scan_rank"], rw["scan_score_pct"] = int(r["rank"]), r.score_pct
                rows.append(rw)
                prog.progress((i + 1) / int(kver))
            st.session_state["rk_full"] = pd.DataFrame(rows)
        c4.download_button("⬇ Ranked points CSV", res.to_csv(index=False).encode(), "lunarsouth_best_points.csv", key="rk_dl")
        if "rk_full" in st.session_state:
            fr = st.session_state["rk_full"].sort_values("score_pct", ascending=False).reset_index(drop=True)
            fr.insert(0, "full_rank", range(1, len(fr) + 1))
            st.markdown("**Full-model verification** - the same weights applied to this mission window's M1–M5 results "
                        f"({days} d from {start_iso[:10]}). The scan uses multi-year averages, so the order can change.")
            st.dataframe(fr[["full_rank", "scan_rank", "point_name", "score_pct", "scan_score_pct", "feasibility", "sunlit_pct", "longest_dark_h",
                             "earth_visible_pct", "longest_dte_blackout_h", "gen_total_kwh", "min_soc_pct", "survival_status", "lat", "lon"]].round(2),
                         width="stretch", hide_index=True,
                         column_config={"score_pct": st.column_config.ProgressColumn("Full score %", min_value=0, max_value=100, format="%.1f%%"),
                                        "scan_score_pct": st.column_config.NumberColumn("Scan score %", format="%.1f")})
    elif res is not None:
        st.warning("No cell passes the filters - relax the slope limit or widen the search area.")

    # ---- comparison basket
    st.markdown("#### 3 · Compare chosen points side by side")
    bc = st.columns([1.2, 1, 1, 1.4])
    if bc[0].button(f"➕ Add the selected point ({site.id})", key="bk_cur"):
        _add_pt(site.id, site.name, site.lat, site.lon, site.elev_m)
    mlat = bc[1].number_input("Lat", -90.0, -80.0, -85.0, 0.01, format="%.4f", key="bk_lat")
    mlon = bc[2].number_input("Lon °E", -180.0, 360.0, 0.0, 0.1, format="%.4f", key="bk_lon")
    if bc[1].button("➕ Add coordinates", key="bk_add_xy"):
        _add_pt(f"PT-{len(basket) + 1:02d}", f"Point {mlat:.3f}°, {mlon:.3f}°E", mlat, mlon)
    lib = {s.id: s for s in ALL.values() if s.lat <= -80}
    pick_lib = bc[3].multiselect("Add known sites", list(lib), key="bk_lib", format_func=lambda i: f"{i} · {lib[i].name}")
    if bc[3].button("➕ Add selected sites", key="bk_add_lib"):
        for i in pick_lib:
            _add_pt(i, lib[i].name, lib[i].lat, lib[i].lon, lib[i].elev_m)
    if basket:
        bdf = pd.DataFrame(basket)
        st.dataframe(bdf[["id", "name", "lat", "lon"]], width="stretch", hide_index=True, height=min(38 * (len(bdf) + 1), 300))
        r1, r2, r3 = st.columns([2, 1, 1])
        rm = r1.multiselect("Remove", [b["id"] for b in basket], key="bk_rm")
        if r2.button("Remove selected", key="bk_rm_go") and rm:
            st.session_state["basket"] = [b for b in basket if b["id"] not in rm]
            st.rerun()
        if r3.button("Clear all", key="bk_clear"):
            st.session_state["basket"] = []
            st.session_state.pop("bk_res", None)
            st.rerun()
        if st.button(f"⚖️ Compare {len(basket)} points with the full models", type="primary", key="bk_go"):
            rows = []
            prog = st.progress(0.0)
            for i, p in enumerate(basket):
                rows.append(_full_point(p, weights))
                prog.progress((i + 1) / len(basket))
            st.session_state["bk_res"] = pd.DataFrame(rows)
    else:
        st.info("The basket is empty - add the selected point, typed coordinates, known sites, or the scan results above.")
    if st.session_state.get("bk_res") is not None and len(st.session_state["bk_res"]):
        br = st.session_state["bk_res"].sort_values("score_pct", ascending=False).reset_index(drop=True)
        br.insert(0, "rank", range(1, len(br) + 1))
        best = br.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Best point", str(best.point_id), f"{best.point_name}")
        kpi(k2, "Best score", fmt(best.score_pct, 1, "%"), f"runner-up {fmt(br.score_pct.iloc[1], 1, '%')}" if len(br) > 1 else "")
        kpi(k3, "Spread", fmt(br.score_pct.max() - br.score_pct.min(), 1, " pts"), "best − worst")
        kpi(k4, "Survive the window", f"{int((~br.survival_status.str.startswith('FAIL')).sum())}/{len(br)}", "points that keep critical loads powered")
        cols = ["rank", "point_id", "point_name", "score_pct", "feasibility", "sunlit_pct", "longest_dark_h", "earth_visible_pct",
                "longest_dte_blackout_h", "gen_total_kwh", "min_soc_pct", "ops_supported_pct", "terrain_score", "psr_km", "survival_status", "lat", "lon"]
        hib = {"score_pct", "feasibility", "sunlit_pct", "earth_visible_pct", "gen_total_kwh", "min_soc_pct", "ops_supported_pct", "terrain_score"}
        lob = {"longest_dark_h", "longest_dte_blackout_h"}

        def _hl(col):
            if col.name in hib:
                return ["background-color: rgba(61,220,151,.28)" if v == col.max() else "" for v in col]
            if col.name in lob:
                return ["background-color: rgba(61,220,151,.28)" if v == col.min() else "" for v in col]
            return [""] * len(col)
        st.dataframe(br[cols].round(2).style.apply(_hl).format(precision=1), width="stretch", hide_index=True)
        st.caption("Green = best value in each column. Score % uses the weights above on this mission window's full-model results.")
        g1, g2 = st.columns(2)
        radar = go.Figure()
        labs = [scan_mod.CRITERIA[k][0] for k in scan_mod.FULL_CRITERIA]
        for _, r in br.head(8).iterrows():
            vals = [r.get(f"c_{k}") if r.get(f"c_{k}") is not None and np.isfinite(r.get(f"c_{k}")) else 0 for k in scan_mod.FULL_CRITERIA]
            radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labs + labs[:1], name=f"#{r['rank']} {r.point_id}", fill="toself", opacity=0.55))
        radar.update_layout(**PLOT, height=430, title="Criterion profile (0–100 %, top 8)", polar=dict(radialaxis=dict(range=[0, 100])))
        g1.plotly_chart(radar, width="stretch")
        fb = go.Figure(go.Bar(x=br.score_pct, y=[f"#{r} {i}" for r, i in zip(br["rank"], br.point_id)], orientation="h",
                              marker=dict(color=br.score_pct, colorscale="Viridis", cmin=0, cmax=100),
                              text=[f"{v:.1f}%" for v in br.score_pct], textposition="outside"))
        fb.update_layout(**PLOT, height=430, title="Ranking (score %)", yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 105]))
        g2.plotly_chart(fb, width="stretch")
        d1, d2 = st.columns([1, 1])
        d1.download_button("⬇ Comparison CSV", br.to_csv(index=False).encode(), "lunarsouth_point_comparison.csv", key="bk_dl")
        go_id = d2.selectbox("Open a point in all model tabs", list(br.point_id), key="bk_open")
        if d2.button("📍 Analyse this point", key="bk_open_go"):
            r = br[br.point_id == go_id].iloc[0]
            st.session_state["_pending_pick"] = (round(float(r.lat), 5), round(float(r.lon), 5))
            st.rerun()

with tabs[T_DIFF]:
    st.markdown("### 🔀 Scenario diff - two dates, sites or scenarios side by side")
    st.caption("Both sides use the sidebar's array, battery, loads and link settings; change location, start date or scenario per side.")
    pool_d = {**dict(ALL), **({site.id: site} if site.id not in ALL else {})}
    ids_d = list(pool_d)
    sc_keys = list(en.SCENARIOS)
    base_dt = datetime.fromisoformat(start_iso)

    def diff_side(col, tag, d_shift):
        with col:
            st.markdown(f"#### {tag}")
            sid_ = st.selectbox("Location", ids_d, index=ids_d.index(site.id), key=f"df_site_{tag}", format_func=lambda i: f"{i} · {pool_d[i].name}")
            sh_ = st.number_input("Start shift vs sidebar (days)", -365.0, 365.0, d_shift, 0.25, key=f"df_shift_{tag}")
            sc_ = st.selectbox("Scenario", sc_keys, index=sc_keys.index(scenario), key=f"df_sc_{tag}", format_func=lambda k: en.SCENARIOS[k])
            st.caption(f"Start {(base_dt + timedelta(days=sh_)):%Y-%m-%d %H:%M} UTC · {days} d")
        return pool_d[sid_], sh_, sc_

    ca, cb = st.columns(2)
    sa, sha, sca = diff_side(ca, "A", 0.0)
    sb, shb, scb = diff_side(cb, "B", 1.0)
    with st.spinner("Running A and B through M1→M5 …"):
        ra = run_for(sa, sca, start=(base_dt + timedelta(days=sha)).isoformat())
        rb = run_for(sb, scb, start=(base_dt + timedelta(days=shb)).isoformat())
    dtab = ho.diff_table(ho.diff_metrics(ra), ho.diff_metrics(rb), "A", "B")
    st.session_state["diff_tab"] = dtab

    def _vcol(v):
        return "color:#3ddc97" if str(v).startswith("B better") else "color:#ffc857" if str(v).startswith("A better") else ""
    st.dataframe(dtab.style.format({"A": "{:,.1f}", "B": "{:,.1f}", "Δ (B−A)": "{:+,.1f}"}, na_rep="").map(_vcol, subset=["verdict"]),
                 width="stretch", hide_index=True)
    figd2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=["Generation (W)", "Battery SoC (%)"])
    for rr_, nm, colr, dash in ((ra, "A", "#38d9ff", "solid"), (rb, "B", "#ffc857", "dash")):
        el = (rr_.timeline.time_utc - rr_.timeline.time_utc.iloc[0]).dt.total_seconds() / 86400
        figd2.add_trace(go.Scatter(x=el, y=rr_.timeline.gen_w, name=f"{nm} generation", line=dict(color=colr, width=1.2, dash=dash)), 1, 1)
        figd2.add_trace(go.Scatter(x=el, y=rr_.timeline.soc_pct, name=f"{nm} SoC", line=dict(color=colr, width=2, dash=dash)), 2, 1)
    figd2.update_layout(**PLOT, height=480, hovermode="x unified", title="A vs B on elapsed mission time", legend=dict(orientation="h", y=-0.12))
    figd2.update_xaxes(title_text="elapsed days", row=2, col=1)
    st.plotly_chart(figd2, width="stretch")

    st.markdown("---\n### 🎯 Small-change sensitivity - 0.1° site shift and ±24 h launch delay")
    st.caption("At the pole a 0.1° change in latitude moves the site ~3 km, while 0.1° of longitude moves it only tens of metres - but either "
               "can change which ridge blocks the grazing Sun. Each case is a full M1→M5 run with a horizon ray-cast for the shifted point.")
    sens_key = run_args(site)
    if st.button("Run sensitivity (6 cases)", key="sens_go"):
        st.session_state["sens_key"] = sens_key
    if st.session_state.get("sens_key") == sens_key:
        cases = [("baseline", site, 0.0), ("start −24 h", site, -1.0), ("start +24 h", site, 1.0)]
        if site.lat + 0.1 <= -80.0 or abs(site.lat) < 80:
            cases.append(("lat +0.1° (equatorward)", shifted_site(site, 0.1, 0.0), 0.0))
        if site.lat - 0.1 >= -90.0:
            cases.append(("lat −0.1° (poleward)", shifted_site(site, -0.1, 0.0), 0.0))
        cases.append(("lon +0.1°", shifted_site(site, 0.0, 0.1), 0.0))
        rows_s, prog = [], st.progress(0.0)
        for i, (lab, s_, dd) in enumerate(cases):
            rr_ = run_for(s_, start=(base_dt + timedelta(days=dd)).isoformat())
            m_ = ho.diff_metrics(rr_)
            rows_s.append({"case": lab, "moved (m)": round(shift_m(site, s_)), "sunlit %": m_["sunlit_pct"], "longest dark h": m_["longest_dark_h"],
                           "Earth %": m_["earth_visible_pct"], "longest blackout h": m_["longest_dte_blackout_h"], "gen kWh": m_["gen_total_kwh"],
                           "min SoC %": m_["min_soc_pct"], "first sunset h": m_["first_sunset_h"], "verdict": m_["status"]})
            prog.progress((i + 1) / len(cases))
        sens = pd.DataFrame(rows_s)
        base_row = sens.iloc[0]
        for col_ in ("sunlit %", "gen kWh", "min SoC %", "longest dark h"):
            sens[f"Δ {col_}"] = sens[col_] - base_row[col_]
        st.session_state["sens_tab"] = sens
        st.dataframe(sens.round(1), width="stretch", hide_index=True)
        figs2 = go.Figure([go.Bar(x=sens.case[1:], y=sens["Δ sunlit %"][1:], name="Δ sunlit (pts)", marker_color="#ffd54a"),
                           go.Bar(x=sens.case[1:], y=100 * sens["Δ gen kWh"][1:] / max(base_row["gen kWh"], 1e-9), name="Δ generation (%)", marker_color="#ff9f43"),
                           go.Bar(x=sens.case[1:], y=sens["Δ min SoC %"][1:], name="Δ battery floor (pts)", marker_color="#3ddc97")])
        figs2.update_layout(**PLOT, height=340, barmode="group", title="What a small change does to power", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(figs2, width="stretch")
    lib = ho.libration_stats(tl)
    c = st.columns(4)
    kpi(c[0], "Earth elevation range", f"{lib['earth_alt_min']:.1f}° … {lib['earth_alt_max']:.1f}°", "libration over this window")
    kpi(c[1], "Earth elevation swing", fmt(lib["earth_alt_swing"], 1, "°"), "~±7° optical libration")
    kpi(c[2], "Earth azimuth swing", fmt(lib["earth_az_swing"], 1, "°"))
    kpi(c[3], "Earth within 1° of skyline", fmt(lib["near_horizon_pct"], 1, "%"), "edge case: visibility flips on small errors")
    st.caption("Libration is included in the SPICE geometry (MOON_PA_DE421 orientation). When Earth sits within ~1° of the terrain skyline, "
               "a small horizon or pointing error flips the link on or off, so those hours carry the most schedule risk.")

    st.markdown("---\n### 📦 Mission profiles & handoff package")
    p1, p2, p3 = st.columns(3)
    with p1:
        preset_p = st.selectbox("Preset mission profile", list(ho.PROFILE_PRESETS), key="prof_preset")
        st.caption(ho.PROFILE_PRESETS[preset_p]["note"])
        if st.button("Load preset into the sidebar", key="prof_load"):
            st.session_state["_pending_profile"] = {"name": preset_p, "settings": ho.PROFILE_PRESETS[preset_p]["settings"]}
            st.rerun()
    with p2:
        up_p = st.file_uploader("Load a teammate's profile (.json)", type=["json"], key="prof_up")
        if up_p is not None and st.button("Apply uploaded profile", key="prof_apply"):
            try:
                d_ = ho.validate_profile(json.load(up_p))
                st.session_state["_pending_profile"] = d_
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load profile: {e}")
    with p3:
        pname = st.text_input("Profile name", f"{site.id} · {start_iso[:10]} · {days} d", key="prof_name")
    notes = st.text_area("Trade-off notes (they travel with the profile and the handoff note)", key="tradeoff_notes", height=130,
                         placeholder="e.g. B (+24 h) gains 6 pts sunlit but its first Earthset comes 30 h earlier - prefer A for the commissioning pass.")
    prof = ho.profile_from_state(st.session_state, PROFILE_KEYS, pname, notes)
    hand = ho.handoff_markdown(site, start_iso, days, prof["settings"], R, st.session_state.get("diff_tab"), notes, st.session_state.get("sens_tab"))
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Mission profile (.json)", json.dumps(prof, indent=1, default=str).encode(), f"profile_{site.id}.json", "application/json")
    d2.download_button("⬇ Handoff note (.md)", hand.encode(), f"handoff_{site.id}.md", "text/markdown")
    d3.caption("Full timeline CSV: 📄 Report tab.")
    with st.expander("Preview handoff note"):
        st.markdown(hand)

# ------------------------------------------------------------------ guide (accessibility)
with tabs[T_GUIDE]:
    st.markdown("### How to read this planner")
    st.markdown("1. **Pick a location** (map click, discovery, candidate or mission preset). 2. **Set dates.** 3. Read the banner: "
                "energy verdict, terrain verdict, feasibility. 4. Open a model tab for the evidence; switch on 🧮 *Show calculations* for the maths.")
    st.markdown("#### Finding and ranking the best landing points")
    st.markdown("- **🏆 Rank & compare points → Scan and rank** scores every 1 km cell of the cap with your weights (sunlight, Earth view, "
                "flat ground, night length, solar power, blackouts, distance to water-ice cold traps) and returns well-separated winners with a %.\n"
                "- **Run full M1–M5 on top K** re-checks the winners with the full physics for *your* dates, because the scan uses multi-year averages.\n"
                "- Put any mix of points (scan results, typed coordinates, known sites, the selected point) in the **comparison basket** to see them "
                "side by side: best values are highlighted, a radar shows each point's profile, and one click opens a point in every model tab.")
    st.markdown("#### Flying real hardware")
    st.markdown("- **🔋 M5 → Real-equipment loadout** lists real instruments and systems (VIPER NSS/NIRVSS, PRIME-1 TRIDENT/MSolo, Blue Ghost LISTER/LMS, "
                "MOXIE, Iris radio, RAD750, MMRTG, Kilopower …) with rounded public power and mass figures. Apply a loadout and every model, "
                "the dashboard and the Mission Sim use it. RTGs/reactors add constant power and RHUs add heat, which shrinks the survival-heater load.\n"
                "- In **🎮 M5 Mission Sim** the rover runs the VIPER/PRIME-1 instruments, and ⏭ buttons jump to the next sunrise or sunset with full physics.")
    st.markdown("#### Earth vs Moon - why intuition from Earth fails here")
    st.dataframe(guide.EARTH_VS_MOON, width="stretch", hide_index=True)
    st.markdown("#### Coordinates, time and why small changes matter")
    st.markdown(guide.COORDS_MD)
    st.markdown(f"#### Coordinate sensitivity at this point ({site.lat:.4f}°, {site.lon:.4f}°E)")
    if viz_pole.stack() and abs(site.lat) >= 80:
        ss = guide.spatial_sensitivity(viz_pole.value_at, site.lat, site.lon)
        st.dataframe(ss.round(3), width="stretch", hide_index=True)
        figss = go.Figure()
        for L, colr in (("Δ sun", "#ffd54a"), ("Δ earth", "#4aa8ff"), ("Δ le10", "#3ddc97")):
            figss.add_trace(go.Bar(x=ss.move, y=ss[L], name=L, marker_color=colr))
        figss.update_layout(**PLOT, height=320, barmode="group", title="How much the map values change if the landing point moves (1 km stack)")
        st.plotly_chart(figss, width="stretch")
        st.caption("Δ sun / Δ earth are fractions (0.05 = 5 percentage points of time); Δ le10 is the change in the share of ≤10° terrain.")
    else:
        st.caption("Spatial sensitivity uses the 80°S pole-wide stack; this point lies outside it.")
    st.markdown("#### Schedule sensitivity")
    st.markdown("Use **Compare · seasons → Date sensitivity** to slip the landing date, and **Mission clock** to see the same site at two "
                "dates side by side. At the pole, a slip of one or two weeks can swap a sunlit landing for a two-week night.")

# ------------------------------------------------------------------ extensions (Objective A drop-ins)
for i, ext in enumerate(EXT):
    with tabs[16 + i]:
        try:
            ext.render({"st": st, "site": site, "result": R, "timeline": tl, "run_for": run_for, "all_sites": ALL, "start_iso": start_iso,
                        "days": days, "step_min": step, "plot_layout": PLOT, "viz_pole": viz_pole, "viz_power": viz_power, "viz3d": viz3d, "root": ROOT})
        except Exception as e:
            st.error(f"Extension '{ext.TAB_TITLE}' failed: {e}")

# ------------------------------------------------------------------ data sources
with tabs[T_DATA]:
    st.markdown("### How each priority resource is implemented")
    c = st.columns(4)
    cards = [
        ("① NASA MOON TREK", "Illumination & Earth visibility", [
            "Layers <b>AVGVISIB_85S_060M_201608_EARTH_SP</b> and <b>WAC_POLE_ILL_PCT_SOUTH_100M</b> located in Trek's south-polar WMTS",
            "Same products ingested as numeric rasters (Trek tiles are PNG renderings)",
            "Used in: Explorer Sun/Earth layers, suitability index, Model 1/2 validation (r ≈ 0.97)"]),
        ("② ASU LROC QUICKMAP", "Terrain & observed illumination", [
            "<b>NAC_DTM_SHACKRDGE02</b> 3.5 m DTM → near-field horizon & slopes at Connecting Ridge",
            "<b>WAC_POLE_ILL_PCT_SOUTH_100M</b> → observed-illumination check",
            "LOLA polar 80/240 m → 200 km horizon ray-casting for any point"]),
        ("③ JPL HORIZONS", "Sun & Earth ephemerides", [
            "API CENTER=coord@301 topocentric; 16 sites × Sun/Earth × 361 epochs stored raw",
            "Agreement with runtime SPICE: max |Δel| 1.3×10⁻⁵°",
            "Drives every Sun/Earth angle, sunrise/Earthrise, 1/r² irradiance and the whole-pole power lattice"]),
        ("④ CSA", "Power, thermal & operations requirements", [
            "Lunar Utility Vehicle: survive ~2-week night at <b>−200 °C</b>, ≥10-yr life",
            "−200 °C → survival-heater sink (Model 5); 14-day darkness scenario",
            "No CSA power ratings exist - every load/array/battery value is a labelled assumption"]),
    ]
    for col, (t, sub, items) in zip(c, cards):
        col.markdown(f"<div class='card'><div class='p'>PRIORITY RESOURCE</div><h4>{t}</h4><div class='src'>{sub}</div><ul>" +
                     "".join(f"<li>{i}</li>" for i in items) + "</ul></div>", unsafe_allow_html=True)
    st.markdown("### Additional datasets (supplied local archive + PDS)")
    st.dataframe(pd.DataFrame([
        ("LDEM_80S_20MPP_ADJ", "NASA GSFC PGDA (LOLA)", "20 m", "80°S→pole", "pole-wide slope at native 20 m, 3D mesh, near-field horizons", "used"),
        ("LPSR_80S_20MPP_ADJ", "NASA GSFC PGDA", "20 m", "80°S→pole", "permanent-shadow fraction, suitability penalty", "used"),
        ("LDEC_80S_20MPP_ADJ", "NASA GSFC PGDA", "20 m", "80°S→pole", "supplied copy damaged (3 660 unreadable tile reads) - re-download needed", "excluded (corrupt)"),
        ("LDEM_80S_20MPP_ADJ gaps", "NASA GSFC PGDA", "20 m", "~2% of cap", "54 damaged tiles: slope left blank, elevation filled from PDS 80 m (flagged)", "handled"),
        ("ldem_87s_5mpp / ldec_87s_5mpp", "NASA GSFC PGDA", "5 m", "87.5°S→pole", "point terrain metrics & near-field horizons", "used"),
        ("Site01_final_adj_5mpp_surf", "NASA GSFC PGDA", "5 m", "Connecting ridge", "point terrain metrics", "used"),
        ("Diviner Tbol max/min anomaly", "LRO Diviner", "0.5°", "global", "thermal-environment layer", "used (display)"),
        ("NAC_DTM_VIKRAMSITE1", "ASU LROC", "3 m", "69.4°S", "outside the 80°S analysis cap", "documented, not used"),
        ("NAC_DTM_TYCHOPK", "ASU LROC", "2 m", "43°S (Tycho)", "not a south-polar product", "excluded"),
        ("de421.bsp, moon_pa_de421, moon_080317.tf, pck, naif0012.tls", "NASA NAIF", "-", "-", "SPICE runtime ephemeris", "used"),
    ], columns=["product", "source", "resolution", "coverage", "role", "status"]), width="stretch", hide_index=True)
    st.markdown("### Coverage footprints")
    from clps_engine.rasters import latlon_to_ps as _ps
    fp = go.Figure()
    for lat_c, name, colr in ((-80, "LOLA 20 m / 80 m / PSR (80°S)", "#38d9ff"), (-85, "LOLA AVGVISIB Sun & Earth (85°S)", "#ffd54a"),
                              (-88, "WAC illumination (88°S)", "#ff8c42")):
        t = np.linspace(0, 2 * np.pi, 200)
        x, y = _ps(np.full(t.size, lat_c), np.degrees(t))
        fp.add_trace(go.Scatter(x=x / 1000, y=y / 1000, mode="lines", name=name, line=dict(color=colr, width=2)))
    fp.add_trace(go.Scatter(x=[-100, 100, 100, -100, -100], y=[-100, -100, 100, 100, -100], mode="lines", name="PGDA 5 m (87.5°S box)",
                            line=dict(color="#3ddc97", dash="dash")))
    fp.add_trace(go.Scatter(x=[-13.7, -7.6, -7.6, -13.7, -13.7], y=[-13.2, -13.2, -8.2, -8.2, -13.2], mode="lines", fill="toself",
                            name="NAC DTM SHACKRDGE02 3.5 m", line=dict(color="#ff5d73")))
    fp.add_trace(go.Scatter(x=[-19, -3, -3, -19, -19], y=[-20, -20, -4, -4, -20], mode="lines", name="PGDA Site01 5 m", line=dict(color="#9b5cff")))
    xx, yy = _ps(site.lat, site.lon)
    fp.add_trace(go.Scatter(x=[xx / 1000], y=[yy / 1000], mode="markers", name="analysed point", marker=dict(size=12, color="#fff", symbol="x")))
    mk_mod.add_pole2d(fp, MK, gl=False)
    fp.update_layout(**PLOT, height=520, title="Spatial coverage of integrated products (polar stereographic, km)", yaxis_scaleanchor="x")
    st.plotly_chart(fp, width="stretch")

# ------------------------------------------------------------------ validation
with tabs[T_VAL]:
    def show_csv(title, p, **kw):
        if p.exists():
            st.markdown(f"**{title}** · `{p.relative_to(ROOT)}`")
            st.dataframe(pd.read_csv(p).round(4), width="stretch", hide_index=True, **kw)
    v = PROC / "illumination_validation.csv"
    if v.exists():
        vv = pd.read_csv(v).dropna(subset=["lola_sun_vis", "lola_earth_vis"])
        figv = go.Figure()
        figv.add_trace(go.Scatter(x=vv.lola_sun_vis, y=vv.model_sun_disc_vis_18p6yr, mode="markers+text", text=vv.site_id, name="Sun",
                                  marker=dict(size=10, color="#ffd54a"), textposition="top center", textfont=dict(size=9)))
        figv.add_trace(go.Scatter(x=vv.lola_earth_vis, y=vv.model_earth_disc_vis_18p6yr, mode="markers", name="Earth", marker=dict(size=10, color="#4aa8ff")))
        figv.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#7fa7b8"), name="1:1"))
        figv.update_layout(**PLOT, height=420, title="Model 1/2 (18.6-yr) vs Moon Trek/LOLA AVGVISIB at 16 sites", xaxis_title="LOLA product",
                           yaxis_title="LunarSouth model")
        st.plotly_chart(figv, width="stretch")
    lv = PROC / "polewide" / "lattice_validation.csv"
    if lv.exists():
        show_csv("Whole-pole lattice vs LOLA AVGVISIB (every lattice point inside 85°S)", lv)
    show_csv("Model 1/2 validation table", v)
    p = PROC / "jpl_horizons" / "horizons_vs_spice.csv"
    if p.exists():
        hv = pd.read_csv(p)
        figh = go.Figure([go.Box(y=(hv.hz_sun_el - hv.spice_sun_el) * 3600, name="Sun Δel (arcsec)", marker_color="#ffd54a"),
                          go.Box(y=(hv.hz_earth_el - hv.spice_earth_el) * 3600, name="Earth Δel (arcsec)", marker_color="#4aa8ff")])
        figh.update_layout(**PLOT, height=320, title="JPL Horizons − SPICE residuals, 16 sites × 361 epochs (arcseconds)")
        st.plotly_chart(figh, width="stretch")
    show_csv("Dataset statistics (full products)", PROC / "dataset_statistics.csv")
    pv = PROC / "provenance.json"
    if pv.exists():
        with st.expander("Provenance records"):
            st.json(json.loads(pv.read_text()))

# ------------------------------------------------------------------ report
with tabs[T_REP]:
    T = R.terrain
    kp = [("Sunlit", fmt(R.solar["sunlit_pct"], 1, "%")), ("Earth visible", fmt(R.dte["earth_visible_pct"], 1, "%")),
          ("Generated", fmt(R.generation["total_gen_kwh"], 1, " kWh")), ("Min SoC", fmt(surv["min_soc_pct"], 0, "%")),
          ("Longest dark", fmt(R.solar["longest_darkness_h"], 0, " h")), ("Longest blackout", fmt(R.dte["longest_dte_blackout_h"], 0, " h")),
          ("Terrain", str(T.get("status", "n/a")).split(" - ")[0]), ("Feasibility", f"{feasibility_score(row)}/100")]
    secs = [
        ("Configuration", f"<table><tr><td>Location</td><td>{site.lat:.5f}°, {site.lon:.5f}°E, {site.elev_m:.1f} m ({site.source})</td></tr>"
                          f"<tr><td>Window</td><td>{start_iso} UTC + {days} d, step {step} min</td></tr>"
                          f"<tr><td>Scenario</td><td>{en.SCENARIOS[scenario]}</td></tr><tr><td>Horizon</td><td>{R.horizon_source}</td></tr>"
                          f"<tr><td>Array</td><td>{orient}, {area} m², η {eff}, dust {dust}, aging {degr}, MPPT {mppt} → {R.generation['nameplate_w_at_1au']:.0f} W nameplate</td></tr>"
                          f"<tr><td>Battery</td><td>{cap:.0f} Wh, SoC₀ {soc0}, DoD {dod}, safe {safe}</td></tr></table>"),
        ("Energy survival verdict", f"<div class='st'><b>{surv['status']}</b> - demand {surv['total_demand_kwh']:.1f} kWh vs generated "
                                     f"{surv['total_generated_kwh']:.1f} kWh; unserved {surv['energy_deficit_kwh']:.2f} kWh; time to critical SoC "
                                     f"{surv['hours_to_critical_soc'] if surv['hours_to_critical_soc'] is not None else 'never'} h; largest consumer "
                                     f"{surv['top_consumer']}.</div>"),
        ("Illumination & communication", f"<p>Sunlit {R.solar['sunlit_pct']:.1f}% (mean disc {R.solar['mean_disc_fraction_pct']:.1f}%), longest darkness "
                                         f"{R.solar['longest_darkness_h']:.0f} h · Earth visible {R.dte['earth_visible_pct']:.1f}%, longest blackout "
                                         f"{R.dte['longest_dte_blackout_h']:.0f} h, {R.dte['n_windows']} DTE windows (geometric only).</p>"),
        ("Terrain (Model 3)", f"<p>{T.get('best_source','')} · index {T.get('terrain_score')} · {T.get('status')}</p><ul>" +
                              "".join(f"<li>⚠ {h}</li>" for h in T.get("hazards", [])) + "".join(f"<li>{u}</li>" for u in T.get("uncertainty", [])) + "</ul>"),
        ("Independent products at this point", f"<p>LOLA 18.6-yr Sun visibility {fmt(100*viz_pole.value_at('sun', site.lat, site.lon),1,'%')} · "
                                               f"Earth {fmt(100*viz_pole.value_at('earth', site.lat, site.lon),1,'%')} · WAC {fmt(viz_pole.value_at('wac', site.lat, site.lon),1,'%')} · "
                                               f"20 m terrain ≤10° {fmt(100*viz_pole.value_at('le10', site.lat, site.lon),1,'%')} (1 km cell) · "
                                               f"lattice mean power {fmt(viz_pole.value_at('power', site.lat, site.lon),0,' W/m²')}</p>"),
    ]
    st.markdown(f"### Mission assessment - {site.name}")
    for h, body in secs:
        st.markdown(f"**{h}**")
        st.markdown(body, unsafe_allow_html=True)
    if st.button("📄 Generate full HTML report for this point", type="primary"):
        figs = [("Power timeline", viz_power.power_timeline(tl)), ("Energy loss chain", viz_power.loss_waterfall(tl, R.array, step)),
                ("Energy flow", viz_power.energy_sankey(tl, R.subsystem_summary, step)), ("Power rose", viz_power.power_rose(tl, step)),
                ("Generation fingerprint", viz_power.calendar_heatmap(tl))]
        st.session_state["report_html"] = build_html(f"Mission assessment · {site.name}", kp, secs, figs)
        st.session_state["report_name"] = f"LunarSouth_{site.id}_{site.lat:.3f}_{site.lon:.3f}.html"
    if "report_html" in st.session_state:
        st.download_button("⬇ Download HTML report", st.session_state["report_html"].encode(), st.session_state["report_name"], "text/html")
    st.download_button("⬇ Download full timeline (CSV)", tl.to_csv(index=False).encode(), f"{site.id}_timeline.csv")

st.markdown(f"<div class='footer'>{TEAM} · NASA SPACE APPS CHALLENGE 2026 · DATA: NASA MOON TREK · ASU LROC QUICKMAP · JPL HORIZONS · CSA · NASA PGDA</div>",
            unsafe_allow_html=True)
