#!/usr/bin/env python3
"""Validation of the upgraded models (M1-M5) and integrated data products.

Requires the processed products (python tools/ingest_sources.py all) and kernels/.
Run:  python tests/test_upgraded_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from clps_engine import energy as en
from clps_engine import power_gen as pg
from clps_engine.candidates import all_candidates
from clps_engine.geometry import compute_geometry
from clps_engine.mission import run_mission, terrain_assessment, horizon_mask
from clps_engine.rasters import latlon_to_ps, ps_to_latlon, product_registry
from clps_engine.solar import disc_fraction_above
from clps_engine.sites import custom_site

PROC = ROOT / "data" / "processed"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    fails += not bool(cond)


print("A. Data products & georeferencing")
reg = product_registry()
for k in ("LOLA_DEM_80M", "LOLA_DEM_240M", "LOLA_AVG_SUN_VIS_60M", "LOLA_AVG_EARTH_VIS_60M", "WAC_ILLUM_PCT_100M", "NAC_DTM_SHACKRDGE02"):
    check(f"{k} present", k in reg)
x, y = latlon_to_ps(-85.4381, 37.36667)
la, lo = ps_to_latlon(x, y)
check("PS projection round-trip", abs(la + 85.4381) < 1e-9 and abs(lo - 37.36667) < 1e-9)
z = float(reg["LOLA_DEM_80M"].sample_latlon(-85.4381, 37.36667))
check("LOLA 80 m elevation at Site06 matches catalogue (5946 m)", abs(z - 5946.1) < 10, f"{z:.1f} m")
zn = float(reg["NAC_DTM_SHACKRDGE02"].sample_latlon(-89.444, -137.291))
zl = float(reg["LOLA_DEM_80M"].sample_latlon(-89.444, -137.291))
check("NAC DTM vs LOLA at CR06 agree < 10 m", abs(zn - zl) < 10, f"NAC {zn:.1f} LOLA {zl:.1f}")
check("NAC returns NaN outside its footprint", not np.isfinite(reg["NAC_DTM_SHACKRDGE02"].sample_latlon(-85.4, 37.0)))

print("\nB. All 16 candidates processed")
allsites_all = all_candidates()
allsites = [s for s in allsites_all if s.lat <= -79.9]  # polar cap: fully pre-processed
sites = [s for s in allsites if s.source in ("baseline", "research")]
check("16 project candidate sites preserved", len(sites) == 16, ",".join(s.id for s in sites))
check("discovered sites loaded alongside", sum(s.source == "discovered" for s in allsites) >= 30)
hz = pd.read_csv(PROC / "horizons" / "horizon_profiles_lola.csv")
check("200 km horizon for every site incl. discoveries", set(hz.site_id) >= {s.id for s in allsites})
tm = pd.read_csv(PROC / "terrain_metrics_all_sites.csv")
check("80 m terrain metrics for every site", set(tm[tm["product"] == "LDEM_80S_80M"].site_id) >= {s.id for s in allsites})
from clps_engine.rasters import product_registry as _pr
_d20 = _pr().get("LDEM_20M")
_gap = {s.id for s in allsites if not np.isfinite(_d20.sample_latlon(s.lat, s.lon))}
check("20 m PGDA terrain metrics for every site outside damaged-tile gaps", set(tm[tm["product"] == "LDEM_80S_20M"].site_id) >= {s.id for s in allsites} - _gap,
      f"sites in 20 m damaged tiles (use 80 m/5 m): {sorted(_gap)}")
check("NAC 3.5 m metrics for Connecting Ridge sites", set(tm[tm["product"] == "NAC_DTM_SHACKRDGE02"].site_id) == {"CR06", "CR07", "CR001"})
check("3D scene for every site", all((PROC / "terrain3d" / f"{s.id}_scene.npz").exists() for s in allsites))

print("\nC. Model 1 / 2 validation against independent products")
v = pd.read_csv(PROC / "illumination_validation.csv").dropna(subset=["lola_sun_vis", "lola_earth_vis"])
r_sun = np.corrcoef(v.model_sun_disc_vis_18p6yr, v.lola_sun_vis)[0, 1]
r_earth = np.corrcoef(v.model_earth_disc_vis_18p6yr, v.lola_earth_vis)[0, 1]
check("Sun visibility vs LOLA AVGVISIB correlation > 0.9", r_sun > 0.9, f"r={r_sun:.3f}")
check("Earth visibility vs LOLA AVGVISIB_EARTH correlation > 0.9", r_earth > 0.9, f"r={r_earth:.3f}")
check("Earth visibility mean abs diff < 0.05", v.earth_vis_diff_model_minus_lola.abs().mean() < 0.05)
hv = pd.read_csv(PROC / "jpl_horizons" / "horizons_vs_spice.csv")
check("JPL Horizons vs SPICE Sun elevation < 0.001 deg", (hv.hz_sun_el - hv.spice_sun_el).abs().max() < 1e-3)
check("JPL Horizons vs SPICE Earth elevation < 0.001 deg", (hv.hz_earth_el - hv.spice_earth_el).abs().max() < 1e-3)
check("disc fraction: 0 below, 0.5 at, 1 above", np.allclose(disc_fraction_above(np.array([-1, 0, 1.0]), np.array([0.27] * 3)), [0, 0.5, 1]))
old = pd.read_csv(ROOT / "data" / "horizon_profiles.csv")
new = hz[hz.site_id == "LM1"]
o = np.interp(new.azimuth_deg, old.azimuth_deg, old.horizon_elev_deg, period=360)
check("LM1 new horizon >= legacy 10 km horizon (adds far terrain)", (new.horizon_elev_deg.values - o).min() > -0.5)

print("\nD. Model 4 generation physics")
s = [x for x in sites if x.id == "LM1"][0]
g = compute_geometry(s, "2026-10-15T00:00:00", 24 * 30, 60)
from clps_engine.solar import apply_solar_visibility
g = apply_solar_visibility(g, horizon_mask(s)[0])
cfg = pg.SolarArrayConfig()
o4 = pg.apply_generation(g, cfg, 60)
check("zero generation when disc fully below terrain", (o4.loc[o4.sun_disc_fraction == 0, "gen_w"] == 0).all())
check("generation never exceeds irradiance x area", (o4.gen_w <= o4.irradiance_w_m2 * cfg.area_m2 + 1e-6).all())
check("irradiance within 1361/r^2 range (1300-1420 W/m2)", o4.irradiance_w_m2.between(1300, 1420).all())
o_dirty = pg.apply_generation(g, pg.SolarArrayConfig(dust_factor=0.5), 60)
check("dust factor scales generation linearly", np.isclose(o_dirty.gen_w.sum() * 0.95 / 0.5, o4.gen_w.sum(), rtol=1e-6))
oh = pg.apply_generation(g, pg.SolarArrayConfig(orientation="horizontal"), 60)
check("vertical tracking beats horizontal at the pole", o4.gen_w.sum() > 5 * oh.gen_w.sum())

print("\nE. Model 5 energy balance")
bat = en.BatteryConfig()
tl, summ = en.simulate(o4, en.default_subsystems(), bat, en.ThermalConfig(), 60)
check("SoC within [1-DoD, 1]", tl.soc.min() >= 1 - bat.max_dod - 1e-9 and tl.soc.max() <= 1 + 1e-9)
h = 1.0
e_in = (tl.batt_charge_w * bat.charge_efficiency).sum() * h
e_out = (tl.batt_discharge_w / bat.discharge_efficiency).sum() * h
sd = tl.self_discharge_wh
bal = bat.initial_soc * bat.capacity_wh + e_in - e_out - sd.sum() - tl.soc.iloc[-1] * bat.capacity_wh
check("battery energy conservation (Wh)", abs(bal) < 1.0, f"residual {bal:.3f} Wh")
check("served <= demand", (tl.load_served_w <= tl.load_demand_w + 1e-6).all())
dark = run_mission(s, "2026-10-15T00:00:00", 24 * 30, 60, scenario="extended_dark")
check("14-day extended darkness drains default battery", dark.survival["min_soc_pct"] < 30)
cr7 = [x for x in sites if x.id == "CR07"][0]
big = run_mission(cr7, "2026-10-15T00:00:00", 24 * 30, 60, scenario="extended_dark",
                  battery=en.BatteryConfig(capacity_wh=60000, max_charge_w=5000, max_discharge_w=5000))
check("CR07 (continuous sun after the event) survives with 60 kWh", big.survival["survives"])
for sc in en.SCENARIOS:
    r = run_mission(s, "2026-10-15T00:00:00", 24 * 10, 60, scenario=sc)
    check(f"scenario {sc} runs", r.survival["status"] != "")

print("\nF. Site/date changes & missing data")
a = run_mission(s, "2026-10-15T00:00:00", 24 * 10, 60)
b = run_mission(s, "2026-11-01T00:00:00", 24 * 10, 60)
check("date change changes geometry", not np.isclose(a.timeline.sun_az.iloc[0], b.timeline.sun_az.iloc[0]))
c = run_mission(custom_site(-89.45, -137.3, 1950.0, "x", "CUSTOMTEST"), "2026-10-15T00:00:00", 48, 60)
check("custom site gets live LOLA horizon", "LOLA" in c.horizon_source, c.horizon_source)
far = run_mission(custom_site(-70.0, 0.0, 0.0, "far", "FAR"), "2026-10-15T00:00:00", 48, 60)
check("site outside polar DEMs uses global LOLA horizon (labelled, not flat)", "LDEM_16" in far.horizon_source, far.horizon_source[:50])
check("terrain assessment refuses landing verdict outside polar DEMs", terrain_assessment(custom_site(-70, 0, 0, "f", "FAR")).get("status", "").startswith("NOT ASSESSED"))

print("\nG. Pole-wide coverage (80S -> pole)")
PW = PROC / "polewide"
m = np.load(PW / "mosaic1km.npz")
inside = np.hypot(*np.meshgrid(m["x_m"], m["y_m"])) < 300000
check("1 km mosaic covers >= 97% of the 80S cap with 20 m-derived slope", np.isfinite(m["le10"])[inside].mean() >= 0.97,
      f"{100*np.isfinite(m['le10'])[inside].mean():.1f}%")
check("elevation complete over the cap (gaps filled from PDS 80 m, flagged)", np.isfinite(m["elev_mean"])[inside].mean() > 0.999)
lat_ = pd.read_csv(PW / "sim_lattice.csv")
check("whole-pole lattice > 10 000 points", len(lat_) > 10000, f"{len(lat_)}")
check("lattice power physically bounded (0..400 W/m2)", lat_.mean_specific_power_w_m2.between(0, 400).all())
lv = pd.read_csv(PW / "lattice_validation.csv").set_index("quantity")
check("lattice Sun vs LOLA AVGVISIB r > 0.8", lv.loc["sun", "pearson_r"] > 0.8, f"r={lv.loc['sun','pearson_r']:.3f}")
check("lattice Earth vs LOLA AVGVISIB_EARTH r > 0.8", lv.loc["earth", "pearson_r"] > 0.8, f"r={lv.loc['earth','pearson_r']:.3f}")
disc = pd.read_csv(ROOT / "data" / "discovered_sites.csv")
check(">= 30 discovered candidate sites", len(disc) >= 30, f"{len(disc)}")
from clps_engine.rasters import latlon_to_ps as _ps
_x, _y = _ps(disc.latitude_deg.values, disc.longitude_deg.values)
_d = np.hypot(_x[:, None] - _x[None, :], _y[:, None] - _y[None, :])
dmin = _d[np.triu_indices(len(disc), 1)].min()
check("discoveries separated >= 7.5 km", dmin >= 7500, f"min {dmin/1000:.1f} km")
pt = run_mission(custom_site(-88.2, 160.0, 0.0, "pick", "POINT"), "2026-10-15T00:00:00", 48, 60)
check("arbitrary map point runs full chain", pt.survival["status"] != "" and pt.terrain.get("status", "").split()[0] in ("SUITABLE", "CONDITIONAL", "POOR"))
from clps_engine import viz_power, viz_pole
for fn in ("loss_waterfall", "energy_sankey", "calendar_heatmap", "power_rose", "power_timeline"):
    try:
        f = {"loss_waterfall": lambda: viz_power.loss_waterfall(pt.timeline, pt.array, 60),
             "energy_sankey": lambda: viz_power.energy_sankey(pt.timeline, pt.subsystem_summary, 60),
             "calendar_heatmap": lambda: viz_power.calendar_heatmap(pt.timeline),
             "power_rose": lambda: viz_power.power_rose(pt.timeline, 60),
             "power_timeline": lambda: viz_power.power_timeline(pt.timeline)}[fn]()
        check(f"power visual {fn}", f is not None)
    except Exception as e:
        check(f"power visual {fn}", False, str(e))
wf = viz_power.loss_waterfall(pt.timeline, pt.array, 60).data[0]
check("loss waterfall closes (sum of steps = delivered)", abs(sum(wf.y[:-1]) - wf.y[-1]) < 1e-6 * max(1, abs(wf.y[-1])) + 1e-6)
for L in viz_pole.LAYERS:
    viz_pole.pole_map(L, [], disc, (-88.2, 160.0))
check("all pole-wide map layers render", True)
viz_pole.pole_3d("suitability", [], disc, (-88.2, 160.0))
check("pole-wide 3D renders", True)

print("\nH. Objective-A gap fixes (presets, link budget, events, global horizon, extensions)")
pres = [x for x in allsites_all if x.source.startswith("preset")]
check(">= 20 historic/commercial/international presets", len(pres) >= 20, f"{len(pres)}")
P = {x.id: x for x in pres}
from clps_engine.mission import horizon_mask as _hm
m15, src15 = _hm(P["A15"])
check("Apollo 15 gets a real (global DEM) horizon - Apennines > 5 deg", m15 is not None and max(m15[1]) > 5, f"{max(m15[1]):.1f} deg; {src15[:30]}")
ce4 = run_mission(P["CE4"], "2026-10-15T00:00:00", 24 * 10, 60)
check("far-side Chang'e 4: Earth never visible, link never closes", ce4.dte["earth_visible_pct"] == 0 and ce4.link["link_closed_pct"] == 0)
a11 = run_mission(P["A11"], "2026-10-15T00:00:00", 24 * 10, 60)
fs = a11.timeline.fspl_db.mean()
check("X-band FSPL at lunar distance ~ 222-224 dB", 222 < fs < 224.5, f"{fs:.2f} dB")
check("link closed <= geometric visibility", a11.link["link_closed_pct"] <= a11.dte["earth_visible_pct"] + 1e-9)
check("DSN handoffs ~ 2-4 per day for nearside site", 1.5 * 10 <= a11.link["dsn_handoffs"] <= 4.5 * 10, f"{a11.link['dsn_handoffs']} in 10 d")
from clps_engine import events as _ev
check("event log contains Sun, Earth and link events", {"SUNRISE", "SUNSET", "LINK AOS", "LINK LOS"} <= set(a11.events.event) | set(ce4.events.event))
cd = _ev.countdowns(a11.events, a11.timeline.time_utc.iloc[0])
check("countdowns are non-negative", (cd.t_minus_h.dropna() >= 0).all())
from clps_engine import extensions as _ext
check("extension loader works (template ignored)", isinstance(_ext.discover(), list))
check("terrain refuses landing-scale verdict on 1.9 km global DEM", a11.terrain["status"].startswith("NOT ASSESSED"))

print(f"\n=== {'PASS' if fails == 0 else 'FAIL'} ({fails} failure(s)) ===")
sys.exit(1 if fails else 0)
