"""Model 5 Mission Sim: payload + browser physics engine checked against the Python models.

    .venv/bin/python tests/test_mission_sim.py

Needs Node.js for the engine checks (skipped with a notice if `node` is not on PATH).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clps_engine import activities as act, energy as en, power_gen as pg, rf_link, sim_payload as sp, viz3d  # noqa: E402
from clps_engine.candidates import all_candidates  # noqa: E402
from clps_engine.mission import run_mission  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    site = next(s for s in all_candidates() if s.id == "LM1")
    arr, bat, th, lbc = pg.SolarArrayConfig(), en.BatteryConfig(), en.ThermalConfig(), rf_link.LinkBudgetConfig()
    subs = en.default_subsystems()
    no_ops = act.ActivityPlan(comms=[], bursts=[])   # subsystems only, so the game's untouched outpost must equal energy.simulate
    R = run_mission(site, "2026-10-15T00:00:00", 15 * 24, 30.0, array=arr, subsystems=subs, battery=bat, thermal=th, link_budget=lbc,
                    activities=no_ops)
    scene = viz3d.scene_for(site)
    P = sp.build_payload(R, scene, site, array=arr, battery=bat, thermal=th, lbc=lbc, subsystems=subs)
    js = sp.payload_json(P)

    print("\nA. Payload")
    check("payload is strict JSON", json.loads(js)["n"] == len(R.timeline))
    check("payload size < 900 KB", len(js) < 900_000, f"{len(js) // 1024} KB")
    check("far-field mask from LOLA ray-cast", P["far_src"].startswith("LOLA"), P["far_src"])
    full = np.interp(np.arange(0, 360.0), *R.mask, period=360.0)
    check("far-field horizon <= full site horizon (+0.05 deg)", bool(np.all(np.array(P["far_hz"]) <= full + 0.05)))
    check(">= 4 science targets from the data", len(P["pois"]) >= 4, ", ".join(f"{p['id']}:{p['kind']}" for p in P["pois"]))
    check("targets on drivable ground (<= 22 deg)", all(p["slope"] <= 22.0 for p in P["pois"]))

    print("\nB. Python reference horizon vs Model 1")
    hz = sp.cell_horizon(scene, 0.0, 0.0, np.array(P["far_hz"]))
    tl = R.timeline
    hzs = np.interp(tl.sun_az, np.arange(0, 360.0), hz, period=360.0)
    lit = (tl.sun_alt + tl.sun_radius).to_numpy() > hzs
    agree = 100 * float((lit == tl.sunlit.to_numpy()).mean())
    check("rover-at-site sunlit agrees with M1 >= 97%", agree >= 97.0, f"{agree:.2f}%")

    print("\nD. Ray-traced illumination, shadow physics and handoff (clps_engine.handoff)")
    from clps_engine import handoff as ho
    im = ho.illumination_map(scene, tl, np.array(P["far_hz"]), n_samples=48)
    d = abs(100 * im["site_frac"] - R.solar["sunlit_pct"])
    check("ray-traced site pixel within 5 pts of M1 sunlit", d < 5.0, f"{100 * im['site_frac']:.1f}% vs {R.solar['sunlit_pct']:.1f}%")
    check("best spot within 1 km >= site", im["best_1000"]["frac"] >= im["site_frac"] - 1e-9)
    check("PSR boundary distance finite", np.isfinite(ho.psr_boundary_km(scene)), f"{ho.psr_boundary_km(scene):.2f} km")
    g = ho.grazing_stats(tl)
    check("grazing stats computed", 0 <= g["grazing_pct"] <= 100, f"median margin {g['median_margin_deg']:.2f} deg")
    lib = ho.libration_stats(tl)
    check("libration swing physical (5-20 deg)", 5 < lib["earth_alt_swing"] < 20, f"{lib['earth_alt_swing']:.1f} deg")
    m = ho.diff_metrics(R)
    dt_ = ho.diff_table(m, m)
    check("diff of identical runs is all 'same'", set(dt_.verdict[:-1]) <= {"same", "n/a"})
    md = ho.handoff_markdown(site, "2026-10-15T00:00:00", 15, {"cap": 4000.0}, R, dt_, "test note")
    check("handoff note contains results, diff and notes", all(k in md for k in ("Key results", "Scenario diff", "test note")))
    for name, pr in ho.PROFILE_PRESETS.items():
        ho.validate_profile(pr)
    check(">= 4 preset mission profiles", len(ho.PROFILE_PRESETS) >= 4)

    print("\nE. Best-point scan, equipment catalogue, RTG / RHU physics")
    from clps_engine import scan as scn, equipment as eqm
    r = scn.scan(top_n=10, min_sep_km=8)
    check("scan returns 10 ranked cells", len(r) == 10, f"best {r.score_pct.iloc[0]:.1f}%")
    check("scan scores sorted, 0-100", bool((r.score_pct.diff().dropna() <= 1e-9).all() and r.score_pct.between(0, 100).all()))
    from clps_engine.rasters import latlon_to_ps
    xy = np.array([latlon_to_ps(a, b) for a, b in zip(r.lat, r.lon)])
    dmin = min(np.hypot(*(xy[i] - xy[j])) for i in range(len(xy)) for j in range(i))
    check("ranked points respect min separation", dmin >= 7_999, f"{dmin / 1000:.1f} km")
    check("ranked points honour slope filter", bool((r.slope_deg <= 15).all()))
    rr = scn.scan(top_n=5, center=(site.lat, site.lon), radius_km=40)
    dd = [np.hypot(*(np.array(latlon_to_ps(a, b)) - np.array(latlon_to_ps(site.lat, site.lon)))) for a, b in zip(rr.lat, rr.lon)]
    check("radius search stays inside radius", len(rr) > 0 and max(dd) <= 40_500, f"max {max(dd) / 1000:.1f} km")
    sc_, parts = scn.full_score({**R.summary_row(), "gen_total_kwh_per_day": R.generation["total_gen_kwh"] / 15, "psr_km": 3.0})
    check("full-model score in 0-100", 0 <= sc_ <= 100, f"{sc_:.1f}%")
    for nm, q in eqm.LOADOUTS.items():
        assert all(k in eqm.BY_ID for k in q), nm
    check(">= 25 real devices and 6 loadouts", len(eqm.CATALOG) >= 25 and len(eqm.LOADOUTS) >= 6)
    q = eqm.LOADOUTS["Night-surviving seismic station (RTG)"]
    src = eqm.loadout_sources(q)
    check("MMRTG + RHUs give 110 We / 166 Wt", src == {"rps_w": 110.0, "heat_w": 166.0}, str(src))
    arr2 = pg.SolarArrayConfig(rps_w=110.0)
    R2 = run_mission(site, "2026-10-15T00:00:00", 15 * 24, 30.0, array=arr2, subsystems=subs, battery=bat, thermal=th, link_budget=lbc, activities=no_ops)
    dg = float((R2.timeline.gen_w - R.timeline.gen_w).mean())
    check("RTG adds a constant 110 W to the bus", abs(dg - 110.0) < 1e-6, f"{dg:.3f} W")
    check("RTG raises minimum SoC", R2.survival["min_soc_pct"] >= R.survival["min_soc_pct"] - 1e-9,
          f"{R.survival['min_soc_pct']:.1f} -> {R2.survival['min_soc_pct']:.1f} %")
    th2 = en.ThermalConfig(waste_heat_w=40.0)
    lit = np.array([True, False])
    check("waste heat offsets heaters 1:1 (never negative)", np.allclose(th.demand_w(lit) - th2.demand_w(lit), np.minimum(th.demand_w(lit), 40.0)))

    node = shutil.which("node")
    if not node:
        print("\nC. Browser engine - SKIPPED (node not found)")
    else:
        print("\nC. Browser engine (Node) vs Python")
        pts = [(0.0, 0.0), (4000.0, -3000.0), (-6500.0, 5200.0)]
        with tempfile.TemporaryDirectory() as td:
            pj, qj = Path(td) / "p.json", Path(td) / "q.json"
            pj.write_text(js)
            qj.write_text(json.dumps(pts))
            res = json.loads(subprocess.run([node, str(ROOT / "tests" / "sim_engine_check.js"), str(pj), str(qj)],
                                            capture_output=True, text=True, check=True, timeout=120).stdout)
        for (x, y), prof in zip(pts, res["profiles"]):
            ref = sp.cell_horizon(scene, x, y, np.array(P["far_hz"]))
            d = float(np.max(np.abs(np.array(prof) - ref)))
            check(f"JS horizon == Python at ({x/1000:.1f}, {y/1000:.1f}) km", d < 0.05, f"max |diff| {d:.4f} deg")
        soc_py = tl.soc_pct.to_numpy()
        d = float(np.max(np.abs(np.array(res["soc"]) - soc_py)))
        check("JS outpost battery replay == energy.simulate SoC", d < 0.05, f"max |diff| {d:.4f} %")
        d2 = float(np.max(np.abs(np.array(res["served"]) - tl.load_served_w.to_numpy())))
        check("JS served load == energy.simulate", d2 < 0.1, f"max |diff| {d2:.4f} W")
        ref_drive = [act.Traverse(slope_deg=s, speed_m_s=0.1).drive_w() for s in (0, 5, 15)]
        check("JS drive power == activities.Traverse.drive_w", np.allclose(res["drive"], ref_drive, atol=1e-6),
              " / ".join(f"{v:.1f} W" for v in res["drive"]))
        check("line of sight at the outpost is clear", res["los_self"])

    print(f"\n=== {'PASS' if not FAIL else 'FAIL'} ({len(FAIL)} failure(s)) ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
