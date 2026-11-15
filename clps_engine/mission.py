"""Integrated mission chain: Model 1 -> 2 -> 3 -> 4 -> 5 for any site and date.

One call gives the planner every model's result on the same timeline, so a
change of site, date, array or load propagates through all of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import dte as dte_mod
from . import energy as en
from . import power_gen as pg
from . import rf as rf_mod
from . import rf_link
from . import events as ev_mod
from . import activities as act_mod
from . import pv_electrical as pve
from . import solar as solar_mod
from .geometry import compute_geometry
from .horizon import load_horizon_table, mask_for_site
from .sites import Site
from .windows import runs_table

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


# ------------------------------------------------------------------ inputs
@lru_cache(maxsize=1)
def _hz_table():
    p = PROC / "horizons" / "horizon_profiles_lola.csv"
    return load_horizon_table(p) if p.exists() else None


def horizon_mask(site: Site):
    """LOLA 200-km terrain horizon if processed; else compute from DEM on the fly (custom sites). Cached per location."""
    return _horizon_mask_cached(site.id, round(site.lat, 6), round(site.lon, 6))


@lru_cache(maxsize=256)
def _horizon_mask_cached(site_id: str, lat: float, lon: float):
    site = Site(site_id, site_id, lat, lon)
    t = _hz_table()
    if t is not None:
        m = mask_for_site(t, site.id)
        if m is not None:
            return m, "LOLA LDEM_80S_80M + LDEM_75S_240M ray-cast, 200 km"
    try:
        from .horizon_build import ray_horizon
        from .rasters import product_registry

        from .horizon_build import near_field_from_raster

        reg = product_registry()
        if "LOLA_DEM_80M" in reg and reg["LOLA_DEM_80M"].covers(site.lat, site.lon):
            z = float(reg["LOLA_DEM_80M"].sample_latlon(site.lat, site.lon))
            h = ray_horizon(site.lat, site.lon, reg["LOLA_DEM_80M"], reg.get("LOLA_DEM_240M"), site_elev_m=z)
            az, el = h.azimuth_deg.to_numpy(), h.horizon_elev_deg.to_numpy()
            src = "LOLA 80/240 m ray-cast, 200 km (computed live)"
            for key, rng in (("LDEM_5M_87S", 1000.0), ("LDEM_20M", 2000.0)):
                d = reg.get(key)
                if d is not None and d.covers(site.lat, site.lon):
                    zn = float(d.sample_latlon(site.lat, site.lon))
                    if np.isfinite(zn):
                        nf, _ = near_field_from_raster(site.lat, site.lon, d, az, rng, zn + 2.0)
                        el = np.maximum(el, nf)
                        src += f" + {d.scale_m:.0f} m near field ({key})"
                        break
            return (az, el), src
        from .horizon_build import ray_horizon_global
        from .rasters import global_dem

        gd = global_dem()
        if gd is not None:
            h = ray_horizon_global(site.lat, site.lon, gd)
            return (h.azimuth_deg.to_numpy(), h.horizon_elev_deg.to_numpy()), \
                "LOLA LDEM_16 global 1.9 km ray-cast, 300 km (non-polar; near field < 2 km unresolved)"
    except Exception:
        pass
    return None, "NONE - flat 0 deg horizon (no DEM coverage) - results optimistic"


@lru_cache(maxsize=1)
def terrain_metrics() -> pd.DataFrame:
    p = PROC / "terrain_metrics_all_sites.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def site_products() -> pd.DataFrame:
    p = PROC / "site_product_statistics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def legacy_model3() -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in (ROOT / "data" / "model3_site_comparison.csv", ROOT / "data" / "model3_research_comparison.csv") if p.exists()]
    return pd.concat(parts).drop_duplicates("site_id") if parts else pd.DataFrame()


def product_value(site_id: str, key: str, col: str = "site_value"):
    sp = site_products()
    if sp.empty:
        return np.nan
    r = sp[(sp.site_id == site_id) & (sp["product"] == key)]
    if not len(r) or not bool(r["covered"].iloc[0]) or col not in r.columns:
        return np.nan
    return float(r[col].iloc[0])


# ------------------------------------------------------------------ Model 3
def live_terrain_metrics(site: Site, radius_m: float = 1000.0) -> pd.DataFrame:
    """Terrain metrics for ANY point from every covering DEM, each at its own native resolution."""
    from .horizon_build import terrain_window
    from .rasters import product_registry

    reg = product_registry()
    rows = []
    for key, prod in (("NAC_DTM_SHACKRDGE02", "NAC_DTM_SHACKRDGE02"), ("SITE01_5M", "Site01_5MPP"), ("LDEM_5M_87S", "LDEM_87S_5M"),
                      ("LDEM_20M", "LDEM_80S_20M"), ("LOLA_DEM_80M", "LDEM_80S_80M")):
        d = reg.get(key)
        if d is None or not d.covers(site.lat, site.lon):
            continue
        w = terrain_window(d, site.lat, site.lon, radius_m)
        h = w["height"].astype(float)
        if not np.isfinite(h).any():
            continue
        gy, gx = np.gradient(h, w["mpp"])
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        yy, xx = np.meshgrid(w["y_m"], w["x_m"], indexing="ij")
        circ = np.hypot(xx, yy) <= radius_m
        m = circ & np.isfinite(sl)
        if m.sum() < 10:
            continue
        v, hv = sl[m], h[m]
        row = {"site_id": site.id, "product": prod, "resolution_m": d.scale_m, "radius_m": radius_m,
               "valid_pct": 100.0 * m.sum() / circ.sum(), "n_cells": int(m.sum()),
               "elev_site_m": float(d.sample_latlon(site.lat, site.lon)), "elev_mean_m": float(hv.mean()),
               "elev_std_m": float(hv.std()), "relief_m": float(hv.max() - hv.min()),
               "slope_mean_deg": float(v.mean()), "slope_median_deg": float(np.median(v)),
               "slope_p90_deg": float(np.percentile(v, 90)), "slope_max_deg": float(v.max())}
        for t in (5, 10, 15, 20):
            row[f"pct_slope_le_{t}deg"] = float(100 * (v <= t).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def terrain_assessment(site: Site, mask=None) -> dict:
    """Evidence-based terrain suitability with hazards and uncertainty, resolution-aware (cached per location)."""
    import copy

    key = (site.id, round(site.lat, 6), round(site.lon, 6), None if mask is None else float(np.asarray(mask[1]).sum()))
    if key not in _TERRAIN_CACHE:
        _TERRAIN_CACHE[key] = _terrain_assessment(site, mask)
    return copy.deepcopy(_TERRAIN_CACHE[key])


_TERRAIN_CACHE: dict = {}


def _terrain_assessment(site: Site, mask=None) -> dict:
    tm = terrain_metrics()
    out = {"site_id": site.id, "hazards": [], "uncertainty": []}
    live = live_terrain_metrics(site)
    if len(live):
        tm = pd.concat([tm[tm.site_id != site.id] if not tm.empty else tm, live, tm[(tm.site_id == site.id) & (tm["product"] == "LDEM_80S_80M") & (tm.radius_m == 2000.0)] if not tm.empty else tm])
        out["all_resolutions"] = live
    if tm.empty or site.id not in set(tm.site_id):
        out["status"] = ("NOT ASSESSED - only the 1.9 km global DEM covers this point (too coarse for landing slopes)"
                         if abs(site.lat) < 80 else "NO TERRAIN DATA COVERS THIS POINT - cannot assess")
        out["uncertainty"].append("landing-scale terrain requires a polar LOLA product or a NAC DTM at this location")
        return out
    c = tm[(tm.site_id == site.id) & (tm["product"] == "LDEM_80S_80M") & (tm.radius_m == 1000.0)].iloc[0]
    out["common_80m"] = c.to_dict()
    fine = tm[(tm.site_id == site.id) & (tm["product"] != "LDEM_80S_80M") & (tm.radius_m == 1000.0)].sort_values("resolution_m")
    best = c.to_dict()
    best_src = "LDEM_80S_80M (80 m)"
    if len(fine):
        best = fine.iloc[0].to_dict()
        best_src = f"{best['product']} ({best['resolution_m']:g} m)"
    lg = legacy_model3()
    if not lg.empty and site.id in set(lg.site_id):
        r = lg[lg.site_id == site.id].iloc[0]
        out["legacy_native"] = {"resolution_m": float(r.cell_size_m), "source": r.terrain_source,
                                "pct_slope_le_10deg": float(r.pct_slope_le_10deg), "mean_slope_deg": float(r.mean_slope_deg),
                                "flat_circle_le_10deg_m": float(r.largest_flat_circle_le_10deg_diameter_m),
                                "relief_m": float(r.relief_m), "roughness_p90_m": float(r.roughness_p90_m)}
        if best_src.startswith("LDEM_80S_80M"):
            best = {**best, "pct_slope_le_10deg": float(r.pct_slope_le_10deg), "slope_mean_deg": float(r.mean_slope_deg),
                    "slope_p90_deg": float(r.p90_slope_deg), "relief_m": float(r.relief_m),
                    "resolution_m": float(r.cell_size_m)}
            best_src = f"{r.terrain_source} ({r.cell_size_m:.0f} m, legacy Model 3 grid)"
    out["best_source"] = best_src
    out["best"] = best
    res = float(best.get("resolution_m", 80.0))
    le10 = float(best["pct_slope_le_10deg"])
    p90 = float(best["slope_p90_deg"])
    relief = float(best["relief_m"])
    # hazards
    if p90 > 15:
        out["hazards"].append(f"10% of the 1 km footprint exceeds {p90:.1f} deg slope")
    if le10 < 50:
        out["hazards"].append(f"only {le10:.0f}% of footprint is <=10 deg")
    if relief > 300:
        out["hazards"].append(f"local relief {relief:.0f} m within 1 km")
    psr = product_value(site.id, "LOLA_PSR_60M", "r5km_mean")
    if np.isfinite(psr) and psr > 0.05:
        out["hazards"].append(f"{100*psr:.0f}% of 5 km radius is permanently shadowed (thermal/trafficability)")
    if mask is not None:
        hz = np.asarray(mask[1])
        out["horizon_mean_deg"] = float(hz.mean())
        out["horizon_pct_above_2deg"] = float(100 * (hz > 2.0).mean())
        if (hz > 2.0).mean() > 0.25:
            out["hazards"].append(f"terrain horizon >2 deg over {100*(hz>2).mean():.0f}% of azimuths (Sun/Earth blockage)")
    # uncertainty
    if res >= 20:
        out["uncertainty"].append(f"slopes at {res:.0f} m baseline underestimate lander-scale (<5 m) slopes; boulders unresolved")
    if res >= 80:
        out["uncertainty"].append("80 m LOLA grid interpolates between tracks - candidate needs NAC DTM before certification")
    if float(c["valid_pct"]) < 99:
        out["uncertainty"].append(f"valid coverage {c['valid_pct']:.0f}%")
    # provisional score (0-100); weights are a documented planning heuristic
    score = 0.5 * le10 + 0.3 * max(0.0, 100 - 4 * max(p90 - 5, 0)) + 0.2 * max(0.0, 100 - relief / 5)
    out["terrain_score"] = round(float(score), 1)
    out["resolution_m"] = res
    out["status"] = ("SUITABLE (screening)" if score >= 70 and not out["hazards"] else
                     "CONDITIONAL - hazards present" if score >= 45 else "POOR - high slope / relief")
    return out


# ------------------------------------------------------------------ chain
@dataclass
class MissionResult:
    site: Site
    timeline: pd.DataFrame
    step_min: float
    horizon_source: str
    mask: Optional[tuple]
    solar: dict
    dte: dict
    rf: dict
    generation: dict
    survival: dict
    subsystem_summary: pd.DataFrame
    terrain: dict
    scenario: str
    array: pg.SolarArrayConfig
    battery: en.BatteryConfig
    solar_windows: pd.DataFrame = field(default_factory=pd.DataFrame)
    dte_windows: pd.DataFrame = field(default_factory=pd.DataFrame)
    link: dict = field(default_factory=dict)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    sessions: pd.DataFrame = field(default_factory=pd.DataFrame)
    traverse: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: dict = field(default_factory=dict)
    electrical: dict = field(default_factory=dict)
    plan: object = None

    def summary_row(self) -> dict:
        return {
            "site_id": self.site.id, "site_name": self.site.name, "lat": self.site.lat, "lon": self.site.lon,
            "elev_m": self.site.elev_m, "scenario": self.scenario,
            "sunlit_pct": self.solar["sunlit_pct"], "mean_sun_disc_pct": self.solar.get("mean_disc_fraction_pct"),
            "longest_dark_h": self.solar["longest_darkness_h"], "longest_sun_h": self.solar["longest_sunlight_h"],
            "earth_visible_pct": self.dte["earth_visible_pct"], "longest_dte_blackout_h": self.dte["longest_dte_blackout_h"],
            "longest_dte_window_h": self.dte["longest_dte_h"],
            "link_closed_pct": self.link.get("link_closed_pct"), "data_volume_gbit": self.link.get("data_volume_gbit"),
            "dsn_handoffs": self.link.get("dsn_handoffs"),
            "sun_and_earth_pct": float(100 * (self.timeline.sunlit & self.timeline.earth_visible).mean()),
            "gen_total_kwh": self.generation["total_gen_kwh"], "gen_mean_w": self.generation["mean_gen_w"],
            "gen_peak_w": self.generation["peak_gen_w"], "longest_zero_gen_h": self.generation["longest_zero_gen_h"],
            "demand_total_kwh": self.survival["total_demand_kwh"], "min_soc_pct": self.survival["min_soc_pct"],
            "energy_deficit_kwh": self.survival["energy_deficit_kwh"],
            "ops_supported_pct": self.survival["operational_time_supported_pct"],
            "hours_to_critical_soc": self.survival["hours_to_critical_soc"],
            "survival_status": self.survival["status"],
            "terrain_score": self.terrain.get("terrain_score"), "terrain_status": self.terrain.get("status"),
            "terrain_resolution_m": self.terrain.get("resolution_m"),
            "lola_avg_sun_vis_pct": 100 * product_value(self.site.id, "LOLA_AVG_SUN_VIS_60M"),
            "lola_avg_earth_vis_pct": 100 * product_value(self.site.id, "LOLA_AVG_EARTH_VIS_60M"),
            "wac_illum_pct": product_value(self.site.id, "WAC_ILLUM_PCT_100M"),
            "horizon_source": self.horizon_source,
        }


def _solar_report(df, step):
    rep = solar_mod.solar_report(df, step)
    rep["mean_disc_fraction_pct"] = float(100 * df["sun_disc_fraction"].mean())
    rep["partial_pct"] = float(100 * ((df.sun_disc_fraction > 0) & (df.sun_disc_fraction < 0.999)).mean())
    runs = solar_mod.solar_windows(df, step)
    rises = runs[runs.state == "sunlit"]["start_utc"]
    sets = runs[runs.state == "sunlit"]["end_utc"]
    rep["n_sunrises"] = int(len(rises))
    rep["first_sunrise_utc"] = str(rises.iloc[1]) if len(rises) > 1 else (str(rises.iloc[0]) if len(rises) and not df.sunlit.iloc[0] else "")
    rep["first_sunset_utc"] = str(sets.iloc[0]) if len(sets) and df.sunlit.iloc[0] else ""
    rep["sunlit_hours"] = float(df.sunlit.sum() * step / 60)
    rep["dark_hours"] = float((~df.sunlit).sum() * step / 60)
    return rep


def run_mission(
    site: Site,
    start_iso: str,
    hours: float,
    step_min: float = 30.0,
    *,
    array: Optional[pg.SolarArrayConfig] = None,
    subsystems: Optional[list] = None,
    battery: Optional[en.BatteryConfig] = None,
    thermal: Optional[en.ThermalConfig] = None,
    scenario: str = "nominal",
    earth_rule: str = "any_part",
    geometry: Optional[pd.DataFrame] = None,
    link: Optional[rf_mod.LinkConfig] = None,
    link_budget: Optional[rf_link.LinkBudgetConfig] = None,
    activities: Optional[act_mod.ActivityPlan] = None,
    cell: Optional[pve.CellConfig] = None,
) -> MissionResult:
    array = array or pg.SolarArrayConfig()
    subsystems = subsystems if subsystems is not None else en.default_subsystems()
    battery = battery or en.BatteryConfig()
    thermal = thermal or en.ThermalConfig()
    mask, hsrc = horizon_mask(site)

    g = geometry if geometry is not None else compute_geometry(site, start_iso, hours, step_min)
    # Model 1 - illumination against terrain horizon (centre-of-disc flag + disc fraction)
    g = solar_mod.apply_solar_visibility(g, mask, sun_disc_radius_deg=0.0)
    # Model 2 - Earth visibility against terrain horizon
    g = dte_mod.apply_dte_visibility(g, mask, earth_rule)
    g = rf_mod.apply_rf_quality(g, link or rf_mod.LinkConfig())
    solar = _solar_report(g, step_min)
    dte = dte_mod.dte_report(g, step_min)
    rf = rf_mod.rf_report(g, step_min, link or rf_mod.LinkConfig())
    lb = link_budget or rf_link.LinkBudgetConfig()
    g = rf_link.apply_link_budget(g, lb, step_min)
    linkrep = rf_link.link_report(g, lb, step_min)
    # Scenario perturbations (applied after the geometric truth is recorded)
    gs, subs, over = en.apply_scenario(scenario, g, subsystems, step_min)
    arr = pg.SolarArrayConfig(**{**array.to_dict(), **over})
    # Model 4 - generation
    gs = pg.apply_generation(gs, arr, step_min)
    gen = pg.generation_report(gs, arr, step_min)
    # Model 5 - demand / battery / survival
    plan = activities if activities is not None else act_mod.ActivityPlan()
    extra, sessions, trav = act_mod.build_activity_loads(gs, plan, step_min)
    tl, summ = en.simulate(gs, subs, battery, thermal, step_min, extra_loads=extra)
    cell = cell or pve.CellConfig()
    tl = pve.apply_electrical(tl, arr, cell)
    surv = en.survival_report(tl, summ, battery, step_min)
    surv["deficit_vs_generation_kwh"] = float(summ["demand_kwh"].sum() - gen["total_gen_kwh"])
    terrain = terrain_assessment(site, mask)
    events = ev_mod.build_events(tl, max(battery.min_safe_soc, 1 - battery.max_dod))
    return MissionResult(
        site=site, timeline=tl, step_min=step_min, horizon_source=hsrc, mask=mask, solar=solar, dte=dte, rf=rf,
        generation=gen, survival=surv, subsystem_summary=summ, terrain=terrain, scenario=scenario,
        array=arr, battery=battery,
        solar_windows=solar_mod.solar_windows(g, step_min), dte_windows=dte_mod.dte_windows(g, step_min),
        link=linkrep, events=events, sessions=sessions, traverse=trav,
        stability=act_mod.outpost_stability(tl, battery, step_min), electrical=pve.electrical_report(tl, cell), plan=plan,
    )


def feasibility_score(row: dict) -> float:
    """Transparent multi-criteria index (0-100) for ranking; weights documented in reports/."""
    def c(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0, 1)) if x is not None and np.isfinite(x) else 0.0
    s = (
        0.25 * c(row["sunlit_pct"], 20, 90)
        + 0.15 * (1 - c(row["longest_dark_h"], 24, 400))
        + 0.20 * c(row["earth_visible_pct"], 20, 100)
        + 0.10 * (1 - c(row["longest_dte_blackout_h"], 24, 400))
        + 0.15 * c(row["ops_supported_pct"], 50, 100)
        + 0.15 * c(row.get("terrain_score") or 0, 30, 90)
    )
    return round(100 * s, 1)
