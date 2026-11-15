"""Model 5 Mission Sim - payload builder for the playable rover + outpost simulator (static/mission_sim.html).

Everything the browser game needs is packed from the SAME shared timeline the other models use:

  * Sun / Earth az-el, irradiance, cell efficiency           <- Model 1 / Model 4 (SPICE geometry)
  * outpost generation gen_w                                  <- Model 4 (full PV chain)
  * DSN coverage + link-rate potential (C/N0 link budget)     <- Model 2 link budget
  * 24 x 24 km LOLA 80 m DEM, slope, PSR, LOLA Sun/Earth vis  <- Model 3 scene grid
  * far-field horizon (12-200 km) ray-cast from the site      <- LOLA 80 m + 240 m DEMs
  * outpost battery / thermal / subsystem table               <- Model 5 configuration (sidebar + load table)

The rover's local illumination is NOT taken from the site: the browser ray-marches the scene DEM from the
rover's own position (near field) and combines it with the far-field mask, exactly as cell_horizon() below
does in Python (the reference used by the tests).
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from .rasters import MOON_R_M

AZ_STEP = 1.0          # deg, horizon profile resolution used by the game
OBS_H = 1.5            # rover mast / panel height above terrain (m)
SCENE_HALF = 12000.0   # scene half width (m) - terrain beyond this comes from the far-field mask


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _r(a, nd=3):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


@lru_cache(maxsize=64)
def _far_field_cached(lat: float, lon: float, elev: float, half: float):
    from .horizon_build import ray_horizon
    from .rasters import product_registry

    reg = product_registry()
    dem = reg.get("LOLA_DEM_80M")
    if dem is None or not dem.covers(lat, lon):
        return None
    h = ray_horizon(lat, lon, dem, reg.get("LOLA_DEM_240M"), az_step_deg=AZ_STEP, near_min_m=half,
                    site_elev_m=elev if np.isfinite(elev) else None, observer_height_m=OBS_H)
    return tuple(h.horizon_elev_deg.round(4))


def far_field_mask(site, scene: dict, full_mask=None) -> tuple[np.ndarray, str]:
    """Horizon from terrain beyond the scene window only. Falls back to the full site mask (conservative)."""
    h = scene["height_m"]
    c = h.shape[0] // 2
    elev = float(h[c, c]) if np.isfinite(h[c, c]) else float("nan")
    half = float(max(abs(scene["x_m"][0]), abs(scene["x_m"][-1])))
    try:
        ff = _far_field_cached(round(site.lat, 6), round(site.lon, 6), round(elev, 2), half)
    except Exception:  # noqa: BLE001 - missing rasters on a slim deployment
        ff = None
    az = np.arange(0.0, 360.0, AZ_STEP)
    if ff is not None:
        return np.asarray(ff, float), f"LOLA 80/240 m ray-cast {half/1000:.0f}-200 km (far field)"
    if full_mask is not None:
        return np.interp(az, np.asarray(full_mask[0]), np.asarray(full_mask[1]), period=360.0), "site 200 km mask (fallback, conservative)"
    return np.zeros(az.size), "flat (no DEM)"


def encode_scale(h: np.ndarray) -> tuple[float, float]:
    """int16 height encoding: offset + the finest scale (0.1 m upward) that fits the scene's relief."""
    lo, hi = float(np.nanmin(h)), float(np.nanmax(h))
    off = round((lo + hi) / 2.0, 1)
    for sc in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        if max(hi - off, off - lo) / sc < 32000:
            return off, sc
    return off, 4.0


def _grid_sampler(scene):
    h = np.where(np.isfinite(scene["height_m"]), scene["height_m"], np.nanmin(scene["height_m"])).astype(float)
    x0 = float(scene["x_m"][0])
    y0 = float(scene["y_m"][0])          # row 0 = north edge (y decreasing down the rows)
    mpp = float(abs(scene["x_m"][1] - scene["x_m"][0]))

    def sample(x, y):
        c = np.clip((np.asarray(x) - x0) / mpp, 0, h.shape[1] - 1.001)
        r = np.clip((y0 - np.asarray(y)) / mpp, 0, h.shape[0] - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        return (h[r0, c0] * (1 - fc) * (1 - fr) + h[r0, c0 + 1] * fc * (1 - fr)
                + h[r0 + 1, c0] * (1 - fc) * fr + h[r0 + 1, c0 + 1] * fc * fr)

    return sample, mpp, x0, y0, h


def cell_horizon(scene: dict, x: float, y: float, far: np.ndarray, obs_h: float = OBS_H) -> np.ndarray:
    """Reference (Python) for the game's per-position horizon: max(near-field DEM ray-march, far-field mask).

    Same spherical-curvature elevation as horizon_build: el = atan2(h_t - h_o - d^2/2R, d).
    """
    sample, mpp, x0, y0, _ = _grid_sampler(scene)
    h_o = float(sample(x, y)) + obs_h
    az = np.radians(np.arange(0.0, 360.0, AZ_STEP))[:, None]
    d = np.arange(mpp, 2 * SCENE_HALF * 1.5, mpp)[None, :]
    xs, ys = x + np.sin(az) * d, y + np.cos(az) * d
    inside = (np.abs(xs) <= abs(x0)) & (np.abs(ys) <= abs(y0))
    ht = sample(xs, ys)
    el = np.degrees(np.arctan2(ht - h_o - d * d / (2 * MOON_R_M), d))
    el = np.where(inside, el, -90.0).max(axis=1)
    return np.maximum(el, far)


def local_alt(alt_deg, az_deg, x, y):
    """Body elevation at an offset (x east, y north metres) from the site: the local vertical tilts by d/R."""
    a = np.radians(az_deg)
    return np.asarray(alt_deg) + np.degrees((x * np.sin(a) + y * np.cos(a)) / MOON_R_M)


def points_of_interest(scene: dict, max_r_m: float = 9000.0) -> list[dict]:
    """Science targets picked from the real co-registered layers (deterministic)."""
    x, y = np.meshgrid(scene["x_m"], scene["y_m"])
    r = np.hypot(x, y)
    slope = scene["slope_deg"]
    out = []

    def pick(score, mask, kind, label, value_mb, n, sep=2500.0):
        sc = np.where(mask & np.isfinite(score), score, -np.inf)
        flat = np.argsort(sc, axis=None)[::-1]
        got = []
        for k in flat[:20000]:
            if not np.isfinite(sc.flat[k]) or len(got) >= n:
                break
            px, py = float(x.flat[k]), float(y.flat[k])
            if all(np.hypot(px - q["x"], py - q["y"]) > sep for q in out + got):
                got.append({"x": round(px), "y": round(py), "kind": kind, "label": label, "value_mb": value_mb,
                            "slope": round(float(slope.flat[k]), 1), "dist_km": round(float(r.flat[k]) / 1000, 2)})
        out.extend(got)

    base = (r > 1200) & (r < max_r_m) & (slope < 14)
    psr = scene.get("lola_psr_60m")
    if psr is not None and np.isfinite(psr).any():
        pick(-r, base & (psr >= 0.5), "psr", "Cold trap (PSR) - volatile sampling", 600, 3)
    sv = scene.get("lola_avg_sun_vis_60m")
    if sv is not None and np.isfinite(sv).any():
        pick(sv, base & (r > 2500), "ridge", "Sunlit ridge - long-duration survey", 250, 2)
    ev = scene.get("lola_avg_earth_vis_60m")
    if ev is not None and np.isfinite(ev).any():
        pick(ev, base & (r > 2500), "earth", "Earth-view spot - direct-to-Earth uplink", 150, 1)
    pick(slope, (r > 1500) & (r < max_r_m) & (slope > 14) & (slope < 22), "outcrop", "Crater-wall outcrop - imaging", 300, 2)
    for i, p in enumerate(out):
        p["id"] = f"T{i + 1}"
    return out


def rate_potential(tl: pd.DataFrame, lbc) -> np.ndarray:
    """DTE rate a transmitter with Earth in view WOULD get (C/N0 budget + DSN coverage), independent of site terrain."""
    cn0 = tl["cn0_dbhz"].to_numpy(float)
    rate = np.minimum(10 ** ((cn0 - lbc.required_ebn0_db - lbc.required_margin_db) / 10.0), lbc.max_rate_bps)
    dsn = np.zeros(len(tl), bool)
    for k in ("Goldstone", "Madrid", "Canberra"):
        c = f"dsn_el_{k}"
        if c in tl:
            dsn |= tl[c].to_numpy(float) > lbc.gs_elev_mask_deg
    return np.where(dsn, rate, 0.0), dsn


def _dsn_station(tl, lbc):
    names = np.array(["none", "Goldstone", "Madrid", "Canberra"])
    el = np.vstack([tl.get(f"dsn_el_{k}", pd.Series(np.full(len(tl), -90.0))).to_numpy(float) for k in names[1:]])
    best = np.argmax(el, axis=0)
    return np.where(el.max(axis=0) > lbc.gs_elev_mask_deg, best + 1, 0).astype(int).tolist(), names.tolist()


def plan_series(R) -> dict:
    """The batch M5 run (planned schedule) as category-stacked series for the dashboard."""
    tl = R.timeline
    cats = dict(zip(R.subsystem_summary.subsystem, R.subsystem_summary.category))
    by = {}
    for c in tl.columns:
        if c.startswith("load::"):
            k = cats.get(c[6:], "other")
            by[k] = by.get(k, 0) + tl[c].to_numpy(float)
    area = float(R.array.area_m2)
    return {
        "harvest_w": _r(tl["panel_irradiance_w_m2"].to_numpy(float) * area * (tl["sun_disc_fraction"].to_numpy(float)), 1),
        "gen_w": _r(tl["gen_w"], 1),
        "use_w": {k: _r(v, 1) for k, v in sorted(by.items())},
        "demand_w": _r(tl["load_demand_w"], 1),
        "deficit_w": _r(tl["energy_deficit_w"], 1),
        "clip_w": _r(tl["gen_clipped_w"], 1),
        "soc_pct": _r(tl["soc_pct"], 2),
        "sunlit": tl["sunlit"].astype(int).tolist(),
    }


def build_payload(R, scene: Optional[dict], site, *, array, battery, thermal, lbc, subsystems, mode: str = "sim") -> dict:
    tl = R.timeline
    t0 = pd.Timestamp(tl["time_utc"].iloc[0])
    rp, dsn = rate_potential(tl, lbc)
    relay = np.zeros(len(tl))
    if lbc.relay_enabled:
        t_h = np.arange(len(tl)) * R.step_min / 60.0
        relay = np.where(np.mod(t_h, 12.0) < 12.0 * lbc.relay_availability, lbc.relay_rate_bps, 0.0)
    st_idx, st_names = _dsn_station(tl, lbc)
    P = {
        "mode": mode,
        "version": 1,
        "site": {"id": site.id, "name": site.name, "lat": site.lat, "lon": site.lon},
        "t0_ms": int(t0.value // 1_000_000),
        "step_min": float(R.step_min),
        "n": int(len(tl)),
        "s": {
            "sun_alt": _r(tl.sun_alt, 4), "sun_az": _r(tl.sun_az, 3), "sun_r": _r(tl.sun_radius, 4),
            "earth_alt": _r(tl.earth_alt, 4), "earth_az": _r(tl.earth_az, 3), "earth_r": _r(tl.earth_radius, 4),
            "irr": _r(tl.irradiance_w_m2, 0), "cell_eff": _r(tl.cell_eff, 4),
            "site_sunfrac": _r(tl.sun_disc_fraction, 3), "site_earth": tl.earth_visible.astype(int).tolist(),
            "site_hz_sun": _r(tl.sun_horizon_deg, 3),
            "out_gen": _r(tl.gen_w, 2), "out_harvest": _r(tl.panel_irradiance_w_m2 * array.area_m2 * tl.sun_disc_fraction, 1),
            "dsn": dsn.astype(int).tolist(), "rate_pot": _r(rp, 0), "relay": _r(relay, 0),
            "station": st_idx,
        },
        "stations": st_names,
        "plan": plan_series(R),
        "outpost": {
            "array": {"area_m2": array.area_m2, "orientation": array.orientation},
            "battery": battery.to_dict(),
            "thermal": {k: getattr(thermal, k) for k in thermal.__dataclass_fields__},
            "subsystems": [s.to_dict() for s in subsystems],
            "comms": {"ref_rf_w": lbc.tx_power_w, "sspa_eff": 0.35, "exciter_w": 6.0, "max_rate_bps": lbc.max_rate_bps,
                      "band": lbc.band, "gain_dbi": lbc.tx_antenna_gain_dbi},
        },
        "rover": {   # ASSUMPTIONS - editable in the game
            "mass_kg": 250.0, "crr": 0.15, "drive_eff": 0.55, "drive_avionics_w": 35.0, "speed_m_s": 0.10,
            "max_slope_deg": 25.0, "panel_m2": 1.0, "panel_eff_scale": 1.0,
            "dust": float(array.dust_factor), "age": float(array.degradation_factor), "mppt": float(array.mppt_efficiency),
            "battery_wh": 1500.0, "radiating_m2": 1.5, "mli_e": 0.03, "set_c": -20.0,
            "dte_rf_w": 2.0, "dte_gain_dbi": 6.0, "uhf_range_m": 8000.0, "uhf_rate_bps": 2.0e5, "uhf_dc_w": 15.0,
            "dock_m": 80.0, "dock_w": 150.0,
        },
    }
    if scene is not None:
        h = scene["height_m"].astype(float)
        hmin = float(np.nanmin(h))
        hh = np.where(np.isfinite(h), h, hmin)
        far, far_src = far_field_mask(site, scene, R.mask)

        def u8(k, scale):  # display layers at half resolution (slope is recomputed from the DEM in the browser)
            a = scene.get(k)
            if a is None:
                return None
            return _b64(np.clip(np.nan_to_num(np.asarray(a, float)[::2, ::2], nan=0.0) * scale, 0, 255).round().astype("uint8"))

        h_off, h_scale = encode_scale(hh)
        P["dem"] = {
            "n": int(h.shape[0]), "m": int(h.shape[1]), "mpp": float(abs(scene["x_m"][1] - scene["x_m"][0])),
            "x0": float(scene["x_m"][0]), "y0": float(scene["y_m"][0]),
            "h_i16": _b64(np.round((hh - h_off) / h_scale).astype("<i2")), "h_off": h_off, "h_scale": h_scale,
            "lay_n": int(scene["height_m"][::2, ::2].shape[0]), "lay_m": int(scene["height_m"][::2, ::2].shape[1]),
            "psr_u8": u8("lola_psr_60m", 255.0),
            "sunvis_u8": u8("lola_avg_sun_vis_60m", 255.0),
            "earthvis_u8": u8("lola_avg_earth_vis_60m", 255.0),
        }
        P["far_hz"] = _r(far, 3)
        P["far_src"] = far_src
        P["pois"] = points_of_interest(scene)
    return P


def payload_json(P: dict) -> str:
    return json.dumps(P, separators=(",", ":"), allow_nan=False)
