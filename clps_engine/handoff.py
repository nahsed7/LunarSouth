"""Cross-team handoff, scenario diffing, small-change sensitivity, and ray-traced terrain illumination.

  * diff_metrics / diff_table   - two runs (dates, sites or scenarios) side by side with deltas
  * PROFILE_PRESETS             - named mission profiles (sidebar settings + load table) for one-click setup / exchange
  * handoff_markdown            - a self-contained handoff note: configuration, results, diff, trade-off notes
  * illumination_map            - fraction of the mission each scene pixel is lit (DEM cast shadows + far-field horizon)
  * psr_boundary_km             - distance from the site to the nearest LOLA permanently-shadowed pixel
  * grazing_stats               - how close to the horizon the Sun is while the site counts as lit
  * libration_stats             - Earth elevation / azimuth swing over the window (libration made visible)
"""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import events as ev_mod

# --------------------------------------------------------------------------- diff
METRICS = [  # key, label, unit, better ("hi" | "lo")
    ("sunlit_pct", "Sunlit (terrain)", "%", "hi"),
    ("longest_dark_h", "Longest darkness", "h", "lo"),
    ("earth_visible_pct", "Earth visible", "%", "hi"),
    ("longest_dte_blackout_h", "Longest Earth blackout", "h", "lo"),
    ("link_closed_pct", "Link closed", "%", "hi"),
    ("gen_total_kwh", "Energy generated", "kWh", "hi"),
    ("min_soc_pct", "Battery floor (min SoC)", "%", "hi"),
    ("energy_deficit_kwh", "Unserved energy", "kWh", "lo"),
    ("first_sunset_h", "First sunset after start", "h", "hi"),
    ("first_earthset_h", "First Earthset after start", "h", "hi"),
]


def diff_metrics(R) -> dict:
    row = R.summary_row()
    out = {k: row.get(k) for k, *_ in METRICS if k in row}
    t0 = R.timeline.time_utc.iloc[0]
    cd = ev_mod.countdowns(R.events, t0).set_index("event") if len(R.events) else pd.DataFrame()
    for key, ev in (("first_sunset_h", "SUNSET"), ("first_earthset_h", "EARTHSET")):
        out[key] = float(cd.loc[ev, "t_minus_h"]) if ev in getattr(cd, "index", []) else np.nan
    out["status"] = R.survival["status"].split(" - ")[0]
    return out


def diff_table(a: dict, b: dict, label_a="A", label_b="B") -> pd.DataFrame:
    rows = []
    for k, lab, unit, better in METRICS:
        va, vb = a.get(k), b.get(k)
        va = np.nan if va is None else float(va)
        vb = np.nan if vb is None else float(vb)
        d = vb - va
        if not np.isfinite(d) or abs(d) < 1e-9:
            verdict = "same" if np.isfinite(d) else "n/a"
        else:
            verdict = f"{label_b} better" if (d > 0) == (better == "hi") else f"{label_a} better"
        rows.append({"metric": f"{lab} ({unit})", label_a: va, label_b: vb, "Δ (B−A)": d, "verdict": verdict})
    rows.append({"metric": "Energy verdict", label_a: np.nan, label_b: np.nan, "Δ (B−A)": np.nan,
                 "verdict": f"{a.get('status')} → {b.get('status')}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- profiles
PROFILE_PRESETS = {
    "Small CLPS lander (default)": {
        "settings": {"orient": "vertical_tracking", "area": 1.2, "eff": 0.295, "cap": 4000.0, "soc0": 0.9, "dod": 0.8, "safe": 0.3,
                     "pmax": 1000.0, "mli": 0.02, "tint": -20, "scenario": "nominal", "ptx": 10.0, "gtx": 20.0, "relay": False},
        "note": "Representative small CLPS payload lander: ~1.2 m² vertical array, 4 kWh battery, X-band DTE.",
    },
    "Long-duration outpost (survive the night)": {
        "settings": {"orient": "vertical_tracking", "area": 6.0, "eff": 0.30, "cap": 40000.0, "soc0": 1.0, "dod": 0.85, "safe": 0.25,
                     "pmax": 4000.0, "mli": 0.015, "tint": -25, "scenario": "nominal", "ptx": 20.0, "gtx": 24.0, "relay": True,
                     "len_mode": "Days", "days": 365, "step": 60},
        "note": "Sized for multi-lunation operations: large array + 40 kWh storage, relay enabled, one-year window.",
    },
    "Hibernation / survival-first": {
        "settings": {"orient": "vertical_fixed", "area": 1.0, "eff": 0.28, "cap": 6000.0, "soc0": 1.0, "dod": 0.9, "safe": 0.2,
                     "pmax": 800.0, "mli": 0.01, "tint": -35, "scenario": "survival", "ptx": 5.0, "gtx": 15.0, "relay": False},
        "note": "Minimal loads, low heater set point, survival scenario - tests whether the platform outlasts a polar night.",
    },
    "High-rate science + video": {
        "settings": {"orient": "dual_axis", "area": 2.5, "eff": 0.32, "cap": 8000.0, "soc0": 0.95, "dod": 0.8, "safe": 0.35,
                     "pmax": 2000.0, "mli": 0.02, "tint": -20, "scenario": "high_science", "ptx": 40.0, "gtx": 30.0, "band": "Ka (32 GHz)",
                     "rmax": 1e7},
        "note": "Power-hungry science and Ka-band video: shows where generation and link margin become the bottleneck.",
    },
}


def _jsonable(v):
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def profile_from_state(state, keys, name: str, notes: str = "") -> dict:
    settings = {k: _jsonable(state[k]) for k in keys if k in state}
    prof = {"format": "lunarsouth-mission-profile/1", "name": name, "notes": notes, "settings": settings}
    if "subs" in state:
        prof["subs"] = json.loads(pd.DataFrame(state["subs"]).to_json(orient="records"))
    if "plan" in state:
        prof["plan"] = state["plan"]
    if state.get("equip_qty"):
        prof["equipment"] = {"qty": dict(state["equip_qty"]), "sources": dict(state.get("equip_src", {}))}
    if state.get("basket"):
        prof["comparison_points"] = list(state["basket"])
    return prof


def validate_profile(d: dict) -> dict:
    if not isinstance(d, dict) or "settings" not in d or not isinstance(d["settings"], dict):
        raise ValueError("not a LunarSouth mission profile (missing 'settings')")
    return d


# --------------------------------------------------------------------------- handoff note
def handoff_markdown(site, start_iso, days, settings: dict, R, diff: pd.DataFrame | None, notes: str, sens: pd.DataFrame | None = None) -> str:
    m = diff_metrics(R)
    lines = [f"# Mission handoff - {site.name} ({site.id})", "",
             f"- Location: {site.lat:.5f}°, {site.lon:.5f}°E, {site.elev_m:,.0f} m",
             f"- Window: {start_iso[:16]} UTC + {days} d · scenario `{R.scenario}`",
             f"- Generated: {datetime.utcnow():%Y-%m-%d %H:%M} UTC by the LunarSouth CLPS Mission Planner", "",
             "## Key results", "", "| metric | value |", "|---|---|"]
    for k, lab, unit, _ in METRICS:
        v = m.get(k)
        lines.append(f"| {lab} | {'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:,.1f} {unit}'} |")
    lines += [f"| Energy verdict | {m['status']} |", ""]
    if diff is not None and len(diff):
        lines += ["## Scenario diff", "", diff.round(2).to_markdown(index=False), ""]
    if sens is not None and len(sens):
        lines += ["## Small-change sensitivity", "", sens.round(2).to_markdown(index=False), ""]
    lines += ["## Trade-off notes", "", notes.strip() or "_none recorded_", "", "## Configuration", "", "```json",
              json.dumps({k: _jsonable(v) for k, v in settings.items()}, indent=1), "```", "",
              "All subsystem powers, battery and RF parameters are engineering assumptions unless stated otherwise."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- terrain illumination physics
def _cast_lit(H, x0, y0, mpp, Xo, Yo, Ho, alt, az, obs_h, half):
    """Sun-centre visibility at observer points (Xo, Yo, ground Ho) against the full-resolution scene DEM H."""
    from scipy import ndimage
    from .rasters import MOON_R_M

    a = np.radians(az)
    sx, sy = np.sin(a), np.cos(a)
    alt_px = alt + np.degrees((Xo * sx + Yo * sy) / MOON_R_M)
    tan_e = np.tan(np.radians(alt_px))
    h0 = Ho + obs_h
    lit = np.ones(Xo.shape, bool)
    hmax = float(np.nanmax(H))
    d = mpp
    while d < 3 * half:
        px, py = Xo + sx * d, Yo + sy * d
        inside = (np.abs(px) <= half) & (np.abs(py) <= half)
        if not inside.any():
            break
        need = h0 + d * tan_e + d * d / (2 * MOON_R_M)
        act = inside & lit & (need < hmax + 50)
        if act.any():
            ht = ndimage.map_coordinates(H, [np.clip((y0 - py[act]) / mpp, 0, H.shape[0] - 1),
                                             np.clip((px[act] - x0) / mpp, 0, H.shape[1] - 1)], order=1)
            blk = ht > need[act]
            idx = np.flatnonzero(act.ravel())[blk]
            lit.ravel()[idx] = False
        d += mpp if d < 2000 else 2 * mpp if d < 6000 else 4 * mpp
    return lit, alt_px


def illumination_map(scene: dict, tl: pd.DataFrame, far_hz: np.ndarray, n_samples: int = 48, stride: int = 2,
                     obs_h: float = 2.0) -> dict:
    """Fraction of mission time each pixel (on a stride-decimated grid) sees the Sun's centre.

    Near field: rays marched over the full-resolution 80 m scene DEM from 2 m above each pixel (same observer height as the
    Model 1 horizon); far field: the 12-200 km LOLA mask; local-vertical tilt d/R per pixel. Validated against Model 1 at the
    site pixel in tests/test_mission_sim.py.
    """
    H = np.where(np.isfinite(scene["height_m"]), scene["height_m"], np.nanmin(scene["height_m"])).astype(float)
    mpp = float(abs(scene["x_m"][1] - scene["x_m"][0]))
    x0, y0, half = float(scene["x_m"][0]), float(scene["y_m"][0]), float(abs(scene["x_m"][0]))
    x = scene["x_m"][::stride]
    y = scene["y_m"][::stride]
    X, Y = np.meshgrid(x, y)
    Ho = H[::stride, ::stride]
    idx = np.unique(np.linspace(0, len(tl) - 1, n_samples).round().astype(int))
    lit_sum = np.zeros(X.shape)
    az_grid = np.arange(0.0, 360.0, 360.0 / len(far_hz))
    for i in idx:
        alt, az = float(tl.sun_alt.iloc[i]), float(tl.sun_az.iloc[i])
        far = float(np.interp(az, az_grid, far_hz, period=360.0))
        if alt + 0.4 < far:                      # below the far horizon everywhere in the window (|d/R| < 0.4 deg)
            continue
        lit, alt_px = _cast_lit(H, x0, y0, mpp, X, Y, Ho, alt, az, obs_h, half)
        lit_sum += lit & (alt_px > far)
    frac = lit_sum / max(len(idx), 1)
    c = frac.shape[0] // 2
    r = np.hypot(X, Y)
    out = {"frac": frac, "x_m": x, "y_m": y, "site_frac": float(frac[c, c]), "n_samples": int(len(idx))}
    for rad in (500.0, 1000.0, 3000.0):
        m = r <= rad
        k = np.argmax(np.where(m, frac, -1))
        out[f"best_{int(rad)}"] = {"frac": float(frac.flat[k]), "x": float(X.flat[k]), "y": float(Y.flat[k]), "dist_m": float(r.flat[k])}
    return out


def psr_boundary_km(scene: dict) -> float:
    psr = scene.get("lola_psr_60m")
    if psr is None or not np.isfinite(psr).any():
        return float("nan")
    X, Y = np.meshgrid(scene["x_m"], scene["y_m"])
    m = psr >= 0.5
    return float(np.hypot(X[m], Y[m]).min() / 1000.0) if m.any() else float("inf")


def grazing_stats(tl: pd.DataFrame) -> dict:
    lit = tl.sunlit.to_numpy(bool)
    if not lit.any():
        return {"lit_steps": 0}
    m = tl.loc[lit, "solar_elevation_above_horizon_deg"].to_numpy(float)
    return {"lit_steps": int(lit.sum()), "median_margin_deg": float(np.median(m)), "p10_margin_deg": float(np.percentile(m, 10)),
            "grazing_pct": float(100 * (m < 0.5).mean()), "partial_disc_pct": float(100 * (tl.loc[lit, "sun_disc_fraction"] < 0.999).mean()),
            "max_sun_alt_deg": float(tl.sun_alt.max())}


def libration_stats(tl: pd.DataFrame) -> dict:
    ea, ez = tl.earth_alt.to_numpy(float), tl.earth_az.to_numpy(float)
    zr = np.degrees(np.angle(np.exp(1j * np.radians(ez - np.degrees(np.angle(np.mean(np.exp(1j * np.radians(ez)))))))))
    return {"earth_alt_min": float(ea.min()), "earth_alt_max": float(ea.max()), "earth_alt_swing": float(ea.max() - ea.min()),
            "earth_az_swing": float(zr.max() - zr.min()),
            "near_horizon_pct": float(100 * (np.abs(tl.visibility_margin_deg.to_numpy(float)) < 1.0).mean()) if "visibility_margin_deg" in tl else np.nan}
