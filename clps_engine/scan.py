"""Best-point scan: rank every 1 km cell of the 80°S cap with a transparent, weighted percentage score.

Inputs are the packaged pole-wide products (no new downloads):
  * suitability1km.npz  sun_merged / earth_merged  (LOLA AVGVISIB products, Model 1/2 lattice beyond 85°S)
  * mosaic1km.npz       le10 (fraction of 20 m pixels <= 10°), slope_mean, psr_frac, valid_frac, elevation, lat/lon
  * sim_lattice.csv     5 km Model 1/2/4 lattice: longest_dark_h, longest_blackout_h, mean_specific_power_w_m2
Each criterion is normalised to 0-1, multiplied by a user weight and summed; the score is the weighted mean x 100 %.
Non-maximum suppression keeps the best cell in every `min_sep_km` neighbourhood so the list is not ten copies of one ridge.
The scan is a fast screening step; `refine` in the app re-runs the full M1-M5 models on the top cells.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .rasters import latlon_to_ps

PW = Path(__file__).resolve().parent.parent / "data" / "processed" / "polewide"

CRITERIA = {
    # key: (label, help)
    "sun": ("Sunlight", "Average fraction of time the Sun is visible (LOLA 18.6-yr product; Model 1 lattice beyond 85°S)"),
    "earth": ("Earth visibility", "Average fraction of time Earth is above the horizon (direct-to-Earth comms)"),
    "flat": ("Flat ground", "Fraction of 20 m pixels in the 1 km cell with slope <= 10° (safe landing / driving)"),
    "night": ("Short nights", "Longest continuous darkness from the 5 km Model 1 lattice (24 h -> 1, >= 400 h -> 0)"),
    "power": ("Solar power", "Mean specific power of a vertical sun-tracking array from the 5 km Model 4 lattice"),
    "comms": ("Short blackouts", "Longest direct-to-Earth blackout from the 5 km Model 2 lattice (0 h -> 1, >= 400 h -> 0)"),
    "ice": ("Near cold traps", "Distance to the nearest permanently shadowed cell (<= 2 km -> 1, >= 20 km -> 0) - water-ice access"),
}
DEFAULT_WEIGHTS = {"sun": 30, "earth": 20, "flat": 15, "night": 15, "power": 10, "comms": 5, "ice": 5}
PRESET_WEIGHTS = {
    "Balanced outpost": DEFAULT_WEIGHTS,
    "Power-first (solar)": {"sun": 40, "earth": 5, "flat": 10, "night": 25, "power": 20, "comms": 0, "ice": 0},
    "Comms-first (DTE)": {"sun": 15, "earth": 40, "flat": 10, "night": 5, "power": 5, "comms": 25, "ice": 0},
    "Water-ice prospecting": {"sun": 20, "earth": 10, "flat": 15, "night": 10, "power": 5, "comms": 5, "ice": 35},
    "Safe landing": {"sun": 20, "earth": 10, "flat": 50, "night": 10, "power": 5, "comms": 5, "ice": 0},
}


def _clip01(a):
    return np.clip(a, 0.0, 1.0)


@lru_cache(maxsize=1)
def grid() -> dict | None:
    pm, ps = PW / "mosaic1km.npz", PW / "suitability1km.npz"
    if not (pm.exists() and ps.exists()):
        return None
    m, s = np.load(pm), np.load(ps)
    x, y = m["x_m"].astype(float), m["y_m"].astype(float)
    X, Y = np.meshgrid(x, y)                       # rows follow y, columns follow x (verified against latlon_to_ps)
    g = {"x": X, "y": Y, "lat": m["lat"].astype(float), "lon": m["lon"].astype(float), "elev": m["elev_mean"].astype(float),
         "slope": m["slope_mean"].astype(float), "le10": m["le10"].astype(float), "psr": m["psr_frac"].astype(float),
         "valid": m["valid_frac"].astype(float), "sun": s["sun_merged"].astype(float), "earth": s["earth_merged"].astype(float)}
    # distance to the nearest PSR cell (km)
    try:
        from scipy.ndimage import distance_transform_edt
        cell_km = abs(float(x[1] - x[0])) / 1000.0
        g["psr_km"] = distance_transform_edt(~(np.nan_to_num(g["psr"]) >= 0.3)) * cell_km
    except Exception:  # noqa: BLE001
        g["psr_km"] = np.full(X.shape, np.nan)
    # 5 km lattice -> nearest lattice point for every 1 km cell
    lp = PW / "sim_lattice.csv"
    for k in ("dark_h", "blackout_h", "spow"):
        g[k] = np.full(X.shape, np.nan)
    if lp.exists():
        L = pd.read_csv(lp)
        try:
            from scipy.spatial import cKDTree
            d, idx = cKDTree(np.c_[L.x_m, L.y_m]).query(np.c_[X.ravel(), Y.ravel()], distance_upper_bound=7_500.0)
            ok = np.isfinite(d)
            for k, col in (("dark_h", "longest_dark_h"), ("blackout_h", "longest_blackout_h"), ("spow", "mean_specific_power_w_m2")):
                v = np.full(X.size, np.nan)
                v[ok] = L[col].to_numpy(float)[idx[ok]]
                g[k] = v.reshape(X.shape)
        except Exception:  # noqa: BLE001
            pass
    return g


def components(g: dict) -> dict:
    """Each criterion normalised to 0-1 (NaN where the input product has no data)."""
    return {
        "sun": _clip01(g["sun"] / 0.70),                       # 0.70 ~ best average visibility anywhere on the cap
        "earth": _clip01(g["earth"]),
        "flat": _clip01(g["le10"]),
        "night": 1.0 - _clip01((g["dark_h"] - 24.0) / (400.0 - 24.0)),
        "power": _clip01(g["spow"] / 300.0),
        "comms": 1.0 - _clip01(g["blackout_h"] / 400.0),
        "ice": 1.0 - _clip01((g["psr_km"] - 2.0) / 18.0),
    }


def scan(weights: dict | None = None, *, top_n: int = 15, min_sep_km: float = 8.0, max_slope_deg: float = 15.0,
         center: tuple | None = None, radius_km: float | None = None, lat_max: float = -80.0) -> pd.DataFrame:
    """Rank the cap. Returns one row per selected cell, best first, with the score and every component in %."""
    g = grid()
    if g is None:
        return pd.DataFrame()
    w = {k: float(v) for k, v in (weights or DEFAULT_WEIGHTS).items() if k in CRITERIA and float(v) > 0}
    if not w:
        w = dict(DEFAULT_WEIGHTS)
    comp = components(g)
    num = np.zeros(g["x"].shape)
    den = np.zeros(g["x"].shape)
    for k, wk in w.items():
        c = comp[k]
        ok = np.isfinite(c)
        num += np.where(ok, wk * c, 0.0)
        den += np.where(ok, wk, 0.0)
    score = np.where(den > 0, 100.0 * num / np.maximum(den, 1e-9), np.nan)
    keep = (np.nan_to_num(g["valid"]) >= 0.5) & np.isfinite(g["sun"]) & (np.nan_to_num(g["slope"], nan=99) <= max_slope_deg) \
        & (g["lat"] <= lat_max) & (np.nan_to_num(g["psr"]) < 0.5)
    if center is not None and radius_km:
        cx, cy = latlon_to_ps(center[0], center[1])
        keep &= np.hypot(g["x"] - cx, g["y"] - cy) <= radius_km * 1000.0
    score = np.where(keep, score, np.nan)
    flat_idx = np.argsort(np.nan_to_num(score, nan=-1).ravel())[::-1]
    xs, ys = g["x"].ravel(), g["y"].ravel()
    chosen = []
    sep = (min_sep_km * 1000.0) ** 2
    for i in flat_idx:
        if not np.isfinite(score.ravel()[i]):
            break
        if all((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 >= sep for j in chosen):
            chosen.append(i)
            if len(chosen) >= top_n:
                break
    rows = []
    for r, i in enumerate(chosen, 1):
        row = {"rank": r, "lat": round(float(g["lat"].ravel()[i]), 5), "lon": round(float(g["lon"].ravel()[i]), 5),
               "score_pct": round(float(score.ravel()[i]), 1), "elev_m": round(float(g["elev"].ravel()[i]), 0),
               "sun_vis_pct": round(100 * float(g["sun"].ravel()[i]), 1), "earth_vis_pct": round(100 * float(g["earth"].ravel()[i]), 1),
               "flat_le10_pct": round(100 * float(g["le10"].ravel()[i]), 1), "slope_deg": round(float(g["slope"].ravel()[i]), 1),
               "longest_dark_h": round(float(g["dark_h"].ravel()[i]), 0), "longest_blackout_h": round(float(g["blackout_h"].ravel()[i]), 0),
               "psr_km": round(float(g["psr_km"].ravel()[i]), 1)}
        for k in CRITERIA:
            v = comp[k].ravel()[i]
            row[f"c_{k}"] = round(100 * float(v), 1) if np.isfinite(v) else None
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["weights"] = w
    out.attrs["cells_considered"] = int(np.isfinite(score).sum())
    return out


def score_map(weights: dict | None = None, max_slope_deg: float = 15.0) -> tuple:
    """(x_km, y_km, score) at 2 km for a heat-map (every other cell)."""
    g = grid()
    if g is None:
        return None
    w = weights or DEFAULT_WEIGHTS
    comp = components(g)
    num = sum(np.where(np.isfinite(comp[k]), float(v) * comp[k], 0.0) for k, v in w.items() if k in comp)
    den = sum(np.where(np.isfinite(comp[k]), float(v), 0.0) for k, v in w.items() if k in comp)
    sc = np.where(np.asarray(den) > 0, 100.0 * num / np.maximum(den, 1e-9), np.nan)
    sc = np.where((np.nan_to_num(g["valid"]) >= 0.5) & (np.nan_to_num(g["slope"], nan=99) <= max_slope_deg) & np.isfinite(g["sun"]), sc, np.nan)
    return g["x"][0, ::2] / 1000.0, g["y"][::2, 0] / 1000.0, sc[::2, ::2]


# ------------------------------------------------------------------ full-model point scoring (for the comparison basket)
FULL_CRITERIA = {
    "sun": ("sunlit_pct", 20.0, 90.0, False),
    "night": ("longest_dark_h", 24.0, 400.0, True),
    "earth": ("earth_visible_pct", 20.0, 100.0, False),
    "comms": ("longest_dte_blackout_h", 0.0, 400.0, True),
    "power": ("gen_total_kwh_per_day", 0.0, 12.0, False),
    "flat": ("terrain_score", 30.0, 90.0, False),
    "ice": ("psr_km", 2.0, 20.0, True),
}


def full_score(row: dict, weights: dict | None = None) -> tuple[float, dict]:
    """Score (0-100 %) of one point from full M1-M5 results, with the same criteria and weights as the scan."""
    w = weights or DEFAULT_WEIGHTS
    parts, num, den = {}, 0.0, 0.0
    for k, (col, lo, hi, inverse) in FULL_CRITERIA.items():
        v = row.get(col)
        if v is None or not np.isfinite(v) or float(w.get(k, 0)) <= 0:
            parts[k] = None if v is None or not np.isfinite(v) else round(100 * float(np.clip((v - lo) / (hi - lo), 0, 1) if not inverse else 1 - np.clip((v - lo) / (hi - lo), 0, 1)), 1)
            continue
        c = float(np.clip((v - lo) / (hi - lo), 0, 1))
        c = 1 - c if inverse else c
        parts[k] = round(100 * c, 1)
        num += float(w[k]) * c
        den += float(w[k])
    return (round(100 * num / den, 1) if den else 0.0), parts


def point_context(lat: float, lon: float) -> dict:
    """Pole-wide product values of the 1 km cell containing (lat, lon) - used to give full-model points a PSR distance."""
    g = grid()
    if g is None or lat > -79.9:
        return {}
    x, y = latlon_to_ps(lat, lon)
    xs, ys = g["x"][0], g["y"][:, 0]
    j, i = int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))
    return {k: float(g[k][i, j]) for k in ("psr_km", "sun", "earth", "le10", "slope", "dark_h", "blackout_h", "spow")}
