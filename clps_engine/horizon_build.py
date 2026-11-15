"""Build 360-degree terrain horizon masks and terrain windows from LOLA/NAC rasters.

Horizon elevation along azimuth A from a site with observer height h_o:
    elev = atan2((R+h_t)cos(g) - (R+h_o), (R+h_t)sin(g)),  g = d/R
i.e. the full spherical-curvature form (no flat-Earth approximation).
Near field (<= 40 km) is ray-cast on the 80 m LOLA DEM, far field (40-200 km)
on the 240 m LOLA DEM. Obstacles smaller than ~2 DEM cells (~160 m) close to
the site are NOT resolved - recorded in provenance as a known limitation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .rasters import MOON_R_M, PolarRaster, latlon_to_ps, local_enu_axes


def ray_horizon(
    lat: float,
    lon: float,
    dem_near: PolarRaster,
    dem_far: Optional[PolarRaster] = None,
    *,
    az_step_deg: float = 0.5,
    near_max_m: float = 40000.0,
    far_max_m: float = 200000.0,
    observer_height_m: float = 2.0,
    site_elev_m: Optional[float] = None,
    near_min_m: float = 0.0,
) -> pd.DataFrame:
    """near_min_m > 0 skips terrain closer than that (far-field-only mask, e.g. beyond a local scene window)."""
    az = np.arange(0.0, 360.0, az_step_deg)
    east, north = local_enu_axes(lon)
    x0, y0 = latlon_to_ps(lat, lon)
    if site_elev_m is None or not np.isfinite(site_elev_m):
        site_elev_m = float(dem_near.sample_xy(np.array([x0]), np.array([y0]))[0])
    h_o = site_elev_m + observer_height_m

    segs = [(np.arange(max(dem_near.scale_m, near_min_m), near_max_m, dem_near.scale_m), dem_near)]
    if dem_far is not None:
        segs.append((np.arange(near_max_m, far_max_m, dem_far.scale_m), dem_far))

    best = np.full(az.size, -90.0)
    best_d = np.zeros(az.size)
    cover = np.zeros(az.size)
    total = 0
    a = np.radians(az)[:, None]
    dir_x = np.sin(a) * east[0] + np.cos(a) * north[0]
    dir_y = np.sin(a) * east[1] + np.cos(a) * north[1]
    for dist, dem in segs:
        # Map-plane distance ~= surface distance near the pole (PS scale factor
        # <0.6% inside 85S); recorded as a limitation.
        xs = x0 + dir_x * dist[None, :]
        ys = y0 + dir_y * dist[None, :]
        ht = dem.sample_xy(xs, ys)
        g = dist[None, :] / MOON_R_M
        rt = MOON_R_M + ht
        el = np.degrees(np.arctan2(rt * np.cos(g) - (MOON_R_M + h_o), rt * np.sin(g)))
        el = np.where(np.isfinite(el), el, -90.0)
        idx = np.argmax(el, axis=1)
        e = el[np.arange(az.size), idx]
        upd = e > best
        best[upd] = e[upd]
        best_d[upd] = dist[idx[upd]]
        cover += np.isfinite(ht).sum(axis=1)
        total += dist.size
    return pd.DataFrame(
        {
            "azimuth_deg": az,
            "horizon_elev_deg": best,
            "horizon_dist_km": best_d / 1000.0,
            "ray_valid_pct": 100.0 * cover / max(total, 1),
        }
    )


def _near_field(az_deg, sample_en, mpp: float, max_m: float, h_o: float):
    """Max horizon elevation over 0..max_m from a high-resolution local sampler f(E, N)->abs height."""
    a = np.radians(az_deg)[:, None]
    dist = np.arange(mpp, max_m, mpp)
    E, N = np.sin(a) * dist[None, :], np.cos(a) * dist[None, :]
    ht = sample_en(E, N)
    g = dist[None, :] / MOON_R_M
    rt = MOON_R_M + ht
    el = np.degrees(np.arctan2(rt * np.cos(g) - (MOON_R_M + h_o), rt * np.sin(g)))
    el = np.where(np.isfinite(el), el, -90.0)
    return el.max(axis=1), np.isfinite(ht).mean(axis=1)


def near_field_from_raster(lat, lon, dem: PolarRaster, az_deg, max_m: float, h_o: float):
    east, north = local_enu_axes(lon)
    x0, y0 = latlon_to_ps(lat, lon)
    f = lambda E, N: dem.sample_xy(x0 + E * east[0] + N * north[0], y0 + E * east[1] + N * north[1])
    return _near_field(az_deg, f, dem.scale_m, max_m, h_o)


def near_field_from_grid(tr: dict, az_deg, h_site_abs: float, h_o: float):
    """Legacy datakit terrain grid (heights relative to site, grid-oriented with north_xy/east_xy)."""
    from scipy import ndimage

    meta, h = tr["meta"], tr["height"].astype(float)
    mpp = float(meta["metres_per_pixel"])
    px, py = meta.get("site_pixel", [h.shape[1] / 2, h.shape[0] / 2])
    nx, ny = meta["north_xy"]
    ex, ey = meta["east_xy"]
    valid = np.isfinite(h)
    hf = np.where(valid, h, 0.0)

    def f(E, N):
        gx = E * ex + N * nx
        gy = E * ey + N * ny
        col, row = px + gx / mpp, py - gy / mpp
        ok = (row >= 0) & (row <= h.shape[0] - 1) & (col >= 0) & (col <= h.shape[1] - 1)
        v = ndimage.map_coordinates(hf, [np.clip(row, 0, h.shape[0] - 1), np.clip(col, 0, h.shape[1] - 1)], order=1)
        w = ndimage.map_coordinates(valid.astype(float), [np.clip(row, 0, h.shape[0] - 1), np.clip(col, 0, h.shape[1] - 1)], order=1)
        return np.where(ok & (w > 0.999), v + h_site_abs, np.nan)

    half = float(meta.get("grid_half_width_m", mpp * min(h.shape) / 2))
    return _near_field(az_deg, f, mpp, half * 0.98, h_o)


def terrain_window(dem: PolarRaster, lat: float, lon: float, half_width_m: float, out_px: int = 0):
    """Local terrain window, rotated so +y = local north, +x = local east.

    Returns dict(height (m, absolute), x_m, y_m, mpp, native_mpp, valid_pct).
    out_px>0 resamples onto an out_px x out_px grid (for 3D display / common
    comparison resolution); native_mpp is always retained.
    """
    east, north = local_enu_axes(lon)
    x0, y0 = latlon_to_ps(lat, lon)
    n = out_px or int(2 * half_width_m / dem.scale_m) + 1
    u = np.linspace(-half_width_m, half_width_m, n)
    E, N = np.meshgrid(u, u[::-1])
    xs = x0 + E * east[0] + N * north[0]
    ys = y0 + E * east[1] + N * north[1]
    h = dem.sample_xy(xs, ys)
    return {
        "height": h.astype("float32"),
        "x_m": u,
        "y_m": u[::-1],
        "mpp": float(u[1] - u[0]),
        "native_mpp": dem.scale_m,
        "valid_pct": float(100.0 * np.isfinite(h).mean()),
        "product_id": dem.product_id,
    }


def sample_layer_on_window(layer: PolarRaster, lat: float, lon: float, half_width_m: float, n: int, order: int = 1):
    east, north = local_enu_axes(lon)
    x0, y0 = latlon_to_ps(lat, lon)
    u = np.linspace(-half_width_m, half_width_m, n)
    E, N = np.meshgrid(u, u[::-1])
    return layer.sample_xy(x0 + E * east[0] + N * north[0], y0 + E * east[1] + N * north[1], order=order)


def ray_horizon_global(lat: float, lon: float, gdem, *, az_step_deg: float = 1.0, max_m: float = 300000.0,
                       observer_height_m: float = 2.0, site_elev_m=None) -> pd.DataFrame:
    """Horizon from the global LOLA 16 ppd DEM along great circles (non-polar sites). Near field < ~2 km unresolved."""
    az = np.arange(0.0, 360.0, az_step_deg)
    dist = np.arange(gdem.scale_m, max_m, gdem.scale_m * 0.75)
    la1, lo1 = np.radians(lat), np.radians(lon)
    g = dist[None, :] / MOON_R_M
    A = np.radians(az)[:, None]
    la2 = np.arcsin(np.sin(la1) * np.cos(g) + np.cos(la1) * np.sin(g) * np.cos(A))
    lo2 = lo1 + np.arctan2(np.sin(A) * np.sin(g) * np.cos(la1), np.cos(g) - np.sin(la1) * np.sin(la2))
    ht = gdem.sample_latlon(np.degrees(la2), np.degrees(lo2))
    z0 = float(gdem.sample_latlon(np.array([lat]), np.array([lon]))[0]) if site_elev_m is None else site_elev_m
    rt = MOON_R_M + ht
    el = np.degrees(np.arctan2(rt * np.cos(g) - (MOON_R_M + z0 + observer_height_m), rt * np.sin(g)))
    idx = el.argmax(axis=1)
    return pd.DataFrame({"azimuth_deg": az, "horizon_elev_deg": el[np.arange(az.size), idx],
                         "horizon_dist_km": dist[idx] / 1000.0, "ray_valid_pct": 100.0})
