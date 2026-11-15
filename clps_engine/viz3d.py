"""3D mission environment built from real LOLA / NAC elevation grids.

Scene grids are local east/north/up (metres) around each site, produced by
tools/ingest_sources.py terrain. Shadows are cast by ray marching the Sun
vector over the scene DEM; terrain outside the 24 km window is represented by
the site's 200 km horizon mask (shown as the horizon ring), not by geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .rasters import latlon_to_ps, local_enu_axes

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "processed" / "terrain3d"

LAYERS = {
    "elevation": ("Elevation (m)", "height_m"),
    "slope": ("Slope (deg) - LOLA 80 m", "slope_deg"),
    "sunlight_now": ("Illumination at selected time", None),
    "lola_sun": ("LOLA avg Sun visibility (18.6 yr)", "lola_avg_sun_vis_60m"),
    "lola_earth": ("LOLA avg Earth visibility (18.6 yr)", "lola_avg_earth_vis_60m"),
    "wac_illum": ("LROC WAC illumination % (88-90S)", "wac_illum_pct_100m"),
    "psr": ("LOLA permanent shadow", "lola_psr_60m"),
}

HOLO = dict(bg="#04080f", grid="rgba(80,200,255,0.10)", text="#bfefff", accent="#38d9ff")


def load_scene(site_id: str) -> Optional[dict]:
    p = SCENES / f"{site_id}_scene.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def cast_shadow(h: np.ndarray, mpp: float, sun_alt: float, sun_az: float, max_dist_m: float = 15000.0) -> np.ndarray:
    """Boolean lit mask: True where no scene terrain blocks the Sun centre."""
    if sun_alt <= -2.0:
        return np.zeros(h.shape, bool)
    from scipy import ndimage

    hh = np.where(np.isfinite(h), h, np.nanmin(h))
    az = np.radians(sun_az)
    dx, dy = np.sin(az), np.cos(az)  # east, north
    tan_e = np.tan(np.radians(sun_alt))
    rows, cols = np.indices(h.shape)
    lit = np.ones(h.shape, bool)
    step = mpp
    n = int(max_dist_m / step)
    for k in range(1, n + 1):
        d = k * step
        rr = rows - dy * d / mpp   # row index decreases northwards
        cc = cols + dx * d / mpp
        inside = (rr >= 0) & (rr <= h.shape[0] - 1) & (cc >= 0) & (cc <= h.shape[1] - 1)
        if not inside.any():
            break
        ht = ndimage.map_coordinates(hh, [np.clip(rr, 0, h.shape[0] - 1), np.clip(cc, 0, h.shape[1] - 1)], order=1)
        curv = d * d / (2 * 1737400.0)
        lit &= ~(inside & (ht - curv > hh + d * tan_e))
    return lit


def lambert(h: np.ndarray, mpp: float, sun_alt: float, sun_az: float) -> np.ndarray:
    gy, gx = np.gradient(np.where(np.isfinite(h), h, np.nanmean(h)), mpp)
    nx, ny, nz = -gx, gy, np.ones_like(gx)  # y axis runs north with decreasing row
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    a, z = np.radians(sun_az), np.radians(sun_alt)
    s = np.array([np.cos(z) * np.sin(a), np.cos(z) * np.cos(a), np.sin(z)])
    return np.clip((nx * s[0] + ny * s[1] + nz * s[2]) / nrm, 0, 1)


def neighbours_in_scene(site, others, half_width_m: float = 12000.0):
    """Other candidates that fall inside this site's local scene (east/north metres)."""
    x0, y0 = latlon_to_ps(site.lat, site.lon)
    east, north = local_enu_axes(site.lon)
    out = []
    for o in others:
        if o.id == site.id:
            continue
        x, y = latlon_to_ps(o.lat, o.lon)
        e = (x - x0) * east[0] + (y - y0) * east[1]
        n = (x - x0) * north[0] + (y - y0) * north[1]
        if abs(e) < half_width_m and abs(n) < half_width_m:
            out.append((o, float(e), float(n)))
    return out


def scene_figure(
    site,
    scene: dict,
    *,
    layer: str = "sunlight_now",
    sun_alt: float = 0.0,
    sun_az: float = 0.0,
    earth_alt: float = 0.0,
    earth_az: float = 0.0,
    sunlit_site: Optional[bool] = None,
    earth_visible_site: Optional[bool] = None,
    mask=None,
    neighbours=(),
    vexag: float = 2.0,
    stride: int = 2,
    time_label: str = "",
    height_px: int = 720,
) -> go.Figure:
    h = scene["height_m"][::stride, ::stride].astype(float)
    xs, ys = scene["x_m"][::stride], scene["y_m"][::stride]
    mpp = float(abs(xs[1] - xs[0]))
    z0 = float(np.nanmedian(scene["height_m"][145:156, 145:156]))
    Z = (h - z0) * vexag
    title, key = LAYERS[layer]
    if layer == "sunlight_now":
        lit = cast_shadow(h, mpp, sun_alt, sun_az)
        shade = lambert(h, mpp, sun_alt, sun_az) * lit * (1.0 if sun_alt > -0.3 else 0.0)
        c, cs, cmin, cmax = 0.08 + 0.92 * shade, [[0, "#02050b"], [0.12, "#0a1a2c"], [0.5, "#3c6f8f"], [1, "#e8fbff"]], 0, 1
        cbar = "Lambert x cast shadow"
    else:
        c = scene[key][::stride, ::stride].astype(float)
        if key == "height_m":
            c = h
        cs = {"elevation": "Ice", "slope": [[0, "#0b3d2e"], [0.25, "#1f9d6b"], [0.5, "#f2c14e"], [0.75, "#f06a3b"], [1, "#b3122e"]],
              "lola_sun": "Cividis", "lola_earth": "Blues", "wac_illum": "Cividis", "psr": [[0, "#0d2233"], [1, "#7a2cff"]]}[layer]
        cbar = title
        if not np.isfinite(c).any():
            c = np.zeros_like(h)
            cbar = f"{title}: NO COVERAGE at this site"
        cmin, cmax = (0, 30) if layer == "slope" else (float(np.nanmin(c)), float(np.nanmax(c)) + 1e-9)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs / 1000, y=ys / 1000, z=Z, surfacecolor=c, colorscale=cs, cmin=cmin, cmax=cmax,
        colorbar=dict(title=dict(text=cbar, font=dict(color=HOLO["text"])), tickfont=dict(color=HOLO["text"]), len=0.6),
        lighting=dict(ambient=0.9, diffuse=0.2, specular=0.0), showscale=True,
        contours=dict(z=dict(show=True, usecolormap=False, color="rgba(56,217,255,0.10)", width=1,
                             start=float(np.nanmin(Z)), end=float(np.nanmax(Z)), size=max(50.0, float(np.nanmax(Z) - np.nanmin(Z)) / 25))),
        hovertemplate="E %{x:.2f} km<br>N %{y:.2f} km<br>%{surfacecolor:.3f}<extra></extra>",
        name="terrain",
    ))
    # Landing site
    zs = 30.0
    col = "#7dffb0" if sunlit_site else "#ff6b6b" if sunlit_site is not None else HOLO["accent"]
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[zs], mode="markers+text", marker=dict(size=7, color=col, symbol="diamond"),
                               text=[f"{site.id}"], textfont=dict(color="#ffffff", size=13), name=f"{site.id} landing site",
                               hovertext=f"{site.name}<br>{site.lat:.4f}, {site.lon:.4f}<br>{site.elev_m:.0f} m", hoverinfo="text"))
    for o, e, n in neighbours:
        fig.add_trace(go.Scatter3d(x=[e / 1000], y=[n / 1000], z=[zs], mode="markers+text", marker=dict(size=4, color="#ffd27d"),
                                   text=[o.id], textfont=dict(color="#ffd27d", size=11), name=o.id, showlegend=False))
    # Horizon ring (200 km mask projected to a 6 km ring for display)
    if mask is not None:
        az = np.radians(np.asarray(mask[0]))
        el = np.radians(np.asarray(mask[1]))
        R = 6.0
        fig.add_trace(go.Scatter3d(x=R * np.sin(az), y=R * np.cos(az), z=zs + R * 1000 * np.tan(el) * 0.5,
                                   mode="lines", line=dict(color="rgba(56,217,255,0.8)", width=3),
                                   name="Terrain horizon (200 km mask)"))
    # Sun & Earth direction rays
    def ray(alt, az_, L=9.0):
        a, z_ = np.radians(az_), np.radians(alt)
        return L * np.cos(z_) * np.sin(a), L * np.cos(z_) * np.cos(a), zs + L * 1000 * np.sin(z_) * 0.5
    for alt, az_, nm, colr, vis in ((sun_alt, sun_az, "Sun", "#ffd54a", sunlit_site), (earth_alt, earth_az, "Earth", "#4aa8ff", earth_visible_site)):
        x1, y1, z1 = ray(alt, az_)
        fig.add_trace(go.Scatter3d(x=[0, x1], y=[0, y1], z=[zs, z1], mode="lines", line=dict(color=colr, width=6,
                                   dash="solid" if vis else "dash"), name=f"{nm} dir ({alt:.2f} deg el, {az_:.1f} az)"))
        glow = "rgba(255,213,74," if nm == "Sun" else "rgba(74,168,255,"
        base = 22 if nm == "Sun" else 16
        for s_, a in ((base * 3.0, 0.08), (base * 1.9, 0.20), (base * 1.25, 0.42)):
            fig.add_trace(go.Scatter3d(x=[x1], y=[y1], z=[z1], mode="markers", marker=dict(size=s_, color=f"{glow}{a if vis else a * 0.4})"),
                                       hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x1], y=[y1], z=[z1], mode="markers+text", marker=dict(size=base * 0.7, color=colr, line=dict(color="#ffffff", width=1)),
                                   text=[f"{'☀' if nm == 'Sun' else '🌍'} {nm.upper()} {'VISIBLE' if vis else 'BLOCKED'} · el {alt:+.2f}° az {az_:.1f}°"],
                                   textposition="top center", textfont=dict(color=colr, size=12, family="JetBrains Mono, monospace"), showlegend=False))
    ax = dict(backgroundcolor=HOLO["bg"], gridcolor=HOLO["grid"], zerolinecolor=HOLO["grid"], color=HOLO["text"], showbackground=True)
    fig.update_layout(
        height=height_px, paper_bgcolor=HOLO["bg"], plot_bgcolor=HOLO["bg"], font=dict(color=HOLO["text"], family="JetBrains Mono, monospace"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{site.name} ({site.id}) | {title} | {time_label} | vertical x{vexag:g}", font=dict(size=14)),
        scene=dict(xaxis=dict(title="East (km)", **ax), yaxis=dict(title="North (km)", **ax), zaxis=dict(title="Relative height x VE (m)", **ax),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35), camera=dict(eye=dict(x=1.2, y=-1.4, z=0.9))),
        legend=dict(bgcolor="rgba(4,8,15,0.7)", bordercolor=HOLO["grid"], font=dict(size=11)),
        uirevision=site.id,
    )
    return fig


def regional_figure(sites, highlight: Optional[str] = None, layer: str = "elevation", height_px: int = 680) -> Optional[go.Figure]:
    p = SCENES / "regional_scene.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        H, xs, ys = d["height_m"].astype(float), d["x_m"], d["y_m"]
        C = {"elevation": H, "lola_sun": d["sun_vis"].astype(float), "lola_earth": d["earth_vis"].astype(float)}[layer]
    fig = go.Figure(go.Surface(x=xs / 1000, y=ys / 1000, z=H * 1.0, surfacecolor=C,
                               colorscale={"elevation": "Ice", "lola_sun": "Cividis", "lola_earth": "Blues"}[layer],
                               lighting=dict(ambient=0.9, diffuse=0.3), colorbar=dict(len=0.6, tickfont=dict(color=HOLO["text"])),
                               hovertemplate="x %{x:.0f} km<br>y %{y:.0f} km<br>%{z:.0f} m<extra></extra>", name="LOLA 240 m"))
    from .rasters import product_registry
    dem = product_registry().get("LOLA_DEM_240M")
    for s in sites:
        x, y = latlon_to_ps(s.lat, s.lon)
        if abs(x) > 150000 or abs(y) > 150000:
            continue
        z = float(dem.sample_xy(np.array([x]), np.array([y]))[0]) if dem else 0
        hl = s.id == highlight
        fig.add_trace(go.Scatter3d(x=[x / 1000], y=[y / 1000], z=[z + 600], mode="markers+text",
                                   marker=dict(size=8 if hl else 5, color="#7dffb0" if hl else ("#38d9ff" if s.source == "baseline" else "#ffd27d")),
                                   text=[s.id], textfont=dict(color="#ffffff" if hl else "#bfefff", size=12 if hl else 10),
                                   name=s.id, showlegend=False, hovertext=f"{s.name}<br>{s.lat:.3f}, {s.lon:.3f}", hoverinfo="text"))
    ax = dict(backgroundcolor=HOLO["bg"], gridcolor=HOLO["grid"], color=HOLO["text"])
    fig.update_layout(height=height_px, paper_bgcolor=HOLO["bg"], font=dict(color=HOLO["text"]), margin=dict(l=0, r=0, t=40, b=0),
                      title="South polar region 300 x 300 km (LOLA LDEM_75S_240M, polar stereographic; pole at 0,0)",
                      scene=dict(xaxis=dict(title="PS x (km)", **ax), yaxis=dict(title="PS y (km)", **ax), zaxis=dict(title="Height (m)", **ax),
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.18)))
    return fig


def scene_for(site) -> Optional[dict]:
    """Precomputed scene if available, otherwise build one on the fly from LOLA 80 m + co-registered layers."""
    sc = load_scene(site.id)
    if sc is not None:
        return sc
    from .rasters import product_registry
    from .horizon_build import terrain_window, sample_layer_on_window

    reg = product_registry()
    dem = reg.get("LOLA_DEM_80M")
    if dem is None or not dem.covers(site.lat, site.lon):
        return None
    w = terrain_window(dem, site.lat, site.lon, 12000.0, out_px=301)
    gy, gx = np.gradient(w["height"].astype(float), w["mpp"])
    out = {"height_m": w["height"], "x_m": w["x_m"], "y_m": w["y_m"],
           "slope_deg": np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")}
    for k in ("LOLA_AVG_SUN_VIS_60M", "LOLA_AVG_EARTH_VIS_60M", "WAC_ILLUM_PCT_100M", "LOLA_PSR_60M"):
        if k in reg:
            out[k.lower()] = sample_layer_on_window(reg[k], site.lat, site.lon, 12000.0, 301).astype("float32")
        else:
            out[k.lower()] = np.full(w["height"].shape, np.nan, "float32")
    return out
