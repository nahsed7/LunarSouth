"""Whole south-pole (80S -> 90S) maps and 3D terrain from the 20 m LOLA mosaic and integrated products."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .rasters import latlon_to_ps, ps_to_latlon

ROOT = Path(__file__).resolve().parent.parent
PW = ROOT / "data" / "processed" / "polewide"
BG, TXT, GRID = "#04080f", "#bfefff", "rgba(80,200,255,0.10)"

# key: (label, colorscale, zmin, zmax, units, source)
LAYERS = {
    "suitability": ("Landing suitability index (composite)", "Viridis", 0, 100, "index", "Model 1+2+3 composite (see Calculations)"),
    "power": ("Mean solar power, vertical-tracking array", [[0, "#05070d"], [0.2, "#3b1d04"], [0.55, "#c26a0a"], [1, "#ffe27a"]], 0, 300, "W/m²",
              "Model 4 on 5 km lattice: SPICE/Horizons geometry × LOLA horizon"),
    "sun": ("Sun visibility (LOLA product, model beyond 85S)", "Cividis", 0, 0.9, "fraction", "Moon Trek AVGVISIB_85S_060M + Model 1 lattice"),
    "earth": ("Earth visibility (LOLA product, model beyond 85S)", "Blues", 0, 1, "fraction", "Moon Trek AVGVISIB_..._EARTH + Model 2 lattice"),
    "le10": ("Terrain ≤10° slope (20 m pixels)", [[0, "#b3122e"], [0.5, "#f2c14e"], [1, "#1f9d6b"]], 0, 1, "fraction", "LDEM_80S_20MPP slope at native 20 m"),
    "elevation": ("Elevation", "Ice", -5000, 7000, "m", "LDEM_80S_20MPP (PGDA LOLA), 1 km mean"),
    "psr": ("Permanently shadowed fraction", [[0, "#0d2233"], [1, "#9b5cff"]], 0, 1, "fraction", "LPSR_80S_20MPP"),
    "dark": ("Longest continuous darkness (2027)", "Inferno_r", 0, 720, "h", "Model 1 lattice"),
    "blackout": ("Longest DTE blackout (2027)", "Magma_r", 0, 720, "h", "Model 2 lattice"),
    "wac": ("LROC WAC illumination %", "Cividis", 0, 70, "%", "QuickMap WAC_POLE_ILL_PCT_SOUTH_100M (88-90S)"),
    "tmax": ("Diviner Tbol max anomaly", "RdBu_r", -60, 60, "K", "LRO Diviner (0.5° grid)"),
    "illum_now": ("Illumination NOW - live Sun & cast shadows", [[0, "#02050b"], [0.02, "#06243a"], [0.35, "#0e6f9a"], [0.7, "#38d9ff"], [1, "#f2fdff"]],
                  0, 1, "lit", "SPICE Sun vector at the selected time × LOLA 20 m-derived mesh"),
}


@lru_cache(maxsize=1)
def stack() -> dict:
    p = PW / "mosaic1km.npz"
    if not p.exists():
        return {}
    m = dict(np.load(p))
    out = {"x": m["x_m"], "y": m["y_m"], "elevation": m["elev_mean"], "le10": m["le10"], "psr": m["psr_frac"],
           "wac": m.get("wac_illum"), "tmax": m.get("div_tmax_anom"), "slope": m["slope_mean"]}
    s = PW / "suitability1km.npz"
    if s.exists():
        q = np.load(s)
        out["suitability"], out["sun"], out["earth"] = q["score"], q["sun_merged"], q["earth_merged"]
    sim = lattice()
    if sim is not None:
        from scipy.interpolate import griddata
        X, Y = np.meshgrid(out["x"], out["y"])
        inside = np.isfinite(out["elevation"])
        for col, key in (("mean_specific_power_w_m2", "power"), ("longest_dark_h", "dark"), ("longest_blackout_h", "blackout")):
            g = griddata((sim.x_m, sim.y_m), sim[col], (X, Y), method="linear")
            out[key] = np.where(inside, g, np.nan).astype(np.float32)
    return out


@lru_cache(maxsize=1)
def lattice():
    p = PW / "sim_lattice.csv"
    return pd.read_csv(p) if p.exists() else None


def value_at(layer: str, lat: float, lon: float) -> float:
    s = stack()
    if not s or s.get(layer) is None:
        return float("nan")
    x, y = latlon_to_ps(lat, lon)
    c = int(np.argmin(np.abs(s["x"] - x)))
    r = int(np.argmin(np.abs(s["y"] - y)))
    return float(s[layer][r, c])


def _rings(fig, is3d=False, z=None):
    for lat in (-80, -82.5, -85, -87.5):
        t = np.linspace(0, 2 * np.pi, 240)
        x, y = latlon_to_ps(np.full(t.size, lat), np.degrees(t))
        if is3d:
            fig.add_trace(go.Scatter3d(x=x / 1000, y=y / 1000, z=np.full(t.size, z), mode="lines",
                                       line=dict(color="rgba(56,217,255,.35)", width=2), showlegend=False, hoverinfo="skip"))
        else:
            fig.add_trace(go.Scattergl(x=x / 1000, y=y / 1000, mode="lines", line=dict(color="rgba(56,217,255,.35)", width=1),
                                       showlegend=False, hoverinfo="skip"))
    for lon in range(0, 360, 30):
        x, y = latlon_to_ps(np.array([-80.0, -90.0]), np.array([lon, lon]))
        if not is3d:
            fig.add_trace(go.Scattergl(x=x / 1000, y=y / 1000, mode="lines", line=dict(color="rgba(56,217,255,.18)", width=1),
                                       showlegend=False, hoverinfo="skip"))
            fig.add_annotation(x=x[0] / 1000 * 1.06, y=y[0] / 1000 * 1.06, text=f"{lon if lon <= 180 else lon - 360}°", showarrow=False,
                               font=dict(size=10, color="#7fa7b8"))


def pole_map(layer: str, sites, discovered: pd.DataFrame, selected=None, pick_step: int = 4, height=760, t_utc=None) -> go.Figure:
    s = stack()
    lab, cs, zmin, zmax, units, src = LAYERS[layer]
    z = s.get(layer)
    if layer == "illum_now":
        t_utc = t_utc if t_utc is not None else "2026-10-15T00:00:00"
        z = np.repeat(np.repeat(pole_illumination_now(pd.Timestamp(t_utc).strftime("%Y-%m-%dT%H:%M:%S"), 2)[2], 2, 0), 2, 1)[:len(s["y"]), :len(s["x"])]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=s["x"] / 1000, y=s["y"] / 1000, z=z, colorscale=cs, zmin=zmin, zmax=zmax, zsmooth="best",
                             colorbar=dict(title=units, len=0.7), hovertemplate="x %{x:.0f} km, y %{y:.0f} km<br>%{z:.3f}<extra></extra>",
                             name=lab))
    # clickable pick lattice (every pick_step km)
    X, Y = np.meshgrid(s["x"][::pick_step], s["y"][::pick_step])
    zz = z[::pick_step, ::pick_step]
    ok = np.isfinite(zz)
    la, lo = ps_to_latlon(X[ok], Y[ok])
    fig.add_trace(go.Scattergl(x=X[ok] / 1000, y=Y[ok] / 1000, mode="markers", marker=dict(size=7, color="rgba(0,0,0,0)"),
                               customdata=np.c_[la, lo, zz[ok]], name="pick",
                               hovertemplate="%{customdata[0]:.4f}°, %{customdata[1]:.4f}°E<br>" + lab + ": %{customdata[2]:.3f}<extra>click to analyse</extra>"))
    _rings(fig)
    if discovered is not None and len(discovered):
        x, y = latlon_to_ps(discovered.latitude_deg.values, discovered.longitude_deg.values)
        fig.add_trace(go.Scattergl(x=x / 1000, y=y / 1000, mode="markers+text", text=discovered.site_id.str.replace("LN-", ""),
                                   textposition="top center", textfont=dict(size=9, color="#7dffb0"),
                                   marker=dict(size=8, symbol="star", color="#7dffb0", line=dict(color="#04080f", width=1)),
                                   name="LunarSouth discoveries", customdata=np.c_[discovered.latitude_deg, discovered.longitude_deg, discovered.suitability_index],
                                   hovertemplate="%{text}<br>%{customdata[0]:.4f}, %{customdata[1]:.4f}<br>index %{customdata[2]:.1f}<extra></extra>"))
    if sites:
        la = np.array([q.lat for q in sites]); lo = np.array([q.lon for q in sites])
        x, y = latlon_to_ps(la, lo)
        fig.add_trace(go.Scattergl(x=x / 1000, y=y / 1000, mode="markers", marker=dict(size=7, symbol="diamond", color="#ffd27d"),
                                   name="Project candidates", text=[q.id for q in sites], customdata=np.c_[la, lo, np.zeros(la.size)],
                                   hovertemplate="%{text}<extra></extra>"))
    if selected is not None:
        x, y = latlon_to_ps(selected[0], selected[1])
        fig.add_trace(go.Scattergl(x=[x / 1000], y=[y / 1000], mode="markers", marker=dict(size=18, symbol="circle-open", color="#ffffff",
                                   line=dict(width=3)), name="Selected point", hoverinfo="skip"))
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG, font=dict(color=TXT),
                      margin=dict(l=10, r=10, t=40, b=10), title=f"{lab} · {src}",
                      xaxis=dict(title="polar stereographic x (km)", showgrid=False, zeroline=False, range=[-310, 310]),
                      yaxis=dict(title="y (km)", showgrid=False, zeroline=False, scaleanchor="x", range=[-310, 310]),
                      legend=dict(orientation="h", y=-0.06), clickmode="event+select", dragmode="pan")
    return fig


def pole_3d(layer: str, sites, discovered, selected=None, step: int = 2, vexag: float = 6.0, height=760, t_utc=None) -> go.Figure:
    s = stack()
    lab, cs, zmin, zmax, units, src = LAYERS[layer]
    H = s["elevation"][::step, ::step].astype(float)
    if layer == "illum_now":
        t_utc = t_utc if t_utc is not None else "2026-10-15T00:00:00"
        C = pole_illumination_now(pd.Timestamp(t_utc).strftime("%Y-%m-%dT%H:%M:%S"), step)[2]
    else:
        C = s[layer][::step, ::step].astype(float)
    x, y = s["x"][::step] / 1000, s["y"][::step] / 1000
    fig = go.Figure(go.Surface(x=x, y=y, z=H * vexag / 1000, surfacecolor=C, colorscale=cs, cmin=zmin, cmax=zmax,
                               colorbar=dict(title=units, len=0.6, tickfont=dict(color=TXT)),
                               lighting=dict(ambient=0.75, diffuse=0.55, specular=0.05, roughness=0.9),
                               lightposition=dict(x=-1e5, y=1e5, z=2e4),
                               hovertemplate="x %{x:.0f} km, y %{y:.0f} km<br>" + lab + ": %{surfacecolor:.3f}<extra></extra>", name=lab))
    zl = np.nanmax(H) * vexag / 1000 * 1.02

    def mark(lat, lon, name, color, size, symbol="diamond"):
        xx, yy = latlon_to_ps(np.asarray(lat), np.asarray(lon))
        cc = np.clip(((xx / 1000 - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1)
        rr = np.clip(((yy / 1000 - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1)
        zz = np.nan_to_num(H[rr, cc]) * vexag / 1000 + 0.8
        return xx / 1000, yy / 1000, zz

    if discovered is not None and len(discovered):
        a, b, c = mark(discovered.latitude_deg.values, discovered.longitude_deg.values, "", "", 0)
        fig.add_trace(go.Scatter3d(x=a, y=b, z=c, mode="markers+text", text=discovered.site_id.str.replace("LN-", ""),
                                   marker=dict(size=4, color="#7dffb0", symbol="diamond"), textfont=dict(size=9, color="#7dffb0"),
                                   name="LunarSouth discoveries"))
    if sites:
        a, b, c = mark([q.lat for q in sites], [q.lon for q in sites], "", "", 0)
        fig.add_trace(go.Scatter3d(x=a, y=b, z=c, mode="markers+text", text=[q.id for q in sites], marker=dict(size=3, color="#ffd27d"),
                                   textfont=dict(size=9, color="#ffd27d"), name="Project candidates"))
    if selected is not None:
        a, b, c = mark([selected[0]], [selected[1]], "", "", 0)
        fig.add_trace(go.Scatter3d(x=a, y=b, z=[c[0], ], mode="markers", marker=dict(size=9, color="#ffffff", symbol="circle-open"),
                                   name="Selected point"))
        fig.add_trace(go.Scatter3d(x=[a[0], a[0]], y=[b[0], b[0]], z=[c[0], c[0] + 12], mode="lines", line=dict(color="#ffffff", width=4),
                                   showlegend=False))
    _rings(fig, True, zl)
    if t_utc is not None:
        sxy = None
        if selected is not None:
            a, b, c = mark([selected[0]], [selected[1]], "", "", 0)
            sxy = (a[0], b[0], c[0] + 12)
        add_sun_earth_3d(fig, t_utc, sxy, z_top=zl, vexag=vexag)
    ax = dict(backgroundcolor=BG, gridcolor=GRID, color=TXT, showbackground=True)
    fig.update_layout(height=height, paper_bgcolor=BG, font=dict(color=TXT), margin=dict(l=0, r=0, t=40, b=0),
                      title=f"South pole 80°S→90°S · {lab} · LOLA 20 m DEM (1 km display mesh, vertical ×{vexag:g})",
                      scene=dict(xaxis=dict(title="x (km)", **ax), yaxis=dict(title="y (km)", **ax), zaxis=dict(title="km ×VE", **ax),
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.22), camera=dict(eye=dict(x=0.9, y=-1.1, z=0.75))),
                      legend=dict(bgcolor="rgba(4,8,15,.7)"), uirevision="pole3d")
    if t_utc is not None:
        fig.update_layout(title=f"South pole 80°S→90°S · {lab} · {pd.Timestamp(t_utc):%Y-%m-%d %H:%M} UTC · Sun & Earth at true azimuth/elevation (vertical ×{vexag:g})")
    return fig


# Approximate IAU feature centres (lat, lon E) for orientation labels only.
CRATERS = [("Shackleton", -89.67, 129.78), ("de Gerlache", -88.5, -87.1), ("Sverdrup", -88.3, -152.0), ("Shoemaker", -88.1, 44.9),
           ("Faustini", -87.3, 77.0), ("Haworth", -87.45, -5.0), ("Cabeus", -85.33, -42.13), ("Nobile", -85.2, 53.5),
           ("Amundsen", -84.5, 82.8), ("Slater", -88.0, 111.0), ("Malapert massif", -86.0, 0.0), ("Scott", -82.1, 48.5),
           ("Wiechert", -84.5, 165.0), ("Mons Mouton", -84.9, 33.0)]


def add_craters(fig):
    la = np.array([c[1] for c in CRATERS]); lo = np.array([c[2] for c in CRATERS])
    x, y = latlon_to_ps(la, lo)
    fig.add_trace(go.Scattergl(x=x / 1000, y=y / 1000, mode="text", text=[c[0] for c in CRATERS], textfont=dict(size=11, color="rgba(230,245,255,.85)"),
                               name="Features (approx. IAU centres)", hoverinfo="text"))
    return fig


@lru_cache(maxsize=1)
def _global_grid():
    from .rasters import global_dem

    g = global_dem()
    if g is None:
        return None
    a = np.asarray(g.arr[::4, ::4], dtype=np.float32) * 0.5  # 0.25 deg display grid
    gy, gx = np.gradient(a)
    shade = np.clip(0.5 + (-gx * 0.7 + gy * 0.7) / 400.0, 0, 1)
    lat = 90 - (np.arange(a.shape[0]) + 0.5) * 0.25
    lon = (np.arange(a.shape[1]) + 0.5) * 0.25
    lon = np.where(lon > 180, lon - 360, lon)
    o = np.argsort(lon)
    return lat, lon[o], a[:, o], shade[:, o]


def global_map(sites, selected=None, height=620):
    gg = _global_grid()
    if gg is None:
        return None
    lat, lon, a, shade = gg
    fig = go.Figure(go.Heatmap(x=lon, y=lat, z=a, colorscale="Ice", zsmooth="best", colorbar=dict(title="m", len=0.7),
                               hovertemplate="%{y:.2f}°, %{x:.2f}°E<br>%{z:.0f} m<extra></extra>"))
    fig.add_trace(go.Heatmap(x=lon, y=lat, z=shade, colorscale=[[0, "rgba(0,0,0,0.45)"], [1, "rgba(255,255,255,0.0)"]], showscale=False, hoverinfo="skip"))
    groups = {}
    for s in sites:
        groups.setdefault(s.source.replace("preset:", ""), []).append(s)
    colors = ["#ffd54a", "#7dffb0", "#ff8c42", "#38d9ff", "#ff5d73", "#9b5cff"]
    for (g, ss), colr in zip(groups.items(), colors * 3):
        fig.add_trace(go.Scattergl(x=[s.lon if s.lon <= 180 else s.lon - 360 for s in ss], y=[s.lat for s in ss], mode="markers+text",
                                   text=[s.id for s in ss], textposition="top center", textfont=dict(size=9, color=colr),
                                   marker=dict(size=8, color=colr, line=dict(color="#04080f", width=1)), name=g.replace("_", " "),
                                   customdata=np.c_[[s.lat for s in ss], [s.lon for s in ss]],
                                   hovertext=[f"{s.name}<br>{', '.join(s.tags[:4])}" for s in ss], hoverinfo="text"))
    fig.add_shape(type="rect", x0=-180, x1=180, y0=-90, y1=-80, line=dict(color="#38d9ff", dash="dot"), fillcolor="rgba(56,217,255,.06)")
    fig.add_annotation(x=0, y=-78, text="80°S polar cap - full 20 m / 80 m analysis", showarrow=False, font=dict(color="#38d9ff", size=11))
    if selected is not None:
        fig.add_trace(go.Scattergl(x=[selected[1] if selected[1] <= 180 else selected[1] - 360], y=[selected[0]], mode="markers",
                                   marker=dict(size=16, symbol="circle-open", color="#fff", line=dict(width=3)), name="analysed point"))
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG, font=dict(color=TXT), margin=dict(l=10, r=10, t=40, b=10),
                      title="Global Moon (LOLA LDEM_16) - historic, commercial & international landing sites", clickmode="event+select",
                      xaxis=dict(title="longitude °E", range=[-180, 180]), yaxis=dict(title="latitude °", range=[-90, 90], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.1))
    return fig


# ------------------------------------------------------------------ live Sun / Earth over the whole pole
MOON_R = 1737400.0


def body_dirs_map(t_utc):
    """Unit Sun and Earth directions (from the Moon centre) in the polar map frame: +x = 90E, +y = 0E, +z = away from Moon at S pole.

    Source: SPICE DE421 + MOON_ME_DE421 (validated vs JPL Horizons). Earth parallax (<=1 deg) is ignored in this
    pole-wide layer only; site analyses use exact topocentric geometry.
    """
    from .geometry import moon_fixed_vectors

    t0 = pd.Timestamp(t_utc).strftime("%Y-%m-%dT%H:%M:%S")
    t1 = (pd.Timestamp(t_utc) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    _, S, E = moon_fixed_vectors(t0, t1, 1.0)
    out = {}
    for name, v in (("sun", S[0]), ("earth", E[0])):
        m = np.array([v[1], v[0], -v[2]]) / np.linalg.norm(v)
        out[name] = m
    return out


@lru_cache(maxsize=64)
def pole_illumination_now(t_key: str, step: int = 2):
    """Cast-shadow illumination over the whole cap at one instant (2 km grid): lit fraction 0..1 with Lambert shading.

    Each cell uses its OWN local vertical (curvature tilt ~10 deg at 300 km) to get the local Sun elevation; terrain
    obstruction by ray-marching the LOLA 20 m-derived mesh with spherical drop d^2/2R.
    """
    from scipy import ndimage

    s = stack()
    H = np.nan_to_num(s["elevation"][::step, ::step].astype(float), nan=np.nanmin(s["elevation"]))
    x = s["x"][::step]
    y = s["y"][::step]
    X, Y = np.meshgrid(x, y)
    mpp = float(abs(x[1] - x[0]))
    d = body_dirs_map(t_key)["sun"]
    rho = np.hypot(X, Y)
    # local up = outward normal of the sphere at (X,Y) in map frame (south-pole centred)
    tilt = rho / MOON_R
    ux = np.where(rho > 0, X / np.maximum(rho, 1) * np.sin(tilt), 0)
    uy = np.where(rho > 0, Y / np.maximum(rho, 1) * np.sin(tilt), 0)
    uz = np.cos(tilt)
    sin_el = d[0] * ux + d[1] * uy + d[2] * uz
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    az = np.degrees(np.arctan2(d[0], d[1]))
    tan_e = np.tan(np.radians(el))
    dx, dy = np.sin(np.radians(az)), np.cos(np.radians(az))
    rows, cols = np.indices(H.shape)
    lit = el > -0.27
    for k in range(1, int(250000 / mpp)):
        dd = k * mpp
        rr, cc = rows - dy * dd / mpp, cols + dx * dd / mpp
        ins = (rr >= 0) & (rr <= H.shape[0] - 1) & (cc >= 0) & (cc <= H.shape[1] - 1)
        if not ins.any():
            break
        ht = ndimage.map_coordinates(H, [np.clip(rr, 0, H.shape[0] - 1), np.clip(cc, 0, H.shape[1] - 1)], order=1)
        lit &= ~(ins & (ht - dd * dd / (2 * MOON_R) > H + dd * tan_e))
    gy, gx = np.gradient(H, mpp)
    nrm = np.sqrt(gx * gx + gy * gy + 1)
    lam = np.clip((-gx * d[0] + gy * d[1] + sin_el) / nrm, 0, 1)
    val = np.where(lit, 0.25 + 0.75 * lam, 0.0)
    val = np.where(np.isfinite(s["elevation"][::step, ::step]), val, np.nan)
    return x, y, val.astype(np.float32), float(np.degrees(np.arcsin(d[2]))), float(az)


def add_sun_earth_3d(fig, t_utc, selected_xy_km=None, z_top=8.0, vexag=6.0, R_km=430.0):
    """Holographic Sun and Earth markers at their true azimuth/elevation (seen from the pole) + rays from the selected point."""
    dirs = body_dirs_map(t_utc)
    for name, col, glow, size in (("sun", "#ffd54a", "rgba(255,213,74,", 26), ("earth", "#4aa8ff", "rgba(74,168,255,", 20)):
        d = dirs[name]
        el = np.degrees(np.arcsin(d[2]))
        az = np.degrees(np.arctan2(d[0], d[1]))
        hx, hy = d[0] / max(np.hypot(d[0], d[1]), 1e-9), d[1] / max(np.hypot(d[0], d[1]), 1e-9)
        X, Y = R_km * hx, R_km * hy
        Z = z_top + R_km * np.tan(np.radians(el)) * vexag / 50.0 + 2.0
        for s_, a in ((size * 3.2, 0.10), (size * 2.0, 0.22), (size * 1.3, 0.45)):
            fig.add_trace(go.Scatter3d(x=[X], y=[Y], z=[Z], mode="markers", marker=dict(size=s_, color=f"{glow}{a})", line=dict(width=0)),
                                       hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=[X], y=[Y], z=[Z], mode="markers+text", marker=dict(size=size * 0.7, color=col, line=dict(color="#ffffff", width=1)),
                                   text=[f"{'☀ SUN' if name == 'sun' else '🌍 EARTH'}  az {az:.1f}°  el {el:+.2f}° (pole)"],
                                   textposition="top center", textfont=dict(color=col, size=13, family="JetBrains Mono, monospace"),
                                   name=f"{name.title()} (true direction)"))
        # direction ring tick on the horizon circle + ray from the analysed point
        if selected_xy_km is not None:
            sx, sy, sz = selected_xy_km
            fig.add_trace(go.Scatter3d(x=[sx, X], y=[sy, Y], z=[sz, Z], mode="lines", line=dict(color=col, width=5, dash="dash"),
                                       opacity=0.8, showlegend=False, hoverinfo="skip"))
        t = np.linspace(0, 2 * np.pi, 200)
    ring = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(go.Scatter3d(x=R_km * np.sin(ring), y=R_km * np.cos(ring), z=np.full(ring.size, z_top), mode="lines",
                               line=dict(color="rgba(56,217,255,.55)", width=2), name="horizon ring (azimuth reference)", hoverinfo="skip"))
    for a in range(0, 360, 30):
        fig.add_trace(go.Scatter3d(x=[R_km * np.sin(np.radians(a)) * 1.04], y=[R_km * np.cos(np.radians(a)) * 1.04], z=[z_top], mode="text",
                                   text=[f"{a if a <= 180 else a - 360}°E"], textfont=dict(color="rgba(191,239,255,.7)", size=10), showlegend=False,
                                   hoverinfo="skip"))
    return fig
