"""User-chosen coordinate marker drawn identically on every map (pole 2D/3D, global, local site scenes, rank + coverage maps)."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from .rasters import latlon_to_ps, local_enu_axes

COLOR = "#ff2bd6"          # hot magenta: not used by any layer colourscale or other marker
RING = "#ffffff"


def _hover(mk) -> str:
    return f"📍 {mk['label']}<br>{mk['lat']:.5f}°, {mk['lon']:.5f}°E<extra>custom marker</extra>"


def ps_km(mk):
    x, y = latlon_to_ps(mk["lat"], mk["lon"])
    return float(x) / 1000.0, float(y) / 1000.0


def local_km(mk, site):
    """Marker position in a site's local east/north frame (km), as used by the 24 km scene maps."""
    x0, y0 = latlon_to_ps(site.lat, site.lon)
    x, y = latlon_to_ps(mk["lat"], mk["lon"])
    east, north = local_enu_axes(site.lon)
    dx, dy = float(x - x0), float(y - y0)
    return (dx * east[0] + dy * east[1]) / 1000.0, (dx * north[0] + dy * north[1]) / 1000.0


def _traces2d(x, y, mk, gl=True):
    S = go.Scattergl if gl else go.Scatter
    return [
        S(x=[x], y=[y], mode="markers", marker=dict(size=30, symbol="circle-open", color=COLOR, line=dict(width=3, color=COLOR)),
          hoverinfo="skip", showlegend=False),
        S(x=[x], y=[y], mode="markers+text", text=[f"📍 {mk['label']}"], textposition="bottom center",
          textfont=dict(size=12, color=COLOR), marker=dict(size=14, symbol="cross-thin-open", color=RING, line=dict(width=3, color=RING)),
          name=f"📍 {mk['label']}", hovertemplate=_hover(mk)),
    ]


def _off_map_note(fig, text):
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.99, xanchor="right", yanchor="top", showarrow=False, text=text,
                       font=dict(size=11, color=COLOR), bgcolor="rgba(4,8,15,.8)", bordercolor=COLOR, borderwidth=1)


def add_pole2d(fig, mk, gl=True):
    """Polar-stereographic km maps (Explorer 2D, rank score map, coverage footprints)."""
    if not mk:
        return fig
    if mk["lat"] > -79.0:
        _off_map_note(fig, f"📍 {mk['label']} ({mk['lat']:.2f}°, {mk['lon']:.2f}°E) is outside the 80°S cap")
        return fig
    x, y = ps_km(mk)
    fig.add_traces(_traces2d(x, y, mk, gl))
    return fig


def add_global(fig, mk):
    if not mk:
        return fig
    lon = mk["lon"] - 360 if mk["lon"] > 180 else mk["lon"]
    fig.add_traces(_traces2d(lon, mk["lat"], mk))
    return fig


def add_local2d(fig, mk, site, half_km=12.0):
    """Site-centred east/north km heatmaps (M3 slope, ray-traced illumination)."""
    if not mk:
        return fig
    e, n = local_km(mk, site)
    if abs(e) <= half_km and abs(n) <= half_km:
        fig.add_traces(_traces2d(e, n, mk, gl=False))
    else:
        d = math.hypot(e, n)
        brg = math.degrees(math.atan2(e, n)) % 360
        _off_map_note(fig, f"📍 {mk['label']} is {d:,.1f} km away (bearing {brg:.0f}°) - outside this 24 km window")
    return fig


def _pin3d(fig, x, y, z0, h, mk):
    fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[z0, z0 + h], mode="lines", line=dict(color=COLOR, width=7),
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z0 + h], mode="markers+text", text=[f"📍 {mk['label']}"], textposition="top center",
                               textfont=dict(size=13, color=COLOR), marker=dict(size=9, color=COLOR, symbol="diamond", line=dict(color=RING, width=2)),
                               name=f"📍 {mk['label']}", hovertemplate=_hover(mk)))


def add_pole3d(fig, mk, elev_m: float, vexag: float):
    if not mk:
        return fig
    if mk["lat"] > -79.0:
        _off_map_note(fig, f"📍 {mk['label']} is outside the 80°S cap")
        return fig
    x, y = ps_km(mk)
    z0 = (0.0 if not np.isfinite(elev_m) else elev_m) * vexag / 1000.0 + 0.8
    _pin3d(fig, x, y, z0, 16.0, mk)
    return fig


def add_scene3d(fig, mk, site, scene, vexag: float, stride: int = 2):
    """Site 3D scene (z in metres x VE relative to the site's median height, site pin at +30 m)."""
    if not mk:
        return fig
    e, n = local_km(mk, site)
    xs, ys = scene["x_m"] / 1000.0, scene["y_m"] / 1000.0
    if not (xs.min() <= e <= xs.max() and ys.min() <= n <= ys.max()):
        d = math.hypot(e, n)
        _off_map_note(fig, f"📍 {mk['label']} is {d:,.1f} km away (bearing {math.degrees(math.atan2(e, n)) % 360:.0f}°) - outside this scene")
        return fig
    H = scene["height_m"]
    z0 = float(np.nanmedian(H[145:156, 145:156]))
    c = int(np.argmin(np.abs(xs - e)))
    r = int(np.argmin(np.abs(ys - n)))
    h = float(H[r, c])
    z = ((h - z0) * vexag if np.isfinite(h) else 0.0) + 30.0
    _pin3d(fig, e, n, z, 900.0, mk)
    return fig
