"""Plain-language guidance: Earth-vs-Moon baseline, coordinate conventions, sensitivity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Values are standard physical constants / well-established figures; sources in the 'basis' column.
EARTH_VS_MOON = pd.DataFrame([
    ("Solar irradiance at the surface", "~1000 W/m² peak at noon (after atmosphere)", "1361 W/m² × 1/r² - no atmosphere", "TSI 1361 W/m² (Kopp & Lean 2011)"),
    ("Length of one day-night cycle", "24 h", "29.53 Earth days (synodic month)", "lunar synodic period"),
    ("Sun elevation at the pole", "up to 23.4° at the pole (axial tilt)", "only ±1.54° at the exact pole (tiny 1.54° tilt) → Sun skims the horizon", "lunar obliquity to ecliptic 1.54°"),
    ("Why terrain matters", "hills block a few minutes of sun", "a 1° ridge 50 km away can cause days of darkness", "grazing Sun + no atmosphere"),
    ("Night temperature", "~ -20 °C to +10 °C typical", "as low as -200 °C (CSA LUV requirement); PSRs ~ -230 °C", "CSA, Diviner"),
    ("Earth in the sky", "-", "Earth hangs almost fixed (librations ±7-8°); behind the limb on the far side", "lunar libration"),
    ("Radio delay to Earth", "~0", "~1.28 s one way (384 400 km)", "speed of light"),
    ("Gravity", "9.81 m/s²", "1.62 m/s²", "IAU"),
    ("Solar panel orientation", "tilted toward the equator", "near-vertical, rotating to follow a Sun that circles the horizon", "geometry"),
], columns=["topic", "Earth", "Moon south pole", "basis"])

COORDS_MD = """
**Coordinates.** Latitude is planetocentric (−90° = south pole). Longitude is **east-positive** (−180…180 or 0…360 are both accepted).
Maps use the **south polar stereographic** projection (centre 90°S, 0°E, sphere R = 1737.4 km): x = ρ·sin(lon), y = ρ·cos(lon),
ρ = 2R·tan((90°+lat)/2). At the pole, **north is the direction away from the pole**; azimuths are measured clockwise from that local north.

**Time.** All times are UTC; ephemerides are computed in TDB internally (SPICE), validated against JPL Horizons.

**Why small changes matter.** Near the pole one degree of latitude is ~30 km, but one degree of *longitude* can be only a few hundred
metres. Moving a site 1 km can put a ridge between it and the Sun, and a few days' shift in landing date can move the whole mission
into lunar night. Use the sensitivity tables below and the date-slip sweep (Compare tab) before trusting any single number.
"""


def spatial_sensitivity(value_at, lat: float, lon: float, layers=("sun", "earth", "power", "le10", "suitability"),
                        offsets_km=(1.0, 5.0, 15.0)) -> pd.DataFrame:
    """Change of map-layer values when the point moves N/E/S/W by each offset (uses the 1 km pole-wide stack)."""
    from .rasters import latlon_to_ps, ps_to_latlon, local_enu_axes

    x0, y0 = latlon_to_ps(lat, lon)
    e, n = local_enu_axes(lon)
    base = {L: value_at(L, lat, lon) for L in layers}
    rows = []
    for d in offsets_km:
        for name, v in (("north", n), ("east", e), ("south", -n), ("west", -e)):
            la, lo = ps_to_latlon(x0 + v[0] * d * 1000, y0 + v[1] * d * 1000)
            r = {"move": f"{d:g} km {name}", "lat": float(la), "lon": float(lo)}
            for L in layers:
                r[f"Δ {L}"] = value_at(L, float(la), float(lo)) - base[L]
            rows.append(r)
    return pd.DataFrame(rows)
