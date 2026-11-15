"""Sun/Earth local geometry: live Skyfield OR ingest precomputed CSVs / NPZ."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .adapters import infer_step_minutes, parse_utc_series, read_table, rename_to_canonical, resolve_column
from .sites import Site

EARTH_RADIUS_KM = 6378.137
SUN_RADIUS_KM = 695700.0  # IAU 2015 nominal solar radius
KERNEL_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "kernels"


def _loader(kernel_dir: Path):
    from skyfield.api import Loader
    from skyfield.planetarylib import PlanetaryConstants

    loader = Loader(str(kernel_dir))
    ts = loader.timescale()
    eph = loader("de421.bsp")
    pc = PlanetaryConstants()
    pc.read_text(loader("moon_080317.tf"))
    pc.read_text(loader("pck00008.tpc"))
    pc.read_binary(loader("moon_pa_de421_1900-2050.bpc"))
    frame = pc.build_frame_named("MOON_ME_DE421")
    return ts, eph, pc, frame


_CACHE = {}


def load_kernels(kernel_dir: Optional[Path] = None):
    kernel_dir = Path(kernel_dir or KERNEL_DIR_DEFAULT)
    key = str(kernel_dir)
    if key not in _CACHE:
        _CACHE[key] = _loader(kernel_dir)
    return _CACHE[key]


def compute_geometry(
    site: Site,
    start_iso: str,
    hours: float,
    step_min: float = 60.0,
    kernel_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute Sun/Earth alt-az at a lunar surface site over a time window."""
    n = max(2, int(round(hours * 60 / step_min)))
    return compute_geometry_n(site, start_iso, n, step_min, kernel_dir)


def compute_geometry_n(
    site: Site,
    start_iso: str,
    n: int,
    step_min: float,
    kernel_dir: Optional[Path] = None,
) -> pd.DataFrame:
    ts, eph, pc, frame = load_kernels(kernel_dir)
    start = datetime.fromisoformat(start_iso.replace("Z", ""))
    t0 = ts.utc(start.year, start.month, start.day, start.hour, start.minute, start.second)
    # Uniform UTC grid (step in UT days; TT-UTC is constant over a mission window)
    t = ts.tt_jd(t0.tt + np.arange(n) * step_min / 1440.0)
    return _geometry_at(site, t, eph, pc, frame)


def _geometry_at(site: Site, t, eph, pc, frame) -> pd.DataFrame:
    place = eph["moon"] + pc.build_latlon_degrees(frame, site.lat, site.lon, elevation_m=site.elev_m)
    obs = place.at(t)
    sun_alt, sun_az, sun_dist = obs.observe(eph["sun"]).apparent().altaz()
    earth_alt, earth_az, earth_dist = obs.observe(eph["earth"]).apparent().altaz()
    radius = np.degrees(np.arcsin(np.minimum(1.0, EARTH_RADIUS_KM / earth_dist.km)))
    sun_radius = np.degrees(np.arcsin(np.minimum(1.0, SUN_RADIUS_KM / sun_dist.km)))

    return pd.DataFrame(
        {
            "time_utc": pd.to_datetime(t.utc_datetime()).tz_localize(None),
            "sun_alt": sun_alt.degrees,
            "sun_az": sun_az.degrees,
            "sun_dist_au": sun_dist.au,
            "sun_radius": sun_radius,
            "earth_alt": earth_alt.degrees,
            "earth_az": earth_az.degrees,
            "earth_dist_km": earth_dist.km,
            "earth_radius": radius,
            "site_id": site.id,
            "geometry_source": "SPICE DE421 + MOON_ME_DE421 (Skyfield)",
        }
    )


def compute_geometry_times(site: Site, times, kernel_dir: Optional[Path] = None) -> pd.DataFrame:
    """SPICE geometry at arbitrary UTC timestamps (used for JPL Horizons cross-checks)."""
    ts, eph, pc, frame = load_kernels(kernel_dir)
    tt = pd.to_datetime(pd.Series(times))
    t = ts.utc(tt.dt.year.values, tt.dt.month.values, tt.dt.day.values,
               tt.dt.hour.values, tt.dt.minute.values, tt.dt.second.values)
    return _geometry_at(site, t, eph, pc, frame)


def moon_fixed_vectors(start_iso: str, stop_iso: str, step_hours: float, kernel_dir: Optional[Path] = None):
    """Sun and Earth position vectors (km) from the Moon centre in MOON_ME_DE421, apparent.

    Used for pole-wide lattices: one SPICE evaluation, then a cheap per-point rotation
    (`local_altaz_from_vectors`), with topocentric parallax applied exactly.
    """
    ts, eph, pc, frame = load_kernels(kernel_dir)
    hours = (pd.Timestamp(stop_iso) - pd.Timestamp(start_iso)).total_seconds() / 3600.0
    start = datetime.fromisoformat(start_iso.replace("Z", ""))
    t0 = ts.utc(start.year, start.month, start.day, start.hour, start.minute, start.second)
    t = ts.tt_jd(t0.tt + np.arange(int(hours / step_hours)) * step_hours / 24.0)
    moon = eph["moon"].at(t)
    sun = moon.observe(eph["sun"]).apparent().frame_xyz(frame).km.T
    earth = moon.observe(eph["earth"]).apparent().frame_xyz(frame).km.T
    return pd.to_datetime(t.utc_datetime()).tz_localize(None), sun, earth


def local_altaz_from_vectors(lat: float, lon: float, elev_m: float, sun_km: np.ndarray, earth_km: np.ndarray) -> pd.DataFrame:
    moon_r_km = 1737.4
    la, lo = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    site = (moon_r_km + elev_m / 1000.0) * up

    def altaz(v):
        d = v - site
        e, n, u = d @ east, d @ north, d @ up
        return np.degrees(np.arctan2(u, np.hypot(e, n))), np.degrees(np.arctan2(e, n)) % 360.0, np.linalg.norm(d, axis=1)

    sa, sz, sd = altaz(sun_km)
    ea, ez, ed = altaz(earth_km)
    return pd.DataFrame({"sun_alt": sa, "sun_az": sz, "sun_dist_au": sd / 149597870.7,
                         "sun_radius": np.degrees(np.arcsin(SUN_RADIUS_KM / sd)),
                         "earth_alt": ea, "earth_az": ez, "earth_dist_km": ed,
                         "earth_radius": np.degrees(np.arcsin(EARTH_RADIUS_KM / ed))})


def compute_geometry_range(site: Site, start_iso: str, stop_iso: str, step_hours: float,
                           kernel_dir: Optional[Path] = None) -> pd.DataFrame:
    hours = (pd.Timestamp(stop_iso) - pd.Timestamp(start_iso)).total_seconds() / 3600.0
    return compute_geometry_n(site, start_iso, int(hours / step_hours), step_hours * 60.0, kernel_dir)


def ingest_geometry_csv(path_or_buf, site_id: Optional[str] = None) -> pd.DataFrame:
    """
    Accept any precomputed geometry table (SPICE/datakit/Horizons exports).

    Required loosely: time + sun_alt + sun_az + earth_alt + earth_az.
    Optional: earth_radius, site_id, sunlit, earth_visible.
    """
    df = read_table(path_or_buf)
    fields = [
        "time_utc",
        "sun_alt",
        "sun_az",
        "earth_alt",
        "earth_az",
        "earth_radius",
        "site_id",
        "sunlit",
        "earth_visible",
    ]
    df = rename_to_canonical(
        df,
        fields,
        required=["time_utc", "sun_alt", "sun_az", "earth_alt", "earth_az"],
    )
    df["time_utc"] = parse_utc_series(df["time_utc"])
    if site_id is not None:
        if "site_id" in df.columns:
            df = df[df["site_id"].astype(str) == str(site_id)].copy()
        df["site_id"] = site_id
    if "earth_radius" not in df.columns:
        # ~0.95 deg typical from lunar surface; better than failing.
        df["earth_radius"] = 0.95
    return df.reset_index(drop=True)


def ingest_geometry_npz(path: Path, site: Site) -> pd.DataFrame:
    """
    Ingest datakit spice ephemeris_*.npz (positions from Moon centre, km)
    and convert to local alt-az at the given site.
    """
    data = np.load(path)
    utc = data["utc"]
    if isinstance(utc[0], bytes):
        utc = [u.decode() for u in utc]

    moon_r_km = 1737.4
    lat, lon = np.radians(site.lat), np.radians(site.lon)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    site_vec = (moon_r_km + site.elev_m / 1000.0) * up

    def local_angles(body_km: np.ndarray):
        topo = body_km - site_vec
        e, n, u = topo @ east, topo @ north, topo @ up
        el = np.degrees(np.arctan2(u, np.hypot(e, n)))
        az = np.degrees(np.arctan2(e, n)) % 360.0
        dist = np.linalg.norm(topo, axis=1)
        return el, az, dist

    sun_el, sun_az, _ = local_angles(data["sun_km"])
    earth_el, earth_az, earth_dist = local_angles(data["earth_km"])
    radius = np.degrees(np.arcsin(np.minimum(1.0, EARTH_RADIUS_KM / earth_dist)))

    return pd.DataFrame(
        {
            "time_utc": parse_utc_series(pd.Series(utc)),
            "sun_alt": sun_el,
            "sun_az": sun_az,
            "earth_alt": earth_el,
            "earth_az": earth_az,
            "earth_radius": radius,
            "site_id": site.id,
        }
    )


def geometry_from_any(
    site: Site,
    *,
    start_iso: Optional[str] = None,
    hours: Optional[float] = None,
    step_min: float = 60.0,
    geometry_csv=None,
    geometry_npz: Optional[Path] = None,
    kernel_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Prefer uploaded/precomputed data; fall back to live ephemeris."""
    if geometry_csv is not None:
        return ingest_geometry_csv(geometry_csv, site_id=site.id)
    if geometry_npz is not None:
        return ingest_geometry_npz(Path(geometry_npz), site)
    if start_iso is None or hours is None:
        raise ValueError("Provide geometry_csv / geometry_npz, or start_iso + hours for live compute.")
    return compute_geometry(site, start_iso, hours, step_min, kernel_dir)


def step_minutes_of(df: pd.DataFrame) -> float:
    return infer_step_minutes(df["time_utc"])
