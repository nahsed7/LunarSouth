"""All 16 project candidate sites: 4 baseline + 12 research candidates.

Research-candidate elevations are not in the source CSV; they are filled from
the LOLA 80 m DEM at the site (recorded as elev_source) when it is available.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .sites import Site, load_sites_csv

DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def _load() -> tuple:
    base = load_sites_csv(DATA / "sample_sites.csv")
    for s in base:
        s.source, s.tags = "baseline", ["baseline", "CLPS"]
    research = load_sites_csv(DATA / "research_candidate_sites.csv")
    try:
        from .rasters import product_registry

        dem = product_registry().get("LOLA_DEM_80M")
    except Exception:
        dem = None
    for s in research:
        s.source, s.tags = "research", ["research"]
        if dem is not None and (not s.elev_m):
            s.elev_m = round(float(dem.sample_latlon(s.lat, s.lon)), 1)
            s.tags.append("elev:LDEM_80S_80M")
    disc = []
    dp = DATA / "discovered_sites.csv"
    if dp.exists():
        disc = load_sites_csv(dp)
        for s in disc:
            s.source, s.tags = "discovered", ["discovered", "LunarSouth"]
    return tuple(base + research + disc + presets())


PRESET_DIRS = [DATA / "presets", DATA.parent / "extensions" / "presets"]


def presets() -> list:
    """Every CSV in data/presets/ and extensions/presets/ (drop-in, e.g. Objective-A site lists).

    Required columns (loose names): site_id, latitude, longitude. Optional: site_name, elevation, mission,
    agency, year, status, region, coord_note. Each file becomes source 'preset:<file stem>'.
    """
    out = []
    for d in PRESET_DIRS:
        for f in sorted(Path(d).glob("*.csv")):
            try:
                sites = load_sites_csv(f)
            except Exception:
                continue
            import pandas as pd

            from .rasters import global_dem, product_registry

            dem80 = product_registry().get("LOLA_DEM_80M")
            gd = global_dem()
            for s in sites:
                if not s.elev_m:
                    if dem80 is not None and dem80.covers(s.lat, s.lon):
                        s.elev_m = round(float(dem80.sample_latlon(s.lat, s.lon)), 1)
                    elif gd is not None:
                        s.elev_m = round(float(gd.sample_latlon([s.lat], [s.lon])[0]), 1)
            meta = pd.read_csv(f)
            for s, (_, r) in zip(sites, meta.iterrows()):
                s.source = f"preset:{f.stem}"
                s.tags = [str(r.get(k)) for k in ("mission", "agency", "year", "status", "region") if k in r and pd.notna(r.get(k))]
                out.append(s)
    return out


def all_candidates() -> list[Site]:
    return [Site(**s.to_dict()) for s in _load()]


def polar_candidates(max_lat: float = -79.9) -> list[Site]:
    """Every site inside the south polar cap (all products and pre-processing apply)."""
    return [s for s in all_candidates() if s.lat <= max_lat]


def project_candidates() -> list[Site]:
    return [s for s in all_candidates() if s.source in ("baseline", "research")]


def discovered() -> list[Site]:
    return [s for s in all_candidates() if s.source == "discovered"]


def baseline_ids() -> list[str]:
    return [s.id for s in _load() if s.source == "baseline"]
