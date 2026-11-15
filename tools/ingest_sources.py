#!/usr/bin/env python3
"""Reproducible ingestion of the four priority sources.

  1. NASA Moon Trek      - WMTS layer discovery for AVGVISIB_85S_060M_201608_EARTH_SP and
                           WAC_POLE_ILL_PCT_SOUTH_100M; numeric rasters pulled from the
                           archives those layers are rendered from (LOLA EXTRAS, LROC PDS).
  2. ASU LROC QuickMap   - WAC polar illumination (PDS), NAC DTM SHACKRDGE02 (PDS), with
                           LOLA GDR polar DEMs (PDS Geosciences) as the base terrain.
  3. JPL Horizons        - topocentric Sun/Earth az/el for every candidate site.
  4. CSA                 - engineering requirements recorded in data/csa_engineering_basis.json
                           (no CSA power numbers are published; see that file).

Raw downloads go to data/raw/ (never modified). Processed products go to
data/processed/. Every product gets a provenance record.

Usage:
  python tools/ingest_sources.py all          # download (cached) + process everything
  python tools/ingest_sources.py download|stats|horizons|terrain|jpl|validate
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clps_engine.rasters import RAW, latlon_to_ps, product_registry  # noqa: E402
from clps_engine.horizon_build import ray_horizon, sample_layer_on_window, terrain_window  # noqa: E402
from clps_engine.candidates import polar_candidates as all_candidates  # noqa: E402  (polar cap only)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

DOWNLOADS = [
    # (local path under data/raw, url, source, product page)
    ("lola_pds/ldem_80s_80m.img", "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_80s_80m.img",
     "PDS Geosciences / LRO LOLA", "https://pds-geosciences.wustl.edu/missions/lro/lola.htm"),
    ("lola_pds/ldem_80s_80m.lbl", "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_80s_80m.lbl", "", ""),
    ("lola_pds/ldem_75s_240m.img", "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_75s_240m.img",
     "PDS Geosciences / LRO LOLA", ""),
    ("lola_pds/ldem_75s_240m.lbl", "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_75s_240m.lbl", "", ""),
    ("lola_illum/AVGVISIB_85S_060M_201608.IMG", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608.IMG",
     "LOLA illumination EXTRAS (Mazarico et al. 2011)", "https://pgda.gsfc.nasa.gov/products/69"),
    ("lola_illum/AVGVISIB_85S_060M_201608.LBL", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608.LBL", "", ""),
    ("lola_illum/AVGVISIB_85S_060M_201608_EARTH.IMG", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608_EARTH.IMG",
     "LOLA illumination EXTRAS (Earth visibility)", "https://trek.nasa.gov/tiles/Moon/SP/AVGVISIB_85S_060M_201608_EARTH_SP/1.0.0/WMTSCapabilities.xml"),
    ("lola_illum/AVGVISIB_85S_060M_201608_EARTH.LBL", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/AVGVISIB_85S_060M_201608_EARTH.LBL", "", ""),
    ("lola_illum/LPSR_85S_060M_201608.IMG", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/LPSR_85S_060M_201608.IMG", "LOLA PSR map", ""),
    ("lola_illum/LPSR_85S_060M_201608.LBL", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/IMG/LPSR_85S_060M_201608.LBL", "", ""),
    ("lola_illum/README_ILLUMINATION_2016.TXT", "https://imbrium.mit.edu/EXTRAS/ILLUMINATION/README_ILLUMINATION_2016.TXT", "", ""),
    ("lroc_wac_illum/WAC_POLE_ILL_PCT_SOUTH_100M.IMG",
     "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_POLE/WAC_POLE_ILL_PCT_SOUTH/WAC_POLE_ILL_PCT_SOUTH_100M.IMG",
     "ASU LROC PDS (Speyerer et al. 2013)", "https://data.lroc.im-ldi.com/lroc/view_rdr_product/WAC_POLE_ILL_PCT_SOUTH_100M"),
    ("lroc_nac_dtm/NAC_DTM_SHACKRDGE02.TIF",
     "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/SHACKRDGE02/NAC_DTM_SHACKRDGE02.TIF",
     "ASU LROC PDS NAC DTM", "https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_DTM_SHACKRDGE02"),
    ("lroc_nac_dtm/NAC_DTM_SHACKRDGE02.LBL",
     "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/SHACKRDGE02/NAC_DTM_SHACKRDGE02.LBL", "", ""),
]
KERNELS = [
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/pck00008.tpc",
]
MOON_TREK_WMTS = {
    "AVGVISIB_85S_060M_201608_EARTH_SP": "https://trek.nasa.gov/tiles/Moon/SP/AVGVISIB_85S_060M_201608_EARTH_SP/1.0.0/WMTSCapabilities.xml",
    "WAC_POLE_ILL_PCT_SOUTH_100M": "https://trek.nasa.gov/tiles/Moon/SP/WAC_POLE_ILL_PCT_SOUTH_100M/1.0.0/WMTSCapabilities.xml",
}

LIMITATIONS = {
    "LOLA_DEM_80M": "Gridded from LOLA tracks; interpolated cells between tracks; ~80 m posting cannot resolve boulders/small craters. Heights relative to 1737.4 km sphere.",
    "LOLA_DEM_240M": "Coarse far-field DEM used only beyond 40 km for horizon ray casting.",
    "LOLA_AVG_SUN_VIS_60M": "18.6-yr average fraction of solar disc visible (2016 release, DEM-derived, not observed). Coverage 85S-pole full, partial to ~82.9S. Linear DEM artifacts noted by producers. Not supported by PDS/LOLA team (EXTRAS).",
    "LOLA_AVG_EARTH_VIS_60M": "18.6-yr average Earth visibility; geometric only, says nothing about RF link closure.",
    "LOLA_PSR_60M": "Binary permanent-shadow flag derived from same DEM.",
    "WAC_ILLUM_PCT_100M": "Percentage of WAC images (one lunar year, 2010-2011) in which pixel was lit; does not correct non-linear temporal sampling; coverage 88S-90S only.",
    "NAC_DTM_SHACKRDGE02": "NAC stereo DTM, 3.5 m/px, covers only the Shackleton-de Gerlache connecting ridge (-89.63 to -89.37, 209.8-239.1E). Confidence map not used.",
}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path, limit: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()


def fetch(url: str, dest: Path, timeout: int = 900) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return True


# ---------------------------------------------------------------- download
def cmd_download():
    for rel, url, src, page in DOWNLOADS:
        dest = RAW / rel
        try:
            got = fetch(url, dest)
            print(("downloaded " if got else "cached     ") + rel)
        except Exception as e:  # record limitation, do not retry forever
            print(f"FAILED {rel}: {e}")
    for url in KERNELS:
        dest = ROOT / "kernels" / url.rsplit("/", 1)[1]
        try:
            fetch(url, dest)
        except Exception as e:
            print(f"FAILED kernel {url}: {e}")
    # Moon Trek layer discovery (capabilities only - tiles are rendered PNGs, not data)
    mt = RAW / "moon_trek"
    mt.mkdir(parents=True, exist_ok=True)
    for layer, url in MOON_TREK_WMTS.items():
        try:
            fetch(url, mt / f"{layer}_WMTSCapabilities.xml", timeout=60)
        except Exception as e:
            print(f"Moon Trek capabilities {layer}: {e}")


# ---------------------------------------------------------------- statistics
def _stats(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"valid_pixels": 0}
    p = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "valid_pixels": int(v.size), "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "median": float(p[4]), "std": float(v.std()),
        "p01": float(p[0]), "p05": float(p[1]), "p10": float(p[2]), "p25": float(p[3]),
        "p75": float(p[5]), "p90": float(p[6]), "p95": float(p[7]), "p99": float(p[8]),
    }


def cmd_stats():
    reg = product_registry()
    sites = all_candidates()
    PROC.mkdir(parents=True, exist_ok=True)
    products, site_rows, prov = [], [], []
    for key, r in reg.items():
        t0 = time.time()
        if r.lines * r.samples > 1e8:  # multi-GB local GeoTIFFs: statistics from a 4000 px decimated read
            try:
                a, _, _ = r.overview(4000)
            except Exception as e:
                print(f"{key}: decimated read failed ({e}); full-product stats skipped (file damaged)")
                a = np.full((2, 2), np.nan)
        else:
            a = r.to_physical(np.asarray(r.raw))
        n_total = a.size
        s = _stats(a.ravel())
        info = r.info()
        row = {"key": key, **{k: info[k] for k in ("product_id", "source", "resolution_m", "units")},
               "grid": f"{r.lines}x{r.samples}", "total_pixels": n_total,
               "invalid_pixels": int(n_total - s["valid_pixels"]),
               "valid_pct": 100.0 * s["valid_pixels"] / n_total, **s}
        if key in ("LOLA_AVG_SUN_VIS_60M", "LOLA_AVG_EARTH_VIS_60M"):
            v = a[np.isfinite(a)]
            for t in (0.2, 0.4, 0.5, 0.6, 0.8):
                row[f"pct_area_ge_{int(t*100)}pct"] = float(100 * (v >= t).mean())
        if key == "WAC_ILLUM_PCT_100M":
            v = a[np.isfinite(a)]
            for t in (20, 40, 60, 80):
                row[f"pct_area_ge_{t}pct"] = float(100 * (v >= t).mean())
        products.append(row)
        del a
        print(f"{key}: stats in {time.time()-t0:.1f}s")
        p = RAW / ("lola_pds" if "DEM" in key and "NAC" not in key else "")
        try:
            lp = str(r.path.relative_to(ROOT))
        except ValueError:
            lp = str(r.path)
        prov.append({
            "key": key, **info, "local_path": lp,
            "download_date": datetime.fromtimestamp(r.path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d"),
            "sha256_first64MiB": sha256(r.path),
            "file_bytes": r.path.stat().st_size,
            "processing": "Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; "
                          "bilinear sampling only where all 4 neighbours valid. Raw file untouched.",
            "coordinate_transformation": "Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), "
                                         "y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.",
            "known_limitations": LIMITATIONS.get(key, ""),
        })
        # per-site statistics
        for s_ in sites:
            cov = r.covers(s_.lat, s_.lon)
            base = {"site_id": s_.id, "product": key, "resolution_m": r.scale_m, "covered": cov}
            if not cov:
                site_rows.append(base)
                continue
            for rad in (1000.0, 5000.0):
                n = max(3, int(2 * rad / r.scale_m) + 1)
                if n > 1200:
                    n = 1201
                w = sample_layer_on_window(r, s_.lat, s_.lon, rad, n, order=0)
                yy, xx = np.meshgrid(np.linspace(-rad, rad, n), np.linspace(-rad, rad, n), indexing="ij")
                inside = np.hypot(xx, yy) <= rad
                st = _stats(w[inside])
                base.update({f"r{int(rad/1000)}km_{k}": v for k, v in st.items()})
                base[f"r{int(rad/1000)}km_valid_pct"] = 100.0 * st.get("valid_pixels", 0) / inside.sum()
            base["site_value"] = float(r.sample_latlon(s_.lat, s_.lon, order=1))
            site_rows.append(base)
    pd.DataFrame(products).to_csv(PROC / "dataset_statistics.csv", index=False)
    pd.DataFrame(site_rows).to_csv(PROC / "site_product_statistics.csv", index=False)
    (PROC / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print("wrote dataset_statistics.csv, site_product_statistics.csv, provenance.json")


# ---------------------------------------------------------------- horizons
def cmd_horizons():
    reg = product_registry()
    near, far = reg["LOLA_DEM_80M"], reg.get("LOLA_DEM_240M")
    from clps_engine.horizon_build import near_field_from_grid, near_field_from_raster
    from clps_engine.terrain import load_terrain_product

    nac = reg.get("NAC_DTM_SHACKRDGE02")
    rows, meta = [], []
    for s in all_candidates():
        z = float(near.sample_latlon(s.lat, s.lon))
        h = ray_horizon(s.lat, s.lon, near, far, observer_height_m=2.0, site_elev_m=z)
        # High-resolution near field (<= ~1 km): NAC 3.5 m (QuickMap) or legacy 5/20 m LOLA grids.
        nf_src = "none (80 m only)"
        nf = None
        if nac is not None and nac.covers(s.lat, s.lon):
            zn = float(nac.sample_latlon(s.lat, s.lon))
            nf, cov = near_field_from_raster(s.lat, s.lon, nac, h.azimuth_deg.values, 1500.0, zn + 2.0)
            nf_src = "NAC_DTM_SHACKRDGE02 3.5 m, 0-1.5 km"
        elif reg.get("LDEM_5M_87S") is not None and reg["LDEM_5M_87S"].covers(s.lat, s.lon) and np.isfinite(reg["LDEM_5M_87S"].sample_latlon(s.lat, s.lon)):
            d5 = reg["LDEM_5M_87S"]
            nf, cov = near_field_from_raster(s.lat, s.lon, d5, h.azimuth_deg.values, 1000.0, float(d5.sample_latlon(s.lat, s.lon)) + 2.0)
            nf_src = "LDEM_87S_5MPP 5 m (PGDA), 0-1 km"
        elif reg.get("LDEM_20M") is not None and np.isfinite(reg["LDEM_20M"].sample_latlon(s.lat, s.lon)):
            d20 = reg["LDEM_20M"]
            nf, cov = near_field_from_raster(s.lat, s.lon, d20, h.azimuth_deg.values, 2000.0, float(d20.sample_latlon(s.lat, s.lon)) + 2.0)
            nf_src = "LDEM_80S_20MPP 20 m (PGDA), 0-2 km"
        else:
            for d in (ROOT / "data" / "terrain", ROOT / "derived" / "model3_research" / "terrain"):
                tr = load_terrain_product(d, s.id.lower())
                if tr is not None:
                    nf, cov = near_field_from_grid(tr, h.azimuth_deg.values, z, z + 2.0)
                    nf_src = f"{tr['meta'].get('dem_file','legacy grid')} {tr['meta']['metres_per_pixel']:.0f} m, 0-1 km"
                    break
        h["horizon_80m_deg"] = h["horizon_elev_deg"]
        if nf is not None:
            h["near_field_deg"] = nf
            h["horizon_elev_deg"] = np.maximum(h["horizon_elev_deg"], nf)
        h["near_field_source"] = nf_src
        h.insert(0, "site_id", s.id)
        rows.append(h)
        meta.append({"site_id": s.id, "dem_site_elev_m": z, "catalog_elev_m": s.elev_m,
                     "horizon_min_deg": h.horizon_elev_deg.min(), "horizon_max_deg": h.horizon_elev_deg.max(),
                     "horizon_mean_deg": h.horizon_elev_deg.mean(),
                     "ray_valid_pct_min": h.ray_valid_pct.min(), "near_field_source": nf_src,
                     "near_field_raises_mean_deg": float((h.horizon_elev_deg - h.horizon_80m_deg).mean())})
        print(f"{s.id}: z={z:.1f} m horizon {h.horizon_elev_deg.min():.2f}..{h.horizon_elev_deg.max():.2f} deg")
    out = PROC / "horizons"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out / "horizon_profiles_lola.csv", index=False, float_format="%.5f")
    pd.DataFrame(meta).to_csv(out / "horizon_summary.csv", index=False)
    (out / "horizon_profiles_lola.csv.provenance.json").write_text(json.dumps({
        "products": ["LDEM_80S_80M (0.08-40 km)", "LDEM_75S_240M (40-200 km)"],
        "method": "ray casting, 0.5 deg azimuth, spherical curvature, observer 2.0 m above DEM surface",
        "azimuth_convention": "clockwise from local north (north = direction away from south pole)",
        "created": now_iso(),
        "limitations": "Near-field obstacles below ~160 m scale not resolved; map-plane distance used for range "
                       "(PS scale error <0.6% inside 85S); legacy LM1 mask (10 km range) retained for comparison.",
    }, indent=2))


# ---------------------------------------------------------------- terrain
def slope_deg(h, mpp):
    gy, gx = np.gradient(h.astype("float64"), mpp)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def cmd_terrain():
    reg = product_registry()
    dem = reg["LOLA_DEM_80M"]
    nac = reg.get("NAC_DTM_SHACKRDGE02")
    out = PROC / "terrain3d"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    layer_keys = [k for k in ("LOLA_AVG_SUN_VIS_60M", "LOLA_AVG_EARTH_VIS_60M", "WAC_ILLUM_PCT_100M", "LOLA_PSR_60M") if k in reg]
    for s in all_candidates():
        # 3D scene: 24 km x 24 km at 80 m (native LOLA posting), 301 x 301
        w = terrain_window(dem, s.lat, s.lon, 12000.0, out_px=301)
        layers = {k: sample_layer_on_window(reg[k], s.lat, s.lon, 12000.0, 301, order=1).astype("float32") for k in layer_keys}
        np.savez_compressed(out / f"{s.id}_scene.npz", height_m=w["height"], x_m=w["x_m"], y_m=w["y_m"],
                            slope_deg=slope_deg(w["height"], w["mpp"]).astype("float32"),
                            **{k.lower(): v for k, v in layers.items()})
        # Model 3 at a COMMON comparison resolution (LOLA 80 m) for every site
        c = terrain_window(dem, s.lat, s.lon, 2000.0)
        sl = slope_deg(c["height"], c["mpp"])
        yy, xx = np.meshgrid(c["y_m"], c["x_m"], indexing="ij")
        r = np.hypot(xx, yy)
        for rad in (1000.0, 2000.0):
            m = (r <= rad) & np.isfinite(sl)
            v, hv = sl[m], c["height"][m]
            row = {"site_id": s.id, "product": "LDEM_80S_80M", "resolution_m": 80.0, "radius_m": rad,
                   "valid_pct": 100.0 * m.sum() / (r <= rad).sum(), "n_cells": int(m.sum()),
                   "elev_site_m": float(dem.sample_latlon(s.lat, s.lon)),
                   "elev_mean_m": float(hv.mean()), "elev_std_m": float(hv.std()),
                   "relief_m": float(hv.max() - hv.min()),
                   "slope_mean_deg": float(v.mean()), "slope_median_deg": float(np.median(v)),
                   "slope_p90_deg": float(np.percentile(v, 90)), "slope_max_deg": float(v.max())}
            for t in (5, 10, 15, 20):
                row[f"pct_slope_le_{t}deg"] = float(100 * (v <= t).mean())
            rows.append(row)
        # Supplied PGDA 20 m / 5 m DEMs (local archive) at native resolution
        from clps_engine.mission import live_terrain_metrics
        lt = live_terrain_metrics(s)
        lt = lt[lt["product"].isin(["LDEM_80S_20M", "LDEM_87S_5M", "Site01_5MPP"])]
        rows.extend(lt.to_dict("records"))
        # NAC 3.5 m where it covers the site (QuickMap high-resolution upgrade)
        if nac is not None and nac.covers(s.lat, s.lon):
            cn = terrain_window(nac, s.lat, s.lon, 1000.0)
            sln = slope_deg(cn["height"], cn["mpp"])
            yy, xx = np.meshgrid(cn["y_m"], cn["x_m"], indexing="ij")
            m = (np.hypot(xx, yy) <= 1000.0) & np.isfinite(sln)
            v, hv = sln[m], cn["height"][m]
            row = {"site_id": s.id, "product": "NAC_DTM_SHACKRDGE02", "resolution_m": nac.scale_m, "radius_m": 1000.0,
                   "valid_pct": 100.0 * m.sum() / (np.hypot(xx, yy) <= 1000.0).sum(), "n_cells": int(m.sum()),
                   "elev_site_m": float(nac.sample_latlon(s.lat, s.lon)),
                   "elev_mean_m": float(hv.mean()), "elev_std_m": float(hv.std()), "relief_m": float(hv.max() - hv.min()),
                   "slope_mean_deg": float(v.mean()), "slope_median_deg": float(np.median(v)),
                   "slope_p90_deg": float(np.percentile(v, 90)), "slope_max_deg": float(v.max())}
            for t in (5, 10, 15, 20):
                row[f"pct_slope_le_{t}deg"] = float(100 * (v <= t).mean())
            # roughness: std of detrended 3x3 neighbourhood at native 3.5 m
            from scipy import ndimage
            hh = np.where(np.isfinite(cn["height"]), cn["height"], np.nan)
            mu = ndimage.uniform_filter(np.nan_to_num(hh), 3)
            mu2 = ndimage.uniform_filter(np.nan_to_num(hh) ** 2, 3)
            rough = np.sqrt(np.clip(mu2 - mu ** 2, 0, None))[m]
            row["roughness_median_m"] = float(np.median(rough))
            row["roughness_p90_m"] = float(np.percentile(rough, 90))
            rows.append(row)
            # Same footprint degraded to 80 m: quantifies resolution effect
            np.savez_compressed(out / f"{s.id}_nac_1km.npz", height_m=cn["height"], x_m=cn["x_m"], y_m=cn["y_m"],
                                slope_deg=sln.astype("float32"))
        print(f"{s.id}: terrain done")
    pd.DataFrame(rows).to_csv(PROC / "terrain_metrics_all_sites.csv", index=False)
    # Regional overview scene: 300 km x 300 km around the pole at 1 km sampling (from 240 m DEM)
    d240 = reg["LOLA_DEM_240M"]
    u = np.linspace(-150000, 150000, 301)
    X, Y = np.meshgrid(u, u[::-1])
    H = d240.sample_xy(X, Y)
    np.savez_compressed(out / "regional_scene.npz", height_m=H.astype("float32"), x_m=u, y_m=u[::-1],
                        sun_vis=reg["LOLA_AVG_SUN_VIS_60M"].sample_xy(X, Y).astype("float32"),
                        earth_vis=reg["LOLA_AVG_EARTH_VIS_60M"].sample_xy(X, Y).astype("float32"))
    print("wrote terrain_metrics_all_sites.csv and 3D scenes")


# ---------------------------------------------------------------- JPL Horizons
HZ_API = "https://ssd.jpl.nasa.gov/api/horizons.api"


def horizons_query(site, body: str, start: str, stop: str, step: str) -> dict:
    params = {
        "format": "json", "COMMAND": f"'{body}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'coord@301'", "COORD_TYPE": "'GEODETIC'",
        "SITE_COORD": f"'{s_lon(site.lon)},{site.lat},{site.elev_m/1000.0:.5f}'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'",
        "QUANTITIES": "'4,20'", "APPARENT": "'AIRLESS'", "CSV_FORMAT": "'YES'",
        "ANG_FORMAT": "'DEG'", "TIME_TYPE": "'UT'", "EXTRA_PREC": "'YES'",
    }
    url = HZ_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        js = json.loads(r.read().decode())
    js["_query_url"] = url
    js["_query_params"] = params
    return js


def s_lon(lon):
    return lon % 360.0


def parse_horizons(result: str) -> pd.DataFrame:
    body = result.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    rows = []
    for ln in body:
        p = [x.strip() for x in ln.split(",")]
        # date, solar-presence, interfering, az, el, delta(AU), deldot
        rows.append({"time_utc": pd.to_datetime(p[0], format="%Y-%b-%d %H:%M"), "az_deg": float(p[3]),
                     "el_deg": float(p[4]), "range_au": float(p[5])})
    return pd.DataFrame(rows)


def cmd_jpl(start="2026-10-15 00:00", stop="2026-11-14 00:00", step="2 h"):
    from clps_engine.geometry import compute_geometry_times

    raw = RAW / "jpl_horizons"
    raw.mkdir(parents=True, exist_ok=True)
    comp = []
    for s in all_candidates():
        dfs = {}
        for body, name in (("10", "sun"), ("399", "earth")):
            f = raw / f"{s.id}_{name}_{start[:10]}_{stop[:10]}.json"
            if not f.exists():
                try:
                    f.write_text(json.dumps(horizons_query(s, body, start, stop, step)))
                except Exception as e:
                    print(f"Horizons {s.id} {name} failed: {e}")
                    continue
            js = json.loads(f.read_text())
            dfs[name] = parse_horizons(js["result"])
        if len(dfs) < 2:
            continue
        t = dfs["sun"]["time_utc"]
        sp = compute_geometry_times(s, t)
        d = pd.DataFrame({
            "site_id": s.id, "time_utc": t,
            "hz_sun_az": dfs["sun"]["az_deg"], "hz_sun_el": dfs["sun"]["el_deg"],
            "hz_earth_az": dfs["earth"]["az_deg"].values, "hz_earth_el": dfs["earth"]["el_deg"].values,
            "hz_earth_range_km": dfs["earth"]["range_au"].values * 149597870.7,
            "spice_sun_az": sp["sun_az"].values, "spice_sun_el": sp["sun_alt"].values,
            "spice_earth_az": sp["earth_az"].values, "spice_earth_el": sp["earth_alt"].values,
            "spice_earth_range_km": sp["earth_dist_km"].values,
        })
        comp.append(d)
        print(f"{s.id}: Horizons vs SPICE sun dEl max {np.abs(d.hz_sun_el-d.spice_sun_el).max():.4f} deg")
    out = PROC / "jpl_horizons"
    out.mkdir(parents=True, exist_ok=True)
    allc = pd.concat(comp)
    allc.to_csv(out / "horizons_vs_spice.csv", index=False, float_format="%.6f")
    # Store Horizons geometry in the engine's canonical geometry schema for reuse
    geo = allc.rename(columns={"hz_sun_az": "sun_az", "hz_sun_el": "sun_alt", "hz_earth_az": "earth_az",
                               "hz_earth_el": "earth_alt"})[["site_id", "time_utc", "sun_alt", "sun_az", "earth_alt", "earth_az"]]
    geo.to_csv(out / "horizons_geometry_all_sites.csv", index=False, float_format="%.6f")


def angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- validation
def cmd_validate():
    """Long-term illumination / Earth visibility vs. LOLA and WAC products, + Horizons."""
    from clps_engine.geometry import compute_geometry_range
    from clps_engine.horizon import load_horizon_table, mask_for_site
    from clps_engine.solar import apply_solar_visibility
    from clps_engine.dte import apply_dte_visibility

    reg = product_registry()
    hz = load_horizon_table(PROC / "horizons" / "horizon_profiles_lola.csv")
    sp = pd.read_csv(PROC / "site_product_statistics.csv")
    rows = []
    for s in all_candidates():
        mask = mask_for_site(hz, s.id)
        # 18.6-yr nodal cycle sampled every 6 h, matching the AVGVISIB averaging period length
        g = compute_geometry_range(s, "2009-07-01T00:00:00", "2028-02-01T00:00:00", step_hours=6.0)
        g = apply_solar_visibility(g, mask, sun_disc_radius_deg=0.0)
        g = apply_dte_visibility(g, mask, "centre")
        g1 = g[(g.time_utc >= "2010-03-01") & (g.time_utc < "2011-03-01")]  # WAC-era lunar year approx
        row = {"site_id": s.id,
               "model_sun_disc_vis_18p6yr": float(g["sun_disc_fraction"].mean()),
               "model_sun_centre_vis_18p6yr": float(g["sunlit"].mean()),
               "model_earth_disc_vis_18p6yr": float(g["earth_disc_fraction"].mean()),
               "model_sun_centre_vis_2010_2011": float(g1["sunlit"].mean())}
        for key, col in (("LOLA_AVG_SUN_VIS_60M", "lola_sun_vis"), ("LOLA_AVG_EARTH_VIS_60M", "lola_earth_vis"),
                         ("WAC_ILLUM_PCT_100M", "wac_illum_pct")):
            r = sp[(sp.site_id == s.id) & (sp["product"] == key)]
            if len(r) and bool(r["covered"].iloc[0]):
                row[col] = float(r["site_value"].iloc[0])
                row[col + "_r1km_p10"] = float(r.get("r1km_p10", pd.Series([np.nan])).iloc[0])
                row[col + "_r1km_p90"] = float(r.get("r1km_p90", pd.Series([np.nan])).iloc[0])
            else:
                row[col] = np.nan
        rows.append(row)
        print(f"{s.id}: model sun {row['model_sun_disc_vis_18p6yr']:.3f} vs LOLA {row.get('lola_sun_vis')}")
    v = pd.DataFrame(rows)
    v["sun_vis_diff_model_minus_lola"] = v["model_sun_disc_vis_18p6yr"] - v["lola_sun_vis"]
    v["earth_vis_diff_model_minus_lola"] = v["model_earth_disc_vis_18p6yr"] - v["lola_earth_vis"]
    v["sun_vis_diff_model2010_minus_wac"] = 100 * v["model_sun_centre_vis_2010_2011"] - v["wac_illum_pct"]
    v.to_csv(PROC / "illumination_validation.csv", index=False)
    print("wrote illumination_validation.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"download": cmd_download, "stats": cmd_stats, "horizons": cmd_horizons, "terrain": cmd_terrain,
             "jpl": cmd_jpl, "validate": cmd_validate}
    if cmd == "all":
        for k in ("download", "stats", "horizons", "terrain", "jpl", "validate"):
            print(f"===== {k}")
            steps[k]()
    else:
        steps[cmd]()
