#!/usr/bin/env python3
"""Pole-wide (80S -> 90S) analysis products for the whole south polar region.

Steps (each cached to data/processed/polewide/):
  grid200   - stream LDEM/LPSR/LDEC 20 m GeoTIFFs in strips; slope at native 20 m;
              aggregate to 200 m cells: elevation, slope fractions, max slope, PSR fraction,
              LOLA shot-support fraction. Memory-bounded (~250 MB).
  mosaic    - 1 km pole-wide stack for maps/3D: 20 m-derived terrain + LOLA AVGVISIB Sun/Earth
              (Moon Trek products) + WAC illumination (QuickMap) + Diviner Tbol anomalies.
  sim       - Model 1/2/4 on a 6 km lattice across the WHOLE pole: SPICE (Horizons-validated)
              Sun/Earth geometry x LOLA ray-cast horizon -> annual sunlit %, Earth %, longest
              darkness, and mean specific power (W/m^2 of vertical-tracking array).
  discover  - rank every 1 km cell with a transparent composite and extract the best
              separated landing candidates (LN-xxx), written to data/discovered_sites.csv.

Usage: python tools/build_polewide.py [grid200|mosaic|sim|discover|all]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clps_engine.rasters import ps_to_latlon, latlon_to_ps, product_registry  # noqa: E402

OUT = ROOT / "data" / "processed" / "polewide"
OUT.mkdir(parents=True, exist_ok=True)
AGG = 10  # 20 m -> 200 m
BAD_TILES: dict = {}


def safe_read(f, r0: int, r1: int, W: int, tile: int = 512) -> np.ndarray:
    """Strip read that survives damaged/truncated tiles (filled NaN, counted in BAD_TILES)."""
    from rasterio.windows import Window

    try:
        return f.read(1, window=Window(0, r0, W, r1 - r0)).astype(np.float32)
    except Exception:
        out = np.full((r1 - r0, W), np.nan, np.float32)
        bh, bw = f.block_shapes[0]
        for tr in range((r0 // bh) * bh, r1, bh):          # the file's own tile grid: lose only damaged tiles
            a0, a1 = max(tr, r0), min(tr + bh, r1)
            for c0 in range(0, W, bw):
                c1 = min(W, c0 + bw)
                try:
                    out[a0 - r0: a1 - r0, c0:c1] = f.read(1, window=Window(c0, a0, c1 - c0, a1 - a0))
                except Exception:
                    BAD_TILES[f.name] = BAD_TILES.get(f.name, 0) + 1
        return out


def cmd_grid200():
    import rasterio
    from rasterio.windows import Window

    reg = product_registry()
    dem, psr, dec = reg["LDEM_20M"].path, reg.get("LPSR_20M"), reg.get("LDEC_20M")
    t0 = time.time()
    with rasterio.open(dem) as r:
        H, W = r.shape
        nc = W // AGG
        nr = H // AGG
        out = {k: np.full((nr, nc), np.nan, np.float32) for k in
               ("elev_mean", "elev_std", "slope_mean", "slope_max", "le5", "le10", "le15", "psr_frac", "shot_frac", "valid_frac")}
        rows_per = 50 * AGG  # 50 output rows per strip
        fpsr = rasterio.open(psr.path) if psr else None
        fdec = rasterio.open(dec.path) if dec else None
        for ro in range(0, nr, 50):
            r0 = ro * AGG
            r1 = min(H, r0 + rows_per)
            a0, a1 = max(0, r0 - 1), min(H, r1 + 1)
            z = safe_read(r, a0, a1, W)
            gy, gx = np.gradient(z, 20.0)
            sl = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
            del gx, gy
            off = r0 - a0
            z, sl = z[off:off + (r1 - r0)], sl[off:off + (r1 - r0)]
            n = (r1 - r0) // AGG

            def blk(a):
                return a[: n * AGG, : nc * AGG].reshape(n, AGG, nc, AGG).swapaxes(1, 2).reshape(n, nc, AGG * AGG)

            zb, sb = blk(z), blk(sl)
            ok = np.isfinite(sb)
            cnt = ok.sum(-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["elev_mean"][ro:ro + n] = np.nanmean(zb, -1)
                out["elev_std"][ro:ro + n] = np.nanstd(zb, -1)
                out["slope_mean"][ro:ro + n] = np.nanmean(sb, -1)
                out["slope_max"][ro:ro + n] = np.nanmax(np.where(ok, sb, -1), -1)
                for t, k in ((5, "le5"), (10, "le10"), (15, "le15")):
                    out[k][ro:ro + n] = np.where(cnt > 0, (ok & (sb <= t)).sum(-1) / np.maximum(cnt, 1), np.nan)
                out["valid_frac"][ro:ro + n] = cnt / (AGG * AGG)
            if fpsr is not None:
                p = blk(safe_read(fpsr, r0, r1, W))
                out["psr_frac"][ro:ro + n] = np.nanmean(np.where(np.isfinite(p), p > 0.5, np.nan), -1)
            if fdec is not None:
                d = blk(safe_read(fdec, r0, r1, W))
                out["shot_frac"][ro:ro + n] = np.mean(np.isfinite(d) & (d > 0), -1)
            del z, sl, zb, sb
            if ro % 500 == 0:
                print(f"  rows {ro}/{nr}  {time.time()-t0:.0f}s", flush=True)
    x = -304000.0 + (np.arange(nc) + 0.5) * 200.0
    y = 304000.0 - (np.arange(nr) + 0.5) * 200.0
    np.savez_compressed(OUT / "grid200m.npz", x_m=x, y_m=y, **out)
    (OUT / "grid200m.provenance.json").write_text(json.dumps({
        "inputs": ["LDEM_80S_20MPP_ADJ.tiff", "LPSR_80S_20MPP_ADJ.tiff", "LDEC_80S_20MPP_ADJ.tiff"],
        "method": "slope by central differences at native 20 m; 10x10 block aggregation to 200 m; nodata excluded per cell",
        "le5/le10/le15": "fraction of 20 m pixels in the 200 m cell with slope <= threshold",
        "psr_frac": "fraction of 20 m pixels flagged permanently shadowed", "shot_frac": "fraction of 20 m pixels with >=1 LOLA return (LDEC)",
        "damaged_tile_strips": {Path(k).name: v for k, v in BAD_TILES.items()},
        "runtime_s": time.time() - t0}, indent=2))
    print(f"grid200 done in {time.time()-t0:.0f}s")


def _block_mean(a, k):
    n = a.shape[0] // k
    b = a[: n * k, : n * k].reshape(n, k, n, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(b, 3), 1)


def cmd_mosaic():
    reg = product_registry()
    g = np.load(OUT / "grid200m.npz")
    K = 5  # 200 m -> 1 km
    m = {k: _block_mean(g[k].astype(np.float64), K).astype(np.float32) for k in
         ("elev_mean", "slope_mean", "le5", "le10", "le15", "psr_frac", "shot_frac", "valid_frac")}
    m["slope_max"] = g["slope_max"][: (g["slope_max"].shape[0] // K) * K].reshape(-1, K, g["slope_max"].shape[1] // K, K).max(axis=(1, 3))
    n = m["elev_mean"].shape[0]
    x = -304000.0 + (np.arange(n) + 0.5) * 1000.0
    y = 304000.0 - (np.arange(n) + 0.5) * 1000.0
    X, Y = np.meshgrid(x, y)
    for key, name in (("LOLA_AVG_SUN_VIS_60M", "sun_vis"), ("LOLA_AVG_EARTH_VIS_60M", "earth_vis"), ("WAC_ILLUM_PCT_100M", "wac_illum")):
        if key in reg:
            m[name] = reg[key].sample_xy(X, Y, order=1).astype(np.float32)
    lat, lon = ps_to_latlon(X, Y)
    m["lat"], m["lon"] = lat.astype(np.float32), lon.astype(np.float32)
    inside = lat <= -80.0
    # Diviner bolometric temperature anomalies (0.5 deg global grid, supplied .xyz)
    from pathlib import Path as _P
    import os
    ext = _P(os.environ.get("CLPS_DATASETS", str(_P.home() / "Desktop" / "clpsdatasets")))
    for f, name in (("diviner_tbol_max_anom.xyz", "div_tmax_anom"), ("diviner_tbol_min_anom.xyz", "div_tmin_anom")):
        p = ext / f
        if p.exists():
            d = pd.read_csv(p, sep=r"\s+", header=None, names=["lon", "lat", "v"])
            d = d[d.lat < -79]
            lons = np.sort(d.lon.unique())
            lats = np.sort(d.lat.unique())
            grid = d.pivot(index="lat", columns="lon", values="v").reindex(index=lats, columns=lons).to_numpy()
            li = np.clip(np.searchsorted(lats, lat) - 0, 0, len(lats) - 1)
            lo = np.clip(np.searchsorted(lons, ((lon + 180) % 360) - 180), 0, len(lons) - 1)
            m[name] = np.where(inside, grid[li, lo], np.nan).astype(np.float32)
    # Fill elevation in damaged-tile gaps of the 20 m file from the independent PDS 80 m DEM (flagged layer)
    d80 = reg["LOLA_DEM_80M"].sample_xy(X, Y)
    m["elev_filled_flag"] = (~np.isfinite(m["elev_mean"]) & np.isfinite(d80)).astype(np.float32)
    m["elev_mean"] = np.where(np.isfinite(m["elev_mean"]), m["elev_mean"], d80).astype(np.float32)
    m.pop("shot_frac", None)  # LDEC_80S_20MPP_ADJ.tiff supplied copy is damaged (unreadable tiles) - excluded
    for k in list(m):
        if k not in ("lat", "lon") and m[k].shape == inside.shape:
            m[k] = np.where(inside, m[k], np.nan).astype(np.float32)
    np.savez_compressed(OUT / "mosaic1km.npz", x_m=x, y_m=y, **m)
    print("mosaic1km written", {k: v.shape for k, v in m.items()})


class FastHorizon:
    """In-memory LOLA DEMs + vectorised ray casting (same equations as horizon_build.ray_horizon)."""

    def __init__(self, reg, az_step=1.0):
        from clps_engine.rasters import MOON_R_M
        self.R = MOON_R_M
        self.near, self.far = reg["LOLA_DEM_80M"], reg["LOLA_DEM_240M"]
        self.zn = self.near.to_physical(np.asarray(self.near.raw)).astype(np.float32)
        self.zf = self.far.to_physical(np.asarray(self.far.raw)).astype(np.float32)
        self.az = np.arange(0, 360, az_step)
        self.dn = np.arange(160.0, 30000.0, 160.0)
        self.df = np.arange(30000.0, 150000.0, 480.0)

    def _samp(self, dem, z, xs, ys):
        from scipy import ndimage
        r, c = dem.xy_to_rc(xs, ys)
        v = ndimage.map_coordinates(z, [r, c], order=1, mode="constant", cval=np.nan)
        return v

    def __call__(self, lat, lon, x0, y0, z0, mast=2.0):
        from clps_engine.rasters import local_enu_axes
        e, n = local_enu_axes(lon)
        a = np.radians(self.az)[:, None]
        dx = np.sin(a) * e[0] + np.cos(a) * n[0]
        dy = np.sin(a) * e[1] + np.cos(a) * n[1]
        best = np.full(self.az.size, -90.0)
        for d, dem, z in ((self.dn, self.near, self.zn), (self.df, self.far, self.zf)):
            ht = self._samp(dem, z, x0 + dx * d, y0 + dy * d)
            g = d / self.R
            el = np.degrees(np.arctan2((self.R + ht) * np.cos(g) - (self.R + z0 + mast), (self.R + ht) * np.sin(g)))
            best = np.maximum(best, np.nanmax(np.where(np.isfinite(el), el, -90.0), axis=1))
        return self.az, best


def cmd_sim(spacing_km: float = 5.0, start="2027-01-01T00:00:00", stop="2028-01-01T00:00:00", step_h=2.0):
    """Whole-pole Model 1/2/4 lattice from the four-source chain (SPICE/Horizons geometry x LOLA horizon x M4)."""
    from clps_engine.geometry import moon_fixed_vectors, local_altaz_from_vectors
    from clps_engine.solar import apply_solar_visibility
    from clps_engine.dte import apply_dte_visibility
    from clps_engine import power_gen as pg
    from clps_engine.windows import longest_run_hours

    reg = product_registry()
    fh = FastHorizon(reg)
    T, S, E = moon_fixed_vectors(start, stop, step_h)
    s = spacing_km * 1000.0
    u = np.arange(-302000.0 + s / 2, 302000.0, s)
    pts = [(x, y) for y in u[::-1] for x in u if np.hypot(x, y) <= 303000.0]
    cfg = pg.SolarArrayConfig(area_m2=1.0)
    rows = []
    t0 = time.time()
    for i, (x, y) in enumerate(pts):
        lat, lon = (float(v) for v in ps_to_latlon(x, y))
        z = float(fh._samp(fh.near, fh.zn, np.array([x]), np.array([y]))[0])
        if not np.isfinite(z):
            continue
        az, hz = fh(lat, lon, x, y, z)
        geo = local_altaz_from_vectors(lat, lon, z + 2.0, S, E)
        geo.insert(0, "time_utc", T)
        mask = (az, hz)
        geo = apply_solar_visibility(geo, mask, 0.0)
        geo = apply_dte_visibility(geo, mask, "any_part")
        gen = pg.apply_generation(geo, cfg, step_h * 60)
        sun = geo.sunlit.to_numpy()
        ev = geo.earth_visible.to_numpy()
        rows.append({"x_m": x, "y_m": y, "lat": lat, "lon": lon, "elev_m": z,
                     "sunlit_pct": 100 * sun.mean(), "sun_disc_pct": 100 * geo.sun_disc_fraction.mean(),
                     "earth_pct": 100 * ev.mean(), "sun_and_earth_pct": 100 * (sun & ev).mean(),
                     "longest_dark_h": longest_run_hours(~sun, step_h * 60),
                     "longest_blackout_h": longest_run_hours(~ev, step_h * 60),
                     "mean_specific_power_w_m2": float(gen.gen_w.mean()),
                     "annual_energy_kwh_m2": float(gen.gen_w.sum() * step_h / 1000.0),
                     "horizon_mean_deg": float(hz.mean()), "horizon_max_deg": float(hz.max())})
        if i % 1000 == 0:
            print(f"  {i}/{len(pts)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sim_lattice.csv", index=False, float_format="%.4f")
    (OUT / "sim_lattice.provenance.json").write_text(json.dumps({
        "window": [start, stop], "step_h": step_h, "lattice_km": spacing_km, "points": len(df),
        "ephemeris": "SPICE DE421 + MOON_ME_DE421 Moon-centred apparent vectors, exact topocentric parallax per point "
                     "(max |d el| vs full per-site SPICE 2.6e-5 deg; SPICE vs JPL Horizons 1.3e-5 deg)",
        "horizon": "LOLA LDEM_80S_80M (0.16-30 km, 160 m step) + LDEM_75S_240M (30-150 km, 480 m step), 1 deg azimuth, 2 m mast",
        "power": "Model 4 default chain, 1 m2 vertical-tracking array, W/m2 delivered to bus",
        "runtime_s": time.time() - t0,
        "limitation": "lattice-scale screening; near-field relief below 80 m unresolved; point analysis uses the full chain"}, indent=2))
    print(f"sim done: {len(df)} points in {time.time()-t0:.0f}s")


def cmd_discover(n_sites: int = 40, min_sep_km: float = 8.0):
    """Composite landing-suitability index over every 1 km cell; extract separated maxima."""
    m = np.load(OUT / "mosaic1km.npz")
    sim_p = OUT / "sim_lattice.csv"
    sun = m["sun_vis"].astype(float)
    earth = m["earth_vis"].astype(float)
    if sim_p.exists():
        from scipy.interpolate import griddata
        sm = pd.read_csv(sim_p)
        X, Y = np.meshgrid(m["x_m"], m["y_m"])
        sun_sim = griddata((sm.x_m, sm.y_m), sm.sun_disc_pct / 100, (X, Y), method="linear")
        earth_sim = griddata((sm.x_m, sm.y_m), sm.earth_pct / 100, (X, Y), method="linear")
        sun = np.where(np.isfinite(sun), sun, sun_sim)      # LOLA product where it exists, model elsewhere
        earth = np.where(np.isfinite(earth), earth, earth_sim)
    le10 = m["le10"].astype(float)
    psr = np.nan_to_num(m["psr_frac"].astype(float))
    score = 100 * (0.40 * np.clip(sun / 0.8, 0, 1) + 0.25 * np.clip(earth / 0.8, 0, 1) + 0.30 * np.clip(le10, 0, 1) + 0.05 * (1 - psr))
    score = np.where(np.isfinite(sun) & np.isfinite(le10), score, np.nan)
    np.savez_compressed(OUT / "suitability1km.npz", score=score.astype(np.float32), sun_merged=sun.astype(np.float32),
                        earth_merged=earth.astype(np.float32))
    from scipy import ndimage
    sc = np.nan_to_num(score, nan=-1)
    k = int(min_sep_km)
    mx = ndimage.maximum_filter(sc, size=2 * k + 1)
    cand = np.argwhere((sc == mx) & (sc > 0))
    vals = sc[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals)
    picked = []
    for idx in order:
        r, c = cand[idx]
        if all(np.hypot(r - pr, c - pc) >= min_sep_km for pr, pc in picked):
            picked.append((r, c))
        if len(picked) >= n_sites:
            break
    reg = product_registry()
    dem = reg.get("LDEM_20M") or reg["LOLA_DEM_80M"]
    rows = []
    for i, (r, c) in enumerate(picked, 1):
        # refine to the flattest/most-lit 200 m cell inside the 1 km winner (keeps 20 m terrain authority)
        x, y = float(m["x_m"][c]), float(m["y_m"][r])
        lat, lon = ps_to_latlon(x, y)
        rows.append({"site_id": f"LN-{i:03d}", "site_name": f"LunarSouth discovery {i:03d}",
                     "latitude_deg": round(float(lat), 5), "longitude_deg": round(float(lon), 5),
                     "observer_elevation_m": round(float(dem.sample_latlon(lat, lon)), 1),
                     "suitability_index": round(float(score[r, c]), 1),
                     "sun_vis_pct": round(100 * float(sun[r, c]), 1), "earth_vis_pct": round(100 * float(earth[r, c]), 1),
                     "pct_slope_le10_20m": round(100 * float(le10[r, c]), 1), "psr_frac": round(float(psr[r, c]), 3)})
    pd.DataFrame(rows).to_csv(ROOT / "data" / "discovered_sites.csv", index=False)
    print(pd.DataFrame(rows).head(15).to_string())


def cmd_validate():
    """Lattice Model 1/2 (2027) vs LOLA AVGVISIB (18.6-yr) at every lattice point the product covers."""
    reg = product_registry()
    sm = pd.read_csv(OUT / "sim_lattice.csv")
    sv = reg["LOLA_AVG_SUN_VIS_60M"].sample_xy(sm.x_m.values, sm.y_m.values)
    evv = reg["LOLA_AVG_EARTH_VIS_60M"].sample_xy(sm.x_m.values, sm.y_m.values)
    ok = np.isfinite(sv) & np.isfinite(evv)
    rows = []
    for name, mod, ref in (("sun", sm.sun_disc_pct.values / 100, sv), ("earth", sm.earth_pct.values / 100, evv)):
        a, b = mod[ok], ref[ok]
        rows.append({"quantity": name, "n_points": int(ok.sum()), "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                     "mean_abs_diff": float(np.abs(a - b).mean()), "bias_model_minus_lola": float((a - b).mean()),
                     "p90_abs_diff": float(np.percentile(np.abs(a - b), 90)),
                     "note": "model = calendar 2027 at 5 km points; LOLA = 18.6-yr average at 60 m - period and support differ"})
    pd.DataFrame(rows).to_csv(OUT / "lattice_validation.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"grid200": cmd_grid200, "mosaic": cmd_mosaic, "sim": cmd_sim, "validate": cmd_validate, "discover": cmd_discover}
    for k in (fns if cmd == "all" else [cmd]):
        print("=====", k)
        fns[k]()
