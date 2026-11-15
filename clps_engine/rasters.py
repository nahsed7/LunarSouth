"""Readers for the south-polar scientific rasters integrated from the four sources.

All products are polar stereographic, centred on 90 S / 0 E, on a 1737.4 km
sphere. Pixel geometry follows each PDS label (LINE/SAMPLE_PROJECTION_OFFSET,
MAP_SCALE); x = rho*sin(lon), y = rho*cos(lon) for the south pole (verified
against the NAC DTM GeoTIFF bounds). Values are returned in physical units
with nodata as NaN - nothing is silently filled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MOON_R_M = 1737400.0
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def latlon_to_ps(lat_deg, lon_deg, r_m: float = MOON_R_M):
    """South polar stereographic (true scale at pole) -> x, y metres."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    rho = 2.0 * r_m * np.tan((np.pi / 2 + lat) / 2.0)
    return rho * np.sin(lon), rho * np.cos(lon)


def ps_to_latlon(x, y, r_m: float = MOON_R_M):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = np.hypot(x, y)
    lat = -90.0 + np.degrees(2.0 * np.arctan(rho / (2.0 * r_m)))
    lon = np.degrees(np.arctan2(x, y))
    return lat, lon


def local_enu_axes(lon_deg: float):
    """Unit vectors (in PS x/y) of local east and north at a south-polar site."""
    lon = np.radians(lon_deg)
    # North at a south-polar site points away from the pole (increasing rho).
    north = np.array([np.sin(lon), np.cos(lon)])
    east = np.array([np.cos(lon), -np.sin(lon)])
    return east, north


@dataclass
class PolarRaster:
    product_id: str
    path: Path
    lines: int
    samples: int
    scale_m: float
    line_offset: float
    sample_offset: float
    units: str
    dtype: str
    scaling: float = 1.0
    offset: float = 0.0
    header_bytes: int = 0
    nodata: Optional[float] = None
    source: str = ""
    description: str = ""
    meta: dict = field(default_factory=dict)
    _arr: Optional[np.ndarray] = None

    # -------------------------------------------------------------- access
    @property
    def raw(self) -> np.ndarray:
        if self._arr is None:
            if self.path.suffix.lower() in (".tif", ".tiff"):
                import rasterio

                with rasterio.open(self.path) as r:
                    self._arr = r.read(1)
            else:
                self._arr = np.memmap(
                    self.path, dtype=self.dtype, mode="r", offset=self.header_bytes,
                    shape=(self.lines, self.samples),
                )
        return self._arr

    def read_block(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Physical values for rows r0:r1, cols c0:c1 (windowed for GeoTIFFs - never loads whole file)."""
        if self.path.suffix.lower() in (".tif", ".tiff"):
            import rasterio
            from rasterio.windows import Window

            with rasterio.open(self.path) as r:
                try:
                    return self.to_physical(r.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)))
                except Exception:  # damaged tile(s): recover every readable 256 px sub-block, NaN the rest
                    out = np.full((r1 - r0, c1 - c0), np.nan)
                    bh, bw = r.block_shapes[0]
                    for tr in range((r0 // bh) * bh, r1, bh):       # iterate the file's own tile grid
                        for tc in range((c0 // bw) * bw, c1, bw):
                            a0, a1 = max(tr, r0), min(tr + bh, r1)
                            b0, b1 = max(tc, c0), min(tc + bw, c1)
                            try:
                                out[a0 - r0: a1 - r0, b0 - c0: b1 - c0] = self.to_physical(
                                    r.read(1, window=Window(b0, a0, b1 - b0, a1 - a0)))
                            except Exception:
                                pass
                    return out
        return self.to_physical(self.raw[r0:r1, c0:c1])

    def overview(self, out_px: int) -> tuple:
        """Decimated full-extent read (average where the file has overviews); returns (array, x_coords, y_coords)."""
        if self.path.suffix.lower() in (".tif", ".tiff"):
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(self.path) as r:
                a = r.read(1, out_shape=(out_px, out_px), resampling=Resampling.average)
            a = self.to_physical(a)
        else:
            step = max(1, self.lines // out_px)
            a = self.to_physical(self.raw[::step, ::step])[:out_px, :out_px]
        x0, x1, y0, y1 = self.extent_xy()
        n = a.shape[0]
        px = (x1 - x0) / n
        xs = x0 + (np.arange(n) + 0.5) * px
        ys = y1 - (np.arange(n) + 0.5) * px
        return a, xs, ys

    def to_physical(self, dn: np.ndarray) -> np.ndarray:
        v = np.asarray(dn).astype("float64")
        bad = ~np.isfinite(v)
        if self.nodata is not None:
            bad |= np.isclose(v, self.nodata)
        if self.dtype.endswith("f4"):
            bad |= v < -1e30  # ISIS special pixels (NULL/LRS/HRS...)
        out = v * self.scaling + self.offset
        out[bad] = np.nan
        return out

    def xy_to_rc(self, x, y):
        col = self.sample_offset + np.asarray(x, float) / self.scale_m
        row = self.line_offset - np.asarray(y, float) / self.scale_m
        return row, col

    def rc_to_xy(self, row, col):
        x = (np.asarray(col, float) - self.sample_offset) * self.scale_m
        y = (self.line_offset - np.asarray(row, float)) * self.scale_m
        return x, y

    def extent_xy(self):
        x0, y0 = self.rc_to_xy(-0.5, -0.5)
        x1, y1 = self.rc_to_xy(self.lines - 0.5, self.samples - 0.5)
        return float(x0), float(x1), float(y1), float(y0)

    def covers(self, lat, lon) -> bool:
        r, c = self.xy_to_rc(*latlon_to_ps(lat, lon))
        return bool(0 <= r <= self.lines - 1 and 0 <= c <= self.samples - 1)

    def sample_xy(self, x, y, order: int = 1) -> np.ndarray:
        """Bilinear (order=1) or nearest (order=0) sampling; NaN outside/nodata."""
        from scipy import ndimage

        x, y = np.asarray(x, float), np.asarray(y, float)
        row, col = self.xy_to_rc(x, y)
        inside = (row >= 0) & (row <= self.lines - 1) & (col >= 0) & (col <= self.samples - 1)
        out = np.full(x.shape, np.nan)
        if not inside.any():
            return out
        r, c = row[inside], col[inside]
        r0, r1 = int(np.floor(r.min())), int(np.ceil(r.max())) + 1
        c0, c1 = int(np.floor(c.min())), int(np.ceil(c.max())) + 1
        r1, c1 = min(r1, self.lines), min(c1, self.samples)
        block = self.read_block(r0, r1, c0, c1)
        valid = np.isfinite(block)
        filled = np.where(valid, block, 0.0)
        vals = ndimage.map_coordinates(filled, [r - r0, c - c0], order=order, mode="nearest")
        if order:
            w = ndimage.map_coordinates(valid.astype(float), [r - r0, c - c0], order=order, mode="nearest")
            vals = np.where(w > 0.999, vals, np.nan)
        else:
            w = ndimage.map_coordinates(valid.astype(float), [r - r0, c - c0], order=0, mode="nearest")
            vals = np.where(w > 0.5, vals, np.nan)
        out[inside] = vals
        return out

    def sample_latlon(self, lat, lon, order: int = 1):
        return self.sample_xy(*latlon_to_ps(lat, lon), order=order)

    def window(self, lat, lon, half_width_m: float):
        """Native-resolution window centred on a site. Returns (array, x_coords, y_coords)."""
        cx, cy = latlon_to_ps(lat, lon)
        rc, cc = self.xy_to_rc(cx, cy)
        n = int(np.ceil(half_width_m / self.scale_m))
        r0, r1 = int(round(float(rc))) - n, int(round(float(rc))) + n + 1
        c0, c1 = int(round(float(cc))) - n, int(round(float(cc))) + n + 1
        rr0, rr1 = max(r0, 0), min(r1, self.lines)
        cc0, cc1 = max(c0, 0), min(c1, self.samples)
        arr = np.full((r1 - r0, c1 - c0), np.nan)
        if rr1 > rr0 and cc1 > cc0:
            arr[rr0 - r0: rr1 - r0, cc0 - c0: cc1 - c0] = self.read_block(rr0, rr1, cc0, cc1)
        xs, _ = self.rc_to_xy(np.zeros(c1 - c0), np.arange(c0, c1))
        _, ys = self.rc_to_xy(np.arange(r0, r1), np.zeros(r1 - r0))
        return arr, xs, ys

    def info(self) -> dict:
        x0, x1, y0, y1 = self.extent_xy()
        return {
            "product_id": self.product_id,
            "file": self.path.name,
            "source": self.source,
            "resolution_m": self.scale_m,
            "grid": [self.lines, self.samples],
            "extent_ps_m": [x0, x1, y0, y1],
            "crs": "Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)",
            "units": self.units,
            "scaling_factor": self.scaling,
            "offset": self.offset,
            "description": self.description,
            **self.meta,
        }


# --------------------------------------------------------------------- labels
def _pds3_label(text: str) -> dict:
    out = {}
    for key in (
        "LINES", "LINE_SAMPLES", "SAMPLE_BITS", "SAMPLE_TYPE", "SCALING_FACTOR", "OFFSET",
        "MAP_SCALE", "LINE_PROJECTION_OFFSET", "SAMPLE_PROJECTION_OFFSET", "RECORD_BYTES",
        "LABEL_RECORDS", "UNIT", "MISSING_CONSTANT", "PRODUCT_CREATION_TIME", "START_TIME",
        "STOP_TIME", "PRODUCT_ID", "PRODUCT_VERSION_ID", "MINIMUM_LATITUDE", "MAXIMUM_LATITUDE",
    ):
        m = re.search(rf"^\s*{key}\s*=\s*\"?([^\"\n<]+)", text, re.M)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"OBJECT\s*=\s*IMAGE\s*\n.*?DESCRIPTION\s*=\s*\"(.*?)\"", text, re.S)
    if m:
        out["DESCRIPTION"] = " ".join(m.group(1).split())
    return out


def _f(lbl, k, default=None):
    v = lbl.get(k)
    if v is None:
        return default
    return float(re.sub(r"[^0-9eE+.\-]", "", v.split()[0]))


def _dtype(lbl) -> str:
    st = lbl.get("SAMPLE_TYPE", "")
    bits = int(_f(lbl, "SAMPLE_BITS", 16))
    if "REAL" in st:
        return "<f4" if bits == 32 else "<f8"
    return {16: "<i2", 32: "<i4", 8: "u1"}[bits]


def open_pds_img(img: Path, lbl: Optional[Path] = None, *, product_id=None, source="", units=None) -> PolarRaster:
    img = Path(img)
    if lbl is not None and Path(lbl).exists():
        text = Path(lbl).read_text(errors="replace")
        header = 0
    else:  # attached label
        text = img.open("rb").read(20000).decode("latin-1", errors="replace")
        lb = _pds3_label(text)
        header = int(_f(lb, "RECORD_BYTES")) * int(_f(lb, "LABEL_RECORDS"))
    lb = _pds3_label(text)
    scale = _f(lb, "MAP_SCALE")
    if scale is not None and scale < 1.0:  # km/pixel labels
        scale *= 1000.0
    return PolarRaster(
        product_id=product_id or lb.get("PRODUCT_ID", img.stem),
        path=img,
        lines=int(_f(lb, "LINES")),
        samples=int(_f(lb, "LINE_SAMPLES")),
        scale_m=float(scale),
        line_offset=_f(lb, "LINE_PROJECTION_OFFSET"),
        sample_offset=_f(lb, "SAMPLE_PROJECTION_OFFSET"),
        units=units or lb.get("UNIT", "unknown"),
        dtype=_dtype(lb),
        scaling=_f(lb, "SCALING_FACTOR", 1.0),
        offset=_f(lb, "OFFSET", 0.0) if "LDEM" not in img.name.upper() else 0.0,
        header_bytes=header,
        source=source,
        description=lb.get("DESCRIPTION", ""),
        meta={k.lower(): lb[k] for k in ("PRODUCT_CREATION_TIME", "START_TIME", "STOP_TIME", "PRODUCT_VERSION_ID",
                                         "MINIMUM_LATITUDE", "MAXIMUM_LATITUDE") if k in lb},
    )


def open_geotiff_ps(path: Path, *, product_id: str, source: str, units: str, dtype: str = "<f4") -> PolarRaster:
    import rasterio

    with rasterio.open(path) as r:
        t = r.transform
        nod = r.nodata
        h, w = r.shape
        dtype = {"float32": "<f4", "uint8": "u1", "uint16": "<u2", "int16": "<i2"}.get(r.dtypes[0], dtype)
    # rasterio transform: x = t.c + (col+0.5)*t.a ; y = t.f + (row+0.5)*t.e
    return PolarRaster(
        product_id=product_id, path=Path(path), lines=h, samples=w, scale_m=float(t.a),
        line_offset=(t.f / t.a) - 0.5, sample_offset=(-t.c / t.a) - 0.5,
        units=units, dtype=dtype, nodata=nod, source=source,
    )


# ---------------------------------------------------------------- registry
def product_registry() -> dict[str, PolarRaster]:
    """Every integrated raster that is present on disk, keyed by short name."""
    reg: dict[str, PolarRaster] = {}
    lp = RAW / "lola_pds"
    for name, key in (("ldem_80s_80m", "LOLA_DEM_80M"), ("ldem_75s_240m", "LOLA_DEM_240M")):
        if (lp / f"{name}.img").exists():
            reg[key] = open_pds_img(lp / f"{name}.img", lp / f"{name}.lbl", product_id=name.upper(),
                                    source="NASA PDS Geosciences Node - LRO LOLA GDR polar (LOLA science team)",
                                    units="m (height above 1737.4 km sphere)")
    li = RAW / "lola_illum"
    for name, key, units in (
        ("AVGVISIB_85S_060M_201608", "LOLA_AVG_SUN_VIS_60M", "fraction of time Sun visible (0-1), 18.6-yr average"),
        ("AVGVISIB_85S_060M_201608_EARTH", "LOLA_AVG_EARTH_VIS_60M", "fraction of time Earth visible (0-1), 18.6-yr average"),
        ("LPSR_85S_060M_201608", "LOLA_PSR_60M", "binary permanent shadow flag"),
    ):
        if (li / f"{name}.IMG").exists():
            reg[key] = open_pds_img(li / f"{name}.IMG", li / f"{name}.LBL", product_id=name,
                                    source="LOLA illumination EXTRAS (MIT imbrium mirror); served by NASA Moon Trek as "
                                           f"{name}_SP (WMTS tiles)", units=units)
    w = RAW / "lroc_wac_illum" / "WAC_POLE_ILL_PCT_SOUTH_100M.IMG"
    if w.exists():
        reg["WAC_ILLUM_PCT_100M"] = open_pds_img(w, None, product_id="WAC_POLE_ILL_PCT_SOUTH_100M",
                                                 source="ASU LROC PDS (QuickMap / Moon Trek layer WAC_POLE_ILL_PCT_SOUTH_100M)",
                                                 units="percent of WAC observations illuminated")
    n = RAW / "lroc_nac_dtm" / "NAC_DTM_SHACKRDGE02.TIF"
    if n.exists():
        reg["NAC_DTM_SHACKRDGE02"] = open_geotiff_ps(n, product_id="NAC_DTM_SHACKRDGE02",
                                                     source="ASU LROC PDS NAC stereo DTM (QuickMap)",
                                                     units="m (height above 1737.4 km sphere)")
    # User-supplied local archive (NASA PGDA / LROC products, ~11 GB) - used when present
    import os

    ext = Path(os.environ.get("CLPS_DATASETS", str(Path.home() / "Desktop" / "clpsdatasets")))
    for fname, key, src, units in (
        ("LDEM_80S_20MPP_ADJ.tiff", "LDEM_20M", "NASA GSFC PGDA LOLA 80S 20 m DEM (adjusted)", "m (height above 1737.4 km sphere)"),
        ("LDEC_80S_20MPP_ADJ.tiff", "LDEC_20M", "NASA GSFC PGDA LOLA 80S 20 m shot count", "LOLA returns per pixel"),
        ("LPSR_80S_20MPP_ADJ.tiff", "LPSR_20M", "NASA GSFC PGDA LOLA 80S 20 m permanent shadow", "PSR flag"),
        ("ldem_87s_5mpp.tif", "LDEM_5M_87S", "NASA GSFC PGDA LOLA 87.5S 5 m DEM", "m (height above 1737.4 km sphere)"),
        ("ldec_87s_5mpp.tif", "LDEC_5M_87S", "NASA GSFC PGDA LOLA 87.5S 5 m shot count", "LOLA returns per pixel"),
        ("Site01_final_adj_5mpp_surf.tif", "SITE01_5M", "NASA GSFC PGDA high-res LOLA Site01 (connecting ridge) 5 m", "m"),
    ):
        f = ext / fname
        if f.exists():
            try:
                reg[key] = open_geotiff_ps(f, product_id=f.stem, source=src, units=units)
            except Exception:
                pass
    return reg


class GlobalDEM:
    """LOLA LDEM_16 (16 px/deg, ~1.9 km at equator) simple-cylindrical global DEM, heights above 1737.4 km.

    Used ONLY where no polar product covers a point (non-polar presets: Apollo, Chang'e, SLIM, ...).
    """

    def __init__(self, img: Path, ppd: int = 16):
        self.path = Path(img)
        self.ppd = ppd
        self.lines, self.samples = 180 * ppd, 360 * ppd
        self.scale_m = MOON_R_M * np.pi / 180.0 / ppd
        self.product_id = self.path.stem.upper()
        self._a = None

    @property
    def arr(self) -> np.ndarray:
        if self._a is None:
            self._a = np.memmap(self.path, dtype="<i2", mode="r", shape=(self.lines, self.samples))
        return self._a

    def sample_latlon(self, lat, lon) -> np.ndarray:
        from scipy import ndimage

        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float) % 360.0
        row = (90.0 - lat) * self.ppd - 0.5
        col = lon * self.ppd - 0.5
        r0, r1 = int(max(np.floor(row.min()) - 1, 0)), int(min(np.ceil(row.max()) + 2, self.lines))
        a = np.asarray(self.arr[r0:r1], dtype=np.float64) * 0.5
        a = np.concatenate([a[:, -1:], a, a[:, :1]], axis=1)  # longitude wrap
        return ndimage.map_coordinates(a, [row - r0, col + 1.0], order=1, mode="nearest")

    def covers(self, lat, lon) -> bool:
        return True


def global_dem() -> Optional[GlobalDEM]:
    p = RAW / "lola_pds" / "ldem_16.img"
    return GlobalDEM(p) if p.exists() else None
