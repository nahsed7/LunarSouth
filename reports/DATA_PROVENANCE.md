# Data provenance

Generated 2026-11-14 20:45 UTC. Raw downloads live in `data/raw/` and are never modified; processed outputs are in `data/processed/`. Re-create everything with `python tools/ingest_sources.py all`.

## Source-priority mapping

| Priority | Resource | What was used | Numeric? |
|---|---|---|---|
| 1 | NASA Moon Trek | WMTS capabilities for `AVGVISIB_85S_060M_201608_EARTH_SP` and `WAC_POLE_ILL_PCT_SOUTH_100M` (saved in data/raw/moon_trek). Trek serves these as rendered PNG tiles only, so the identical products were pulled as numeric rasters from their archives (LOLA EXTRAS / LROC PDS). The Trek catalog API (`TrekServices/ws/index/eq/searchItems`) only exposes the equirectangular index; the south-polar index endpoints returned no data. | rasters: yes; Trek tiles: display only |
| 2 | ASU LROC QuickMap | WAC_POLE_ILL_PCT_SOUTH_100M (PDS IMG, 32-bit), NAC_DTM_SHACKRDGE02 (3.5 m GeoTIFF) - the DTM QuickMap displays over the Shackleton-de Gerlache ridge; LOLA polar GDR DEMs (80 m, 240 m) | yes |
| 3 | JPL Horizons | Observer ephemeris API, CENTER=coord@301 geodetic site coordinates, QUANTITIES 4,20, AIRLESS apparent, UT; 16 sites × Sun & Earth × 361 epochs; raw JSON incl. query URL in data/raw/jpl_horizons | yes |
| 4 | CSA | data/csa_engineering_basis.json - published LUV requirements (-200 °C night, ~2-week night, 10-yr life). No CSA power specs are published; none invented | requirements only |

## Products

### LOLA_DEM_80M - `LDEM_80S_80M`
- Source: NASA PDS Geosciences Node - LRO LOLA GDR polar (LOLA science team)
- Local file: `data/raw/lola_pds/ldem_80s_80m.img` (115,520,000 bytes; sha256 of first 64 MiB `6be82d835a4acf13…`)
- Download date: 2026-11-14 · Product creation/acquisition: 2017-06-15 · observations 2009-07-13T17:33:17 → 2017-02-02T22:18:35
- Resolution: 80.0 m/px · grid 7600×7600 · units: m (height above 1737.4 km sphere) (DN×0.5+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: Gridded from LOLA tracks; interpolated cells between tracks; ~80 m posting cannot resolve boulders/small craters. Heights relative to 1737.4 km sphere.

### LOLA_DEM_240M - `LDEM_75S_240M`
- Source: NASA PDS Geosciences Node - LRO LOLA GDR polar (LOLA science team)
- Local file: `data/raw/lola_pds/ldem_75s_240m.img` (29,062,688 bytes; sha256 of first 64 MiB `d031efe2611b17df…`)
- Download date: 2026-11-14 · Product creation/acquisition: 2017-06-15 · observations 2009-07-13T17:33:17 → 2017-02-02T22:18:35
- Resolution: 240.0 m/px · grid 3812×3812 · units: m (height above 1737.4 km sphere) (DN×0.5+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: Coarse far-field DEM used only beyond 40 km for horizon ray casting.

### LOLA_AVG_SUN_VIS_60M - `AVGVISIB_85S_060M_201608`
- Source: LOLA illumination EXTRAS (MIT imbrium mirror); served by NASA Moon Trek as AVGVISIB_85S_060M_201608_SP (WMTS tiles)
- Local file: `data/raw/lola_illum/AVGVISIB_85S_060M_201608.IMG` (51,166,728 bytes; sha256 of first 64 MiB `82dadf321dee28f1…`)
- Download date: 2026-11-14 · Product creation/acquisition: 2016-09-15T00:00:00 · observations 2009-07-13T17:33:17.246 → 2013-07-18T14:12:13
- Resolution: 60.0 m/px · grid 5058×5058 · units: fraction of time Sun visible (0-1), 18.6-yr average (DN×4e-05+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 18.6-yr average fraction of solar disc visible (2016 release, DEM-derived, not observed). Coverage 85S-pole full, partial to ~82.9S. Linear DEM artifacts noted by producers. Not supported by PDS/LOLA team (EXTRAS).

### LOLA_AVG_EARTH_VIS_60M - `AVGVISIB_85S_060M_201608_EARTH`
- Source: LOLA illumination EXTRAS (MIT imbrium mirror); served by NASA Moon Trek as AVGVISIB_85S_060M_201608_EARTH_SP (WMTS tiles)
- Local file: `data/raw/lola_illum/AVGVISIB_85S_060M_201608_EARTH.IMG` (51,166,728 bytes; sha256 of first 64 MiB `0958313069d6039b…`)
- Download date: 2026-11-14 · Product creation/acquisition: 2016-09-15T00:00:00 · observations 2009-07-13T17:33:17.246 → 2013-07-18T14:12:13
- Resolution: 60.0 m/px · grid 5058×5058 · units: fraction of time Earth visible (0-1), 18.6-yr average (DN×4e-05+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 18.6-yr average Earth visibility; geometric only, says nothing about RF link closure.

### LOLA_PSR_60M - `LPSR_85S_060M_201608`
- Source: LOLA illumination EXTRAS (MIT imbrium mirror); served by NASA Moon Trek as LPSR_85S_060M_201608_SP (WMTS tiles)
- Local file: `data/raw/lola_illum/LPSR_85S_060M_201608.IMG` (51,166,728 bytes; sha256 of first 64 MiB `f8203cab342c817f…`)
- Download date: 2026-11-14 · Product creation/acquisition: 2016-09-15T00:00:00 · observations 2009-07-13T17:33:17.246 → 2013-07-18T14:12:13
- Resolution: 60.0 m/px · grid 5058×5058 · units: binary permanent shadow flag (DN×2.5e-05+0.5)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: Binary permanent-shadow flag derived from same DEM.

### WAC_ILLUM_PCT_100M - `WAC_POLE_ILL_PCT_SOUTH_100M`
- Source: ASU LROC PDS (QuickMap / Moon Trek layer WAC_POLE_ILL_PCT_SOUTH_100M)
- Local file: `data/raw/lroc_wac_illum/WAC_POLE_ILL_PCT_SOUTH_100M.IMG` (5,904,896 bytes; sha256 of first 64 MiB `30b35ae2de695198…`)
- Download date: 2026-11-14 · Product creation/acquisition: 2020-08-31T22:44:07 · observations N/A → N/A
- Resolution: 100.0 m/px · grid 1214×1214 · units: percent of WAC observations illuminated (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: Percentage of WAC images (one lunar year, 2010-2011) in which pixel was lit; does not correct non-linear temporal sampling; coverage 88S-90S only.

### NAC_DTM_SHACKRDGE02 - `NAC_DTM_SHACKRDGE02`
- Source: ASU LROC PDS NAC stereo DTM (QuickMap)
- Local file: `data/raw/lroc_nac_dtm/NAC_DTM_SHACKRDGE02.TIF` (10,073,090 bytes; sha256 of first 64 MiB `89d19c6ba6b0c987…`)
- Download date: 2026-11-14 · Product creation/acquisition: n/a 
- Resolution: 3.5 m/px · grid 1434×1754 · units: m (height above 1737.4 km sphere) (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: NAC stereo DTM, 3.5 m/px, covers only the Shackleton-de Gerlache connecting ridge (-89.63 to -89.37, 209.8-239.1E). Confidence map not used.

### LDEM_20M - `LDEM_80S_20MPP_ADJ`
- Source: NASA GSFC PGDA LOLA 80S 20 m DEM (adjusted)
- Local file: `/home/deshan7/Desktop/clpsdatasets/LDEM_80S_20MPP_ADJ.tiff` (2,696,082,051 bytes; sha256 of first 64 MiB `887d7afa102bbab2…`)
- Download date: 2026-11-05 · Product creation/acquisition: n/a 
- Resolution: 20.0 m/px · grid 30400×30400 · units: m (height above 1737.4 km sphere) (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 

### LDEC_20M - `LDEC_80S_20MPP_ADJ`
- Source: NASA GSFC PGDA LOLA 80S 20 m shot count
- Local file: `/home/deshan7/Desktop/clpsdatasets/LDEC_80S_20MPP_ADJ.tiff` (1,027,440,434 bytes; sha256 of first 64 MiB `685357973510b233…`)
- Download date: 2026-11-05 · Product creation/acquisition: n/a 
- Resolution: 20.0 m/px · grid 30400×30400 · units: LOLA returns per pixel (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 

### LPSR_20M - `LPSR_80S_20MPP_ADJ`
- Source: NASA GSFC PGDA LOLA 80S 20 m permanent shadow
- Local file: `/home/deshan7/Desktop/clpsdatasets/LPSR_80S_20MPP_ADJ.tiff` (154,221,202 bytes; sha256 of first 64 MiB `7df037b047a3f8f0…`)
- Download date: 2026-11-05 · Product creation/acquisition: n/a 
- Resolution: 20.0 m/px · grid 30400×30400 · units: PSR flag (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 

### LDEM_5M_87S - `ldem_87s_5mpp`
- Source: NASA GSFC PGDA LOLA 87.5S 5 m DEM
- Local file: `/home/deshan7/Desktop/clpsdatasets/ldem_87s_5mpp.tif` (2,328,260,115 bytes; sha256 of first 64 MiB `3da068967ef129c8…`)
- Download date: 2026-11-05 · Product creation/acquisition: n/a 
- Resolution: 5.0 m/px · grid 40000×40000 · units: m (height above 1737.4 km sphere) (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 

### LDEC_5M_87S - `ldec_87s_5mpp`
- Source: NASA GSFC PGDA LOLA 87.5S 5 m shot count
- Local file: `/home/deshan7/Desktop/clpsdatasets/ldec_87s_5mpp.tif` (145,895,726 bytes; sha256 of first 64 MiB `e0291d9cc14dafaf…`)
- Download date: 2026-11-05 · Product creation/acquisition: n/a 
- Resolution: 5.0 m/px · grid 40000×40000 · units: LOLA returns per pixel (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 

### SITE01_5M - `Site01_final_adj_5mpp_surf`
- Source: NASA GSFC PGDA high-res LOLA Site01 (connecting ridge) 5 m
- Local file: `/home/deshan7/Desktop/clpsdatasets/Site01_final_adj_5mpp_surf.tif` (40,980,810 bytes; sha256 of first 64 MiB `3ba7b97cb00a2bcf…`)
- Download date: 2026-11-04 · Product creation/acquisition: n/a 
- Resolution: 5.0 m/px · grid 3200×3200 · units: m (DN×1.0+0.0)
- CRS: Polar stereographic, centre 90S/0E, sphere R=1737.4 km (IAU Moon, planetocentric, east-positive)
- Processing: Read at native resolution; DN*SCALING_FACTOR+OFFSET applied; nodata/ISIS special pixels -> NaN; bilinear sampling only where all 4 neighbours valid. Raw file untouched.
- Coordinate transformation: Site lat/lon (planetocentric, east-positive) -> south polar stereographic x=rho*sin(lon), y=rho*cos(lon), rho=2R tan((90+lat)/2); local east/north rotated per site.
- Known limitations: 

### Derived products
- `data/processed/horizons/horizon_profiles_lola.csv` - 200 km ray-cast horizon, 720 azimuths × 16 sites (method in its .provenance.json).
- `data/processed/terrain3d/*_scene.npz` - 24 km × 24 km local east/north grids (301², LOLA 80 m) with co-registered LOLA/WAC layers; `*_nac_1km.npz` NAC 3.5 m grids; `regional_scene.npz` 300 km × 300 km at 1 km (from 240 m).
- `data/processed/terrain_metrics_all_sites.csv` - slope/relief/roughness per site per source resolution.
- `data/processed/jpl_horizons/horizons_vs_spice.csv` - paired Horizons vs SPICE geometry.
- SPICE kernels in `kernels/` from NAIF generic_kernels: de421.bsp, moon_pa_de421_1900-2050.bpc, moon_080317.tf, pck00008.tpc.
- Legacy team products (data/solar, data/dte, data/terrain, model3_*.csv, horizon_profiles.csv) retained unmodified for comparison.
