# LUNARSOUTH · CLPS Mission Planner — Lunar South Pole Mission Engineering System

NASA Space Apps Challenge 2026 · Human Exploration – Moon – Software.

The planner compares south-pole landing sites and dates. For each site and mission window it computes:

- terrain-corrected Sun and Earth geometry;
- solar power generation;
- the energy budget and battery survival;
- terrain suitability.

All five models run on **one shared timeline**, so a change of site, date, array, load or scenario propagates through every result.

## Quick start

```bash
./run.sh            # Linux / macOS  (Windows: double-click run.bat)
# or: pip install -r requirements.txt && streamlit run app.py
```

Opens http://localhost:8501. On the first start the large rasters in `data/packed/` are rebuilt once (SHA-256 checked).
Free hosting on Streamlit Community Cloud: see `DEPLOY.md`.

## New in this release (see CHANGELOG.md)

- **🏆 Rank & compare points:** scan the whole 80°S cap for the best landing points with a weighted percentage score
  (sunlight, Earth view, flat ground, night length, solar power, blackouts, distance to cold traps). Verify the winners with the full
  M1–M5 models, then compare any chosen points side by side (radar, ranking, best-value highlights, CSV).
- **🧰 Real-equipment loadouts in M5:** 29 real devices (VIPER, PRIME-1, Blue Ghost, Mars 2020, Iris, RAD750, MMRTG, Kilopower, FSP…)
  and 6 mission loadouts. RTGs and reactors add constant bus power, and RHUs add heat that reduces the survival-heater load.
- **🎮 Mission Sim:** real rover instruments (TRIDENT, NIRVSS, MSolo, NSS, NavCams), plus ⏭ next sunrise / sunset / +1 day / +1 week
  fast-forward with full physics.
- **Standalone package:** `python tools/build_release.py` writes a GitHub-ready folder and zip in which every file is ≤ 24 MB.
  Run it with `./run.sh` or `run.bat`, or deploy it free on Streamlit Community Cloud (`DEPLOY.md`).

## Whole-pole coverage (80°S → pole)

The planner is not limited to fixed sites. **Any point** on the south polar cap can be clicked on the map and analysed through all five models.

- **20 m terrain everywhere:** `tools/build_polewide.py` streams the PGDA LOLA `LDEM/LPSR_80S_20MPP_ADJ` GeoTIFFs in memory-bounded strips. It computes slope at native 20 m over the full 608 × 608 km cap and aggregates it to 200 m and 1 km.
- **Whole-pole lattice:** Models 1, 2 and 4 run on a 5 km lattice of about 11,000 points for all of 2027. Geometry is SPICE (validated against Horizons); each point's horizon is ray-cast from LOLA.
- **Discovery:** a transparent composite ranks every 1 km cell and extracts 40 separated **LunarSouth discoveries** (`LN-xxx`), which are analysed alongside the 16 project candidates.
- **Local 5 m and 20 m data:** used for terrain metrics and near-field horizons when the local archive is present. Set `CLPS_DATASETS=/path` to point at it (default `~/Desktop/clpsdatasets`).

```bash
.venv/bin/python tools/build_polewide.py all      # grid200 → mosaic → sim → validate → discover
.venv/bin/python tools/ingest_sources.py horizons && .venv/bin/python tools/ingest_sources.py terrain
```

## Quick start

```bash
cd CLPS_mission_planner-main
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-geo.txt
.venv/bin/python tools/ingest_sources.py all     # download (cached) + process all products (~5 min)
.venv/bin/python tools/build_reports.py          # regenerate reports/
.venv/bin/streamlit run app.py                   # open http://localhost:8501
```

`data/raw`, `data/processed` and `kernels/` are already populated in this package, so the app starts immediately. The ingestion step is only needed to rebuild the data.

Tests:

```bash
.venv/bin/python tests/validate_project.py       # legacy suite (team products)
.venv/bin/python tests/test_upgraded_models.py   # upgraded M1-M5, data products, validation thresholds
.venv/bin/python tests/test_mission_sim.py       # M5 Mission Sim payload + browser engine vs Python (needs node)
```

## Workflow in the app

1. **Sidebar → set up the mission:**
   - Site: one of the 16 candidates (4 baseline + 12 research), or custom coordinates with a live LOLA horizon.
   - Mission start and duration.
   - Solar array.
   - Battery.
   - Thermal settings.
   - Operational scenario.
2. **🌐 3D environment** – rotatable/zoomable LOLA terrain. A time slider drives:
   - Sun and Earth direction rays;
   - cast shadows;
   - the 200 km horizon ring;
   - layers: elevation, slope, live illumination, LOLA Sun/Earth visibility, WAC illumination and PSR.

   There is also a 300 km regional view showing all sites.
3. **☀️ M1 Illumination** – solar elevation vs terrain horizon, sky plot, sunrise/sunset windows, and comparison with independent products.
4. **🌍 M2 Earth/DTE** – Earth elevation vs horizon, Earthrise/Earthset windows, blackout durations, and the geometric RF proxy (clearly not a link budget).
5. **⛰ M3 Terrain** – metrics at each source resolution (3.5 / 5 / 20 / 80 m, never mixed), hazards, uncertainty, and slope maps (LOLA and NAC).
6. **⚡ M4 Generation** – power timeline over the illumination state, daily energy, best/worst days, and sensitivity to area, efficiency, dust, aging and orientation.
7. **🔋 M5 Energy survival** – editable subsystem load table, SoC timeline, load shedding, unserved demand, demand by subsystem, and all six scenarios side by side.
8. **🎮 M5 Mission Sim** – a playable rover + outpost on the real LOLA terrain, running on the same M1–M5 timeline:
   - drive the rover by clicking the map, with WASD or with waypoints; its sunlight and Earth view are ray-cast from **its own position**, and drive power follows the local slope;
   - run drill, spectrometer or panorama at science targets picked from the data (PSR cold traps, sunlit ridges, Earth-view spot, outcrops);
   - send signals (telemetry, science, video, beacon or custom RF power), switch outpost subsystems, change the heater set point, dock to charge;
   - live SoC, battery time left, next sunset/sunrise/Earth-set at the rover, survival forecast, event log and score; progress autosaves in the browser;
   - a Power BI-style dashboard (sunlight harvest → generation → usage) with cross-filtering, a time slicer, drill-down by day and isolation by category. The same dashboard is in the M5 tab for the batch plan.
   - The browser engine is checked against the Python models in `tests/test_mission_sim.py` (horizon to 0.02°, battery replay equal to `energy.simulate`).
9. **🔀 Diff & handoff** – A/B scenario diff (two sites, dates or scenarios) with deltas, verdicts and first sunset/Earthset countdowns; a small-change sensitivity panel (0.1° site shift, ±24 h launch delay) plus libration statistics; preset and uploadable mission profiles (JSON), trade-off notes, and a downloadable Markdown handoff note.
   - **⛰ M3** also has a ray-traced illumination map of the 24 km landing area (2 m observer, 80 m DEM + 12–200 km far field), the nearest permanent-shadow boundary, and grazing-light statistics. A banner at the top gives a plain-language Earth-vs-Moon comparison with this site's numbers.
   - **Public link:** `./keep_public.sh` serves the app through a free Cloudflare quick tunnel (no account), with a watchdog that restarts the app if it crashes; `./keep_public.sh url` / `stop`.
10. **⚖️ Compare** – all 16 sites ranked on the same configuration, plus a landing-date slip sweep.
11. **🧪 Data & validation** and **📄 Assessment report** (downloadable Markdown and timeline CSV).

## Data sources (strict priority order)

| # | Source | Products used |
|---|---|---|
| 1 | **NASA Moon Trek** | South-polar WMTS layers `AVGVISIB_85S_060M_201608_EARTH_SP` and `WAC_POLE_ILL_PCT_SOUTH_100M`. Trek serves rendered PNG tiles, so the same products were ingested as numeric rasters from their archives: LOLA `AVGVISIB_85S_060M_201608` (Sun), `…_EARTH`, `LPSR_85S_060M_201608` |
| 2 | **ASU LROC QuickMap** | `WAC_POLE_ILL_PCT_SOUTH_100M` (PDS), `NAC_DTM_SHACKRDGE02` 3.5 m DTM (Connecting Ridge), LOLA polar GDR DEMs `LDEM_80S_80M`, `LDEM_75S_240M` |
| 3 | **JPL Horizons** | Topocentric Sun and Earth az/el/range for all 16 sites (`CENTER=coord@301`), stored raw with query parameters; agrees with SPICE to about 1e-5° |
| 4 | **CSA** | `data/csa_engineering_basis.json`: Lunar Utility Vehicle requirements (−200 °C, two-week night, 10-year life). CSA publishes no power ratings, and none were invented |

Full provenance, statistics and validation are in `reports/`:

- `DATA_PROVENANCE.md`
- `DATASET_STATISTICS.md`
- `MODEL_VALIDATION.md`
- `SITE_COMPARISON.md`

## Models

| Model | File | Summary |
|---|---|---|
| M1 Illumination | `clps_engine/solar.py`, `horizon_build.py` | SPICE Sun geometry against a **200 km LOLA ray-cast horizon** (spherical curvature, 0.5°), with a high-resolution near field (NAC 3.5 m or legacy 5/20 m grids). Outputs the solar disc fraction for partial illumination, sunrise/sunset events and the longest darkness |
| M2 Earth / DTE | `clps_engine/dte.py`, `rf.py` | Earth disc vs horizon, three disc rules, windows and blackouts. Geometric only |
| M3 Terrain | `clps_engine/mission.py::terrain_assessment`, `terrain.py` | Best-resolution slope, p90, relief and roughness; PSR proximity; horizon obstruction; hazard and uncertainty lists; resolution-tagged |
| M4 Generation (new) | `clps_engine/power_gen.py` | S0/r² irradiance, disc fraction, 5 orientations, ASHRAE incidence-angle modifier, radiative cell temperature, temperature coefficient, dust, aging, packing, MPPT and harness losses, sensitivity analysis |
| M5 Energy survival (new) | `clps_engine/energy.py` | Scheduled, prioritised subsystems; MLI survival heaters with a −200 °C sink (CSA); battery with charge/discharge efficiency, DoD floor, power limits and self-discharge; priority load shedding; survival status; six scenarios |

## Engineering integrity

- Every array, battery and load value is a labelled, configurable **assumption**, and none is flight-qualified.
- Visibility is never treated as link closure, illumination never as energy, and generation never as survival.
- Products at different resolutions are reported separately.
- Sites outside DEM coverage are flagged "flat horizon – optimistic" rather than silently assumed flat.
- The legacy UI is kept as `app_legacy.py`, and all legacy team products are retained unmodified.

## Objective A gap fixes (interim — teammate deliverables plug in via `extensions/`)

| Gap | Status | Where |
|---|---|---|
| Landing-site presets beyond CLPS | 21 Apollo / Surveyor / Luna / Chang'e / Chandrayaan-3 / SLIM / IM-1 / IM-2 / Blue Ghost presets; drop more CSVs into `data/presets/` or `extensions/presets/` | `clps_engine/candidates.py::presets` |
| Custom sites without offline CSVs | Main app is fully live. The legacy app now falls back to live SPICE instead of skipping | `app_legacy.py` |
| Fixed offline date range | Any start, duration and step. **Seasons & multi-lunation** small multiples, plus a date-slip sweep | Compare · seasons tab |
| Continuous countdowns, 3D HUD | Event engine (sunrise/sunset, Earthrise/Earthset, link AOS/LOS, DSN handoff, shedding, SoC) with T-minus countdowns, also shown as a 3D HUD overlay | `clps_engine/events.py`, Mission clock, Site 3D |
| Offline Sun-angle proxy | Replaced by real SPICE Sun angles (proxy kept only if kernels are missing, and labelled) | `clps_engine/offline.py` |
| RF geometry only | X/S/Ka link budget: FSPL from SPICE range, EIRP, G/T, C/N0, achievable data rate, DSN Goldstone/Madrid/Canberra visibility, handoffs, optional relay | `clps_engine/rf_link.py`, M2 tab |
| 2D maps | Clickable south-pole map (12 layers, crater labels) and a global LOLA map of all presets | South Pole Explorer |
| Synchronised multi-site / multi-date scrubber | Up to 4 lanes (sites or date shifts) on one clock, with a state timeline and event log | Mission clock tab |
| Accessibility | Guide tab: Earth-vs-Moon baseline, coordinate and time conventions, spatial sensitivity (±1/5/15 km), schedule sensitivity | `clps_engine/guide.py` |
| Non-polar horizons | Global LOLA `LDEM_16` (1.9 km) great-circle ray-cast to 300 km, labelled. Landing-scale terrain is not assessed outside the polar cap | `horizon_build.ray_horizon_global` |

### Dropping in Objective A files

See `extensions/README.md`.

- **New tab:** a file `extensions/<name>.py` with `TAB_TITLE` and `render(ctx)`.
- **New sites:** CSVs in `extensions/presets/`.

No changes to `app.py` are needed. A broken extension stays inside its own tab.


## Data credits and disclaimer

- Terrain, illumination and PSR products: NASA LRO LOLA team / NASA GSFC Planetary Geodesy Data Archive (PGDA), via the
  PDS Geosciences Node. Ephemerides: NASA JPL NAIF SPICE kernels (DE421, lunar PA frame), used through Skyfield.
- Equipment figures are rounded public values from mission papers and press kits. Use them for planning only, not as flight specifications.
- LunarSouth is an independent NASA Space Apps Challenge 2026 entry. It is **not** affiliated with or endorsed by NASA.
- Parts of the code and documentation were written with AI assistance (Claude), and the team reviewed and tested them.
