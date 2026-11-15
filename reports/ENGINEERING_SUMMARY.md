# Engineering summary — what was actually completed

## Data integrated (raw files in data/raw, never modified)
- **Moon Trek (priority 1):** found the south-polar WMTS layers `AVGVISIB_85S_060M_201608_EARTH_SP` and `WAC_POLE_ILL_PCT_SOUTH_100M`.
  - Trek serves these only as rendered PNG tiles, so the identical products were ingested as numeric rasters from their archives:
    - LOLA AVGVISIB Sun and Earth (60 m, 18.6-year average);
    - LPSR permanent-shadow map (60 m).
  - The Trek catalog API returned no south-polar records, which is documented as a limitation.
- **QuickMap / LROC (priority 2):**
  - WAC percentage illumination (100 m, 88–90°S);
  - **NAC DTM SHACKRDGE02 at 3.5 m**, covering CR06, CR07 and CR001;
  - LOLA polar DEMs at 80 m and 240 m.
- **JPL Horizons (priority 3):** 32 topocentric queries (16 sites × Sun/Earth × 361 epochs), stored raw with their query parameters.
- **CSA (priority 4):** the published Lunar Utility Vehicle requirements (−200 °C, two-week night, 10-year life) are recorded in `data/csa_engineering_basis.json` and drive the survival-heater sink and the extended-darkness scenario. CSA publishes no power ratings, and none were invented.

## Models
- **M1 (upgraded):** the horizon was rebuilt for all 16 sites.
  - Before: a 10 km mask for LM1 only, with a flat 0° horizon for every other site in live mode.
  - Now: a 200 km LOLA ray-cast with spherical curvature, plus a 0–1.5 km NAC or 5/20 m near field.
  - Adds solar-disc fraction (partial illumination), sunrise/sunset events and the longest darkness.
- **M2 (upgraded):** Earth visibility uses the same horizon, plus the Earth disc fraction. RF is labelled as geometric only.
- **M3 (upgraded):** uses the best available resolution per site (3.5 / 5 / 20 / 80 m) with common-80 m metrics alongside. It reports:
  - hazards: p90 slope, relief, PSR within 5 km, horizon obstruction;
  - uncertainty and resolution flags.
- **M4 (new):** solar generation (`power_gen.py`).
  - Inputs: S0/r² irradiance from the SPICE Sun distance, and the disc fraction.
  - Five orientations, ASHRAE incidence-angle modifier, radiative cell temperature plus temperature coefficient.
  - Dust, aging, packing, MPPT and harness losses.
  - Daily, best and worst day, partial-illumination energy, sensitivity.
- **M5 (new):** energy survival (`energy.py`).
  - Inputs: scheduled, prioritised subsystems and MLI survival heaters.
  - Battery model: efficiencies, DoD floor, power limits, self-discharge; priority load shedding.
  - Six scenarios; outputs SoC, deficit, time to critical SoC, operations supported and a survival status.
- **Integration:** `mission.run_mission` runs M1→M5 on one timeline. The app (`app.py`) runs the full workflow plus the 3D environment, built from real LOLA and NAC elevation with Sun/Earth rays, cast shadows and the horizon ring.

## Validation (see MODEL_VALIDATION.md)
- **Horizons vs SPICE:** max |Δ elevation| 1.3e-5° and |Δ range| below 1 m. SPICE was retained as the runtime ephemeris.
- **18.6-year model vs LOLA AVGVISIB:** Sun r = 0.974 (mean |Δ| 0.054); Earth r = 0.977 (mean |Δ| 0.029). Residuals are explained in the report.
- **Legacy horizon:** reproduced exactly (r = 0.9999) when truncated to 10 km.
- **Baseline sites vs legacy team Model 1 (89 days):** LM1 55.9 vs 55.96%, Site06 35.4 vs 35.39%, Site20 46.0 vs 45.5%, Site23 50.2 vs 44.5%.
- **Tests:** `tests/validate_project.py` passes with 0 failures. `tests/test_upgraded_models.py` passes with 0 failures, covering energy conservation (0.000 Wh residual), SoC bounds, zero power in darkness, scenarios, site and date changes, and missing-data handling. A headless app run covering site, scenario, comparison and date-sweep interactions raised 0 exceptions.

## Known limitations
- **Near field:** 8 research sites have only the 80 m near field, and they show the largest Sun-visibility residuals.
- **RF:** no RF link budget, relay or ground-station schedule is modelled.
- **Thermal:** a first-order radiative model, not a full thermal analysis.
- **Products not used:** Moon Trek tiles are used for display only; the south-polar Trek catalog endpoints are unavailable; the NAC DTM confidence map is not used.
- **Assumptions:** all array, battery and load parameters are assumptions; none is flight-qualified.
