# LunarSouth changelog - 2026-11-15 release

Each line is a separate, tested change. Tests: `tests/test_mission_sim.py` (sections A–E), `tests/test_upgraded_models.py`,
`tests/validate_project.py`, plus a headless click-through of every tab, map layer, view and location mode.

## 🏆 Rank & compare points (new tab)
1. New tab that scans for the best landing points and compares any chosen points side by side.
2. Whole-cap scan: every 1 km cell of the 80°S cap (~226 000 eligible cells) is scored in about 3 s.
3. Seven transparent criteria, each explained in its help text: sunlight, Earth visibility, flat ground, short nights, solar power, short blackouts and nearness to water-ice cold traps.
4. Five weight presets: balanced outpost, power-first, comms-first, water-ice prospecting, safe landing.
5. Every weight is adjustable, and a caption shows each criterion's effective share.
6. Search area: the whole cap, or a radius around the selected point.
7. Control over how many points to return (5–40).
8. Minimum separation between results (non-maximum suppression), so results are not ten copies of one ridge.
9. Maximum mean-slope filter.
10. Cells inside permanent shadow or with under 50% valid DEM are excluded automatically.
11. Ranked table with score-percentage progress bars and the key physical values.
12. Score heat map (2 km) with the ranked points and the selected point drawn on it.
13. Stacked chart showing why each point scores what it does (weighted contribution of each criterion).
14. CSV download of the ranked points.
15. "Run full M1–M5 on top K": re-scores the winners with the full physics for the chosen dates and re-ranks them.
16. Comparison basket of up to 16 points; duplicates are rejected.
17. Add the currently selected point.
18. Add typed coordinates.
19. Add any of the known sites (discoveries, project candidates, polar presets).
20. Add all scan results in one click.
21. Remove selected points, or clear the basket.
22. Full-model comparison: score %, feasibility index, sunlight, nights, Earth, blackouts, generation, minimum SoC, ops %, terrain, PSR distance and survival status.
23. The best value in each column is highlighted green.
24. KPI row: best point, best score, spread, and how many points survive the window.
25. Radar chart of each point's criterion profile.
26. Horizontal ranking bar chart.
27. CSV download of the comparison.
28. "Analyse this point" opens any compared point in every model tab.
29. Distance to the nearest permanent-shadow cell, from a distance transform of the LOLA PSR fraction.
30. The 5 km Model 1/2/4 lattice (darkness, blackout, power) is mapped onto every 1 km cell.
31. `scan.point_context()` gives any point its pole-wide product values.

## 🧰 Real equipment in Model 5
32. Catalogue of 29 real devices (`clps_engine/equipment.py`) with heritage mission, rounded public power and mass, and a plain-language note.
33. Six real-mission loadouts: VIPER-class rover, PRIME-1, Blue Ghost-style geophysics, RTG seismic station, ISRU pilot plant, Artemis fission base camp.
34. Editable quantity table with a category filter.
35. Loadout budget KPIs: units, mass, peak, mean (schedule-weighted) and night draw.
36. Nuclear power and warm-box heat KPIs.
37. Instant check: can the battery carry this loadout through the longest night at this point?
38. "Apply loadout to all models" turns every device into a Model 5 load with its own schedule and priority.
39. Option to keep or drop the generic bus loads.
40. One click removes all equipment.
41. Banner showing the active nuclear power and heat sources.
42. Model 4: constant RTG/fission output (`SolarArrayConfig.rps_w`) is added to the bus, with new `gen_solar_w` / `gen_rps_w` timeline columns.
43. Generation report splits solar and nuclear kWh.
44. Model 5: RHU and RTG waste heat (`ThermalConfig.waste_heat_w`) offsets the electric survival heaters 1:1, and the heater load never goes negative.
45. The browser physics engine applies the same heat offset, so the game and the batch model stay identical.
46. M4 tab notice when a nuclear source is part of the power mix.

## 🎮 Mission Sim
47. The drill is now the TRIDENT 1 m drill at 200 W (Honeybee, PRIME-1/VIPER).
48. The spectrometer is now NIRVSS with its lamps (VIPER).
49. New MSolo mass-spectrometer task.
50. New NSS neutron-count task.
51. The panorama uses the stereo NavCams.
52. Hint line naming the instrument heritage.
53. ⏭ Next sunrise: fast-forwards with full physics (batteries, heaters, rover).
54. ⏭ Next sunset: same, to the next sunset.
55. +1 day and +1 week skip buttons.
56. The Play button pulses while paused, so a stopped game is obvious.
57. Skip buttons are disabled after the mission ends.
58. Science-target hints name the real instruments.

## Platform, fixes and packaging
59. Fixed the `StreamlitWidgetAlreadyInstantiatedError` crash when clicking the South Pole map.
60. Clicking a non-polar mission on the global map now selects that preset instead of crashing on the latitude limit.
61. Mission profiles save and restore the equipment loadout.
62. Mission profiles save and restore the comparison points.
63. Added an "About" entry to the app menu.
64. Added a sidebar button that resets all settings.
65. Guide: new section on finding and ranking the best points.
66. Guide: new section on flying real hardware.
67. Eleven new regression tests (scan ordering, separation, slope filter, radius, scoring, catalogue, RTG power, SoC gain, heater offset).
68. Renamed the project to LunarSouth across the UI, reports, handoff notes and profile files.
69. Standalone package: big rasters are stored as gzip parts and rebuilt bit-exact (SHA-256) on first start (`clps_engine/data_bootstrap.py`).
70. Every file in the package is ≤ 24 MB, so GitHub's browser upload accepts it.
71. Pinned `requirements.txt`, `run.sh`, `run.bat`, `Dockerfile` and `DEPLOY.md` in the package (`tools/build_release.py`).
72. Left the unused 144 MB intermediate grid out of the package (found by tracing which files the app opens).
73. README quick-start section (run scripts, first-run data rebuild, hosting pointer).
74. README data credits (LOLA/PGDA, PDS, NAIF SPICE), non-endorsement statement and AI-assistance disclosure.
