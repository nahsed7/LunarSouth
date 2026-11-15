# Extensions (drop-in for Objective A files)

1. **New tabs:** copy `_example_extension.py` to `extensions/<feature>.py`, keeping the same `TAB_TITLE` / `render(ctx)` interface, and restart the app. The file appears as a 🧩 tab. Files starting with `_` are ignored.
2. **New site presets:** drop CSV files into `extensions/presets/`. Columns:
   - required: `site_id`, `latitude_deg`, `longitude_deg`
   - optional: `site_name`, `mission`, `agency`, `year`, `status`, `region`, `coord_note`

   They appear under *Mission presets* in the sidebar and in Compare.
3. **Engine code:** `ctx["run_for"](site, scenario=…, start=…, hrs=…)` runs the full M1→M5 chain (cached) for any site or date. No changes to `app.py` are needed.

A broken extension shows its error inside its own tab and never stops the app.
