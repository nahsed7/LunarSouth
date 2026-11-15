"""Drop-in extension mechanism (for Objective-A deliverables or any teammate module).

Put a Python file in  extensions/  that defines:
    TAB_TITLE = "My feature"            # tab label
    ORDER = 50                          # optional sort key
    def render(ctx: dict) -> None: ...  # draw with streamlit; ctx documented below
Optionally add site lists as CSVs in extensions/presets/ (auto-loaded as presets).

ctx keys: st, site, result (MissionResult), timeline, run_for(site, scenario=None, start=None, hrs=None, stp=None),
          all_sites (dict id->Site), start_iso, days, step_min, plot_layout (dict), viz_pole, viz_power, viz3d, root (Path).
A failing extension is shown as an error inside its own tab and never breaks the app.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = ROOT / "extensions"


def discover() -> list:
    mods = []
    for f in sorted(EXT_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"ext_{f.stem}", f)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            if hasattr(m, "render"):
                m.TAB_TITLE = getattr(m, "TAB_TITLE", f.stem)
                mods.append(m)
        except Exception as e:  # keep the app alive; show the problem in a stub tab
            class _Broken:
                TAB_TITLE = f"{f.stem} (error)"
                ORDER = 999
                err = e

                @staticmethod
                def render(ctx, _e=e):
                    ctx["st"].error(f"Extension {f.name} failed to load: {_e}")
            mods.append(_Broken)
    return sorted(mods, key=lambda m: getattr(m, "ORDER", 50))
