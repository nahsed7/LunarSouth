"""Event-driven mission timeline: every state transition becomes a timestamped event.

Events: SUNRISE/SUNSET (terrain-corrected), EARTHRISE/EARTHSET, LINK AOS/LOS (link budget + DSN),
DSN HANDOFF, SHED START/END (Model 5 policy), SOC < SAFE, CRITICAL LOAD LOSS / RESTORED.
`countdowns(events, t)` returns T-minus to the next event of each kind from any mission time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ICON = {"SUNRISE": "☀️↑", "SUNSET": "☀️↓", "EARTHRISE": "🌍↑", "EARTHSET": "🌍↓", "LINK AOS": "📡↑", "LINK LOS": "📡↓",
        "DSN HANDOFF": "🔁", "SHED START": "⚠️", "SHED END": "✅", "SOC < SAFE": "🔋↓", "SOC ≥ SAFE": "🔋↑",
        "CRITICAL LOSS": "⛔", "CRITICAL RESTORED": "🟢"}


def _edges(t, flag, on, off, detail=None):
    f = np.asarray(flag, bool)
    rows = []
    idx = np.flatnonzero(f[1:] != f[:-1]) + 1
    for i in idx:
        rows.append({"time_utc": t.iloc[i], "event": on if f[i] else off, "detail": detail(i) if detail else ""})
    return rows


def build_events(tl: pd.DataFrame, safe_soc: float) -> pd.DataFrame:
    t = tl["time_utc"].reset_index(drop=True)
    tl = tl.reset_index(drop=True)
    ev = []
    ev += _edges(t, tl["sunlit"], "SUNRISE", "SUNSET", lambda i: f"Sun az {tl.sun_az[i]:.1f}°, horizon {tl.sun_horizon_deg[i]:.2f}°")
    ev += _edges(t, tl["earth_visible"], "EARTHRISE", "EARTHSET", lambda i: f"Earth az {tl.earth_az[i]:.1f}°")
    if "link_up" in tl:
        ev += _edges(t, tl["link_up"], "LINK AOS", "LINK LOS", lambda i: f"{tl.dsn_station[i]} · {tl.link_rate_bps[i]/1e3:.0f} kbps")
        st = tl["dsn_station"].to_numpy()
        for i in range(1, len(st)):
            if st[i] != "none" and st[i - 1] != "none" and st[i] != st[i - 1]:
                ev.append({"time_utc": t[i], "event": "DSN HANDOFF", "detail": f"{st[i-1]} → {st[i]}"})
    if "shedding" in tl:
        ev += _edges(t, tl["shedding"], "SHED START", "SHED END")
    if "soc" in tl:
        ev += _edges(t, tl["soc"] >= safe_soc, "SOC ≥ SAFE", "SOC < SAFE", lambda i: f"SoC {100*tl.soc[i]:.1f}%")
    if "critical_unserved" in tl:
        ev += _edges(t, tl["critical_unserved"], "CRITICAL LOSS", "CRITICAL RESTORED")
    e = pd.DataFrame(ev, columns=["time_utc", "event", "detail"]).sort_values("time_utc").reset_index(drop=True)
    e["icon"] = e["event"].map(ICON).fillna("•")
    return e


def countdowns(events: pd.DataFrame, now) -> pd.DataFrame:
    now = pd.Timestamp(now)
    fut = events[events.time_utc > now]
    rows = []
    for kind in ("SUNSET", "SUNRISE", "EARTHSET", "EARTHRISE", "LINK LOS", "LINK AOS", "DSN HANDOFF", "SOC < SAFE", "CRITICAL LOSS"):
        f = fut[fut.event == kind]
        if len(f):
            dt = (f.time_utc.iloc[0] - now).total_seconds() / 3600.0
            rows.append({"event": kind, "icon": ICON.get(kind, ""), "at_utc": f.time_utc.iloc[0], "t_minus_h": dt,
                         "t_minus": f"T−{int(dt // 24)}d {int(dt % 24):02d}h {int((dt * 60) % 60):02d}m", "detail": f.detail.iloc[0]})
        else:
            rows.append({"event": kind, "icon": ICON.get(kind, ""), "at_utc": pd.NaT, "t_minus_h": np.nan,
                         "t_minus": "not in window", "detail": ""})
    return pd.DataFrame(rows)
