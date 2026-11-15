"""Model 5 activity engine: time-varying power usage from real operations, fed into the battery simulation.

Activities (all parameters are configurable ASSUMPTIONS, deterministic - no random numbers):

  Communication sessions  - each session type has a repeat period, duration, RF output power ("amplitude") and
                            SSPA DC efficiency. Sessions are only scheduled while the DTE link is up (link budget +
                            DSN visibility). DC draw = P_rf / eta_SSPA + P_exciter. Data returned per session uses
                            the link-budget rate scaled by RF power (rate ~ P_rf at fixed margin).
  Mobility (rover)        - traverse windows (start day, hours per day, number of days) at speed v on slope theta:
                              P_wheel = m g_moon (Crr cos(theta) + sin(theta)) v / eta_drive
                              P_drive = P_wheel + P_drive_avionics (+ hazard-camera duty)
                            Energy per km E(v) = (P_hotel + P_drive(v)) / v  -> there is an optimum speed.
  Science / drilling      - bursts of power for a duration, repeated every period, optionally sunlit-only.
  Outpost stability       - SoC band holding, rolling 24 h net energy, autonomy, recovery time, SoC oscillation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

G_MOON = 1.62


@dataclass
class CommSession:
    name: str
    period_h: float
    duration_min: float
    rf_power_w: float
    sspa_eff: float = 0.35
    exciter_w: float = 6.0
    priority: int = 2

    def dc_w(self):
        return self.rf_power_w / max(self.sspa_eff, 1e-3) + self.exciter_w


@dataclass
class Traverse:
    enabled: bool = False
    start_day: float = 1.0
    days: float = 5.0
    hours_per_day: float = 4.0
    speed_m_s: float = 0.10
    mass_kg: float = 250.0
    slope_deg: float = 5.0
    crr: float = 0.15                # ASSUMPTION - rolling resistance in loose regolith
    drive_eff: float = 0.55          # ASSUMPTION - motor + gearbox
    drive_avionics_w: float = 35.0   # ASSUMPTION - motor controllers, hazcams, nav compute
    sunlit_only: bool = True
    priority: int = 3

    def drive_w(self, v=None):
        v = self.speed_m_s if v is None else v
        th = np.radians(self.slope_deg)
        return self.mass_kg * G_MOON * (self.crr * np.cos(th) + np.sin(th)) * v / max(self.drive_eff, 1e-3) + self.drive_avionics_w


@dataclass
class Burst:
    name: str = "Drill / instrument burst"
    power_w: float = 150.0
    duration_min: float = 45.0
    period_h: float = 24.0
    sunlit_only: bool = True
    priority: int = 3


@dataclass
class ActivityPlan:
    comms: list = field(default_factory=lambda: [
        CommSession("Telemetry burst", 4.0, 10.0, 5.0),
        CommSession("Science downlink", 12.0, 60.0, 20.0),
        CommSession("Video / high-rate pass", 24.0, 30.0, 40.0, priority=3),
    ])
    traverse: Traverse = field(default_factory=Traverse)
    bursts: list = field(default_factory=lambda: [Burst()])

    def to_json(self):
        return {"comms": [asdict(c) for c in self.comms], "traverse": asdict(self.traverse), "bursts": [asdict(b) for b in self.bursts]}

    @staticmethod
    def from_json(d):
        return ActivityPlan(comms=[CommSession(**c) for c in d.get("comms", [])], traverse=Traverse(**d.get("traverse", {})),
                            bursts=[Burst(**b) for b in d.get("bursts", [])])


def build_activity_loads(tl: pd.DataFrame, plan: ActivityPlan, step_min: float):
    """Returns (extra_loads dict for energy.simulate, sessions DataFrame, traverse DataFrame)."""
    n = len(tl)
    h = step_min / 60.0
    t_h = np.arange(n) * h
    link = tl["link_up"].to_numpy(bool) if "link_up" in tl else tl["earth_visible"].to_numpy(bool)
    rate = tl["dte_rate_bps"].to_numpy(float) if "dte_rate_bps" in tl else np.zeros(n)
    sun = tl["sunlit"].to_numpy(bool)
    loads, sessions = {}, []
    ref_rf = 10.0  # link budget computed at 10 W RF (LinkBudgetConfig default); rate scales ~ linearly with RF power
    for c in plan.comms:
        arr = np.zeros(n)
        steps = max(1, int(round(c.duration_min / step_min)))
        k = 0
        while k * c.period_h < t_h[-1] + h:
            i0 = int(np.searchsorted(t_h, k * c.period_h))
            # slide to the next link-up sample within one period
            j = i0
            lim = min(n, i0 + int(c.period_h / h))
            while j < lim and not link[j]:
                j += 1
            if j < lim:
                seg = np.arange(j, min(n, j + steps))
                seg = seg[link[seg]]
                if len(seg):
                    arr[seg] = c.dc_w()
                    cap = float(rate.max()) if rate.max() > 0 else 0.0  # radio max rate (link budget already capped)
                    vol = float(np.minimum(rate[seg] * c.rf_power_w / ref_rf, cap).sum() * h * 3600 / 8e6)  # MB
                    sessions.append({"session": c.name, "start_utc": tl.time_utc.iloc[seg[0]], "minutes": len(seg) * step_min,
                                     "rf_w": c.rf_power_w, "dc_w": c.dc_w(), "energy_wh": c.dc_w() * len(seg) * h, "data_mb": vol})
            k += 1
        loads[f"Comms · {c.name}"] = {"series": arr, "priority": c.priority, "category": "comms"}
    tr = plan.traverse
    trav = pd.DataFrame()
    if tr.enabled:
        on = np.zeros(n, bool)
        for d in range(int(np.ceil(tr.days))):
            a = (tr.start_day + d) * 24.0
            on |= (t_h >= a) & (t_h < a + tr.hours_per_day)
        if tr.sunlit_only:
            on &= sun
        p = tr.drive_w()
        loads["Mobility · traverse"] = {"series": np.where(on, p, 0.0), "priority": tr.priority, "category": "mobility"}
        dist = np.cumsum(np.where(on, tr.speed_m_s * h * 3600, 0.0))
        trav = pd.DataFrame({"time_utc": tl.time_utc, "driving": on, "distance_km": dist / 1000.0})
    for b in plan.bursts:
        steps = max(1, int(round(b.duration_min / step_min)))
        arr = np.zeros(n)
        k = 0
        while k * b.period_h < t_h[-1] + h:
            i0 = int(np.searchsorted(t_h, k * b.period_h))
            j = i0
            lim = min(n, i0 + int(b.period_h / h))
            while b.sunlit_only and j < lim and not sun[j]:
                j += 1
            if j < lim:
                arr[j:min(n, j + steps)] = b.power_w
            k += 1
        loads[f"Science · {b.name}"] = {"series": arr, "priority": b.priority, "category": "science"}
    return loads, pd.DataFrame(sessions), trav


def speed_energy_curve(tr: Traverse, hotel_w: float, speeds=None) -> pd.DataFrame:
    """Energy per km vs speed: E = (P_hotel + P_drive(v)) / v -> optimum where hotel load stops dominating."""
    speeds = np.linspace(0.01, 0.5, 60) if speeds is None else np.asarray(speeds)
    p = np.array([tr.drive_w(v) for v in speeds])
    e_km = (hotel_w + p) / speeds * (1000.0 / 3600.0)  # Wh per km
    return pd.DataFrame({"speed_m_s": speeds, "drive_w": p, "wh_per_km": e_km, "hours_per_km": 1000.0 / speeds / 3600.0})


def outpost_stability(tl: pd.DataFrame, battery, step_min: float) -> dict:
    h = step_min / 60.0
    soc = tl["soc"].to_numpy()
    safe = max(battery.min_safe_soc, 1 - battery.max_dod)
    net = (tl["gen_w"] - tl["load_served_w"]).to_numpy()
    w = max(1, int(24 / h))
    roll = pd.Series(net).rolling(w, min_periods=w).sum().to_numpy() * h / 1000.0
    roll = roll[np.isfinite(roll)]
    usable = battery.capacity_wh * (battery.initial_soc - (1 - battery.max_dod))
    dark = ~tl["sunlit"].to_numpy(bool)
    night_load = float(tl.loc[dark, "load_demand_w"].mean()) if dark.any() else float(tl["load_demand_w"].mean())
    imin = int(np.argmin(soc))
    rec = np.flatnonzero(soc[imin:] >= safe + 0.05)
    in_band = (soc >= safe) & (soc <= 0.98)
    idx = 100 * (0.4 * in_band.mean() + 0.3 * (roll >= 0).mean() if len(roll) else 0.0) + 30 * (1 - min(1, (soc.max() - soc.min())))
    return {
        "stability_index": float(np.clip(idx, 0, 100)),
        "soc_in_band_pct": float(100 * in_band.mean()),
        "positive_24h_energy_pct": float(100 * (roll >= 0).mean()) if len(roll) else float("nan"),
        "worst_24h_net_kwh": float(roll.min()) if len(roll) else float("nan"),
        "soc_swing_pct": float(100 * (soc.max() - soc.min())),
        "autonomy_h_on_battery": float(usable / max(night_load, 1e-6)),
        "recovery_h_after_min_soc": float(rec[0] * h) if rec.size else None,
        "night_mean_load_w": night_load,
    }
