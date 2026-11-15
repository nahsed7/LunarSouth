"""Model 5 - power consumption, energy storage and mission energy survival.

Consumes the Model 4 generation timeline (gen_w) together with the Model 1/2
flags (sunlit, earth_visible) and walks a battery forward in time with
priority-based load shedding.

Every wattage, schedule and battery figure below is a configurable
ENGINEERING ASSUMPTION. No published CSA (or other agency) subsystem power
ratings exist for these loads; the CSA-sourced items are environmental /
requirement figures only (see data/csa_engineering_basis.json).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

SIGMA = 5.670374419e-8

# Priority: 1 = survival-critical (never shed), 2 = mission-essential, 3 = science, 4 = discretionary
SCHEDULES = ("always", "sunlit", "dark", "earth_visible", "duty", "window")


@dataclass
class Subsystem:
    name: str
    active_w: float
    standby_w: float = 0.0
    schedule: str = "always"         # always | sunlit | dark | earth_visible | duty | window
    duty_period_h: float = 24.0      # for duty schedule
    duty_on_frac: float = 1.0
    start_h: float = 0.0             # operating window relative to mission start
    end_h: float = 1e9
    priority: int = 2
    category: str = "other"

    def to_dict(self):
        return asdict(self)


@dataclass
class ThermalConfig:
    """Survival-heater demand from radiative leak through MLI to the lunar environment."""

    enabled: bool = True
    radiating_area_m2: float = 3.0           # ASSUMPTION
    mli_effective_emittance: float = 0.02    # ASSUMPTION (typical MLI e* 0.01-0.03)
    internal_min_temp_c: float = -20.0       # ASSUMPTION - avionics/battery survival set point
    env_temp_dark_c: float = -200.0          # CSA LUV requirement: "temperatures as low as -200 C"
    env_temp_lit_c: float = -30.0            # ASSUMPTION - grazing-Sun polar surface effective sink
    conductive_leak_w_per_k: float = 0.10    # ASSUMPTION - struts/harness
    heater_efficiency: float = 1.0
    waste_heat_w: float = 0.0                # RHUs + RTG waste heat routed into the warm box (offsets electric heaters)

    def demand_w(self, sunlit: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.zeros(len(sunlit))
        ti = self.internal_min_temp_c + 273.15
        te = np.where(sunlit, self.env_temp_lit_c, self.env_temp_dark_c) + 273.15
        rad = self.mli_effective_emittance * SIGMA * self.radiating_area_m2 * np.clip(ti ** 4 - te ** 4, 0, None)
        cond = self.conductive_leak_w_per_k * np.clip(ti - te, 0, None)
        return np.clip(rad + cond - self.waste_heat_w, 0, None) / max(self.heater_efficiency, 1e-6)


@dataclass
class BatteryConfig:
    capacity_wh: float = 4000.0              # ASSUMPTION - nameplate
    initial_soc: float = 0.90
    charge_efficiency: float = 0.95          # ASSUMPTION Li-ion coulombic+converter
    discharge_efficiency: float = 0.95
    max_dod: float = 0.80                    # ASSUMPTION - SoC never below 1 - max_dod
    min_safe_soc: float = 0.30               # ASSUMPTION - shed non-critical loads below this
    max_charge_w: float = 1000.0
    max_discharge_w: float = 1000.0
    self_discharge_pct_per_day: float = 0.05  # ASSUMPTION
    min_temp_note: str = "Battery assumed held >= internal_min_temp_c by survival heaters"

    def to_dict(self):
        return asdict(self)


def default_subsystems() -> list[Subsystem]:
    """Representative small CLPS-class lander payload (ASSUMPTIONS, editable in the UI)."""
    return [
        Subsystem("Avionics / C&DH computer", 25.0, 12.0, "always", priority=1, category="computing"),
        Subsystem("Survival keep-alive (watchdog, RTC)", 3.0, 3.0, "always", priority=1, category="standby"),
        Subsystem("Comms receiver / command standby", 5.0, 5.0, "always", priority=1, category="comms"),  # TX sessions: activities.py
        Subsystem("Navigation / housekeeping sensors", 8.0, 2.0, "always", priority=2, category="navigation"),
        Subsystem("Science payload suite", 45.0, 4.0, "sunlit", priority=3, category="science"),
        Subsystem("Imaging cameras", 20.0, 0.0, "duty", duty_period_h=12.0, duty_on_frac=0.25, priority=3, category="science"),
        Subsystem("Data processing burst", 30.0, 0.0, "duty", duty_period_h=24.0, duty_on_frac=0.2, priority=4, category="computing"),
    ]


def _schedule_mask(sub: Subsystem, t_h: np.ndarray, sunlit: np.ndarray, earth: np.ndarray) -> np.ndarray:
    win = (t_h >= sub.start_h) & (t_h < sub.end_h)
    s = sub.schedule
    if s == "always" or s == "window":
        on = np.ones_like(win)
    elif s == "sunlit":
        on = sunlit
    elif s == "dark":
        on = ~sunlit
    elif s == "earth_visible":
        on = earth
    elif s == "duty":
        on = (np.mod(t_h, sub.duty_period_h) < sub.duty_on_frac * sub.duty_period_h)
    else:
        raise ValueError(f"unknown schedule {s}")
    return on & win


def simulate(
    df: pd.DataFrame,
    subsystems: list[Subsystem],
    battery: BatteryConfig,
    thermal: ThermalConfig,
    step_min: float,
    extra_loads: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-stepped energy balance. Returns (timeline, per-subsystem summary)."""
    n = len(df)
    h = step_min / 60.0
    t_h = np.arange(n) * h
    sunlit = df["sunlit"].to_numpy(bool)
    earth = df["earth_visible"].to_numpy(bool) if "earth_visible" in df.columns else np.zeros(n, bool)
    gen = df["gen_w"].to_numpy(float)

    loads = {}
    prio = {}
    for s in subsystems:
        on = _schedule_mask(s, t_h, sunlit, earth)
        loads[s.name] = np.where(on, s.active_w, s.standby_w).astype(float)
        prio[s.name] = s.priority
    heater = thermal.demand_w(sunlit)
    loads["Survival heaters"] = heater
    prio["Survival heaters"] = 1
    cats = {s.name: s.category for s in subsystems}
    cats["Survival heaters"] = "thermal"
    for name, spec in (extra_loads or {}).items():  # activity engine (comms sessions, mobility, bursts)
        loads[name] = np.asarray(spec["series"], float)
        prio[name] = int(spec.get("priority", 3))
        cats[name] = spec.get("category", "activity")
    names = list(loads)
    L = np.vstack([loads[k] for k in names])          # (m, n)
    P = np.array([prio[k] for k in names])

    cap = battery.capacity_wh
    floor = cap * (1.0 - battery.max_dod)
    safe = cap * max(battery.min_safe_soc, 1.0 - battery.max_dod)
    sd = battery.self_discharge_pct_per_day / 100.0 / 24.0 * h
    e = cap * battery.initial_soc

    soc = np.zeros(n)
    served = np.zeros_like(L)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    clipped = np.zeros(n)
    deficit = np.zeros(n)
    shed_level = np.zeros(n, dtype=int)
    sd_loss = np.zeros(n)

    for i in range(n):
        e_prev = e
        e = max(e - e * sd, min(e, floor))  # self-discharge does not breach the DoD floor model
        sd_loss[i] = e_prev - e
        # shedding policy: below safe SoC keep only priority <= 2; at floor keep only priority 1
        allowed_p = 4 if e > safe else (2 if e > floor + 1e-9 else 1)
        want = L[:, i] * (P <= allowed_p)
        demand = want.sum()
        net = gen[i] - demand
        if net >= 0:
            c = min(net, battery.max_charge_w, (cap - e) / (battery.charge_efficiency * h) if h else 0)
            c = max(c, 0.0)
            e += c * battery.charge_efficiency * h
            charge[i] = c
            clipped[i] = net - c
            served[:, i] = want
        else:
            need = -net
            avail = max(0.0, (e - floor) * battery.discharge_efficiency / h)
            d = min(need, battery.max_discharge_w, avail)
            e -= d / battery.discharge_efficiency * h
            discharge[i] = d
            short = need - d
            if short > 1e-9:
                # shed from lowest priority upward until the shortfall is covered
                w = want.copy()
                for p in (4, 3, 2, 1):
                    if short <= 1e-9:
                        break
                    idx = np.where(P == p)[0]
                    blk = w[idx].sum()
                    cut = min(blk, short)
                    if blk > 0:
                        w[idx] *= (blk - cut) / blk
                    short -= cut
                served[:, i] = w
                deficit[i] = max(short, 0.0) + (want.sum() - w.sum())
            else:
                served[:, i] = want
        unscheduled_shed = (L[:, i] * (P > allowed_p)).sum()
        deficit[i] += unscheduled_shed
        shed_level[i] = 4 - allowed_p if allowed_p < 4 else 0
        soc[i] = e / cap

    out = df.copy()
    out["load_demand_w"] = L.sum(axis=0)
    out["load_served_w"] = served.sum(axis=0)
    out["heater_w"] = heater
    out["net_w"] = gen - out["load_served_w"].to_numpy()
    out["batt_charge_w"] = charge
    out["batt_discharge_w"] = discharge
    out["gen_clipped_w"] = clipped
    out["energy_deficit_w"] = deficit
    out["self_discharge_wh"] = sd_loss
    for k, name in enumerate(names):  # per-load served power, for stacked visuals
        out[f"load::{name}"] = served[k]
    out["soc"] = soc
    out["soc_pct"] = 100.0 * soc
    out["reserve_wh"] = np.clip(soc * cap - floor, 0, None)
    crit = (L[P == 1].sum(axis=0) - served[P == 1].sum(axis=0)) > 1e-6
    out["critical_unserved"] = crit
    out["shedding"] = shed_level > 0
    rows = []
    for k, name in enumerate(names):
        dem = L[k].sum() * h / 1000.0
        sv = served[k].sum() * h / 1000.0
        rows.append({"subsystem": name, "priority": int(P[k]),
                     "category": cats.get(name, "other"),
                     "demand_kwh": dem, "served_kwh": sv, "served_pct": 100.0 * sv / dem if dem > 0 else 100.0,
                     "mean_w": float(L[k].mean())})
    summ = pd.DataFrame(rows).sort_values("demand_kwh", ascending=False)
    total = summ["demand_kwh"].sum()
    summ["share_of_demand_pct"] = 100.0 * summ["demand_kwh"] / max(total, 1e-9)
    return out, summ


def survival_report(tl: pd.DataFrame, summ: pd.DataFrame, battery: BatteryConfig, step_min: float) -> dict:
    h = step_min / 60.0
    soc = tl["soc"].to_numpy()
    safe = max(battery.min_safe_soc, 1.0 - battery.max_dod)
    below = np.flatnonzero(soc <= safe + 1e-9)
    t_crit = float(below[0] * h) if below.size else None
    crit_fail = bool(tl["critical_unserved"].any())
    shed = tl["shedding"].to_numpy(bool) | (tl["energy_deficit_w"].to_numpy() > 1e-6)
    ops_supported = 100.0 * (1.0 - shed.mean())
    if crit_fail:
        status = "FAIL - critical loads unpowered"
    elif shed.any():
        status = "DEGRADED - survives with load shedding"
    else:
        status = "NOMINAL - all loads supported"
    first_fail = None
    if crit_fail:
        first_fail = str(tl.loc[tl["critical_unserved"], "time_utc"].iloc[0])
    return {
        "status": status,
        "survives": not crit_fail,
        "total_demand_kwh": float(summ["demand_kwh"].sum()),
        "total_served_kwh": float(summ["served_kwh"].sum()),
        "total_generated_kwh": float(tl["gen_w"].sum() * h / 1000.0),
        "generation_clipped_kwh": float(tl["gen_clipped_w"].sum() * h / 1000.0),
        "energy_deficit_kwh": float(tl["energy_deficit_w"].sum() * h / 1000.0),
        "max_deficit_w": float(tl["energy_deficit_w"].max()),
        "min_soc_pct": float(100.0 * soc.min()),
        "end_soc_pct": float(100.0 * soc[-1]),
        "min_reserve_wh": float(tl["reserve_wh"].min()),
        "hours_to_critical_soc": t_crit,
        "first_critical_failure_utc": first_fail,
        "operational_time_supported_pct": float(ops_supported),
        "top_consumer": str(summ.iloc[0]["subsystem"]) if len(summ) else "",
        "mean_demand_w": float(tl["load_demand_w"].mean()),
    }


# ------------------------------------------------------------------ scenarios
SCENARIOS = {
    "nominal": "Nominal mission operations",
    "survival": "Reduced-power survival mode (priority 1-2 only, science off)",
    "high_science": "High-power science operations (science x2.5, continuous imaging)",
    "comm_blackout": "Extended communications blackout (no DTE for 7 days from day 2)",
    "extended_dark": "Extended darkness (zero generation for 14 days from day 2 - CSA two-week lunar night)",
    "array_degraded": "Solar-array degradation (dust 0.70, aging 0.85)",
}


def apply_scenario(name: str, df: pd.DataFrame, subs: list[Subsystem], step_min: float):
    """Return (df, subsystems, array_overrides) for a named scenario."""
    df = df.copy()
    subs = copy.deepcopy(subs)
    over = {}
    n = len(df)
    h = step_min / 60.0
    t_h = np.arange(n) * h
    if name == "survival":
        subs = [s for s in subs if s.priority <= 2]
        for s in subs:
            if s.category == "comms":
                s.schedule, s.duty_period_h, s.duty_on_frac = "duty", 24.0, 0.1
    elif name == "high_science":
        for s in subs:
            if s.category == "science":
                s.active_w *= 2.5
                s.schedule = "always"
    elif name == "comm_blackout":
        blk = (t_h >= 48) & (t_h < 48 + 7 * 24)
        df["earth_visible"] = df["earth_visible"].to_numpy(bool) & ~blk
    elif name == "extended_dark":
        blk = (t_h >= 48) & (t_h < 48 + 14 * 24)
        df["force_dark"] = blk
        df["sunlit"] = df["sunlit"].to_numpy(bool) & ~blk
        if "sun_disc_fraction" in df.columns:
            df["sun_disc_fraction"] = np.where(blk, 0.0, df["sun_disc_fraction"])
    elif name == "array_degraded":
        over = {"dust_factor": 0.70, "degradation_factor": 0.85}
    return df, subs, over
