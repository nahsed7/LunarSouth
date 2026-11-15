"""Model 4 - solar power generation for a lunar lander / surface payload.

Chain per time step (all quantities physical, SI units):

  G      = S0 / r_AU^2                              irradiance at Moon (W/m^2)
  f_disc = fraction of solar disc above terrain horizon (partial illumination)
  cos_i  = n_panel . s_sun                          array orientation / tracking
  IAM    = 1 - b0 (1/cos_i - 1)                     cover-glass incidence-angle modifier
  T_cell = radiative-equilibrium cell temperature (sunlit) [assumption, documented]
  eta(T) = eta_ref [1 + beta (T - T_ref)]           temperature coefficient
  P_dc   = G f_disc A cos_i IAM eta(T) f_dust f_age f_pack
  P_bus  = P_dc * eta_mppt * eta_harness            delivered to the power bus

P_bus is exactly zero whenever the disc is fully below the local terrain
horizon (terrain-obstructed darkness). Nothing here is flight-qualified: every
parameter is a configurable engineering assumption unless marked otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .windows import runs_table

SOLAR_CONSTANT_W_M2 = 1361.0     # TSI at 1 AU (Kopp & Lean 2011; IAU 2015 B3 nominal 1361 W/m^2)
STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass
class SolarArrayConfig:
    area_m2: float = 1.2                  # ASSUMPTION - active cell area
    cell_efficiency: float = 0.295        # ASSUMPTION - BOL triple-junction GaAs, AM0, 28 C
    ref_temp_c: float = 28.0              # AM0 test reference temperature
    temp_coeff_per_c: float = -0.0022     # ASSUMPTION - relative dEta/dT for TJ GaAs (~-0.2..-0.3 %/C)
    packing_factor: float = 0.90          # ASSUMPTION - cell/panel area ratio losses
    orientation: str = "vertical_tracking"  # horizontal | fixed_tilt | vertical_fixed | vertical_tracking | dual_axis
    tilt_deg: float = 90.0                # from horizontal (used by fixed_tilt / vertical_fixed)
    facing_az_deg: float = 0.0            # azimuth of panel normal (fixed modes), clockwise from north
    iam_b0: float = 0.05                  # ASSUMPTION - ASHRAE incidence-angle-modifier coefficient
    dust_factor: float = 0.95             # ASSUMPTION - transmission after landing plume / dust (1 = clean)
    degradation_factor: float = 0.97      # ASSUMPTION - radiation/UV/aging (EOL/BOL)
    mppt_efficiency: float = 0.96         # ASSUMPTION - MPPT / PCDU conversion
    harness_efficiency: float = 0.98      # ASSUMPTION - distribution / harness I^2R
    absorptance: float = 0.88             # ASSUMPTION - front-side solar absorptance (thermal model)
    emittance_front: float = 0.85         # ASSUMPTION
    emittance_back: float = 0.80          # ASSUMPTION
    min_op_temp_c: float = -150.0         # ASSUMPTION - qualification floor; below => 0 W (cold-start limit)
    max_op_temp_c: float = 120.0          # ASSUMPTION
    max_power_w: float = 0.0              # optional PCDU clip (0 = none)
    rps_w: float = 0.0                    # constant electrical output of a radioisotope / fission power system (MMRTG ~110 We BOL)

    def to_dict(self):
        return asdict(self)

    @property
    def nameplate_w(self) -> float:
        return SOLAR_CONSTANT_W_M2 * self.area_m2 * self.cell_efficiency * self.packing_factor


def sun_unit_vectors(sun_alt_deg, sun_az_deg):
    alt, az = np.radians(sun_alt_deg), np.radians(sun_az_deg)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)  # E, N, U


def incidence_cosine(sun_alt_deg, sun_az_deg, cfg: SolarArrayConfig) -> np.ndarray:
    s = sun_unit_vectors(np.asarray(sun_alt_deg, float), np.asarray(sun_az_deg, float))
    mode = cfg.orientation
    if mode == "dual_axis":
        mu = np.ones(len(s))
    elif mode == "vertical_tracking":  # vertical panel rotated about local vertical to face Sun azimuth
        mu = np.cos(np.radians(sun_alt_deg))
    elif mode == "horizontal":
        mu = s[:, 2]
    else:
        tilt = np.radians(90.0 if mode == "vertical_fixed" else cfg.tilt_deg)
        faz = np.radians(cfg.facing_az_deg)
        n = np.array([np.sin(tilt) * np.sin(faz), np.sin(tilt) * np.cos(faz), np.cos(tilt)])
        mu = s @ n
    return np.clip(mu, 0.0, 1.0)


def cell_temperature_c(irr_on_panel_w_m2: np.ndarray, eta_electrical: float, cfg: SolarArrayConfig,
                       sink_temp_k: float = 100.0) -> np.ndarray:
    """Two-sided radiative equilibrium: a G cos_i (1 - eta) = sigma (eps_f+eps_b)(T^4 - T_sink^4).

    Lunar-surface IR back-loading ignored (vertical panels mostly view space at the pole);
    a documented first-order assumption, not a thermal-desktop result.
    """
    q = cfg.absorptance * irr_on_panel_w_m2 * (1.0 - eta_electrical)
    eps = cfg.emittance_front + cfg.emittance_back
    t4 = q / (STEFAN_BOLTZMANN * eps) + sink_temp_k ** 4
    return t4 ** 0.25 - 273.15


def apply_generation(df: pd.DataFrame, cfg: SolarArrayConfig, step_min: float) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    r_au = out["sun_dist_au"].to_numpy(float) if "sun_dist_au" in out.columns else np.full(n, 1.0)
    g = SOLAR_CONSTANT_W_M2 / r_au ** 2
    if "sun_disc_fraction" in out.columns:
        f = out["sun_disc_fraction"].to_numpy(float)
    else:
        f = out["sunlit"].to_numpy(bool).astype(float)
    mu = incidence_cosine(out["sun_alt"].to_numpy(float), out["sun_az"].to_numpy(float), cfg)
    with np.errstate(divide="ignore"):
        iam = np.where(mu > 0.0, np.clip(1.0 - cfg.iam_b0 * (1.0 / np.maximum(mu, 1e-6) - 1.0), 0.0, 1.0), 0.0)
    irr_panel = g * f * mu
    t_cell = cell_temperature_c(irr_panel * iam, cfg.cell_efficiency, cfg)
    eta = cfg.cell_efficiency * (1.0 + cfg.temp_coeff_per_c * (t_cell - cfg.ref_temp_c))
    eta = np.clip(eta, 0.0, None)
    p_dc = irr_panel * iam * cfg.area_m2 * cfg.packing_factor * eta * cfg.dust_factor * cfg.degradation_factor
    temp_ok = (t_cell >= cfg.min_op_temp_c) & (t_cell <= cfg.max_op_temp_c)
    p_dc = np.where(temp_ok | (irr_panel <= 0), p_dc, 0.0)
    p_bus = p_dc * cfg.mppt_efficiency * cfg.harness_efficiency
    if cfg.max_power_w > 0:
        p_bus = np.minimum(p_bus, cfg.max_power_w)
    hours = step_min / 60.0
    out["irradiance_w_m2"] = g
    out["incidence_cos"] = mu
    out["iam"] = iam
    out["panel_irradiance_w_m2"] = irr_panel
    out["cell_temp_c"] = np.where(irr_panel > 0, t_cell, np.nan)
    out["cell_eff"] = np.where(irr_panel > 0, eta, np.nan)
    out["gen_dc_w"] = p_dc
    out["gen_solar_w"] = p_bus
    out["gen_rps_w"] = np.full(n, float(cfg.rps_w))
    p_bus = p_bus + float(cfg.rps_w)          # RTG / fission output adds to the bus day and night
    out["gen_w"] = p_bus
    out["gen_wh_step"] = p_bus * hours
    out["gen_cum_kwh"] = np.cumsum(p_bus * hours) / 1000.0
    out["illum_state"] = np.select([f >= 0.999, f > 0.0], ["full", "partial"], "dark")
    return out


def generation_report(df: pd.DataFrame, cfg: SolarArrayConfig, step_min: float) -> dict:
    p = df["gen_w"].to_numpy(float)
    hours = step_min / 60.0
    dur_h = len(df) * hours
    producing = p > 1e-3
    runs = runs_table(df["time_utc"], producing, step_min, ("gen", "zero"))
    zero = runs[runs.state == "zero"]
    daily = df.set_index("time_utc")["gen_wh_step"].resample("1D").sum() / 1000.0
    st = df["illum_state"].to_numpy()
    partial = st == "partial"
    # loss waterfall (energy-weighted)
    lit = df["panel_irradiance_w_m2"].to_numpy(float) > 0
    ideal = (df["irradiance_w_m2"] * cfg.area_m2 * cfg.packing_factor * cfg.cell_efficiency).to_numpy(float)
    e_ideal_always = ideal.sum() * hours / 1000.0
    return {
        "nameplate_w_at_1au": cfg.nameplate_w,
        "mean_gen_w": float(p.mean()),
        "peak_gen_w": float(p.max()),
        "mean_gen_when_producing_w": float(p[producing].mean()) if producing.any() else 0.0,
        "total_gen_kwh": float(p.sum() * hours / 1000.0),
        "rps_gen_kwh": float(cfg.rps_w * dur_h / 1000.0),
        "solar_gen_kwh": float(p.sum() * hours / 1000.0 - cfg.rps_w * dur_h / 1000.0),
        "mean_daily_gen_kwh": float(daily.mean()) if len(daily) else 0.0,
        "best_day": str(daily.idxmax().date()) if len(daily) else "",
        "best_day_kwh": float(daily.max()) if len(daily) else 0.0,
        "worst_day": str(daily.idxmin().date()) if len(daily) else "",
        "worst_day_kwh": float(daily.min()) if len(daily) else 0.0,
        "producing_pct": float(100.0 * producing.mean()),
        "partial_illum_pct": float(100.0 * partial.mean()),
        "gen_in_partial_kwh": float(p[partial].sum() * hours / 1000.0),
        "longest_zero_gen_h": float(zero.hours.max()) if len(zero) else 0.0,
        "capacity_factor_pct": float(100.0 * p.mean() / max(cfg.nameplate_w, 1e-9)),
        "unobstructed_normal_kwh": e_ideal_always,
        "mean_cell_temp_c": float(np.nanmean(df["cell_temp_c"])) if lit.any() else float("nan"),
        "duration_h": dur_h,
    }


def daily_generation(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("time_utc")["gen_wh_step"].resample("1D").sum() / 1000.0
    return d.rename("gen_kwh").reset_index()


def sensitivity(df: pd.DataFrame, cfg: SolarArrayConfig, step_min: float) -> pd.DataFrame:
    """One-at-a-time sensitivity of total generation to the main design parameters."""
    base = generation_report(apply_generation(df, cfg, step_min), cfg, step_min)["total_gen_kwh"]
    rows = []
    cases = [
        ("area_m2", [0.5, 1.5]), ("cell_efficiency", [0.8, 1.1]), ("dust_factor", [0.8, 1.0 / max(cfg.dust_factor, 1e-6)]),
        ("degradation_factor", [0.85, 1.0 / max(cfg.degradation_factor, 1e-6)]),
    ]
    for name, mults in cases:
        for m in mults:
            c = SolarArrayConfig(**{**cfg.to_dict(), name: min(getattr(cfg, name) * m, 1.0 if name != "area_m2" else 1e9)})
            e = generation_report(apply_generation(df, c, step_min), c, step_min)["total_gen_kwh"]
            rows.append({"parameter": name, "value": getattr(c, name), "total_gen_kwh": e,
                         "change_pct": 100.0 * (e - base) / max(base, 1e-9)})
    for o in ("horizontal", "vertical_fixed", "vertical_tracking", "dual_axis"):
        c = SolarArrayConfig(**{**cfg.to_dict(), "orientation": o})
        e = generation_report(apply_generation(df, c, step_min), c, step_min)["total_gen_kwh"]
        rows.append({"parameter": "orientation", "value": o, "total_gen_kwh": e,
                     "change_pct": 100.0 * (e - base) / max(base, 1e-9)})
    return pd.DataFrame(rows)
