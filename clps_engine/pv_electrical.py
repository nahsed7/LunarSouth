"""Electrical (current / voltage) view of Model 4: how many amps the array delivers from the sunlight it receives.

Chain per time step (all inputs come from Model 1 + Model 4, i.e. SPICE/Horizons Sun geometry, LOLA horizon,
1361/r^2 irradiance, incidence, cell temperature):

  G_eff  = plane-of-array irradiance x IAM x dust x aging          [W/m^2]
  Isc    = Isc_ref (G_eff/1361) [1 + a (T - 28)]                   per cell
  Voc    = Voc_ref + b (T - 28) + n Vt ln(G_eff/1361)               per cell
  I-V    : single-diode curve I = Isc - I0 [exp(V / (n Vt)) - 1] through (0,Isc) and (Voc,0)
  String : Ns cells in series (sized so hot Vmp stays above the bus), Np strings in parallel (fills the array area)
  I_mp   = P_dc / V_mp(string)  - consistent with Model 4's P_dc
  I_bus  = P_bus / V_bus (regulated bus), I_batt = (charge - discharge) / V_batt

Cell figures are ASSUMPTIONS typical of space triple-junction GaAs cells (BOL, AM0, 28 C); they are not a
specific manufacturer's datasheet and not CSA/NASA specifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

K_B, Q = 1.380649e-23, 1.602176634e-19


@dataclass
class CellConfig:
    area_cm2: float = 30.18          # ASSUMPTION - common 3G-class cell size
    isc_ma_cm2: float = 17.0         # ASSUMPTION - AM0 short-circuit current density
    voc_v: float = 2.69              # ASSUMPTION - AM0 open-circuit voltage (3 junctions)
    vmp_v: float = 2.41              # ASSUMPTION
    imp_ma_cm2: float = 16.3         # ASSUMPTION
    disc_alpha_per_c: float = 0.0006  # ASSUMPTION - relative dIsc/dT
    dvoc_v_per_c: float = -0.0062    # ASSUMPTION - dVoc/dT
    ideality_total: float = 3.6      # ASSUMPTION - ~1.2 per junction x 3
    bus_v: float = 28.0              # regulated bus voltage
    battery_v: float = 28.8          # nominal Li-ion pack (8s)
    max_cell_temp_c: float = 120.0   # string sizing temperature

    def to_dict(self):
        return asdict(self)


def string_layout(cfg: CellConfig, array_area_m2: float, packing: float):
    vmp_hot = cfg.vmp_v + cfg.dvoc_v_per_c * (cfg.max_cell_temp_c - 28.0)
    ns = int(np.ceil((cfg.bus_v + 3.0) / max(vmp_hot, 0.5)))          # 3 V headroom for the regulator
    n_cells = int(array_area_m2 * packing * 1e4 // cfg.area_cm2)
    np_ = max(1, n_cells // ns)
    return ns, np_, ns * np_


def apply_electrical(tl: pd.DataFrame, array, cfg: CellConfig = CellConfig()) -> pd.DataFrame:
    out = tl.copy()
    ns, np_, ncell = string_layout(cfg, array.area_m2, array.packing_factor)
    g = (out["panel_irradiance_w_m2"] * out["iam"]).to_numpy(float) * array.dust_factor * array.degradation_factor
    t = np.nan_to_num(out["cell_temp_c"].to_numpy(float), nan=28.0)
    frac = np.clip(g / 1361.0, 0.0, None)
    vt = K_B * (t + 273.15) / Q
    isc_cell = cfg.isc_ma_cm2 * 1e-3 * cfg.area_cm2 * frac * (1 + cfg.disc_alpha_per_c * (t - 28.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        voc_cell = np.where(frac > 1e-4, cfg.voc_v + cfg.dvoc_v_per_c * (t - 28.0) + cfg.ideality_total * vt * np.log(np.maximum(frac, 1e-4)), 0.0)
    voc_cell = np.clip(voc_cell, 0.0, None)
    ff_ref = (cfg.vmp_v * cfg.imp_ma_cm2) / (cfg.voc_v * cfg.isc_ma_cm2)
    vmp_cell = voc_cell * (cfg.vmp_v / cfg.voc_v)
    out["cells_series"], out["strings_parallel"] = ns, np_
    out["isc_array_a"] = isc_cell * np_
    out["voc_string_v"] = voc_cell * ns
    out["vmp_string_v"] = vmp_cell * ns
    with np.errstate(divide="ignore", invalid="ignore"):
        out["imp_array_a"] = np.where(out["vmp_string_v"] > 0.5, out["gen_dc_w"] / out["vmp_string_v"], 0.0)
    out["bus_current_a"] = out["gen_w"] / cfg.bus_v
    if "batt_charge_w" in out:
        out["battery_current_a"] = (out["batt_charge_w"] - out["batt_discharge_w"]) / cfg.battery_v
        out["load_current_a"] = out["load_served_w"] / cfg.bus_v
    out["fill_factor_ref"] = ff_ref
    return out


def iv_curve(isc_a: float, voc_v: float, n_vt_string: float, points: int = 120):
    """Single-diode I-V through (0, Isc) and (Voc, 0); returns V, I, P arrays."""
    if isc_a <= 0 or voc_v <= 0:
        v = np.linspace(0, 1, points)
        return v, np.zeros(points), np.zeros(points)
    i0 = isc_a / (np.exp(voc_v / n_vt_string) - 1.0)
    v = np.linspace(0, voc_v, points)
    i = np.clip(isc_a - i0 * (np.exp(v / n_vt_string) - 1.0), 0, None)
    return v, i, v * i


def electrical_report(tl: pd.DataFrame, cfg: CellConfig) -> dict:
    lit = tl["imp_array_a"] > 1e-3
    return {
        "cells_series": int(tl["cells_series"].iloc[0]), "strings_parallel": int(tl["strings_parallel"].iloc[0]),
        "peak_array_current_a": float(tl["imp_array_a"].max()), "mean_array_current_lit_a": float(tl.loc[lit, "imp_array_a"].mean()) if lit.any() else 0.0,
        "peak_isc_a": float(tl["isc_array_a"].max()), "mean_vmp_string_v": float(tl.loc[lit, "vmp_string_v"].mean()) if lit.any() else 0.0,
        "charge_ah": float((tl["battery_current_a"].clip(lower=0)).sum() * (tl.time_utc.diff().dt.total_seconds().median() / 3600)) if "battery_current_a" in tl else float("nan"),
        "peak_bus_current_a": float(tl["bus_current_a"].max()),
    }
