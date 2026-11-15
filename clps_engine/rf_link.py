"""Direct-to-Earth RF link budget + DSN ground-station coverage, handoffs and optional relay.

Per time step:
  FSPL [dB]   = 20 log10(4 pi d f / c)                       (d from SPICE Earth range)
  EIRP [dBW]  = P_tx[dBW] + G_tx - L_tx - L_point(theta)      (pointing loss vs Earth elevation/horizon margin)
  C/N0 [dBHz] = EIRP - FSPL - L_misc + G/T - k_dB             (k_dB = -228.6 dBW/K/Hz)
  R_max [bps] = 10^((C/N0 - Eb/N0_req - margin_req)/10)       capped at the radio's max rate
The link is CLOSED only if Earth is geometrically visible (Model 2) AND at least one DSN
complex has the Moon above its elevation mask AND the margin >= 0.

Every RF parameter is a configurable ASSUMPTION (typical small-lander X-band values), not a
mission specification. Station coordinates are the published DSN complex locations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

C = 299792458.0
K_DB = -228.6

DSN = {  # name: (lat, lon east, height m) - Goldstone DSS-14, Madrid DSS-63, Canberra DSS-43 (approx. complex centres)
    "Goldstone": (35.4259, -116.8895, 1000.0),
    "Madrid": (40.4313, -4.2480, 830.0),
    "Canberra": (-35.4014, 148.9817, 690.0),
}


@dataclass
class LinkBudgetConfig:
    band: str = "X"
    freq_ghz: float = 8.45                 # X-band downlink
    tx_power_w: float = 10.0               # ASSUMPTION - SSPA RF output
    tx_antenna_gain_dbi: float = 20.0      # ASSUMPTION - small HGA / steerable patch array
    tx_losses_db: float = 1.5              # ASSUMPTION - cables, diplexer
    beamwidth_deg: float = 15.0            # ASSUMPTION - for pointing-loss model
    pointing_error_deg: float = 2.0        # ASSUMPTION
    gs_g_over_t_dbk: float = 50.0          # ASSUMPTION - DSN 34 m BWG at X-band, typical ~50-53 dB/K
    gs_elev_mask_deg: float = 10.0         # station elevation mask
    misc_losses_db: float = 2.0            # polarisation, atmosphere, implementation
    required_ebn0_db: float = 2.5          # ASSUMPTION - turbo/LDPC coded BPSK at BER 1e-6
    required_margin_db: float = 3.0
    max_rate_bps: float = 2.0e6            # radio limit
    relay_enabled: bool = False            # optional lunar relay orbiter
    relay_availability: float = 0.6        # ASSUMPTION - fraction of time relay in view when DTE closed/out
    relay_rate_bps: float = 1.0e5          # ASSUMPTION

    def to_dict(self):
        return asdict(self)


def dsn_moon_elevations(times: pd.Series) -> pd.DataFrame:
    """Moon elevation (deg) at each DSN complex (SPICE DE421 via Skyfield)."""
    from skyfield.api import wgs84

    from .geometry import load_kernels

    ts, eph, _, _ = load_kernels()
    tt = pd.to_datetime(pd.Series(times))
    t = ts.utc(tt.dt.year.values, tt.dt.month.values, tt.dt.day.values, tt.dt.hour.values, tt.dt.minute.values, tt.dt.second.values)
    out = {}
    for name, (la, lo, h) in DSN.items():
        st = eph["earth"] + wgs84.latlon(la, lo, elevation_m=h)
        alt, _, _ = st.at(t).observe(eph["moon"]).apparent().altaz()
        out[name] = alt.degrees
    return pd.DataFrame(out)


def apply_link_budget(df: pd.DataFrame, cfg: LinkBudgetConfig, step_min: float) -> pd.DataFrame:
    out = df.copy()
    d = out["earth_dist_km"].to_numpy(float) * 1000.0 if "earth_dist_km" in out.columns else np.full(len(out), 3.84e8)
    f = cfg.freq_ghz * 1e9
    fspl = 20 * np.log10(4 * np.pi * d * f / C)
    # Pointing loss: Gaussian beam approx 12 (theta/BW)^2 dB; extra off-boresight when Earth hugs the horizon (multipath/blockage)
    margin_geo = out["visibility_margin_deg"].to_numpy(float) if "visibility_margin_deg" in out.columns else np.full(len(out), 5.0)
    l_point = 12.0 * (cfg.pointing_error_deg / cfg.beamwidth_deg) ** 2 + np.where(margin_geo < 1.0, 3.0, 0.0)
    eirp = 10 * np.log10(cfg.tx_power_w) + cfg.tx_antenna_gain_dbi - cfg.tx_losses_db - l_point
    cn0 = eirp - fspl - cfg.misc_losses_db + cfg.gs_g_over_t_dbk - K_DB
    el = dsn_moon_elevations(out["time_utc"])
    up = el.to_numpy() > cfg.gs_elev_mask_deg
    best = np.where(up.any(axis=1), np.array(list(DSN))[np.argmax(np.where(up, el.to_numpy(), -99), axis=1)], "none")
    geo = out["earth_visible"].to_numpy(bool)
    rate = 10 ** ((cn0 - cfg.required_ebn0_db - cfg.required_margin_db) / 10.0)
    rate = np.minimum(rate, cfg.max_rate_bps)
    closed = geo & up.any(axis=1) & (rate > 0)
    out["fspl_db"] = fspl
    out["eirp_dbw"] = eirp
    out["cn0_dbhz"] = cn0
    out["link_margin_db"] = np.where(geo, cn0 - 10 * np.log10(np.maximum(rate, 1)) - cfg.required_ebn0_db, np.nan)
    for k in DSN:
        out[f"dsn_el_{k}"] = el[k].to_numpy()
    out["dsn_station"] = np.where(closed, best, "none")
    out["dte_rate_bps"] = np.where(closed, rate, 0.0)
    relay = np.zeros(len(out))
    if cfg.relay_enabled:
        # deterministic availability pattern (no random numbers): relay in view for the first `availability` of each 12 h orbit
        t_h = np.arange(len(out)) * step_min / 60.0
        in_view = (np.mod(t_h, 12.0) < 12.0 * cfg.relay_availability)
        relay = np.where(~closed & in_view, cfg.relay_rate_bps, 0.0)
    out["relay_rate_bps"] = relay
    out["link_rate_bps"] = out["dte_rate_bps"] + relay
    out["link_up"] = out["link_rate_bps"] > 0
    return out


def link_report(df: pd.DataFrame, cfg: LinkBudgetConfig, step_min: float) -> dict:
    from .windows import longest_run_hours

    h = step_min / 60.0
    up = df["link_up"].to_numpy(bool)
    st = df["dsn_station"].to_numpy()
    served = st[st != "none"]
    handoffs = int(np.sum(served[1:] != served[:-1])) if served.size > 1 else 0
    geo = df["earth_visible"].to_numpy(bool)
    return {
        "config": cfg.to_dict(),
        "link_closed_pct": float(100 * up.mean()),
        "geometric_visible_pct": float(100 * geo.mean()),
        "visible_but_no_station_pct": float(100 * (geo & (st == "none") & (df["relay_rate_bps"].to_numpy() == 0)).mean()),
        "longest_link_outage_h": float(longest_run_hours(~up, step_min)),
        "data_volume_gbit": float((df["link_rate_bps"].to_numpy() * h * 3600).sum() / 1e9),
        "mean_rate_when_up_kbps": float(df.loc[up, "link_rate_bps"].mean() / 1e3) if up.any() else 0.0,
        "median_cn0_dbhz": float(np.nanmedian(df.loc[geo, "cn0_dbhz"])) if geo.any() else float("nan"),
        "mean_fspl_db": float(df["fspl_db"].mean()),
        "dsn_handoffs": handoffs,
        "station_share_pct": {k: float(100 * (st == k).mean()) for k in DSN},
        "note": "Link budget uses configurable assumptions (10 W, 20 dBi, DSN 34 m G/T 50 dB/K, X-band). Not a mission link design.",
    }
