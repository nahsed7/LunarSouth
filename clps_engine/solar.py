"""Model 1 (generic): solar availability against a terrain horizon for ANY site/dataset."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .horizon import HorizonMask, horizon_at
from .windows import runs_table, summarise_flags, time_remaining_hours


def apply_solar_visibility(
    geometry: pd.DataFrame,
    mask: Optional[HorizonMask] = None,
    sun_disc_radius_deg: float = 0.25,
) -> pd.DataFrame:
    """
    Mark samples where the Sun clears the local terrain horizon.

    `sun_disc_radius_deg` ~0.25° (half of ~0.5° solar disc). Use 0.0 for
    centre-of-disc rule (matches most existing Model 1 scripts).
    """
    df = geometry.copy()
    hz = horizon_at(mask, df["sun_az"].to_numpy(float))
    df["sun_horizon_deg"] = hz
    df["sunlit"] = (df["sun_alt"].to_numpy(float) + sun_disc_radius_deg) > hz
    df["solar_elevation_above_horizon_deg"] = df["sun_alt"] - hz
    r = df["sun_radius"].to_numpy(float) if "sun_radius" in df.columns else np.full(len(df), 0.2666)
    # Fraction of the solar disc above the local terrain horizon (partial illumination)
    df["sun_disc_fraction"] = disc_fraction_above(df["sun_alt"].to_numpy(float) - hz, r)
    if "earth_alt" in df.columns and "earth_az" in df.columns:
        ehz = horizon_at(mask, df["earth_az"].to_numpy(float))
        er = df["earth_radius"].to_numpy(float) if "earth_radius" in df.columns else np.full(len(df), 0.95)
        df["earth_disc_fraction"] = disc_fraction_above(df["earth_alt"].to_numpy(float) - ehz, er)
    return df


def disc_fraction_above(centre_minus_horizon_deg: np.ndarray, radius_deg: np.ndarray) -> np.ndarray:
    """Area fraction of a disc above a locally straight horizon line (circular segment)."""
    d = np.clip(np.asarray(centre_minus_horizon_deg, float) / np.maximum(radius_deg, 1e-9), -1.0, 1.0)
    return 1.0 - (np.arccos(d) - d * np.sqrt(1.0 - d * d)) / np.pi


def solar_report(df: pd.DataFrame, step_min: float) -> dict:
    """Summary metrics for power/illumination planning."""
    sun = df["sunlit"].to_numpy(bool)
    runs = runs_table(df["time_utc"], sun, step_min, ("sunlit", "dark"))
    lit_h = runs.loc[runs["state"] == "sunlit", "hours"]
    dark_h = runs.loc[runs["state"] == "dark", "hours"]
    base = {
        "sunlit_pct": float(100.0 * sun.mean()),
        "longest_sunlight_h": float(lit_h.max()) if len(lit_h) else 0.0,
        "longest_darkness_h": float(dark_h.max()) if len(dark_h) else 0.0,
        "mean_sun_alt_when_lit_deg": float(df.loc[df["sunlit"], "sun_alt"].mean()) if sun.any() else float("nan"),
        "max_sun_alt_deg": float(df["sun_alt"].max()),
        "min_sun_alt_deg": float(df["sun_alt"].min()),
    }
    # Hours remaining at start of window
    if len(df):
        base["hours_to_sunset_at_t0"] = time_remaining_hours(sun, 0, step_min) if sun[0] else 0.0
        base["hours_to_sunrise_at_t0"] = time_remaining_hours(~sun, 0, step_min) if not sun[0] else 0.0
    return base


def solar_windows(df: pd.DataFrame, step_min: float) -> pd.DataFrame:
    return runs_table(df["time_utc"], df["sunlit"], step_min, labels=("sunlit", "dark"))
