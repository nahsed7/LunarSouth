"""Catalogue of real lunar-surface equipment for the Model 5 loadout builder.

Power and mass are *rounded public figures* (mission fact sheets, conference papers, press kits) intended for
concept-level budgets. They are approximate by design - verify against the instrument ICD before using them for a
real mission. Each item becomes a Model 5 subsystem (load) or, for power sources, adds constant bus power
(`SolarArrayConfig.rps_w`) and/or warm-box heat (`ThermalConfig.waste_heat_w`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .energy import Subsystem


@dataclass(frozen=True)
class Equipment:
    id: str
    name: str
    category: str            # science | sampling | comms | computing | mobility | isru | thermal | infrastructure | power
    heritage: str
    active_w: float
    standby_w: float = 0.0
    schedule: str = "always"  # Model 5 schedule
    duty_period_h: float = 24.0
    duty_on_frac: float = 1.0
    priority: int = 3
    mass_kg: float = 0.0
    rps_w: float = 0.0        # electrical output (power sources)
    heat_w: float = 0.0       # heat delivered to the warm box (RHU / RTG waste heat)
    note: str = ""

    def to_dict(self):
        return asdict(self)


CATALOG: list[Equipment] = [
    # ---- science instruments
    Equipment("NSS", "Neutron Spectrometer System (NSS)", "science", "VIPER · Lunar Prospector heritage", 5, 1, "always", priority=3, mass_kg=1.6,
              note="Detects hydrogen (water ice) to ~1 m depth while driving."),
    Equipment("NIRVSS", "Near-IR Volatiles Spectrometer System (NIRVSS)", "science", "VIPER (NASA Ames)", 18, 2, "sunlit", priority=3, mass_kg=3.5,
              note="Surface hydration / mineralogy; has its own lamps for shadowed ground."),
    Equipment("MSOLO", "Mass Spectrometer Observing Lunar Operations (MSolo)", "science", "PRIME-1 · VIPER (KSC/INFICON)", 25, 3, "duty",
              duty_period_h=24, duty_on_frac=0.3, priority=3, mass_kg=6.0, note="Identifies volatiles released by drilling and outgassing."),
    Equipment("LISTER", "Lunar Instrumentation for Subsurface Thermal Exploration (LISTER)", "science", "Blue Ghost M1 (Texas Tech / Honeybee)",
              30, 2, "duty", duty_period_h=48, duty_on_frac=0.15, priority=3, mass_kg=6.0, note="Pneumatic heat-flow probe to ~3 m."),
    Equipment("LMS", "Lunar Magnetotelluric Sounder (LMS)", "science", "Blue Ghost M1 (SwRI)", 6, 1, "always", priority=3, mass_kg=6.5,
              note="Electric/magnetic fields to probe the interior conductivity."),
    Equipment("SEIS", "Broadband seismometer", "science", "Farside Seismic Suite · Apollo PSE heritage", 6, 6, "always", priority=2, mass_kg=8.0,
              note="Needs continuous power and a stable warm box through the night."),
    Equipment("NGLR", "Next Generation Lunar Retroreflector (NGLR-1)", "science", "Blue Ghost M1 (Univ. Maryland)", 0, 0, "always", priority=4,
              mass_kg=1.5, note="Passive laser target - zero power, survives any night."),
    Equipment("LND", "Radiation dosimeter (LND class)", "science", "Chang'e-4 LND heritage", 5, 1, "always", priority=3, mass_kg=1.0,
              note="Surface radiation environment for crew planning."),
    Equipment("ROLSES", "Radio-wave Observations at the Lunar Surface (ROLSES)", "science", "IM-1 Odysseus (NASA GSFC)", 10, 1, "always",
              priority=4, mass_kg=4.0, note="Low-frequency radio spectrometer."),
    Equipment("SCALPSS", "Stereo cameras (SCALPSS)", "science", "IM-1 · Blue Ghost M1 (NASA LaRC)", 15, 0, "duty", duty_period_h=12,
              duty_on_frac=0.1, priority=4, mass_kg=2.5, note="Plume-surface interaction and site imaging."),
    Equipment("NAVCAM", "Navigation / hazard cameras (pair)", "mobility", "VIPER · Yutu-2 heritage", 6, 0, "duty", duty_period_h=6,
              duty_on_frac=0.5, priority=2, mass_kg=1.2, note="Stereo navigation imaging."),
    Equipment("LIGHTS", "LED floodlights for shadowed regions", "mobility", "VIPER headlights", 40, 0, "dark", priority=4, mass_kg=1.5,
              note="Needed to drive and image inside permanently shadowed regions."),
    # ---- sampling / excavation
    Equipment("TRIDENT", "TRIDENT 1 m rotary-percussive drill", "sampling", "PRIME-1 · VIPER (Honeybee Robotics)", 200, 5, "duty",
              duty_period_h=72, duty_on_frac=0.04, priority=4, mass_kg=20.0, note="Brings up cuttings from 1 m in 10 cm bites."),
    Equipment("PVAC", "Lunar PlanetVac pneumatic sampler", "sampling", "Blue Ghost M1 (Honeybee Robotics)", 50, 0, "duty", duty_period_h=168,
              duty_on_frac=0.02, priority=4, mass_kg=5.0, note="Gas-jet regolith sample collection."),
    Equipment("IPEX", "Regolith excavator (RASSOR / IPEx class)", "isru", "NASA KSC Swamp Works", 150, 5, "duty", duty_period_h=24,
              duty_on_frac=0.25, priority=4, mass_kg=30.0, note="Counter-rotating bucket drums; low reaction force."),
    # ---- ISRU
    Equipment("MOXIE", "Solid-oxide oxygen generator (MOXIE class)", "isru", "Mars 2020 MOXIE heritage", 300, 10, "sunlit", priority=4,
              mass_kg=17.0, note="Makes O2 (MOXIE: ~6-12 g/h on Mars); lunar oxide-reduction plants are far bigger."),
    Equipment("SUBLIM", "Ice sublimation / volatile extraction oven", "isru", "Concept (NASA ISRU roadmap)", 250, 5, "duty", duty_period_h=24,
              duty_on_frac=0.2, priority=4, mass_kg=15.0, note="Heats icy regolith and captures water vapour."),
    # ---- communications (transmit sessions are scheduled in the activity planner)
    Equipment("IRIS", "Iris V2 X-band transponder (receive / standby)", "comms", "JPL - MarCO, Artemis I CubeSats", 12, 12, "always",
              priority=1, mass_kg=1.2, note="Deep-space transponder; TX power is set in the activity planner."),
    Equipment("SBAND", "S-band transceiver (standby)", "comms", "CLPS landers", 8, 8, "always", priority=1, mass_kg=1.0,
              note="TT&C link to DSN / commercial ground stations."),
    Equipment("UHF", "UHF proximity radio (rover / relay link)", "comms", "Apollo LRV · Mars relay heritage", 10, 1, "duty", duty_period_h=4,
              duty_on_frac=0.25, priority=2, mass_kg=0.8, note="Short-range rover-to-lander link."),
    # ---- computing
    Equipment("RAD750", "RAD750 flight computer", "computing", "LRO · Curiosity · Perseverance", 10, 10, "always", priority=1, mass_kg=1.0,
              note="Radiation-hardened single-board computer."),
    Equipment("HPSC", "High-performance processor (HPSC class)", "computing", "NASA HPSC chiplet (in development)", 15, 1, "duty",
              duty_period_h=24, duty_on_frac=0.2, priority=4, mass_kg=0.5, note="Onboard image processing / autonomy bursts."),
    # ---- thermal
    Equipment("TCE", "Thermal control electronics + heat-switches", "thermal", "Generic lander bus", 3, 3, "always", priority=1, mass_kg=1.5,
              note="Heater controllers and thermostats."),
    Equipment("RHU8", "Radioisotope heater units ×8 (1 Wt each)", "thermal", "Apollo · Yutu · Mars rovers", 0, 0, "always", priority=1,
              mass_kg=0.4, heat_w=8.0, note="Plutonium-238 pellets: 1 W of heat each, no electricity, no switching."),
    # ---- infrastructure
    Equipment("DUSTSH", "Electrodynamic Dust Shield (EDS)", "infrastructure", "Blue Ghost M1 (NASA KSC)", 5, 0, "duty", duty_period_h=24,
              duty_on_frac=0.05, priority=4, mass_kg=1.0, note="Lifts dust off radiators, lenses and panels with electric fields."),
    Equipment("CHARGER", "Micro-rover charging port", "infrastructure", "MoonRanger / CADRE class rovers", 60, 0, "sunlit", priority=4,
              mass_kg=2.0, note="Recharges a small rover parked at the outpost."),
    # ---- power sources (constant generation and/or heat)
    Equipment("MMRTG", "MMRTG radioisotope generator", "power", "Curiosity · Perseverance · Dragonfly", 0, 0, "always", priority=1,
              mass_kg=45.0, rps_w=110.0, heat_w=150.0, note="~110 We at start of life (~2 kW of heat; part routed to the warm box)."),
    Equipment("KILOPOWER", "Kilopower 1 kWe fission reactor", "power", "KRUSTY ground test 2018 (NASA/DOE)", 0, 0, "always", priority=1,
              mass_kg=400.0, rps_w=1000.0, note="Stirling-converter fission unit; sized for night-long outposts."),
    Equipment("FSP40", "Fission Surface Power 40 kWe", "power", "NASA Fission Surface Power project (target)", 0, 0, "always", priority=1,
              mass_kg=6000.0, rps_w=40000.0, note="Artemis base-camp class power plant (design target)."),
]
BY_ID = {e.id: e for e in CATALOG}
CATEGORIES = sorted({e.category for e in CATALOG})

LOADOUTS = {
    "VIPER-class prospecting rover": {"NSS": 1, "NIRVSS": 1, "MSOLO": 1, "TRIDENT": 1, "NAVCAM": 1, "LIGHTS": 1, "SBAND": 1, "RAD750": 1, "TCE": 1},
    "PRIME-1 drill + mass spectrometer": {"TRIDENT": 1, "MSOLO": 1, "SBAND": 1, "RAD750": 1, "TCE": 1},
    "Blue Ghost-style geophysics lander": {"LISTER": 1, "LMS": 1, "NGLR": 1, "SCALPSS": 1, "PVAC": 1, "DUSTSH": 1, "LND": 1, "SBAND": 1,
                                           "RAD750": 1, "TCE": 1},
    "Night-surviving seismic station (RTG)": {"SEIS": 1, "LND": 1, "IRIS": 1, "RAD750": 1, "TCE": 1, "RHU8": 2, "MMRTG": 1},
    "ISRU pilot plant": {"IPEX": 1, "MOXIE": 1, "SUBLIM": 1, "MSOLO": 1, "IRIS": 1, "RAD750": 1, "HPSC": 1, "TCE": 1, "CHARGER": 1},
    "Artemis base-camp power (fission)": {"KILOPOWER": 1, "MOXIE": 2, "SUBLIM": 1, "IPEX": 1, "IRIS": 1, "UHF": 1, "RAD750": 1, "TCE": 1,
                                          "LIGHTS": 2, "CHARGER": 1},
}


def catalog_frame() -> pd.DataFrame:
    df = pd.DataFrame([e.to_dict() for e in CATALOG])
    return df[["id", "name", "category", "heritage", "active_w", "standby_w", "schedule", "priority", "mass_kg", "rps_w", "heat_w", "note"]]


def average_w(e: Equipment, sunlit_frac: float = 0.5) -> float:
    """Expected mean draw of one unit for a rough budget (schedule-weighted)."""
    on = {"always": 1.0, "sunlit": sunlit_frac, "dark": 1 - sunlit_frac, "earth_visible": 0.6, "duty": e.duty_on_frac}.get(e.schedule, 1.0)
    return e.active_w * on + e.standby_w * (1 - on)


def loadout_items(qty: dict) -> list[tuple[Equipment, int]]:
    return [(BY_ID[k], int(n)) for k, n in qty.items() if k in BY_ID and int(n) > 0]


def loadout_subsystems(qty: dict) -> list[Subsystem]:
    subs = []
    for e, n in loadout_items(qty):
        if e.category == "power" or (e.active_w == 0 and e.standby_w == 0):
            continue
        label = f"⚙ {e.name}" + (f" ×{n}" if n > 1 else "")
        subs.append(Subsystem(label, e.active_w * n, e.standby_w * n, e.schedule, e.duty_period_h, e.duty_on_frac,
                              priority=e.priority, category=e.category))
    return subs


def loadout_sources(qty: dict) -> dict:
    items = loadout_items(qty)
    return {"rps_w": float(sum(e.rps_w * n for e, n in items)), "heat_w": float(sum(e.heat_w * n for e, n in items))}


def loadout_summary(qty: dict, sunlit_frac: float = 0.5) -> dict:
    items = loadout_items(qty)
    return {
        "items": int(sum(n for _, n in items)),
        "mass_kg": float(sum(e.mass_kg * n for e, n in items)),
        "peak_w": float(sum(e.active_w * n for e, n in items)),
        "mean_w": float(sum(average_w(e, sunlit_frac) * n for e, n in items)),
        "night_w": float(sum((e.active_w if e.schedule in ("always", "dark") else e.standby_w) * n for e, n in items)),
        **loadout_sources(qty),
    }
