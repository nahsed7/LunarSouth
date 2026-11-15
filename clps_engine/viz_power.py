"""Power-generation and energy-flow visuals (Model 4 + Model 5) - dashboard grade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

C = dict(sun="#ffd54a", gen="#ffb02e", load="#ff5d73", batt="#3ddc97", earth="#4aa8ff", cyan="#38d9ff",
         dim="#7fa7b8", bg="rgba(0,0,0,0)", panel="rgba(4,8,15,.6)", txt="#d7f6ff")
BASE = dict(template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
            font=dict(family="Segoe UI, JetBrains Mono, sans-serif", size=12, color=C["txt"]), margin=dict(t=50, l=50, r=20, b=40))


def loss_waterfall(tl: pd.DataFrame, cfg, step_min: float) -> go.Figure:
    """Energy-weighted loss chain from unobstructed sunlight to energy delivered on the bus (kWh)."""
    h = step_min / 60.0
    G = tl["irradiance_w_m2"].to_numpy()
    A = cfg.area_m2
    f = tl["sun_disc_fraction"].to_numpy()
    mu = tl["incidence_cos"].to_numpy()
    iam = tl["iam"].to_numpy()
    eta = np.nan_to_num(tl["cell_eff"].to_numpy(), nan=cfg.cell_efficiency)
    e0 = (G * A).sum() * h / 1000  # sunlight on the aperture if never obstructed, normal incidence
    e1 = (G * A * f).sum() * h / 1000
    e2 = (G * A * f * mu).sum() * h / 1000
    e3 = (G * A * f * mu * iam).sum() * h / 1000
    e4 = e3 * cfg.packing_factor
    e5 = (G * A * f * mu * iam * cfg.packing_factor * eta).sum() * h / 1000
    e6 = e5 * cfg.dust_factor * cfg.degradation_factor
    e7 = tl["gen_w"].sum() * h / 1000
    steps = [("Sunlight on array (no terrain, normal)", e0, "absolute"),
             ("Terrain obstruction / partial disc", e1 - e0, "relative"),
             ("Orientation (cos i)", e2 - e1, "relative"),
             ("Cover-glass IAM", e3 - e2, "relative"),
             ("Packing factor", e4 - e3, "relative"),
             ("Cell conversion (η·T)", e5 - e4, "relative"),
             ("Dust + aging", e6 - e5, "relative"),
             ("MPPT + harness", e7 - e6, "relative"),
             ("Delivered to bus", e7, "total")]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=[s[2] for s in steps], x=[s[0] for s in steps], y=[s[1] for s in steps],
        text=[f"{s[1]:,.1f}" for s in steps], textposition="outside",
        connector=dict(line=dict(color="rgba(127,167,184,.4)")),
        increasing=dict(marker=dict(color=C["batt"])), decreasing=dict(marker=dict(color=C["load"])),
        totals=dict(marker=dict(color=C["sun"]))))
    fig.update_layout(**BASE, height=420, title="Energy loss chain - where the sunlight goes (kWh over mission)", yaxis_title="kWh",
                      xaxis=dict(tickangle=-25))
    return fig


def energy_sankey(tl: pd.DataFrame, summ: pd.DataFrame, step_min: float) -> go.Figure:
    h = step_min / 60.0
    gen = tl["gen_w"].to_numpy()
    served = tl["load_served_w"].to_numpy()
    ch, dis, clip = tl["batt_charge_w"].to_numpy(), tl["batt_discharge_w"].to_numpy(), tl["gen_clipped_w"].to_numpy()
    direct = np.minimum(gen, served)
    E = lambda a: float(a.sum() * h / 1000)
    loads = summ[summ.served_kwh > 0].sort_values("served_kwh", ascending=False)
    nodes = ["Solar array", "Battery", "Clipped / unused", "Power bus"] + list(loads.subsystem) + ["Unserved demand"]
    src, dst, val, col = [], [], [], []

    def link(a, b, v, c):
        if v > 1e-6:
            src.append(nodes.index(a)); dst.append(nodes.index(b)); val.append(v); col.append(c)

    link("Solar array", "Power bus", E(direct), "rgba(255,213,74,.45)")
    link("Solar array", "Battery", E(ch), "rgba(61,220,151,.45)")
    link("Solar array", "Clipped / unused", E(clip), "rgba(127,167,184,.35)")
    link("Battery", "Power bus", E(dis), "rgba(61,220,151,.45)")
    for _, r in loads.iterrows():
        link("Power bus", r.subsystem, float(r.served_kwh), "rgba(56,217,255,.35)")
    for _, r in summ.iterrows():
        un = float(r.demand_kwh - r.served_kwh)
        if un > 1e-3 and r.subsystem in nodes:
            link(r.subsystem, "Unserved demand", un, "rgba(255,93,115,.5)")
    ncol = [C["sun"], C["batt"], C["dim"], C["cyan"]] + ["#5fb7d4"] * len(loads) + [C["load"]]
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=nodes, color=ncol, pad=14, thickness=16,
                                                            line=dict(color="rgba(0,0,0,0)")),
                              link=dict(source=src, target=dst, value=val, color=col,
                                        hovertemplate="%{source.label} → %{target.label}: %{value:.2f} kWh<extra></extra>")))
    fig.update_layout(**BASE, height=460, title="Energy flow over the mission (kWh) - array → battery / bus → subsystems")
    return fig


def calendar_heatmap(tl: pd.DataFrame) -> go.Figure:
    d = tl[["time_utc", "gen_w"]].copy()
    d["day"] = d.time_utc.dt.floor("D")
    d["hour"] = d.time_utc.dt.hour + d.time_utc.dt.minute / 60
    d["hr"] = (d["hour"] // 1).astype(int)
    pv = d.pivot_table(index="hr", columns="day", values="gen_w", aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=[[0, "#060b14"], [0.15, "#2a1a05"], [0.5, "#b8650a"], [1, "#ffe27a"]],
                               colorbar=dict(title="W"), hovertemplate="%{x|%Y-%m-%d} %{y}:00 UTC<br>%{z:.0f} W<extra></extra>"))
    fig.update_layout(**BASE, height=330, title="Generation fingerprint - day × hour (UTC); dark = no power", yaxis_title="hour UTC")
    return fig


def power_rose(tl: pd.DataFrame, step_min: float) -> go.Figure:
    h = step_min / 60.0
    b = (tl["sun_az"] // 15 * 15).astype(int)
    e = (tl["gen_w"] * h / 1000).groupby(b).sum().reindex(range(0, 360, 15), fill_value=0)
    blocked = ((tl["sun_alt"] > 0) & (tl["sun_disc_fraction"] < 0.01)).groupby(b).mean().reindex(range(0, 360, 15), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=e.values, theta=e.index + 7.5, width=14, marker_color=C["sun"], name="Energy kWh",
                              hovertemplate="Sun az %{theta:.0f}°<br>%{r:.2f} kWh<extra></extra>"))
    fig.add_trace(go.Barpolar(r=blocked.values * (e.max() if e.max() > 0 else 1), theta=blocked.index + 7.5, width=14,
                              marker_color="rgba(255,93,115,.45)", name="Sun up but terrain-blocked (share)"))
    fig.update_layout(**BASE, height=420, title="Power rose - energy by Sun azimuth & terrain blockage",
                      polar=dict(angularaxis=dict(direction="clockwise", rotation=90), bgcolor=C["panel"]))
    return fig


def gauges(gen_report: dict, surv: dict, battery) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3])
    floor = 100 * (1 - battery.max_dod)
    fig.add_trace(go.Indicator(mode="gauge+number", value=gen_report["capacity_factor_pct"], number=dict(suffix="%"),
                               title=dict(text="Array capacity factor"), gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=C["sun"]))), 1, 1)
    fig.add_trace(go.Indicator(mode="gauge+number", value=surv["min_soc_pct"], number=dict(suffix="%"), title=dict(text="Minimum battery SoC"),
                               gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=C["batt"]),
                                          steps=[dict(range=[0, floor], color="rgba(255,93,115,.35)"),
                                                 dict(range=[floor, 100 * max(battery.min_safe_soc, 1 - battery.max_dod)], color="rgba(255,200,87,.25)")])), 1, 2)
    fig.add_trace(go.Indicator(mode="gauge+number", value=surv["operational_time_supported_pct"], number=dict(suffix="%"),
                               title=dict(text="Operations supported"), gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=C["cyan"]))), 1, 3)
    fig.update_layout(**{**BASE, "margin": dict(t=30, l=30, r=30, b=10)}, height=230)
    return fig


def power_timeline(tl: pd.DataFrame) -> go.Figure:
    """Generation vs demand with the battery flow and illumination state ribbon."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.3, 0.15], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.gen_w, name="Generated (bus) W", line=dict(color=C["sun"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(255,213,74,.18)"), 1, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.load_demand_w, name="Demand W", line=dict(color=C["load"], width=1.2, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.load_served_w, name="Served W", line=dict(color=C["cyan"], width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.batt_charge_w, name="Charging W", line=dict(color=C["batt"], width=0), fill="tozeroy",
                             fillcolor="rgba(61,220,151,.35)"), 2, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=-tl.batt_discharge_w, name="Discharging W", line=dict(color=C["load"], width=0), fill="tozeroy",
                             fillcolor="rgba(255,93,115,.35)"), 2, 1)
    state = tl["illum_state"].map({"full": 2, "partial": 1, "dark": 0})
    fig.add_trace(go.Heatmap(x=tl.time_utc, y=["Sun"], z=[state.values], colorscale=[[0, "#11182a"], [0.5, "#b8650a"], [1, "#ffe27a"]],
                             showscale=False, hovertemplate="%{x}<br>0 dark · 1 partial · 2 full: %{z}<extra></extra>"), 3, 1)
    fig.add_trace(go.Heatmap(x=tl.time_utc, y=["Earth"], z=[tl.earth_visible.astype(int).values], colorscale=[[0, "#11182a"], [1, C["earth"]]],
                             showscale=False), 3, 1)
    fig.update_layout(**BASE, height=560, hovermode="x unified", legend=dict(orientation="h", y=-0.08),
                      title="Power timeline - generation, demand, battery flow, Sun/Earth state")
    fig.update_yaxes(title_text="W", row=1, col=1)
    fig.update_yaxes(title_text="battery W", row=2, col=1)
    return fig


# ------------------------------------------------------------------ electrical (amps) visuals
def current_timeline(tl: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.30, 0.25], vertical_spacing=0.04,
                        subplot_titles=["Array current from sunlight (A)", "Battery current (+ charge / − discharge, A)", "Sunlight on the array (W/m²)"])
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.isc_array_a, name="Short-circuit Isc", line=dict(color="rgba(255,213,74,.5)", width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.imp_array_a, name="Operating current I_mp", line=dict(color=C["sun"], width=2), fill="tozeroy",
                             fillcolor="rgba(255,213,74,.15)"), 1, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.load_current_a, name="Load current (28 V bus)", line=dict(color=C["cyan"], width=1)), 1, 1)
    bc = tl.battery_current_a
    fig.add_trace(go.Bar(x=tl.time_utc, y=bc.clip(lower=0), name="charging A", marker_color=C["batt"]), 2, 1)
    fig.add_trace(go.Bar(x=tl.time_utc, y=bc.clip(upper=0), name="discharging A", marker_color=C["load"]), 2, 1)
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.panel_irradiance_w_m2, name="plane-of-array W/m²", line=dict(color="#ff8c42", width=1), fill="tozeroy",
                             fillcolor="rgba(255,140,66,.12)"), 3, 1)
    fig.update_layout(**BASE, height=640, hovermode="x unified", bargap=0, legend=dict(orientation="h", y=-0.08),
                      title="From sunlight to amps - array, bus and battery currents over the mission")
    return fig


def irradiance_current_scatter(tl: pd.DataFrame) -> go.Figure:
    d = tl[tl.panel_irradiance_w_m2 > 1]
    fig = go.Figure(go.Scattergl(x=d.panel_irradiance_w_m2, y=d.imp_array_a, mode="markers",
                                 marker=dict(size=5, color=d.cell_temp_c, colorscale="Turbo", showscale=True, colorbar=dict(title="cell °C")),
                                 customdata=np.c_[d.sun_alt, d.sun_disc_fraction * 100],
                                 hovertemplate="G %{x:.0f} W/m²<br>I %{y:.2f} A<br>Sun el %{customdata[0]:.2f}° · disc %{customdata[1]:.0f}%<extra></extra>"))
    fig.update_layout(**BASE, height=380, xaxis_title="sunlight on panel (W/m²)", yaxis_title="array current I_mp (A)",
                      title="How much current the sunlight produces (colour = cell temperature: hotter → lower voltage)")
    return fig


def iv_figure(tl: pd.DataFrame, i: int, cfg) -> go.Figure:
    from plotly.subplots import make_subplots
    r = tl.iloc[i]
    ns = int(r.cells_series)
    vt = 1.380649e-23 * (np.nan_to_num(r.cell_temp_c, nan=28.0) + 273.15) / 1.602176634e-19
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    from .pv_electrical import iv_curve
    for frac, a in ((1.0, 0.25), (0.5, 0.18), (0.25, 0.12)):
        v, cur, pw = iv_curve(cfg.isc_ma_cm2 * 1e-3 * cfg.area_cm2 * frac * r.strings_parallel, (cfg.voc_v + cfg.ideality_total * vt * np.log(frac)) * ns,
                              cfg.ideality_total * vt * ns)
        fig.add_trace(go.Scatter(x=v, y=cur, line=dict(color=f"rgba(127,167,184,{a*2})", dash="dot"), name=f"{int(frac*1361)} W/m² ref"), secondary_y=False)
    v, cur, pw = iv_curve(float(r.isc_array_a), float(r.voc_string_v), cfg.ideality_total * vt * ns)
    fig.add_trace(go.Scatter(x=v, y=cur, line=dict(color=C["sun"], width=3), name="I-V now"), secondary_y=False)
    fig.add_trace(go.Scatter(x=v, y=pw, line=dict(color="#ff8c42", width=2), name="P-V now"), secondary_y=True)
    if r.vmp_string_v > 0:
        fig.add_trace(go.Scatter(x=[r.vmp_string_v], y=[r.imp_array_a], mode="markers+text", marker=dict(size=13, color="#fff", symbol="star"),
                                 text=[f"MPP {r.imp_array_a:.2f} A @ {r.vmp_string_v:.1f} V"], textposition="top right", name="max power point"), secondary_y=False)
    fig.update_layout(**BASE, height=380, title=f"Array I-V / P-V at {r.time_utc:%Y-%m-%d %H:%M} UTC · {ns}s × {int(r.strings_parallel)}p cells · "
                      f"G {r.panel_irradiance_w_m2:.0f} W/m² · {np.nan_to_num(r.cell_temp_c, nan=0):.0f} °C")
    fig.update_xaxes(title_text="string voltage (V)")
    fig.update_yaxes(title_text="current (A)", secondary_y=False)
    fig.update_yaxes(title_text="power (W)", secondary_y=True)
    return fig


def iv_animation(tl: pd.DataFrame, cfg, n_frames: int = 48) -> go.Figure:
    from .pv_electrical import iv_curve
    idx = np.linspace(0, len(tl) - 1, n_frames).astype(int)
    ns = int(tl.cells_series.iloc[0])
    vmax = float(tl.voc_string_v.max()) * 1.05 or 1
    imax = float(tl.isc_array_a.max()) * 1.1 or 1

    def curve(i):
        r = tl.iloc[i]
        vt = 1.380649e-23 * (np.nan_to_num(r.cell_temp_c, nan=28.0) + 273.15) / 1.602176634e-19
        v, cur, _ = iv_curve(float(r.isc_array_a), float(r.voc_string_v), cfg.ideality_total * vt * ns)
        return v, cur, r

    v0, c0, r0 = curve(idx[0])
    fig = go.Figure(data=[go.Scatter(x=v0, y=c0, line=dict(color=C["sun"], width=3), fill="tozeroy", fillcolor="rgba(255,213,74,.15)", name="I-V"),
                          go.Scatter(x=[r0.vmp_string_v], y=[r0.imp_array_a], mode="markers", marker=dict(size=14, color="#fff", symbol="star"), name="MPP")])
    frames = []
    for i in idx:
        v, cur, r = curve(i)
        frames.append(go.Frame(data=[go.Scatter(x=v, y=cur), go.Scatter(x=[r.vmp_string_v], y=[r.imp_array_a])], name=str(i),
                               layout=go.Layout(title_text=f"{r.time_utc:%Y-%m-%d %H:%M} UTC · {r.imp_array_a:.2f} A · {r.gen_w:.0f} W · Sun el {r.sun_alt:+.2f}° · "
                                                           f"{'SUNLIT' if r.sunlit else 'DARK (terrain)'}")))
    fig.frames = frames
    fig.update_layout(**BASE, height=430, xaxis=dict(range=[0, vmax], title="string voltage (V)"), yaxis=dict(range=[0, imax], title="current (A)"),
                      title="Animated: the array's I-V curve breathing with the Sun through the mission",
                      updatemenus=[dict(type="buttons", x=0.0, y=1.15, buttons=[
                          dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=True)]),
                          dict(label="⏸", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                      sliders=[dict(steps=[dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                                                label=f"{tl.time_utc.iloc[i]:%m-%d}") for i in idx], x=0.05, len=0.9, y=-0.12)])
    return fig


def sun_to_amps_sankey(tl: pd.DataFrame, cfg, array, step_min: float) -> go.Figure:
    """Ah view: sunlight charge equivalent -> array current -> bus -> loads / battery."""
    h = step_min / 60.0
    ah = lambda a: float(np.nansum(a) * h)
    arr_ah = ah(tl.imp_array_a)
    load_ah = ah(tl.load_current_a)
    ch_ah = ah(tl.battery_current_a.clip(lower=0))
    dis_ah = ah(-tl.battery_current_a.clip(upper=0))
    bus_ah = ah(tl.bus_current_a)
    nodes = [f"Array output {arr_ah:,.0f} Ah @ ~{np.nanmean(tl.vmp_string_v[tl.vmp_string_v > 0]) if (tl.vmp_string_v > 0).any() else 0:.0f} V",
             f"28 V bus {bus_ah:,.0f} Ah", f"Loads {load_ah:,.0f} Ah", f"Battery charge {ch_ah:,.0f} Ah", f"Battery discharge {dis_ah:,.0f} Ah"]
    fig = go.Figure(go.Sankey(node=dict(label=nodes, color=[C["sun"], C["cyan"], "#5fb7d4", C["batt"], C["load"]], pad=18, thickness=18),
                              link=dict(source=[0, 1, 1, 4], target=[1, 2, 3, 2], value=[max(bus_ah, 1e-6), max(load_ah - dis_ah, 1e-6), max(ch_ah, 1e-6), max(dis_ah, 1e-6)],
                                        color=["rgba(255,213,74,.4)", "rgba(56,217,255,.35)", "rgba(61,220,151,.4)", "rgba(255,93,115,.4)"])))
    fig.update_layout(**BASE, height=330, title="Charge flow (amp-hours) - array → bus → loads & battery")
    return fig


# ------------------------------------------------------------------ Model 5 activity visuals
def load_stack(tl: pd.DataFrame) -> go.Figure:
    cols = [c for c in tl.columns if c.startswith("load::")]
    order = sorted(cols, key=lambda c: -tl[c].sum())
    pal = ["#38d9ff", "#4aa8ff", "#ffd54a", "#3ddc97", "#ff8c42", "#9b5cff", "#ff5d73", "#7fa7b8", "#f2c14e", "#5fb7d4", "#c084fc", "#94e2d5", "#fab387"]
    fig = go.Figure()
    for k, c in enumerate(order):
        fig.add_trace(go.Scatter(x=tl.time_utc, y=tl[c], name=c.replace("load::", ""), stackgroup="one", line=dict(width=0.5, color=pal[k % len(pal)])))
    fig.add_trace(go.Scatter(x=tl.time_utc, y=tl.gen_w, name="Generation", line=dict(color="#ffffff", width=1.5, dash="dot")))
    fig.update_layout(**BASE, height=460, hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                      title="Power usage by activity & subsystem (stacked, served W) vs generation")
    return fig


def comm_sessions_fig(sessions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if len(sessions):
        for name, g in sessions.groupby("session"):
            fig.add_trace(go.Scatter(x=g.start_utc, y=g.dc_w, mode="markers", name=name,
                                     marker=dict(size=np.clip(g.minutes / 3, 6, 40), opacity=0.75, line=dict(color="#fff", width=0.5)),
                                     customdata=np.c_[g.minutes, g.rf_w, g.data_mb, g.energy_wh],
                                     hovertemplate="%{x}<br>%{customdata[0]:.0f} min · RF %{customdata[1]:.0f} W · DC %{y:.0f} W"
                                                   "<br>%{customdata[2]:,.0f} MB · %{customdata[3]:.0f} Wh<extra>" + name + "</extra>"))
    fig.update_layout(**BASE, height=360, yaxis_title="DC power while transmitting (W)",
                      title="Communication sessions - amplitude (y), duration (bubble size), scheduled only when the link is up")
    return fig


def speed_curve_fig(curve: pd.DataFrame, v_sel: float, hotel_w: float) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=curve.speed_m_s, y=curve.wh_per_km, name="energy per km (Wh/km)", line=dict(color=C["sun"], width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=curve.speed_m_s, y=curve.drive_w, name="drive power (W)", line=dict(color=C["load"], dash="dot")), secondary_y=True)
    i = int(curve.wh_per_km.idxmin())
    fig.add_trace(go.Scatter(x=[curve.speed_m_s[i]], y=[curve.wh_per_km[i]], mode="markers+text", marker=dict(size=13, color="#3ddc97", symbol="star"),
                             text=[f"optimum {curve.speed_m_s[i]:.2f} m/s"], textposition="top center", name="optimum"), secondary_y=False)
    fig.add_vline(x=v_sel, line_color="#fff", annotation_text=f"planned {v_sel:.2f} m/s")
    fig.update_layout(**BASE, height=360, title=f"How speed matters: E/km = (hotel {hotel_w:.0f} W + drive(v)) / v")
    fig.update_xaxes(title_text="speed (m/s)")
    return fig


def stability_fig(tl: pd.DataFrame, step_min: float) -> go.Figure:
    from plotly.subplots import make_subplots
    h = step_min / 60.0
    w = max(1, int(24 / h))
    net = (tl.gen_w - tl.load_served_w)
    roll = net.rolling(w, min_periods=w).sum() * h / 1000
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], subplot_titles=["Rolling 24 h net energy (kWh)", "Phase portrait: SoC vs net power"])
    fig.add_trace(go.Scatter(x=tl.time_utc, y=roll, fill="tozeroy", line=dict(color=C["batt"]), name="24 h net"), 1, 1)
    fig.add_hline(y=0, line_color="#ff5d73", line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scattergl(x=tl.soc_pct, y=net, mode="markers", marker=dict(size=4, color=np.arange(len(tl)), colorscale="Viridis", showscale=False),
                               name="state trajectory"), 1, 2)
    fig.update_xaxes(title_text="SoC %", row=1, col=2)
    fig.update_yaxes(title_text="net W", row=1, col=2)
    fig.update_layout(**BASE, height=360, title="Outpost energy stability", showlegend=False)
    return fig
