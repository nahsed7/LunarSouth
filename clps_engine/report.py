"""Self-contained HTML mission assessment report for any selected point."""

from __future__ import annotations

import html
from datetime import datetime, timezone

import plotly.io as pio

CSS = """body{font-family:'Segoe UI',system-ui,sans-serif;background:#04080f;color:#d7f6ff;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding:24px}h1,h2{font-family:ui-monospace,monospace;color:#e9fbff}
h1{letter-spacing:.04em}.brand{color:#38d9ff;font-size:13px;letter-spacing:.3em}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.k{background:#0a1422;border:1px solid rgba(56,217,255,.25);
border-radius:10px;padding:10px}.k b{display:block;font-size:22px;color:#fff}.k span{font-size:11px;color:#7fa7b8;text-transform:uppercase}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid rgba(56,217,255,.15);padding:4px 8px;text-align:left}
.st{padding:10px 14px;border-radius:10px;border:1px solid #38d9ff;margin:10px 0}.note{font-size:12px;color:#7fa7b8}"""


def build_html(title: str, kpis: list, sections: list, figures: list) -> str:
    """sections: [(heading, html_body)]; figures: [(heading, plotly Figure)]."""
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class='wrap'>",
             "<div class='brand'>LUNARSOUTH · CLPS MISSION PLANNER</div>", f"<h1>{html.escape(title)}</h1>",
             f"<div class='note'>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · data: NASA Moon Trek, ASU LROC QuickMap, "
             "JPL Horizons / NAIF SPICE, CSA requirements, NASA PGDA LOLA 20 m/5 m.</div>",
             "<div class='kpis'>" + "".join(f"<div class='k'><span>{html.escape(a)}</span><b>{html.escape(b)}</b></div>" for a, b in kpis) + "</div>"]
    for h, body in sections:
        parts.append(f"<h2>{html.escape(h)}</h2>{body}")
    first = True
    for h, fig in figures:
        parts.append(f"<h2>{html.escape(h)}</h2>")
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if first else False))
        first = False
    parts.append("<p class='note'>All array, battery and load values are engineering assumptions; screening results only - not flight "
                 "qualification or landing certification.</p></div></body></html>")
    return "\n".join(parts)
