"""Template for teammate (Objective A) modules - copy to extensions/<name>.py (no leading underscore) to activate."""

TAB_TITLE = "Example extension"
ORDER = 50


def render(ctx):
    st = ctx["st"]
    r = ctx["result"]
    st.markdown(f"### {ctx['site'].name}")
    st.metric("Sunlit %", f"{r.solar['sunlit_pct']:.1f}")
    st.line_chart(ctx["timeline"].set_index("time_utc")[["gen_w", "load_served_w"]])
