"""LunaPSR visual identity — mission-control dark theme.

Restrained scientific-imaging aesthetic: dark space-inspired surfaces,
a single cyan accent, monospace reserved for numeric/telemetry values.
All styling is injected once per session and targets stable Streamlit
data-testids so it survives Streamlit updates.
"""

from __future__ import annotations

import streamlit as st

ACCENT = "#4fc3f7"
BG = "#0b0e14"
PANEL = "#111726"
BORDER = "#1d2637"

_CSS = f"""
<style>
/* ---- global surface ------------------------------------------------ */
.stApp {{
    background: {BG};
    color: #d9e2ec;
}}
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}

/* ---- typography ---------------------------------------------------- */
html, body, [class*="css"] {{ font-family: "Inter", "Segoe UI", system-ui,
    -apple-system, sans-serif; }}
h1, h2, h3, h4 {{ letter-spacing: 0.04em; color: #eaf2fb; }}
h4 {{ text-transform: none; }}
code, [data-testid="stMetricValue"], .lunapsr-mono {{
    font-family: "JetBrains Mono", "Cascadia Code", "SF Mono", Consolas,
        "Liberation Mono", monospace !important;
}}
[data-testid="stMetricValue"] {{ color: {ACCENT}; font-variant-numeric: tabular-nums; }}

/* ---- metric cards --------------------------------------------------- */
[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px 16px 10px 16px;
}}
[data-testid="stMetricLabel"] p {{
    text-transform: uppercase; font-size: 0.68rem !important;
    letter-spacing: 0.09em; color: #8fa3bd !important; margin-bottom: 2px;
}}

/* ---- tabs ----------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.09em;
    color: #8fa3bd; padding: 8px 14px; border-radius: 6px 6px 0 0;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important; border-bottom-color: {ACCENT};
    box-shadow: inset 0 -2px 0 {ACCENT};
}}

/* ---- buttons / inputs ----------------------------------------------- */
.stButton > button {{
    border: 1px solid {BORDER}; border-radius: 6px;
    background: {PANEL}; color: #d9e2ec;
}}
.stButton > button:hover {{
    border-color: {ACCENT}; color: {ACCENT};
}}
.stButton > button[kind="primary"], .stButton > button[data-testid="stFormSubmitButton"] {{
    background: rgba(79, 195, 247, 0.12); border-color: rgba(79, 195, 247, 0.45);
}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] {{
    border-color: {BORDER} !important; background: {PANEL};
}}
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important; border-radius: 8px;
    background: rgba(17, 23, 38, 0.5);
}}
hr {{ border-color: {BORDER}; }}

/* ---- hide streamlit chrome (safe, stable) --------------------------- */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---- mission header banner ------------------------------------------ */
.lunapsr-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; margin: 0 0 14px 0;
    background: linear-gradient(180deg, {PANEL} 0%, {BG} 100%);
    border: 1px solid {BORDER}; border-radius: 10px;
}}
.lunapsr-title {{
    font-size: 1.55rem; font-weight: 700; letter-spacing: 0.28em;
    color: #eaf2fb; line-height: 1.1;
}}
.lunapsr-subtitle {{
    font-size: 0.78rem; color: #8fa3bd; margin-top: 3px; letter-spacing: 0.05em;
}}
.lunapsr-badge {{
    display: inline-block; margin-top: 6px; padding: 2px 10px;
    font-size: 0.62rem; letter-spacing: 0.14em; color: {ACCENT};
    border: 1px solid rgba(79, 195, 247, 0.4); border-radius: 999px;
}}
.lunapsr-status {{
    display: flex; align-items: center; gap: 8px;
    font-size: 0.72rem; letter-spacing: 0.12em; color: #9fb3c8;
    white-space: nowrap;
}}
.lunapsr-dot {{
    width: 9px; height: 9px; border-radius: 50%;
    background: #37d67a; box-shadow: 0 0 6px rgba(55, 214, 122, 0.8);
}}
.lunapsr-dot.warn {{ background: #f5a623; box-shadow: 0 0 6px rgba(245, 166, 35, 0.8); }}

/* ---- section label ---------------------------------------------------- */
.lunapsr-sectionlabel {{
    font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #8fa3bd; margin: 4px 0 2px 0;
}}
</style>
"""


def apply_theme() -> None:
    """Inject the global theme CSS exactly once per session."""
    if not st.session_state.get("_lunapsr_theme_applied"):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["_lunapsr_theme_applied"] = True


def render_header(status_text: str = "SYSTEM READY", warn: bool = False) -> None:
    """Mission-control application header, identical on every page."""
    dot_cls = "lunapsr-dot warn" if warn else "lunapsr-dot"
    st.markdown(
        f"""
<div class="lunapsr-header">
  <div>
    <div class="lunapsr-title">LUNAPSR</div>
    <div class="lunapsr-subtitle">Lunar Permanently Shadowed Region Analysis</div>
    <span class="lunapsr-badge">CHANDRAYAAN-2 &bull; OHRC</span>
  </div>
  <div class="lunapsr-status">
    <span class="{dot_cls}"></span>{status_text}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Small uppercase section label used above card groups."""
    st.markdown(f'<div class="lunapsr-sectionlabel">{text}</div>',
                unsafe_allow_html=True)
