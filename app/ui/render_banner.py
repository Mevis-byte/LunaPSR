"""LunaPSR banner rendering."""

from __future__ import annotations

import streamlit as st


def render_banner() -> None:
    st.markdown(
        """
<div style='text-align:center'>
<h1 style='margin-bottom:0'>🌑 LunaPSR</h1>
<p style='margin-top:0;opacity:0.7'>Chandrayaan-2 OHRC — Permanently Shadowed Region Image Enhancement</p>
</div>
<hr/>
""",
        unsafe_allow_html=True,
    )