"""Tab 5: Processing History.

Lists persisted processing records, shows their JSON, allows export, and lets
the user push a record back into the Enhancement tab to reproduce it.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.exporters import json_bytes
from app.ui.state import get_history
from core.history import ProcessingRecord


def render() -> None:
    st.subheader("Processing History")
    hist = get_history()

    if not hist.records:
        st.info("No processing runs yet. Apply an enhancement to create a record.")
        return

    st.markdown(f"{len(hist.records)} record(s). Records are stored as JSON in `data/history/`.")

    rows = []
    for r in hist.records:
        rows.append({
            "ID": r.processing_id,
            "Timestamp (UTC)": r.timestamp,
            "Input": r.input_name,
            "Pipeline": r.pipeline_name,
            "Stages": " → ".join(s.get("stage", "?") for s in r.pipeline_stages),
            "Warnings": len(r.warnings),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Reproduce a record")
    selected_id = st.selectbox(
        "Select a record",
        [r.processing_id for r in hist.records],
        format_func=lambda i: f"{i} — {hist.get(i).pipeline_name} — {hist.get(i).timestamp}",
    )
    rec = hist.get(selected_id)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.download_button("Export record JSON", json_bytes(rec.to_dict()),
                           file_name=f"{rec.processing_id}.json", mime="application/json")
    with c2:
        if st.button("↩ Load into Enhancement tab", use_container_width=True):
            resume = record_to_resume_params(rec)
            st.session_state.pop("enhanced", None)
            st.session_state["resume_pipeline"] = resume
            st.success("Controls loaded. Go to the Enhancement tab and press Apply.")
    with c3:
        st.download_button("Export all records JSON",
                           json_bytes({"records": [r.to_dict() for r in hist.records]}),
                           file_name="lunapsr_history.json", mime="application/json")

    st.divider()
    st.markdown("### Record preview")
    st.json(json.loads(rec.to_json()))


def record_to_resume_params(rec: ProcessingRecord) -> dict:
    """Convert a record into widget defaults for the Enhancement tab."""
    resume: dict = {"method": rec.pipeline_name}
    for stage in rec.pipeline_stages:
        params = stage.get("parameters", {})
        name = stage.get("stage")
        if name == "gamma":
            resume["gamma"] = params.get("gamma", 0.75)
        elif name == "clahe":
            resume["clip_limit"] = params.get("clip_limit", 2.0)
            resume["tile_size"] = params.get("tile_size", 8)
        elif name == "retinex":
            resume["retinex_sigma"] = params.get("sigma", 30.0)
        elif name == "denoise":
            resume["denoise"] = params.get("method", "none")
            resume["denoise_strength"] = params.get("strength", 3.0)
        elif name == "detail":
            resume["detail_strength"] = params.get("strength", 0.0)
        elif name == "normalize":
            pcts = params.get("percentiles", [0.5, 99.5])
            resume["pct_lo"] = pcts[0]
            resume["pct_hi"] = pcts[1]

    if rec.pipeline_name == "custom":
        resume["custom_stages"] = [
            s.get("stage") for s in rec.pipeline_stages
            if s.get("stage") != "normalize"
        ]
    return resume