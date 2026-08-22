"""Session state helpers.

Central accessors keep keys consistent across pages.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def get_image():
    return st.session_state.get("image", None)


def set_image(img, revision: Optional[int] = None) -> None:
    st.session_state["image"] = img
    # Invalidate cached processing when the image changes.
    st.session_state["image_revision"] = (
        revision if revision is not None else 0
    )


def bump_image_revision() -> int:
    """Increment the image revision (invalidates processing caches)."""
    current = st.session_state.get("image_revision", 0)
    st.session_state["image_revision"] = current + 1
    return current + 1


def image_key() -> str:
    rev = st.session_state.get("image_revision", None)
    img = get_image()
    if img is None:
        return "none"
    return f"{img.source_name}|{img.shape[0]}x{img.shape[1]}|{rev}"


def get_enhanced():
    return st.session_state.get("enhanced", None)


def set_enhanced(run_result) -> None:
    st.session_state["enhanced"] = run_result


def get_history():
    from core.history import ProcessingHistory

    if "history" not in st.session_state:
        st.session_state["history"] = ProcessingHistory()
    return st.session_state["history"]


def get_roi() -> Optional[tuple]:
    r = st.session_state.get("roi", None)
    return r


def set_roi(x0: int, y0: int, x1: int, y1: int) -> None:
    st.session_state["roi"] = (int(x0), int(y0), int(x1), int(y1))