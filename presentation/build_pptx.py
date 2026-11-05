#!/usr/bin/env python3
"""Build the 6-slide LunaPSR overview deck.

Requires: python-pptx, Pillow.
The deck is deliberately plain-language: a reader who has never seen the
project should understand what it does, what problem it solves, and which
features matter by the end of slide 3.

Run:
    pip install python-pptx Pillow
    python presentation/build_pptx.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
PRES = ROOT / "presentation"
ASSETS = PRES / "assets"
OUT = PRES / "lunapsr_6slide.pptx"

# ---------------------------------------------------------------------------
# Palette / typography
# ---------------------------------------------------------------------------
BG = RGBColor(0x05, 0x07, 0x0B)
PANEL = RGBColor(0x0B, 0x0F, 0x16)
PANEL_2 = RGBColor(0x10, 0x16, 0x22)
LINE = RGBColor(0x1D, 0x28, 0x39)
TEXT = RGBColor(0xE8, 0xF1, 0xFA)
MUTED = RGBColor(0x93, 0xA8, 0xBF)
FAINT = RGBColor(0x68, 0x7B, 0x92)
ACCENT = RGBColor(0x4F, 0xC3, 0xF7)
GREEN = RGBColor(0x3C, 0xD6, 0x8A)
AMBER = RGBColor(0xF5, 0xB0, 0x42)

FONT = "Arial"
MONO = "Consolas"

W, H = 13.333, 7.5


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def add_rect(slide, x, y, w, h, fill=PANEL, line=LINE, rounded=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    if rounded:
        try:
            shape.adjustments[0] = 0.08
        except Exception:
            pass
    return shape


def add_text(slide, text, x, y, w, h, size=12, color=TEXT, bold=False,
             font=FONT, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
             line_spacing=1.0, space_after=0, italic=False, margins=(0, 0, 0, 0)):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def add_rich_paragraphs(slide, items, x, y, w, h, size=11, color=TEXT,
                        bullet_color=ACCENT, space_after=7, line_spacing=1.04,
                        margins=(0, 0, 0, 0)):
    """items: list of lists of (text, bold, color_or_None)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    tf.vertical_anchor = MSO_ANCHOR.TOP

    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        b = p.add_run()
        b.text = "•  "
        b.font.name = FONT
        b.font.size = Pt(size)
        b.font.bold = True
        b.font.color.rgb = bullet_color
        for text, bold, col in item:
            r = p.add_run()
            r.text = text
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = col or color
    return box


def fit_picture(slide, path, x, y, w, h, alt=""):
    """Place an image fully inside the box, preserving aspect ratio."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2
    py = y + (h - ph) / 2
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    if alt:
        try:
            pic._element.nvPicPr.cNvPr.set("descr", alt)
        except Exception:
            pass
    return pic


def add_figure(slide, path, x, y, w, h, caption=None, alt="", pad=0.12,
               caption_h=0.55):
    add_rect(slide, x, y, w, h, fill=PANEL, line=LINE)
    image_h = h - 2 * pad - (caption_h if caption else 0)
    fit_picture(slide, path, x + pad, y + pad, w - 2 * pad, image_h, alt=alt)
    if caption:
        add_text(slide, caption, x + pad, y + h - caption_h - 0.02,
                 w - 2 * pad, caption_h - 0.06, size=7.8, color=MUTED,
                 line_spacing=1.08)


def add_header(slide, kicker, title, subtitle=None, number=None):
    # top rule + small accent mark
    add_rect(slide, 0, 0, W, 0.028, fill=ACCENT, line=None, rounded=False)
    add_rect(slide, 0.62, 0.54, 0.055, 0.42, fill=ACCENT, line=None, rounded=False)
    add_text(slide, kicker.upper(), 0.82, 0.48, 9.3, 0.28, size=8.5, color=ACCENT,
             bold=True)
    add_text(slide, title, 0.82, 0.73, 11.6, 0.68, size=27, color=TEXT, bold=True,
             line_spacing=0.95)
    if subtitle:
        add_text(slide, subtitle, 0.82, 1.36, 11.8, 0.58, size=11.2, color=MUTED,
                 line_spacing=1.1)
    if number:
        add_text(slide, f"LunaPSR  ·  {number}/6", 11.55, 0.48, 1.2, 0.25,
                 size=7.5, color=FAINT, align=PP_ALIGN.RIGHT)


def add_footer(slide, source, tag):
    add_rect(slide, 0.62, 7.02, 12.1, 0.012, fill=LINE, line=None, rounded=False)
    add_text(slide, source, 0.62, 7.1, 8.4, 0.22, size=6.8, color=FAINT)
    add_text(slide, tag, 8.9, 7.1, 3.82, 0.22, size=6.8, color=FAINT,
             align=PP_ALIGN.RIGHT)


def add_card(slide, x, y, w, h, label, body, label_color=ACCENT,
             body_size=8.8, label_size=7.0, fill=PANEL_2):
    add_rect(slide, x, y, w, h, fill=fill, line=LINE)
    add_text(slide, label.upper(), x + 0.13, y + 0.09, w - 0.26, 0.17,
             size=label_size, color=label_color, bold=True)
    add_text(slide, body, x + 0.13, y + 0.27, w - 0.26, h - 0.32,
             size=body_size, color=MUTED, line_spacing=1.02)


def add_pill(slide, text, x, y, w, h, color=MUTED, fill=PANEL_2):
    add_rect(slide, x, y, w, h, fill=fill, line=LINE)
    add_text(slide, text, x + 0.05, y + 0.01, w - 0.10, h - 0.02,
             size=7.1, color=color, align=PP_ALIGN.CENTER,
             valign=MSO_ANCHOR.MIDDLE)


def add_stat(slide, x, y, w, h, number, label, color=TEXT):
    add_rect(slide, x, y, w, h, fill=PANEL_2, line=LINE)
    add_text(slide, number, x + 0.14, y + 0.10, w - 0.28, 0.40, size=23,
             color=color, bold=True, line_spacing=0.9)
    add_text(slide, label, x + 0.14, y + 0.52, w - 0.28, h - 0.56, size=7.2,
             color=MUTED, line_spacing=1.0)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    blank = prs.slide_layouts[6]

    # =====================================================================
    # 1 — Title / what this is
    # =====================================================================
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    s.shapes.add_picture(str(ASSETS / "ohrc_hero.jpg"), 0, 0, Inches(W), Inches(H))
    add_rect(s, 0, 0, W, 0.035, fill=ACCENT, line=None, rounded=False)

    add_text(s, "CHANDRAYAAN-2  ·  OHRC  ·  LUNAR PSR", 0.78, 0.62, 8.6, 0.30,
             size=9, color=ACCENT, bold=True)
    add_text(s, "LunaPSR", 0.75, 0.92, 7.2, 1.15, size=64, color=TEXT, bold=True,
             line_spacing=0.9)
    add_text(s,
             "A read-only enhancement and analysis system for the Moon’s darkest places.",
             0.78, 2.12, 8.4, 0.72, size=20, color=TEXT, bold=True,
             line_spacing=1.0)
    add_text(s,
             "It makes faint Chandrayaan-2 OHRC images easier to study — and shows exactly "
             "what each enhancement step added, without pretending to recover detail that "
             "is not in the data.",
             0.80, 2.88, 7.9, 0.86, size=11.8, color=MUTED, line_spacing=1.12)

    # Three-at-a-glance cards.
    labels = [
        ("WHAT IT IS", "A software lab for real Chandrayaan-2 OHRC lunar images."),
        ("WHAT IT DOES", "Enhances, compares, measures, and records every derived step."),
        ("WHY IT MATTERS", "Permanently shadowed regions are science-rich — but only a few DN above the noise."),
    ]
    for i, (lab, body) in enumerate(labels):
        add_card(s, 0.78 + i * 3.98, 4.22, 3.72, 1.42, lab, body,
                 body_size=10.4, label_size=7.2)

    add_text(s,
             "For lunar scientists, image analysts, and reviewers who need to know what an "
             "enhancement changed.",
             0.80, 5.88, 11.7, 0.28, size=9.2, color=MUTED, italic=True)
    add_text(s,
             "HONEST BY DESIGN  ·  All enhanced imagery is a derived representation. "
             "We do not claim recovered geology, measured SNR, or information the sensor did not collect.",
             0.80, 6.32, 11.7, 0.34, size=8.2, color=FAINT, line_spacing=1.05)
    add_text(s, "6-slide overview", 11.0, 7.12, 1.6, 0.22, size=7.5, color=FAINT,
             align=PP_ALIGN.RIGHT)

    s.notes_slide.notes_text_frame.text = (
        "One-sentence version: LunaPSR is a read-only lab that enhances faint Chandrayaan-2 "
        "OHRC lunar images and documents exactly what the enhancement added. "
        "The next two slides explain the problem and the features; slide 5 gives the honest benchmark."
    )

    # =====================================================================
    # 2 — Problems solved
    # =====================================================================
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    add_header(
        s,
        "01 · The problem it solves",
        "The signal is barely above the noise.",
        "Permanently Shadowed Regions (PSRs) near the Moon’s poles receive no direct sunlight. "
        "OHRC (the Orbiter High Resolution Camera) can image them, but the data are extremely "
        "faint — and very easy to over-enhance.",
        number=2,
    )

    add_rich_paragraphs(
        s,
        [
            [("Weak signal: ", True, TEXT),
             ("pixel values rise only 1–5 DN above black in the studied region "
              "(mean 3.056). DN is the raw brightness unit.", False, MUTED)],
            [("Noise: ", True, TEXT),
             ("the Phase 4B physics model predicts about 0.511 DN of total noise — "
              "a model-based SNR of about 6, not a measured SNR.", False, MUTED)],
            [("False detail: ", True, TEXT),
             ("a global brightness stretch mostly reveals noise. Aggressive Retinex "
              "can create edges, textures, and isolated components that were not in the data.", False, MUTED)],
            [("No audit trail: ", True, TEXT),
             ("comparing methods on a 1.2-gigapixel product is slow, inconsistent, "
              "and hard to reproduce.", False, MUTED)],
        ],
        0.82, 2.08, 6.08, 2.30, size=10.9, space_after=8, line_spacing=1.06,
    )

    add_rect(s, 0.82, 4.58, 6.05, 1.08, fill=PANEL_2, line=LINE)
    add_text(s, "The real problem", 0.99, 4.70, 5.7, 0.19, size=7.2, color=ACCENT, bold=True)
    add_text(s,
             "An enhanced image can look better while becoming less trustworthy.",
             0.99, 4.94, 5.7, 0.46, size=13.4, color=TEXT, bold=True, line_spacing=1.0)

    add_text(s, "PROBLEMS IT SOLVES", 0.82, 5.90, 4.0, 0.18, size=6.8, color=FAINT, bold=True)
    pills = ["Weak contrast", "Noise amplification", "False detail", "No ground truth", "No audit trail"]
    for i, p in enumerate(pills):
        add_pill(s, p, 0.82 + i * 1.205, 6.12, 1.16, 0.30, color=MUTED)

    add_figure(
        s,
        ROOT / "screenshots" / "adaptive_retinex_maps_confidence_alpha_noise.png",
        7.18, 2.02, 5.42, 3.48,
        caption="Model-based confidence, adaptive strength, and modeled noise — diagnostics, not measurements.",
        alt="Adaptive Retinex diagnostic maps: confidence, adaptive strength, and modeled noise.",
        caption_h=0.52,
    )
    add_text(s, "MODEL-BASED  ·  NOT MEASURED", 7.20, 5.62, 5.38, 0.20,
             size=7.1, color=FAINT, align=PP_ALIGN.RIGHT)

    add_footer(s, "Source: README §2, §12; docs/phase5a_ssr_investigation.md",
               "AN ENHANCEMENT IS NOT EVIDENCE BY ITSELF")

    s.notes_slide.notes_text_frame.text = (
        "Plain-language problem statement: the lunar surface here is almost black. "
        "A normal brightness stretch makes noise visible; a strong Retinex can invent structure. "
        "LunaPSR exists to enhance the data while keeping the evidence about what changed."
    )

    # =====================================================================
    # 3 — What it does + features
    # =====================================================================
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    add_header(
        s,
        "02 · What it does + features",
        "One workflow: from raw pixels to evidence you can review.",
        "LunaPSR is not a single filter. It is a complete lab for reading, enhancing, "
        "comparing, and documenting real lunar images — without modifying the source product.",
        number=3,
    )

    pipe = ["Real OHRC product", "Mask & normalize", "Noise model", "Enhance", "Diagnostics & record"]
    for i, label in enumerate(pipe):
        x = 0.82 + i * 2.36
        add_rect(s, x, 1.98, 2.12, 0.50,
                 fill=RGBColor(0x0E, 0x24, 0x33) if i == 2 else PANEL_2, line=LINE)
        add_text(s, label, x + 0.05, 2.03, 2.02, 0.40, size=7.6, color=TEXT,
                 bold=(i == 2), align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE,
                 line_spacing=0.95)
        if i < len(pipe) - 1:
            add_text(s, "→", x + 2.13, 2.07, 0.22, 0.30, size=12, color=FAINT,
                     align=PP_ALIGN.CENTER)

    add_text(s, "FEATURES THAT MATTER", 0.82, 2.72, 4.0, 0.20, size=7.2, color=ACCENT, bold=True)
    features = [
        ("READ-ONLY", "1.2-gigapixel product, memory-mapped; source array never changed."),
        ("NOISE-AWARE", "Model-based SNR and confidence, not ad-hoc contrast."),
        ("MULTI-METHOD", "Gamma, CLAHE, SSR, denoisers, LunaPSR, Adaptive Retinex."),
        ("DIAGNOSTICS", "Noise, confidence, structure, and edge-preservation maps."),
        ("AUDITABLE", "Metrics, processing history, JSON / CSV / PNG exports."),
        ("USABLE", "9-tab Streamlit lab; 304 tests keep the workflow honest."),
    ]
    for i, (lab, body) in enumerate(features):
        col, row = i % 2, i // 2
        add_card(s, 0.82 + col * 3.07, 3.02 + row * 0.88, 2.96, 0.76,
                 lab, body, body_size=8.6, label_size=6.8)

    add_text(s, "The six cards are the practical capabilities — not separate products.",
             0.82, 5.72, 6.05, 0.28, size=8.6, color=MUTED, italic=True)

    add_figure(
        s,
        ROOT / "screenshots" / "adaptive_retinex_section.png",
        7.18, 2.72, 5.42, 3.55,
        caption="The lab: documented controls, Original / SSR / LunaPSR views, and headline diagnostics.",
        alt="LunaPSR Adaptive Retinex lab section with controls and Original, SSR, and LunaPSR views.",
        caption_h=0.52,
    )
    add_rect(s, 7.18, 6.40, 5.42, 0.42, fill=PANEL_2, line=LINE)
    add_text(s, "Every output is derived; every derived step has a record.",
             7.34, 6.48, 5.10, 0.28, size=9.0, color=MUTED, italic=True,
             line_spacing=1.0)
    add_footer(s, "Source: README §4–§6; app.py; core/ohrc/",
               "READ-ONLY SOURCE  ·  DERIVED OUTPUTS")

    s.notes_slide.notes_text_frame.text = (
        "This is the slide that answers 'what does the project do?' "
        "It runs one consistent workflow: read the real product, mask and normalize, model noise, "
        "enhance, then measure and save. The six feature cards are the practical capabilities."
    )

    # =====================================================================
    # 4 — How it works
    # =====================================================================
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    add_header(
        s,
        "03 · How it works",
        "Enhance only where the signal can support it.",
        "A normal enhancement applies one curve everywhere. LunaPSR uses a physics-based noise "
        "model to decide how much enhancement each part of the image can justify.",
        number=4,
    )

    pipe = ["Real OHRC product", "Mask & normalize", "Noise model",
            "Adaptive enhancement", "Diagnostics & record"]
    for i, label in enumerate(pipe):
        x = 0.78 + i * 2.36
        add_rect(s, x, 1.98, 2.12, 0.58, fill=PANEL_2 if i not in (2, 3) else RGBColor(0x0E, 0x24, 0x33),
                 line=LINE)
        add_text(s, label, x + 0.06, 2.02, 2.00, 0.48, size=7.6, color=TEXT,
                 bold=(i in (2, 3)), align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE,
                 line_spacing=0.95)
        if i < len(pipe) - 1:
            add_text(s, "→", x + 2.13, 2.08, 0.22, 0.30, size=12, color=FAINT,
                     align=PP_ALIGN.CENTER)

    add_rich_paragraphs(
        s,
        [
            [("Noise is predicted, not guessed. ", True, TEXT),
             ("The model combines shot noise, read noise, and quantization noise "
              "at each brightness level.", False, MUTED)],
            [("Two-factor confidence. ", True, TEXT),
             ("Is the local signal strong? Is local variation above the modeled noise? "
              "Both must agree before enhancement is released.", False, MUTED)],
            [("Adaptive strength α. ", True, TEXT),
             ("It stays conservative where the data look like noise, and increases only "
              "where real structure is supported.", False, MUTED)],
            [("Adaptive Retinex. ", True, TEXT),
             ("It separates illumination, robust response, and medium-scale structure "
              "instead of applying one global curve to the whole scene.", False, MUTED)],
        ],
        0.82, 2.82, 6.05, 2.55, size=10.2, space_after=7, line_spacing=1.04,
    )

    add_rect(s, 0.82, 5.48, 3.12, 0.60, fill=PANEL_2, line=LINE)
    add_text(s, "σ(DN) = √(DN/g + (σread/g)² + 1/12)", 0.96, 5.56, 2.86, 0.24,
             size=8.3, color=ACCENT, font=MONO)
    add_text(s, "gain g is inferred from metadata — not an official calibration claim",
             0.96, 5.80, 2.86, 0.20, size=6.2, color=FAINT, line_spacing=1.0)

    add_rect(s, 4.08, 5.48, 2.80, 0.60, fill=PANEL_2, line=LINE)
    add_text(s, "IN PLAIN TERMS", 4.22, 5.55, 2.5, 0.16, size=6.2, color=ACCENT, bold=True)
    add_text(s, "The noise model is the gate, not a post-hoc excuse.", 4.22, 5.72, 2.52, 0.30,
             size=7.7, color=MUTED, line_spacing=1.0)

    add_pill(s, "Phase 5B prototype — validation ongoing", 0.82, 6.28, 3.12, 0.30,
             color=AMBER, fill=RGBColor(0x20, 0x18, 0x08))
    add_text(s, "MODEL-BASED SNR  ·  NOT A MEASURED SNR", 4.08, 6.32, 2.80, 0.20,
             size=6.6, color=FAINT, align=PP_ALIGN.CENTER)

    add_figure(
        s,
        ROOT / "screenshots" / "adaptive_retinex_maps_illumination_response_structure.png",
        7.18, 2.82, 5.42, 3.45,
        caption="Illumination, robust Retinex response, and medium-scale structure used as variation evidence.",
        alt="Adaptive Retinex maps: illumination, response, and structure.",
        caption_h=0.52,
    )
    add_footer(s, "Source: docs/phase4b_noise_model.md; docs/phase5b_lunapsr_retinex.md",
               "THE NOISE MODEL IS THE GATE")

    s.notes_slide.notes_text_frame.text = (
        "The method in one sentence: estimate what the sensor noise should look like, then "
        "hold enhancement back wherever the image variation is consistent with that noise. "
        "The formula is shown for completeness; the audience only needs the idea of a gate."
    )

    # =====================================================================
    # 5 — Evidence
    # =====================================================================
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    add_header(
        s,
        "04 · Evidence",
        "We built ground truth — and report where it loses.",
        "Phase 5C injects known craters and boulders into eight real 256×256 OHRC dark tiles. "
        "Five methods, one generic detector, per-tile 5% false-positive calibration.",
        number=5,
    )

    stats = [
        ("0.882", "Raw AUC — best overall", TEXT),
        ("0.868", "LunaPSR AUC", ACCENT),
        ("0.704", "LunaPSR boulder AUC ≤ 0.5 DN", GREEN),
        ("0.638", "Raw boulder AUC ≤ 0.5 DN", TEXT),
    ]
    for i, (num, lab, col) in enumerate(stats):
        add_stat(s, 0.82 + i * 3.04, 1.98, 2.92, 0.90, num, lab, color=col)

    add_rich_paragraphs(
        s,
        [
            [("Raw wins overall. ", True, TEXT),
             ("AUC 0.882 vs 0.868. Enhancement is not required to detect these features.", False, MUTED)],
            [("LunaPSR has a narrow win. ", True, TEXT),
             ("At ≤ 0.5 DN it ties raw overall (0.744 vs 0.742) and leads on faint "
              "12–20 px boulders (0.704 vs 0.638; SSR 0.559).", False, MUTED)],
            [("It also loses. ", True, TEXT),
             ("LunaPSR is the worst method for 6 px boulders and less sensitive than raw "
              "for faint craters. SSR amplifies noise ~50×; LunaPSR ~14.5×.", False, MUTED)],
        ],
        0.82, 3.12, 6.05, 2.10, size=10.4, space_after=8, line_spacing=1.05,
    )

    add_rect(s, 0.82, 5.28, 6.05, 0.92, fill=PANEL_2, line=LINE)
    add_text(s, "THE HONEST RESULT", 0.98, 5.39, 4.0, 0.17, size=6.8, color=ACCENT, bold=True)
    add_text(s, "A narrow, measurable win — not a clean sweep.",
             0.98, 5.58, 5.70, 0.44, size=13.0, color=TEXT, bold=True, line_spacing=1.0)
    add_text(s, "AUC measures how well a method separates real features from background: "
                "0.5 is chance, 1.0 is perfect.",
             0.82, 6.34, 6.05, 0.32, size=7.3, color=FAINT, line_spacing=1.05)

    add_figure(
        s,
        ROOT / "outputs" / "phase5c" / "curves_boulder.png",
        7.18, 1.98, 5.42, 1.68,
        caption="Detection versus injected boulder contrast. The hard regime is low input SNR.",
        alt="Phase 5C detection-versus-contrast curves for boulders.",
        caption_h=0.30, pad=0.08,
    )
    add_figure(
        s,
        ROOT / "outputs" / "phase5c" / "montage_boulder_d12.png",
        7.18, 3.78, 5.42, 2.92,
        caption="Injected boulders at 12 px: the visual task behind the curves.",
        alt="Phase 5C montage of injected boulder targets at 12 pixels.",
        caption_h=0.32, pad=0.08,
    )
    add_footer(s, "Source: docs/phase5c_ground_truth_benchmark.md; outputs/phase5c/",
               "PHASE 5C COMPLETE  ·  8 TILES  ·  5 METHODS  ·  5% FP")

    s.notes_slide.notes_text_frame.text = (
        "The benchmark is deliberately adversarial to the project: raw data wins overall. "
        "LunaPSR's win is narrow and size-dependent. Say that clearly; it is more credible "
        "than claiming a clean sweep."
    )

    # =====================================================================
    # 6 — Use / demo / next
    # =====================================================================
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    add_header(
        s,
        "05 · Use it",
        "What you can do with LunaPSR.",
        "Use it to inspect a real product, compare enhancement methods, and see exactly what "
        "each step changed — before you trust the result.",
        number=6,
    )

    add_rich_paragraphs(
        s,
        [
            [("Inspect ", True, TEXT), ("a real OHRC product read-only.", False, MUTED)],
            [("Compare ", True, TEXT),
             ("gamma, CLAHE, SSR, denoisers, LunaPSR, and Adaptive Retinex side by side.", False, MUTED)],
            [("See ", True, TEXT),
             ("confidence, noise, structure, and edge diagnostics for every view.", False, MUTED)],
            [("Export ", True, TEXT),
             ("derived images, metrics, and a reproducible run record.", False, MUTED)],
            [("Run the lab ", True, TEXT),
             ("locally with the full 1.2 Gpx product; the cloud demo uses the bundled sample.", False, MUTED)],
        ],
        0.82, 2.04, 6.05, 2.25, size=10.7, space_after=8, line_spacing=1.05,
    )

    add_rect(s, 0.82, 4.56, 6.05, 1.22, fill=PANEL_2, line=LINE)
    add_text(s, "CURRENT STATUS", 0.98, 4.67, 4.0, 0.18, size=6.8, color=ACCENT, bold=True)
    for i, line in enumerate([
        "Phase 5C benchmark: complete.",
        "Phase 5B Adaptive Retinex: prototype — validation ongoing.",
        "Live demo: private; the owner can make it public.",
    ]):
        add_text(s, "•  " + line, 0.98, 4.88 + i * 0.22, 5.65, 0.22,
                 size=7.9, color=MUTED, line_spacing=1.0)

    add_rect(s, 0.82, 5.94, 6.05, 0.75, fill=PANEL_2, line=LINE)
    add_text(s, "NEXT", 0.98, 6.03, 1.0, 0.16, size=6.8, color=ACCENT, bold=True)
    add_text(s, "Validate across more regions · add downstream detection · "
                "publish one guided real-tile before/after.",
             0.98, 6.20, 5.65, 0.36, size=7.9, color=MUTED, line_spacing=1.02)

    add_figure(
        s,
        ROOT / "screenshots" / "lunapsr_diagnostic_maps.png",
        7.18, 2.04, 5.42, 3.80,
        caption="Diagnostic maps: local model-based confidence, modeled noise, and adaptive strength.",
        alt="LunaPSR diagnostic maps: local model-based confidence, modeled noise, and adaptive strength.",
        caption_h=0.52,
    )
    add_text(s, "The goal is not a prettier image. It is a more trustworthy analysis.",
             7.18, 6.08, 5.42, 0.42, size=11.0, color=TEXT, bold=True,
             line_spacing=1.0, align=PP_ALIGN.RIGHT)
    add_footer(s, "github.com/Mevis-byte/LunaPSR  ·  README §10, §14",
               "DERIVED REPRESENTATION  ·  NOT RECOVERED GEOLOGY")

    s.notes_slide.notes_text_frame.text = (
        "Close on use and status. The project is not claiming a finished product: Phase 5C is complete, "
        "Phase 5B is still a prototype, and the live demo is private. The ask is to try it and "
        "report where it fails."
    )

    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    out = build()
    print(out)
    print(f"{out.stat().st_size / 1e6:.2f} MB")
