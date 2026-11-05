# LunaPSR — 6-slide presentation

A reader-friendly overview of what LunaPSR does, what problem it solves, and which
features matter. The deck is deliberately honest: it includes the Phase 5C result
where raw data wins overall and where LunaPSR only has a narrow, size-dependent win.

## Files

| File | Purpose |
|---|---|
| `lunapsr_6slide.pptx` | **Primary PowerPoint deck.** 16:9, 6 slides, images fit inside their boxes. |
| `lunapsr_6slide.html` | Browser-based 16:9 deck. Arrow keys / space navigate, `F` fullscreen. |
| `lunapsr_6slide.pdf` | 6-page 16:9 PDF export, generated from the HTML with Chromium. |
| `speaker_notes.md` | Slide-by-slide script, timing, Q&A guardrails, demo choreography. |
| `build_pptx.py` | Rebuilds the PowerPoint from the source images and copy. |

## The six slides

1. **Title / what it is** — one-sentence description plus what it is, what it does, and why it matters.
2. **Problems it solves** — weak signal, noise, false detail, missing ground truth, and no audit trail.
3. **What LunaPSR does + features** — the read-only pipeline and six practical capabilities.
4. **How it works** — noise-aware enhancement and why the noise model is the gate.
5. **Evidence** — Phase 5C ground truth, including where LunaPSR loses.
6. **What you can do** — inspect, compare, diagnose, export, and the current status.

## Presenting

Open `lunapsr_6slide.pptx` in PowerPoint, or open `lunapsr_6slide.html` in a browser and use:

- **← / →** or **Space** to change slides
- **F** to enter/exit fullscreen
- **Ctrl/Cmd+P** to export a PDF

The PPTX embeds short speaker notes on every slide.

## Image sources

The deck uses all four screenshots embedded in the README:

- `screenshots/adaptive_retinex_maps_confidence_alpha_noise.png`
- `screenshots/adaptive_retinex_section.png`
- `screenshots/adaptive_retinex_maps_illumination_response_structure.png`
- `screenshots/lunapsr_diagnostic_maps.png`

It also uses the OHRC browse strip as the title background and the Phase 5C
boulder curve — plus the 12 px boulder montage in the PPTX — as evidence figures.

## Important

The deck uses relative image paths. Keep the `presentation/` folder in place
inside the repository, or open the HTML from its current location. The source
OHRC product is never modified; all enhanced imagery is a derived
representation. The noise estimate is model-based, not measured from the scene.
